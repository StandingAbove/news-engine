"""Vanilla TimesFM 2 walk-forward baseline.

Part 4 of the news-conditioned plan. Singh's framing was: "TimesFM 2 can run
in the background … from last year or five years ago … and you can see on a
day-to-day basis how well it is tracking." That's the baseline — no news,
just TimesFM 2 fed the price history strictly before "today" and asked to
forecast the next ``H`` steps. We replay this day-by-day across the holdout
window and log per-day forecasts + errors.

The conditioned model (Parts 5–7) plugs into exactly this loop, swapping the
forecast call for a news-conditioned one. So this file's responsibilities
are:

1. Provide a uniform :class:`Forecaster` adapter around TimesFM 2 (and a
   trivial naive-last fallback for sanity-checking the harness without the
   foundation model loaded).
2. A ``replay()`` walk-forward loop that consumes
   ``backend.conditioning.alignment.AlignedSample`` and produces a
   :class:`ReplayResult` with per-sample forecasts, errors, and a few
   summary metrics.

This module imports from ``backend.forecast.models.timesfm_model`` because
that's where Sam already wrapped the TimesFM 2.5 main-branch API. Re-using
that wrapper is intentional — it keeps Singh's "decoupled sub-component A"
genuinely decoupled (forecast/ doesn't import here; we import from it).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .alignment import AlignedSample

log = logging.getLogger(__name__)


# --- forecaster protocol ------------------------------------------------------
class Forecaster(Protocol):
    """Anything that can take a 1-D float array of past closes and return an
    H-step point forecast (close prices, not returns)."""

    name: str

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray: ...


# --- naive fallback -----------------------------------------------------------
@dataclass
class NaiveLastForecaster:
    """Repeats the last observed close H times. Useful as a sanity floor that
    the conditioned model has to clear before we believe any improvement."""
    name: str = "naive_last"

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, float(history[-1]), dtype=np.float32)


# --- TimesFM 2 wrapper --------------------------------------------------------
@dataclass
class TimesFMBaseline:
    """Lazy wrapper around the TimesFM 2.5 forecaster Sam already wired in
    ``backend.forecast.models.timesfm_model``.

    The first call loads the checkpoint (~900 MB). Subsequent calls reuse it
    via the underlying model's per-process cache. If the dependency isn't
    installed or the checkpoint can't be fetched, instantiation will raise —
    callers can catch that and fall back to ``NaiveLastForecaster``.
    """
    name: str = "timesfm_2.5_vanilla"
    context_len: int = 512

    def __post_init__(self) -> None:
        # Lazy-import so this module is still importable in environments
        # without the timesfm dependency.
        from backend.forecast.models.timesfm_model import TimesFMForecaster
        self._impl = TimesFMForecaster(context_len=self.context_len)
        # Probe availability up-front so a bad config fails fast.
        if hasattr(self._impl, "is_available") and not self._impl.is_available():
            raise RuntimeError("TimesFM dependency unavailable on this machine.")

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        # Delegate to the existing per-model API. fit() is a no-op for
        # TimesFM (foundation model) but we call it to keep the protocol
        # uniform with the other forecasters in backend/forecast/.
        self._impl.fit(history.astype(np.float32))
        out = self._impl.forecast(int(horizon))
        # The wrapper returns a Forecast object with .point; defend against
        # it returning a bare ndarray too.
        point = getattr(out, "point", out)
        return np.asarray(point[:horizon], dtype=np.float32)


# --- replay result ------------------------------------------------------------
@dataclass
class ReplayResult:
    forecaster: str
    horizons: int
    n_samples: int
    t: np.ndarray                  # (N,)   int64 ns
    forecast: np.ndarray           # (N, H) close-price forecasts
    target: np.ndarray             # (N, H) actual closes (excludes the t=0 seed price)
    forecast_returns: np.ndarray   # (N, H) close-to-close forecast returns
    target_returns: np.ndarray     # (N, H) actual close-to-close returns
    elapsed_s: float

    # --- derived metrics ------------------------------------------------------
    def per_sample_mae(self, in_returns: bool = True) -> np.ndarray:
        a, b = (self.forecast_returns, self.target_returns) if in_returns \
               else (self.forecast, self.target)
        return np.mean(np.abs(a - b), axis=1)

    def per_sample_rmse(self, in_returns: bool = True) -> np.ndarray:
        a, b = (self.forecast_returns, self.target_returns) if in_returns \
               else (self.forecast, self.target)
        return np.sqrt(np.mean((a - b) ** 2, axis=1))

    def per_sample_dir_acc(self) -> np.ndarray:
        """Directional accuracy of the forecast returns vs the actual."""
        sign_match = np.sign(self.forecast_returns) == np.sign(self.target_returns)
        return sign_match.mean(axis=1)

    def summary(self) -> dict:
        return {
            "forecaster": self.forecaster,
            "n_samples": self.n_samples,
            "horizon": self.horizons,
            "elapsed_s": round(self.elapsed_s, 2),
            "mae_returns_mean": float(self.per_sample_mae(True).mean()),
            "rmse_returns_mean": float(self.per_sample_rmse(True).mean()),
            "mae_price_mean": float(self.per_sample_mae(False).mean()),
            "dir_acc_mean": float(self.per_sample_dir_acc().mean()),
        }


# --- core walk-forward loop ---------------------------------------------------
def replay(
    samples: Sequence[AlignedSample],
    forecaster: Forecaster,
    *,
    log_every: int = 25,
) -> ReplayResult:
    """Replay ``forecaster`` over a sequence of :class:`AlignedSample`.

    On each sample we feed ``sample.history`` into ``forecaster.forecast`` and
    compare the resulting H-step forecast against ``sample.target[1:]`` (the
    actual closes after t). Errors are reported in two units: price space
    (raw $ MAE / RMSE) and return space (close-to-close), the latter being
    the comparable thing across regimes.

    The samples are not assumed to share a horizon — we use the configured H
    from the *first* sample and validate the rest match.
    """
    if not samples:
        raise ValueError("replay() requires at least one sample")
    H = samples[0].forward_returns.size
    for s in samples:
        if s.forward_returns.size != H:
            raise ValueError(
                f"inconsistent horizon: sample @ {s.t.date()} has H={s.forward_returns.size}, expected {H}"
            )

    N = len(samples)
    fc = np.zeros((N, H), dtype=np.float32)
    tgt = np.zeros((N, H), dtype=np.float32)
    fc_ret = np.zeros((N, H), dtype=np.float32)
    tgt_ret = np.zeros((N, H), dtype=np.float32)
    t_arr = np.zeros(N, dtype=np.int64)

    t0 = time.time()
    for i, s in enumerate(samples):
        t_arr[i] = s.t.value
        seed = float(s.target[0])  # close on day t (the seed price)
        fc_path = forecaster.forecast(s.history, H)
        if fc_path.size != H:
            raise RuntimeError(
                f"forecaster '{getattr(forecaster, 'name', '?')}' returned "
                f"{fc_path.size} steps, expected {H}"
            )
        fc[i] = fc_path
        tgt[i] = s.target[1:]
        # close-to-close returns. For step k we compare:
        #   forecast: fc[k] / (fc[k-1] if k>0 else seed) - 1
        #   target  : tgt[k] / (tgt[k-1] if k>0 else seed) - 1
        fc_prev = np.concatenate([[seed], fc_path[:-1]])
        tgt_prev = np.concatenate([[seed], tgt[i, :-1]])
        fc_ret[i] = fc_path / fc_prev - 1.0
        tgt_ret[i] = tgt[i] / tgt_prev - 1.0

        if log_every and (i + 1) % log_every == 0:
            log.info("replay %4d / %d  (%s)", i + 1, N, getattr(forecaster, "name", "?"))

    elapsed = time.time() - t0
    return ReplayResult(
        forecaster=getattr(forecaster, "name", forecaster.__class__.__name__),
        horizons=H,
        n_samples=N,
        t=t_arr,
        forecast=fc,
        target=tgt,
        forecast_returns=fc_ret,
        target_returns=tgt_ret,
        elapsed_s=elapsed,
    )
