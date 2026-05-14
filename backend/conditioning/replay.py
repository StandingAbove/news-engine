"""Walk-forward evaluation of the news-conditioned forecaster (Part 7).

Mirrors :func:`backend.conditioning.baseline.replay` but injects the trained
adapter between the vanilla forecast and the comparison against actuals.

For each holdout sample t we:

1. Get the vanilla return path from the frozen forecaster (TimesFM 2).
2. Embed the news in [t - K_news, t) using the configured ``NewsEncoder``.
3. Compute the history features at t.
4. Run the adapter to get the H-step return adjustment.
5. Form the conditioned price path by compounding (vanilla + adjustment) from
   the seed price ``sample.target[0]``.
6. Score it against ``sample.target[1:]`` in both price and return space.

This is the function that produces the head-to-head comparison Singh wants:
"how well it is tracking on a day-to-day basis." The output is a
:class:`baseline.ReplayResult`, same shape as the vanilla one, so Part 8 can
plot the two on the same axes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False
    torch = None  # type: ignore

from .adapter import AdapterConfig, ConditioningAdapter, history_features
from .alignment import AlignedSample
from .baseline import Forecaster, ReplayResult
from .embed import NewsEncoder

log = logging.getLogger(__name__)


@dataclass
class ConditionedForecaster:
    """Wraps a vanilla forecaster + a trained adapter + an encoder behind a
    single ``forecast(history, horizon)``-style call. *Not* a drop-in for the
    plain :class:`Forecaster` protocol because the adapter needs the news +
    history features alongside the price history. We provide a richer call
    instead, used by :func:`replay_conditioned` below.
    """
    name: str
    vanilla: Forecaster
    adapter: "ConditioningAdapter"      # torch module
    encoder: NewsEncoder
    device: str = "cpu"

    def forecast_path(
        self,
        sample: AlignedSample,
        horizon: int,
    ) -> np.ndarray:
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not available")
        seed = float(sample.target[0])
        vanilla_path = self.vanilla.forecast(sample.history, horizon)
        prev = np.concatenate([[seed], vanilla_path[:-1]])
        vanilla_ret = vanilla_path / prev - 1.0

        news_emb = self.encoder.encode(sample.news, sample.t.to_pydatetime())
        hist_feat = history_features(sample.history)

        with torch.no_grad():
            v = torch.from_numpy(vanilla_ret.astype(np.float32)).unsqueeze(0).to(self.device)
            n = torch.from_numpy(news_emb.astype(np.float32)).unsqueeze(0).to(self.device)
            h = torch.from_numpy(hist_feat.astype(np.float32)).unsqueeze(0).to(self.device)
            adjust = self.adapter(v, n, h).squeeze(0).cpu().numpy()

        cond_ret = vanilla_ret + adjust
        growth = np.cumprod(1.0 + cond_ret)
        return (seed * growth).astype(np.float32)


# --- replay loop --------------------------------------------------------------
def replay_conditioned(
    samples: Sequence[AlignedSample],
    forecaster: ConditionedForecaster,
    *,
    log_every: int = 25,
) -> ReplayResult:
    """Walk-forward replay using the adapter-conditioned forecaster.

    Returns a :class:`ReplayResult` with the same shape as
    :func:`backend.conditioning.baseline.replay`, so the two can be diffed.
    """
    if not samples:
        raise ValueError("replay_conditioned() requires at least one sample")
    H = samples[0].forward_returns.size
    for s in samples:
        if s.forward_returns.size != H:
            raise ValueError(
                f"inconsistent horizon: {s.t.date()} has H={s.forward_returns.size}, expected {H}"
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
        seed = float(s.target[0])
        path = forecaster.forecast_path(s, H)
        fc[i] = path
        tgt[i] = s.target[1:]
        prev = np.concatenate([[seed], path[:-1]])
        fc_ret[i] = path / prev - 1.0
        prev_t = np.concatenate([[seed], tgt[i, :-1]])
        tgt_ret[i] = tgt[i] / prev_t - 1.0
        if log_every and (i + 1) % log_every == 0:
            log.info("conditioned replay %4d / %d  (%s)", i + 1, N, forecaster.name)

    elapsed = time.time() - t0
    return ReplayResult(
        forecaster=forecaster.name,
        horizons=H,
        n_samples=N,
        t=t_arr,
        forecast=fc,
        target=tgt,
        forecast_returns=fc_ret,
        target_returns=tgt_ret,
        elapsed_s=elapsed,
    )


# --- comparison helper --------------------------------------------------------
@dataclass
class CompareResult:
    """Side-by-side metrics for a vanilla and a conditioned replay.

    ``per_day`` arrays are length-N (one row per replay day) so Part 8 can
    plot the rolling difference. ``mae_*`` and ``dir_acc_*`` are scalars.
    """
    n_samples: int
    horizon: int
    mae_vanilla: float
    mae_conditioned: float
    mae_delta: float
    rmse_vanilla: float
    rmse_conditioned: float
    rmse_delta: float
    dir_acc_vanilla: float
    dir_acc_conditioned: float
    dir_acc_delta: float
    win_rate: float          # share of days where conditioned MAE < vanilla MAE
    per_day_mae_vanilla: np.ndarray
    per_day_mae_conditioned: np.ndarray
    per_day_t: np.ndarray

    def to_text(self) -> str:
        return (
            f"n={self.n_samples}  H={self.horizon}\n"
            f"            vanilla     conditioned    Δ\n"
            f"MAE (ret)   {self.mae_vanilla:.5f}    {self.mae_conditioned:.5f}    {self.mae_delta:+.5f}\n"
            f"RMSE (ret)  {self.rmse_vanilla:.5f}    {self.rmse_conditioned:.5f}    {self.rmse_delta:+.5f}\n"
            f"Dir-acc     {self.dir_acc_vanilla:.4f}     {self.dir_acc_conditioned:.4f}     {self.dir_acc_delta:+.4f}\n"
            f"win rate (cond beats vanilla) : {self.win_rate:.2%}"
        )


def compare(vanilla: ReplayResult, conditioned: ReplayResult) -> CompareResult:
    if vanilla.n_samples != conditioned.n_samples:
        raise ValueError("compare(): replays have different sample counts")
    if not np.array_equal(vanilla.t, conditioned.t):
        raise ValueError("compare(): replays cover different timestamps")
    if vanilla.horizons != conditioned.horizons:
        raise ValueError("compare(): horizon mismatch")

    mae_v = vanilla.per_sample_mae(True)
    mae_c = conditioned.per_sample_mae(True)
    rmse_v = vanilla.per_sample_rmse(True)
    rmse_c = conditioned.per_sample_rmse(True)
    da_v = vanilla.per_sample_dir_acc()
    da_c = conditioned.per_sample_dir_acc()

    return CompareResult(
        n_samples=vanilla.n_samples,
        horizon=vanilla.horizons,
        mae_vanilla=float(mae_v.mean()),
        mae_conditioned=float(mae_c.mean()),
        mae_delta=float(mae_c.mean() - mae_v.mean()),
        rmse_vanilla=float(rmse_v.mean()),
        rmse_conditioned=float(rmse_c.mean()),
        rmse_delta=float(rmse_c.mean() - rmse_v.mean()),
        dir_acc_vanilla=float(da_v.mean()),
        dir_acc_conditioned=float(da_c.mean()),
        dir_acc_delta=float(da_c.mean() - da_v.mean()),
        win_rate=float((mae_c < mae_v).mean()),
        per_day_mae_vanilla=mae_v,
        per_day_mae_conditioned=mae_c,
        per_day_t=vanilla.t,
    )
