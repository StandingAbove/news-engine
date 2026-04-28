"""Forecast-panel API routes.

Surfaces sub-component A (`backend/forecast/`) through the dashboard.
This is a one-way consumption — the API layer imports from forecast,
but forecast never imports anything in v2, which keeps Dr. Singh's
sub-component decoupling intact.

Endpoints
---------
* ``GET /api/forecast/models`` — list of all 9 models with availability
  and a one-line description. The frontend uses this to build the
  model-select checkbox group and grey out unavailable models.
* ``POST /api/forecast/run`` — run a chosen set of models on a chosen
  ticker. Splits the ticker's history into a fit portion + a held-out
  tail of ``eval_tail`` points, fits each model on the fit portion,
  forecasts ``horizon`` points, and (when the forecast overlaps the
  tail) reports MAE/RMSE/dir_acc on the overlap. Returns the history
  array plus per-model forecasts, confidence bands, and metrics.
* ``GET /api/forecast/toy-eval`` — runs the toy harness on the
  deterministic series catalogue and returns the leaderboard (per-series
  + overall) as JSON, in the same shape the CLI prints.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..forecast import metrics as mx
from ..forecast import models as M
from ..forecast import toy
from ..forecast.harness import evaluate
from ..forecast.real_data import load_close_series

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/forecast", tags=["forecast"])


# --- model descriptions -------------------------------------------------------
# Short one-liners surfaced in the UI. Keeping them here (not in the forecast
# package) so backend/forecast/ doesn't need to know about dashboard concerns.
_MODEL_META: Dict[str, Dict[str, str]] = {
    "naive_last":   {"family": "baseline",        "blurb": "Repeat last observed value — random-walk champion."},
    "naive_mean":   {"family": "baseline",        "blurb": "Mean of history — mean-reversion champion."},
    "naive_drift":  {"family": "baseline",        "blurb": "Linear extrapolation from endpoints."},
    "ewma":         {"family": "smoothing",       "blurb": "Exponentially weighted moving average."},
    "holt_winters": {"family": "smoothing",       "blurb": "Holt-Winters: level + trend + optional seasonality."},
    "arima":        {"family": "classical",       "blurb": "Auto-order ARIMA(p,d,q) selected by AIC."},
    "mlp":          {"family": "neural",          "blurb": "Small MLP on a rolling lookback window."},
    "lstm":         {"family": "neural",          "blurb": "Single-layer LSTM on a rolling lookback window."},
    "timesfm":      {"family": "foundation",      "blurb": "Google TimesFM — decoder-only time-series foundation model."},
}


# --- /api/forecast/models ----------------------------------------------------
@router.get("/models")
def list_models() -> List[Dict[str, Any]]:
    """Return every model in the roster with its availability flag."""
    out: List[Dict[str, Any]] = []
    for name in M.list_all():
        cls = M.ALL_MODELS[name]
        avail = getattr(cls, "is_available", lambda: True)
        try:
            is_avail = bool(avail())
        except Exception:  # noqa: BLE001
            is_avail = False
        meta = _MODEL_META.get(name, {"family": "?", "blurb": ""})
        out.append({
            "name": name,
            "family": meta["family"],
            "blurb": meta["blurb"],
            "is_available": is_avail,
        })
    return out


# --- /api/forecast/universe ---------------------------------------------------
@router.get("/universe")
def universe() -> Dict[str, Any]:
    """Expose the configured house universe for the ticker quick-pick."""
    s = get_settings()
    return {
        "universe": s.house_universe,
        "benchmark": s.benchmark_ticker,
    }


# --- /api/forecast/run --------------------------------------------------------
class RunRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. SPY")
    models: List[str] = Field(default_factory=list, description="Model names to run")
    horizon: int = Field(32, ge=1, le=252, description="Forecast length in trading days")
    eval_tail: int = Field(
        32, ge=0, le=252,
        description="Held-out tail length for scoring. 0 = pure forward forecast, no metrics.",
    )
    period: str = Field("2y", description="yfinance history window")


@router.post("/run")
def run_forecast(req: RunRequest) -> Dict[str, Any]:
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker required")

    if not req.models:
        raise HTTPException(400, "at least one model required")
    unknown = [m for m in req.models if m not in M.ALL_MODELS]
    if unknown:
        raise HTTPException(400, f"unknown model(s): {unknown}")

    # --- fetch history ---
    try:
        closes = load_close_series(ticker, period=req.period)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not load {ticker}: {e}") from e

    history = closes.astype(float)
    n = int(history.size)

    eval_tail = min(int(req.eval_tail), max(0, n - 30))  # leave ≥30 pts for fit
    horizon = int(req.horizon)
    fit_len = n - eval_tail

    if fit_len < 30:
        raise HTTPException(
            400,
            f"history too short for eval_tail={eval_tail} "
            f"(n={n}, fit_len={fit_len}); shorten eval_tail or pick a longer period",
        )

    train = history[:fit_len]
    held_out = history[fit_len:] if eval_tail > 0 else np.array([], dtype=float)

    # --- run each model ---
    per_model: List[Dict[str, Any]] = []
    for name in req.models:
        cls = M.ALL_MODELS[name]
        is_avail = True
        try:
            is_avail = bool(getattr(cls, "is_available", lambda: True)())
        except Exception:  # noqa: BLE001
            is_avail = False

        row: Dict[str, Any] = {
            "model": name,
            "family": _MODEL_META.get(name, {}).get("family", "?"),
            "status": "ok",
        }
        if not is_avail:
            row["status"] = "skipped"
            row["error"] = "dependency unavailable"
            per_model.append(row)
            continue

        try:
            model = cls()
            model.fit(train)
            fc = model.forecast(horizon)
        except Exception as e:  # noqa: BLE001
            log.exception("forecast model %s failed on %s", name, ticker)
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            per_model.append(row)
            continue

        point = _sanitize(fc.point[:horizon])
        row["point"] = point.tolist()
        if fc.lower is not None and fc.upper is not None:
            lower = _sanitize(np.asarray(fc.lower[:horizon], dtype=float))
            upper = _sanitize(np.asarray(fc.upper[:horizon], dtype=float))
            row["lower"] = lower.tolist()
            row["upper"] = upper.tolist()

        # Metrics on the overlap between the forecast and the held-out tail.
        if eval_tail > 0 and point.size:
            overlap = min(eval_tail, point.size)
            truth = held_out[:overlap]
            pred = point[:overlap]
            metrics = mx.summarize_forecast(truth, pred, train)
            row["metrics"] = _sanitize_dict(metrics)
        per_model.append(row)

    return {
        "ticker": ticker,
        "period": req.period,
        "history_len": n,
        "fit_len": fit_len,
        "eval_tail": eval_tail,
        "horizon": horizon,
        "history": history.tolist(),
        "forecasts": per_model,
    }


# --- /api/forecast/toy-eval ---------------------------------------------------
@router.get("/toy-eval")
def toy_eval(
    horizon: int = 32,
    n: int = 400,
    models: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the deterministic toy series against every available model.

    Returns the leaderboard in JSON form so the frontend can render the same
    tables the CLI prints.
    """
    horizon = max(1, min(252, int(horizon)))
    n = max(100, min(2000, int(n)))

    model_names = [m.strip() for m in models.split(",")] if models else M.list_all()
    unknown = [m for m in model_names if m not in M.ALL_MODELS]
    if unknown:
        raise HTTPException(400, f"unknown model(s): {unknown}")

    series = toy.all_series(n)
    df = evaluate(series, model_names, horizon=horizon)

    # Per-series breakdown — sorted by MAE among OK rows, with skipped/error
    # preserved as a separate list.
    per_series: List[Dict[str, Any]] = []
    for sname in series.keys():
        grp = df[df["series"] == sname]
        ok_rows = grp[grp["status"] == "ok"].sort_values("mae")
        ok_rows_json = [
            {
                "model": r["model"],
                "mae": _finite(r["mae"]),
                "rmse": _finite(r["rmse"]),
                "mape": _finite(r["mape"]),
                "mase": _finite(r["mase"]),
                "dir_acc": _finite(r["dir_acc"]),
                "elapsed_s": _finite(r["elapsed_s"]),
            }
            for _, r in ok_rows.iterrows()
        ]
        skipped_rows_json = [
            {"model": r["model"], "status": r["status"], "detail": r.get("detail", "")}
            for _, r in grp[grp["status"] != "ok"].iterrows()
        ]
        per_series.append({
            "series": sname,
            "ok": ok_rows_json,
            "skipped": skipped_rows_json,
        })

    # Overall — mean of each metric across OK rows, sorted by MAE.
    ok = df[df["status"] == "ok"]
    if ok.empty:
        overall: List[Dict[str, Any]] = []
    else:
        agg = ok.groupby("model")[["mae", "rmse", "mape", "mase", "dir_acc"]].mean()
        agg = agg.sort_values("mae")
        overall = [
            {
                "model": mname,
                "mae": _finite(row["mae"]),
                "rmse": _finite(row["rmse"]),
                "mape": _finite(row["mape"]),
                "mase": _finite(row["mase"]),
                "dir_acc": _finite(row["dir_acc"]),
            }
            for mname, row in agg.iterrows()
        ]

    return {
        "horizon": horizon,
        "n": n,
        "models": model_names,
        "series": list(series.keys()),
        "per_series": per_series,
        "overall": overall,
    }


# --- helpers ------------------------------------------------------------------
def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0.0 so the JSON encoder doesn't choke."""
    out = np.asarray(arr, dtype=float).copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = 0.0
    return out


def _sanitize_dict(d: Dict[str, float]) -> Dict[str, Optional[float]]:
    return {k: _finite(v) for k, v in d.items()}


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f
