"""Point-forecast metrics.

Standard regression scores plus two forecast-specific ones (MASE and
directional accuracy). Everything is zero-guarded so the harness doesn't
explode on degenerate toy series.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

_EPS = 1e-12


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if t.shape != p.shape:
        n = min(len(t), len(p))
        t, p = t[:n], p[:n]
    mask = np.isfinite(t) & np.isfinite(p)
    return t[mask], p[mask]


def mae(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error.

    Undefined when any true value is zero, so we skip those points.
    Returns a fraction (e.g. 0.05 == 5%), not a percentage — callers
    format for display.
    """
    t, p = _pair(y_true, y_pred)
    mask = np.abs(t) > _EPS
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])))


def mase(y_true, y_pred, history: np.ndarray, season: int = 1) -> float:
    """Mean Absolute Scaled Error.

    Normalizes MAE by the in-sample naive-forecast error. A MASE < 1 means
    the model beats a seasonal-naive baseline on the training history.
    """
    t, p = _pair(y_true, y_pred)
    h = np.asarray(history, dtype=np.float64).reshape(-1)
    if h.size <= season:
        return float("nan")
    naive_err = np.mean(np.abs(h[season:] - h[:-season]))
    if naive_err < _EPS:
        return float("nan")
    return float(np.mean(np.abs(t - p)) / naive_err)


def directional_accuracy(y_true, y_pred, last_observed: float) -> float:
    """Fraction of forecast steps whose *direction* vs the last observed
    value matches the true direction. For a purely point-forecasting model
    this is a soft indicator of whether it's picking up trend at all.
    """
    t, p = _pair(y_true, y_pred)
    if t.size == 0:
        return float("nan")
    true_sign = np.sign(t - last_observed)
    pred_sign = np.sign(p - last_observed)
    # Treat zero as a match only if both are zero.
    matches = (true_sign == pred_sign) | ((true_sign == 0) & (pred_sign == 0))
    return float(np.mean(matches))


def summarize_forecast(
    y_true,
    y_pred,
    history: np.ndarray,
    *,
    season: int = 1,
) -> Dict[str, float]:
    """Roll up the full metric set for one model on one evaluation window."""
    last = float(np.asarray(history, dtype=np.float64).reshape(-1)[-1])
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mase": mase(y_true, y_pred, history, season=season),
        "dir_acc": directional_accuracy(y_true, y_pred, last),
    }
