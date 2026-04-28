"""Forecasting model implementations.

Every model here is a drop-in implementation of `backend.forecast.base.Forecaster`.
Heavy deps (statsmodels, torch, timesfm) are imported lazily inside each
class so the package itself can always be imported — the harness asks each
model `is_available()` before running it, and prints a clean skip message
when a dep is missing.

Use `list_available()` to see which models the current environment supports,
or `list_all()` for the full roster regardless of install state.
"""
from __future__ import annotations

from typing import Dict, List, Type

from .naive import NaiveLast, NaiveMean, NaiveDrift
from .ewma import EWMAForecaster, HoltWintersForecaster
from .arima import ARIMAForecaster
from .mlp import MLPForecaster
from .lstm import LSTMForecaster
from .timesfm_model import TimesFMForecaster


ALL_MODELS: Dict[str, Type] = {
    "naive_last":    NaiveLast,
    "naive_mean":    NaiveMean,
    "naive_drift":   NaiveDrift,
    "ewma":          EWMAForecaster,
    "holt_winters":  HoltWintersForecaster,
    "arima":         ARIMAForecaster,
    "mlp":           MLPForecaster,
    "lstm":          LSTMForecaster,
    "timesfm":       TimesFMForecaster,
}


def list_all() -> List[str]:
    return list(ALL_MODELS.keys())


def list_available() -> List[str]:
    """Return the subset of model names whose deps are importable right now."""
    out = []
    for name, cls in ALL_MODELS.items():
        try:
            available = getattr(cls, "is_available", None)
            if available is None or available():
                out.append(name)
        except Exception:  # noqa: BLE001
            continue
    return out


def build(name: str, **kwargs):
    """Instantiate a model by name."""
    if name not in ALL_MODELS:
        raise KeyError(f"unknown forecaster: {name!r} (known: {list_all()})")
    return ALL_MODELS[name](**kwargs)


__all__ = [
    "ALL_MODELS",
    "list_all",
    "list_available",
    "build",
    "NaiveLast",
    "NaiveMean",
    "NaiveDrift",
    "EWMAForecaster",
    "HoltWintersForecaster",
    "ARIMAForecaster",
    "MLPForecaster",
    "LSTMForecaster",
    "TimesFMForecaster",
]
