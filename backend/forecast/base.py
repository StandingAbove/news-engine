"""Shared `Forecaster` interface.

Every model in `backend.forecast.models` implements this Protocol so the
harness can swap them in and out. Models are evaluated on the *same* series
so results are directly comparable.

Conventions
-----------
* Series are 1-D numpy arrays of floats.
* `fit(history)` is allowed to no-op (e.g. a naive model); models that need
  training should train here and persist any state on `self`.
* `forecast(horizon)` returns a length-`horizon` array of point predictions,
  starting one step *after* the end of `history`.
* Models never see the future. The harness is responsible for giving each
  model exactly the same `history` window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Forecast:
    """Prediction bundle returned from `Forecaster.forecast`.

    Only `point` is required. `lower` / `upper` are optional prediction
    intervals (80% by convention); leave them as `None` for point-only models.
    """
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


@runtime_checkable
class Forecaster(Protocol):
    """Uniform interface every model implements."""

    name: str

    def fit(self, history: np.ndarray) -> None: ...

    def forecast(self, horizon: int) -> Forecast: ...


def as_array(x) -> np.ndarray:
    """Coerce list / Series / ndarray to a 1-D float64 array."""
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty series")
    if not np.all(np.isfinite(arr)):
        # Forward-fill non-finite values — toy tests occasionally include
        # a NaN to stress-test robustness, and real price data has holes.
        mask = np.isfinite(arr)
        if not mask.any():
            raise ValueError("series has no finite values")
        idx = np.where(mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]
    return arr
