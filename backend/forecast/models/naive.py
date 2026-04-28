"""Naive baselines.

`NaiveLast`, `NaiveMean`, `NaiveDrift` — these exist to be beaten. If a
foundation model can't beat naive-last on a sine wave, something is wrong
upstream.
"""
from __future__ import annotations

import numpy as np

from ..base import Forecast, as_array


class NaiveLast:
    """Forecast the last observed value for every horizon step."""

    name = "naive_last"

    def __init__(self) -> None:
        self._last: float = 0.0

    @staticmethod
    def is_available() -> bool:
        return True

    def fit(self, history: np.ndarray) -> None:
        h = as_array(history)
        self._last = float(h[-1])

    def forecast(self, horizon: int) -> Forecast:
        return Forecast(point=np.full(horizon, self._last, dtype=np.float64))


class NaiveMean:
    """Forecast the mean of `history` for every step."""

    name = "naive_mean"

    def __init__(self) -> None:
        self._mu: float = 0.0

    @staticmethod
    def is_available() -> bool:
        return True

    def fit(self, history: np.ndarray) -> None:
        h = as_array(history)
        self._mu = float(np.mean(h))

    def forecast(self, horizon: int) -> Forecast:
        return Forecast(point=np.full(horizon, self._mu, dtype=np.float64))


class NaiveDrift:
    """Linear extrapolation from the first to the last observed value.

    Equivalent to Hyndman's "drift" method: slope = (h[-1] - h[0]) / (n - 1).
    Gives a baseline for trending series — beats `NaiveLast` on obvious trends
    and should lose to it on mean-reverting noise.
    """

    name = "naive_drift"

    def __init__(self) -> None:
        self._last: float = 0.0
        self._slope: float = 0.0

    @staticmethod
    def is_available() -> bool:
        return True

    def fit(self, history: np.ndarray) -> None:
        h = as_array(history)
        self._last = float(h[-1])
        if len(h) <= 1:
            self._slope = 0.0
        else:
            self._slope = float((h[-1] - h[0]) / (len(h) - 1))

    def forecast(self, horizon: int) -> Forecast:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return Forecast(point=self._last + self._slope * steps)
