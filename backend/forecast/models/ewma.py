"""Exponential smoothing baselines.

`EWMAForecaster` — simple exponentially-weighted moving average. Pure numpy,
no external deps; survives on a bare install.

`HoltWintersForecaster` — Holt / Holt-Winters via `statsmodels`. Captures
level + trend (and optionally seasonality). Gracefully reports unavailable
if statsmodels isn't installed.
"""
from __future__ import annotations

import numpy as np

from ..base import Forecast, as_array


class EWMAForecaster:
    """Classic EWMA. `alpha` is the smoothing factor in (0, 1]."""

    name = "ewma"

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._level: float = 0.0

    @staticmethod
    def is_available() -> bool:
        return True

    def fit(self, history: np.ndarray) -> None:
        h = as_array(history)
        a = self.alpha
        level = float(h[0])
        for x in h[1:]:
            level = a * float(x) + (1 - a) * level
        self._level = level

    def forecast(self, horizon: int) -> Forecast:
        # EWMA is a flat forecast at the smoothed level.
        return Forecast(point=np.full(horizon, self._level, dtype=np.float64))


class HoltWintersForecaster:
    """Holt (trend) or Holt-Winters (trend + seasonality) via statsmodels.

    Parameters
    ----------
    trend : {"add", "mul", None}
        Trend component. "add" is a good default for price data.
    seasonal : {"add", "mul", None}
    seasonal_periods : int
        Only used when `seasonal` is set.
    """

    name = "holt_winters"

    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fit = None

    @staticmethod
    def is_available() -> bool:
        try:
            import statsmodels  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, history: np.ndarray) -> None:
        if not self.is_available():
            raise RuntimeError("statsmodels not installed")
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        h = as_array(history)
        # Use multiplicative only if all values are strictly positive.
        trend = self.trend
        seasonal = self.seasonal
        if (trend == "mul" or seasonal == "mul") and np.any(h <= 0):
            trend = "add" if trend == "mul" else trend
            seasonal = "add" if seasonal == "mul" else seasonal

        model = ExponentialSmoothing(
            h,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        # `disp` is a SARIMAX/ARIMA-only kwarg — ExponentialSmoothing takes
        # only `optimized` (and a few method-specific kwargs). Keep this
        # call minimal so it survives across statsmodels versions.
        self._fit = model.fit(optimized=True)

    def forecast(self, horizon: int) -> Forecast:
        if self._fit is None:
            raise RuntimeError("fit() must be called first")
        yhat = np.asarray(self._fit.forecast(horizon), dtype=np.float64)
        return Forecast(point=yhat)
