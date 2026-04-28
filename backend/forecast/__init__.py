"""Forecasting sub-component.

Dr. Singh's Feb-6 guidance, paraphrased:

> Two sub-components, built and tested *separately* on toy data first:
>   (a) a set of forecasting models (TimesFM + a few others),
>   (b) a text-feed synthesizer.
> They don't need to talk right now. Let models fail and/or establish some
> minimum performance.

This package is sub-component (a): a uniform `Forecaster` interface plus
several implementations (naive/drift, ARIMA, EWMA/Holt-Winters, a small MLP,
a small LSTM, and Google's TimesFM). A toy-data harness and an evaluator
drive them on controlled synthetic series first, then on real price data.

Nothing in this package imports from `backend.news`, `backend.signals`, or
`backend.portfolio`. The two sub-components are intentionally decoupled.
"""
from .base import Forecaster, Forecast
from .metrics import (
    mae,
    rmse,
    mape,
    mase,
    directional_accuracy,
    summarize_forecast,
)

__all__ = [
    "Forecaster",
    "Forecast",
    "mae",
    "rmse",
    "mape",
    "mase",
    "directional_accuracy",
    "summarize_forecast",
]
