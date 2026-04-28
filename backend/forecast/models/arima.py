"""ARIMA baseline via statsmodels.

Auto-order selection is intentionally minimal — we grid-search over a small
(p, d, q) cube using AIC, because pulling in pmdarima would add another dep.
The goal here is a competent ARIMA baseline, not state-of-the-art; the
"state of the art" slot in this comparison is TimesFM.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from ..base import Forecast, as_array


class ARIMAForecaster:
    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        *,
        search_p: tuple[int, ...] = (0, 1, 2),
        search_d: tuple[int, ...] = (0, 1),
        search_q: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        self.order = order
        self.search_p = search_p
        self.search_d = search_d
        self.search_q = search_q
        self._fit = None
        self._history: np.ndarray | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import statsmodels  # noqa: F401
            return True
        except Exception:
            return False

    def _select_order(self, h: np.ndarray) -> tuple[int, int, int]:
        if self.order is not None:
            return self.order
        from statsmodels.tsa.arima.model import ARIMA

        best = None
        best_aic = np.inf
        for p, d, q in product(self.search_p, self.search_d, self.search_q):
            if p == 0 and d == 0 and q == 0:
                continue
            try:
                m = ARIMA(h, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
                if m.aic < best_aic:
                    best_aic = m.aic
                    best = (p, d, q)
            except Exception:  # noqa: BLE001
                continue
        return best or (1, 1, 1)

    def fit(self, history: np.ndarray) -> None:
        if not self.is_available():
            raise RuntimeError("statsmodels not installed")
        from statsmodels.tsa.arima.model import ARIMA

        h = as_array(history)
        self._history = h
        order = self._select_order(h)
        self._fit = ARIMA(h, order=order).fit(method_kwargs={"warn_convergence": False})

    def forecast(self, horizon: int) -> Forecast:
        if self._fit is None:
            raise RuntimeError("fit() must be called first")
        res = self._fit.get_forecast(steps=horizon)
        mean = np.asarray(res.predicted_mean, dtype=np.float64)
        try:
            ci = res.conf_int(alpha=0.2)  # 80% PI
            lower = np.asarray(ci[:, 0], dtype=np.float64)
            upper = np.asarray(ci[:, 1], dtype=np.float64)
        except Exception:  # noqa: BLE001
            lower = upper = None
        return Forecast(point=mean, lower=lower, upper=upper)
