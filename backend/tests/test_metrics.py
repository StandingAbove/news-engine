"""Unit tests for metrics library."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.backtest import metrics


def _const_returns(n: int, r: float) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series([r] * n, index=idx)


def test_total_return_compounds_correctly():
    r = _const_returns(5, 0.01)
    # (1.01)^5 - 1
    assert metrics.total_return(r) == pytest.approx(1.01**5 - 1, abs=1e-10)


def test_annual_return_extrapolates_from_daily():
    r = _const_returns(252, 0.0005)
    ar = metrics.annual_return(r)
    # Should be close to (1.0005)^252 - 1
    assert ar == pytest.approx((1.0005**252) - 1, abs=1e-6)


def test_sharpe_zero_for_zero_vol():
    r = _const_returns(60, 0.01)
    assert metrics.sharpe(r) == 0.0


def test_sharpe_positive_for_positive_mean_with_vol():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    r = pd.Series(rng.normal(loc=0.001, scale=0.01, size=500), index=idx)
    s = metrics.sharpe(r)
    assert s > 0.5  # should comfortably be positive


def test_max_drawdown_matches_known_case():
    eq = pd.Series([100, 110, 120, 108, 90, 95, 130], index=pd.date_range("2020-01-01", periods=7))
    # peak before 90 is 120 → dd = -25%
    assert metrics.max_drawdown(eq) == pytest.approx(-0.25, abs=1e-10)


def test_max_drawdown_handles_monotonic_up():
    eq = pd.Series([100, 101, 102, 103], index=pd.date_range("2020-01-01", periods=4))
    assert metrics.max_drawdown(eq) == 0.0


def test_win_rate():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    r = pd.Series([0.01, -0.01, 0.02, -0.005, 0.001, 0.0, 0.03, -0.02, 0.005, 0.004], index=idx)
    # Wins (>0): 1,3,5,7,9,10 = 6. 0.0 is NOT a win (>0 is strict).
    assert metrics.win_rate(r) == pytest.approx(6 / 10)


def test_beta_of_itself_is_one():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    r = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
    b, a = metrics.beta_alpha(r, r)
    assert b == pytest.approx(1.0, abs=1e-10)
    assert a == pytest.approx(0.0, abs=1e-10)


def test_information_ratio_zero_when_identical():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    r = pd.Series(rng.normal(0.0, 0.01, 200), index=idx)
    assert metrics.information_ratio(r, r) == 0.0


def test_summarize_returns_all_keys():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    s = pd.Series(rng.normal(0.0005, 0.01, 252), index=idx)
    b = pd.Series(rng.normal(0.0003, 0.009, 252), index=idx)
    eq = (1 + s).cumprod() * 1_000_000
    out = metrics.summarize(s, eq, b, (1 + b).cumprod() * 1_000_000)
    expected_keys = {
        "total_return", "annual_return", "annual_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "win_rate", "hit_rate",
        "best_month", "worst_month", "beta", "alpha_annual",
        "tracking_error", "information_ratio",
        "benchmark_total_return", "benchmark_sharpe",
    }
    assert expected_keys <= set(out.keys())
    # no NaN / inf
    for k, v in out.items():
        assert np.isfinite(v), f"{k} = {v}"


def test_empty_series_return_zero():
    empty = pd.Series([], dtype=float)
    assert metrics.total_return(empty) == 0.0
    assert metrics.sharpe(empty) == 0.0
    assert metrics.sortino(empty) == 0.0
    assert metrics.max_drawdown(empty) == 0.0
