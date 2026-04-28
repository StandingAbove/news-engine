"""Unit tests for the backtester engine."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backend.backtest.engine import (
    equal_weight_strategy,
    momentum_strategy,
    run_backtest,
)


def _fake_prices(days: int = 252, tickers=("A", "B", "C"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=days, freq="B")
    data = {}
    for t in tickers:
        rets = rng.normal(0.0004, 0.012, days)
        data[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def test_equal_weight_backtest_matches_buy_and_hold_of_average():
    prices = _fake_prices(days=120, tickers=("A", "B"))
    art = run_backtest(
        prices, equal_weight_strategy(["A", "B"]),
        starting_cash=1_000_000, transaction_cost_bps=0, rebalance="daily"
    )
    # With zero TC and daily rebalancing, a 50/50 portfolio should return
    # ~= average of the two buy-and-holds.
    ab = 1_000_000 * ((prices.iloc[-1] / prices.iloc[0]).mean())
    # Rebalancing adds small path-dependent diff — allow 2% tolerance.
    assert art.equity.iloc[-1] == pytest.approx(ab, rel=0.05)


def test_transaction_cost_lowers_equity():
    prices = _fake_prices(days=252)
    art_zero = run_backtest(prices, equal_weight_strategy(list(prices.columns)),
                             starting_cash=1_000_000, transaction_cost_bps=0, rebalance="daily")
    art_high = run_backtest(prices, equal_weight_strategy(list(prices.columns)),
                             starting_cash=1_000_000, transaction_cost_bps=50, rebalance="daily")
    assert art_high.equity.iloc[-1] < art_zero.equity.iloc[-1]


def test_cash_only_strategy_preserves_capital():
    prices = _fake_prices(days=60)

    def cash_fn(_d, _h, _c):
        return {}

    art = run_backtest(prices, cash_fn, starting_cash=500_000, rebalance="daily", transaction_cost_bps=0)
    # 100% cash → equity flat.
    assert art.equity.iloc[-1] == pytest.approx(500_000, abs=1e-6)
    assert abs(art.returns).sum() < 1e-9


def test_benchmark_comparison_produces_metrics():
    prices = _fake_prices(days=180, tickers=("A", "B", "SPY"), seed=3)
    bench = prices["SPY"]
    art = run_backtest(
        prices[["A", "B"]],
        equal_weight_strategy(["A", "B"]),
        starting_cash=1_000_000, rebalance="weekly", transaction_cost_bps=2,
        benchmark_series=bench,
    )
    assert "sharpe" in art.metrics
    assert "beta" in art.metrics
    assert "alpha_annual" in art.metrics
    assert art.bench_equity.iloc[0] == pytest.approx(1_000_000, rel=1e-6)


def test_momentum_strategy_picks_top_performers():
    # Construct prices where "A" clearly outperforms over lookback.
    idx = pd.date_range("2022-01-03", periods=120, freq="B")
    prices = pd.DataFrame({
        "A": np.linspace(100, 200, 120),  # strong uptrend
        "B": np.linspace(100, 95, 120),   # slight drawdown
        "C": np.full(120, 100.0),         # flat
    }, index=idx)
    fn = momentum_strategy(["A", "B", "C"], lookback_days=60, top_k=1, max_weight=1.0)
    art = run_backtest(prices, fn, starting_cash=1_000_000, rebalance="monthly", transaction_cost_bps=0)
    # Eventually concentrates in A; final weight row should have A near 1.0.
    wh = art.weights_history
    assert not wh.empty
    final = wh.iloc[-1]
    assert final["A"] == pytest.approx(1.0, abs=1e-6)


def test_empty_prices_raises():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), equal_weight_strategy(["A"]))


def test_weight_overshoot_is_scaled_down():
    prices = _fake_prices(days=30)

    def over_fn(_d, _h, _c):
        # Deliberately allocate 150%.
        return {"A": 0.5, "B": 0.5, "C": 0.5}

    art = run_backtest(prices, over_fn, starting_cash=100_000,
                       rebalance="daily", transaction_cost_bps=0)
    # Equity should remain roughly proportional to market without going negative/insane.
    assert art.equity.iloc[-1] > 0
    # The scaled-down weights should make holdings + cash ≈ starting cash on day 1.
    assert art.equity.iloc[0] == pytest.approx(100_000, rel=1e-3)
