"""Vectorized backtester.

The backtester takes a target-weight function and runs it over historical
prices. Between rebalances, the portfolio drifts with the market.

Assumptions
-----------
* Daily close-to-close.
* Portfolio starts fully in cash, first rebalance on the start date.
* Transaction costs applied as `transaction_cost_bps` on |Δweight| turnover
  of the portfolio value.
* Weights not in the universe are silently dropped. Excess cash earns 0%.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from . import metrics

log = get_logger(__name__)


WeightFn = Callable[[pd.Timestamp, pd.DataFrame, Dict], Dict[str, float]]


@dataclass
class BacktestArtifacts:
    equity: pd.Series  # strategy equity
    returns: pd.Series  # daily strategy returns
    weights_history: pd.DataFrame  # rows = rebal dates, cols = tickers
    bench_equity: pd.Series
    bench_returns: pd.Series
    metrics: Dict


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    freq = (freq or "weekly").lower()
    if freq == "daily":
        return index
    if freq == "weekly":
        # First trading day of each ISO week
        week = index.to_series().groupby(index.isocalendar().week + index.year * 100).first()
        return pd.DatetimeIndex(sorted(week.values))
    if freq == "monthly":
        return pd.DatetimeIndex(sorted(pd.Series(index).groupby([index.year, index.month]).first().values))
    return index


def run_backtest(
    prices: pd.DataFrame,
    target_fn: WeightFn,
    *,
    starting_cash: float = 1_000_000.0,
    transaction_cost_bps: float = 2.0,
    rebalance: str = "weekly",
    benchmark_series: Optional[pd.Series] = None,
    extra_context: Optional[Dict] = None,
) -> BacktestArtifacts:
    """Run a vectorized backtest.

    `prices` should be a DataFrame indexed by trading date with columns per
    ticker (adjusted close). `target_fn(date, history, ctx)` returns a dict
    of target weights for that date; keys must be a subset of `prices.columns`.
    """
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is empty")

    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = list(prices.columns)
    index = prices.index
    rebal = set(_rebalance_dates(index, rebalance))

    # State
    holdings = pd.Series(0.0, index=tickers)  # in dollars
    cash = float(starting_cash)
    tc = transaction_cost_bps / 10_000.0
    ctx = dict(extra_context or {})

    equity_records: List[float] = []
    weight_rows: List[Dict] = []

    prev_prices = prices.iloc[0].ffill()
    for i, date in enumerate(index):
        px = prices.iloc[i].copy()
        # Drift holdings with today's prices
        if i > 0:
            ret = (px / prev_prices) - 1.0
            ret = ret.fillna(0.0)
            holdings = holdings * (1.0 + ret)
        portfolio_value = holdings.sum() + cash

        # Rebalance if due
        if date in rebal:
            history = prices.loc[:date]
            try:
                targets = target_fn(date, history, ctx) or {}
            except Exception:  # noqa: BLE001
                log.exception("target_fn raised at %s — holding position", date)
                targets = {}
            # Normalize targets — drop tickers not in universe, sum to <=1.
            targets = {t: float(w) for t, w in targets.items() if t in tickers}
            s = sum(max(0.0, w) for w in targets.values())
            if s > 1.0 + 1e-9:
                # Rescale if the allocator over-allocates.
                targets = {t: (w / s) for t, w in targets.items()}

            current_w = (holdings / portfolio_value) if portfolio_value > 0 else holdings * 0
            target_w = pd.Series(0.0, index=tickers)
            for t, w in targets.items():
                target_w[t] = w

            new_holdings = target_w * portfolio_value
            turnover = (new_holdings - holdings).abs().sum()
            cost = turnover * tc
            holdings = new_holdings.copy()
            cash = portfolio_value - holdings.sum() - cost
            portfolio_value = holdings.sum() + cash

            weight_rows.append({"ts": date.isoformat(), **{t: float(target_w[t]) for t in tickers}})

        equity_records.append(holdings.sum() + cash)
        prev_prices = px.ffill()

    equity = pd.Series(equity_records, index=index, name="equity")
    returns = equity.pct_change().fillna(0.0)

    # Benchmark
    if benchmark_series is not None and not benchmark_series.empty:
        bench = benchmark_series.reindex(index).ffill().dropna()
        # Start benchmark with same starting cash for fair comparison.
        bench_equity = (bench / bench.iloc[0]) * starting_cash
        bench_returns = bench_equity.pct_change().fillna(0.0)
    else:
        bench_equity = pd.Series(dtype=float)
        bench_returns = pd.Series(dtype=float)

    m = metrics.summarize(
        strat_returns=returns,
        strat_equity=equity,
        bench_returns=bench_returns if not bench_returns.empty else None,
        bench_equity=bench_equity if not bench_equity.empty else None,
    )

    weights_df = pd.DataFrame(weight_rows).set_index("ts") if weight_rows else pd.DataFrame(columns=tickers)
    return BacktestArtifacts(
        equity=equity,
        returns=returns,
        weights_history=weights_df,
        bench_equity=bench_equity,
        bench_returns=bench_returns,
        metrics=m,
    )


# --- Out-of-the-box strategies ------------------------------------------------

def equal_weight_strategy(universe: List[str]) -> WeightFn:
    def fn(_date, _hist, _ctx):
        if not universe:
            return {}
        w = 1.0 / len(universe)
        return {t: w for t in universe}
    return fn


def momentum_strategy(
    universe: List[str],
    *,
    lookback_days: int = 60,
    top_k: int = 5,
    max_weight: float = 0.30,
) -> WeightFn:
    """Classic cross-sectional momentum — top_k winners, equal-weight within."""
    def fn(date, hist, _ctx):
        available = [t for t in universe if t in hist.columns]
        if not available:
            return {}
        if len(hist) < lookback_days + 1:
            w = 1.0 / len(available)
            return {t: w for t in available}
        end = hist.loc[date] if date in hist.index else hist.iloc[-1]
        start = hist.iloc[-lookback_days - 1]
        rets = (end / start) - 1.0
        ranked = rets.dropna().sort_values(ascending=False)
        picks = list(ranked.index[:top_k])
        if not picks:
            return {}
        w = min(max_weight, 1.0 / len(picks))
        weights = {t: w for t in picks}
        # Normalize in case cap tightened things.
        s = sum(weights.values())
        if s > 0:
            weights = {t: v / s for t, v in weights.items()}
        return weights
    return fn


def news_driven_strategy(
    universe: List[str],
    news_by_date: Dict[pd.Timestamp, List[Dict]],
    *,
    benchmark: str,
    max_weight: float = 0.25,
    active_share: float = 0.7,
    window_days: int = 3,
) -> WeightFn:
    """Translate real news items into a time-varying allocation.

    `news_by_date` maps a trading date → list of {tickers: [..], sentiment: float, ts: datetime}.
    The allocator looks back `window_days` for recent items.
    """
    from ..signals.engine import (
        ScoredNews, aggregate_signals, squash_scores, signals_to_weights, blend_with_benchmark,
    )
    import pandas as _pd

    # Pre-index news by trading date.
    sorted_dates = sorted(news_by_date.keys())

    def fn(date, _hist, _ctx):
        cutoff = date - _pd.Timedelta(days=window_days)
        relevant: List[ScoredNews] = []
        for d in sorted_dates:
            if d < cutoff:
                continue
            if d > date:
                break
            for item in news_by_date[d]:
                relevant.append(
                    ScoredNews(
                        ticker=[t for t in item.get("tickers", []) if t in universe],
                        sentiment=float(item.get("sentiment", 0.0)),
                        ts=item.get("ts", d.to_pydatetime()),
                    )
                )
        if not relevant:
            # No news in window — fall back to benchmark to avoid gap risk.
            return {benchmark: 1.0} if benchmark in universe else {}
        raw = aggregate_signals(relevant, now=date.to_pydatetime())
        scores = squash_scores(raw)
        weights = signals_to_weights(
            scores, universe, long_only=True, max_weight=max_weight, min_abs_signal=0.05,
        )
        if benchmark in universe:
            weights = blend_with_benchmark(weights, benchmark, active_share=active_share)
        return weights

    return fn
