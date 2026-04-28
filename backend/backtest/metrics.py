"""Performance-metrics library.

All functions take a pandas `returns` Series (daily, arithmetic) and return a
scalar. Edge cases (empty series, zero volatility, zero benchmark variance) are
handled and return 0.0 rather than NaN/inf to keep downstream JSON clean.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def annual_return(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    n = len(returns)
    if n == 0:
        return 0.0
    tr = (1.0 + returns).prod()
    return float(tr ** (TRADING_DAYS / n) - 1.0)


def annual_vol(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


_EPS = 1e-12


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns is None or returns.empty:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd is None or np.isnan(sd) or abs(sd) < _EPS:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    if returns is None or returns.empty:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if downside.empty:
        return 0.0
    dd = downside.std(ddof=1)
    if dd is None or np.isnan(dd) or abs(dd) < _EPS:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(returns: pd.Series, equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(annual_return(returns) / mdd)


def win_rate(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def hit_rate_vs_benchmark(strat: pd.Series, bench: pd.Series) -> float:
    if strat is None or bench is None or strat.empty or bench.empty:
        return 0.0
    joined = pd.concat([strat, bench], axis=1, join="inner").dropna()
    if joined.empty:
        return 0.0
    diffs = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float((diffs > 0).sum() / len(diffs))


def best_worst_month(returns: pd.Series) -> tuple[float, float]:
    if returns is None or returns.empty:
        return 0.0, 0.0
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0, 0.0
    return float(monthly.max()), float(monthly.min())


def beta_alpha(strat: pd.Series, bench: pd.Series, rf: float = 0.0) -> tuple[float, float]:
    if strat is None or bench is None or strat.empty or bench.empty:
        return 0.0, 0.0
    joined = pd.concat([strat, bench], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return 0.0, 0.0
    rs = joined.iloc[:, 0] - rf / TRADING_DAYS
    rb = joined.iloc[:, 1] - rf / TRADING_DAYS
    var_b = rb.var(ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return 0.0, 0.0
    cov = np.cov(rs, rb, ddof=1)[0, 1]
    b = float(cov / var_b)
    alpha_daily = float(rs.mean() - b * rb.mean())
    alpha_annual = alpha_daily * TRADING_DAYS
    return b, alpha_annual


def tracking_error(strat: pd.Series, bench: pd.Series) -> float:
    if strat is None or bench is None or strat.empty or bench.empty:
        return 0.0
    joined = pd.concat([strat, bench], axis=1, join="inner").dropna()
    if joined.empty:
        return 0.0
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(strat: pd.Series, bench: pd.Series) -> float:
    te = tracking_error(strat, bench)
    if te == 0:
        return 0.0
    joined = pd.concat([strat, bench], axis=1, join="inner").dropna()
    if joined.empty:
        return 0.0
    excess = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float(excess.mean() * TRADING_DAYS / te)


def summarize(
    strat_returns: pd.Series,
    strat_equity: pd.Series,
    bench_returns: Optional[pd.Series] = None,
    bench_equity: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> dict:
    best_m, worst_m = best_worst_month(strat_returns)
    b, a = beta_alpha(strat_returns, bench_returns) if bench_returns is not None else (0.0, 0.0)
    return {
        "total_return": total_return(strat_returns),
        "annual_return": annual_return(strat_returns),
        "annual_vol": annual_vol(strat_returns),
        "sharpe": sharpe(strat_returns, rf),
        "sortino": sortino(strat_returns, rf),
        "max_drawdown": max_drawdown(strat_equity),
        "calmar": calmar(strat_returns, strat_equity),
        "win_rate": win_rate(strat_returns),
        "hit_rate": hit_rate_vs_benchmark(strat_returns, bench_returns) if bench_returns is not None else 0.0,
        "best_month": best_m,
        "worst_month": worst_m,
        "beta": b,
        "alpha_annual": a,
        "tracking_error": tracking_error(strat_returns, bench_returns) if bench_returns is not None else 0.0,
        "information_ratio": information_ratio(strat_returns, bench_returns) if bench_returns is not None else 0.0,
        "benchmark_total_return": total_return(bench_returns) if bench_returns is not None else 0.0,
        "benchmark_sharpe": sharpe(bench_returns, rf) if bench_returns is not None else 0.0,
    }
