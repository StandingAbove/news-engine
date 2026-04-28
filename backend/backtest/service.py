"""High-level orchestration: given a BacktestConfig, run + persist results."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd

from ..core.db import BacktestRun, NewsItem, get_session_factory
from ..core.logging import get_logger
from ..core.schemas import BacktestConfig, BacktestMetrics, BacktestResult
from ..market.data import get_price_matrix
from .engine import (
    equal_weight_strategy,
    momentum_strategy,
    news_driven_strategy,
    run_backtest,
)

log = get_logger(__name__)


async def _load_news_in_range(start: datetime, end: datetime) -> List[Dict]:
    factory = get_session_factory()
    async with factory() as session:
        from sqlalchemy import and_, select
        stmt = select(NewsItem).where(
            and_(NewsItem.published_at >= start, NewsItem.published_at <= end)
        ).order_by(NewsItem.published_at.asc())
        result = await session.execute(stmt)
        rows = list(result.scalars())
    out: List[Dict] = []
    for r in rows:
        try:
            tickers = json.loads(r.tickers_json or "[]")
        except Exception:  # noqa: BLE001
            tickers = []
        out.append(
            {
                "tickers": tickers,
                "sentiment": float(r.sentiment or 0.0),
                "ts": r.published_at.replace(tzinfo=timezone.utc) if r.published_at.tzinfo is None else r.published_at,
            }
        )
    return out


def _news_by_trading_date(items: List[Dict], index: pd.DatetimeIndex) -> Dict[pd.Timestamp, List[Dict]]:
    """Bucket news items by the next trading day ≥ their timestamp.

    This encodes the "HK at 12am US time → tradeable when the US opens"
    intuition: we don't assume perfect intraday execution; we allocate on
    the next available trading day.
    """
    if not items:
        return {}
    idx = list(index)
    by_date: Dict[pd.Timestamp, List[Dict]] = {}
    idx_i = 0
    items_sorted = sorted(items, key=lambda x: x["ts"])
    for it in items_sorted:
        ts = it["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # advance idx_i to first trading day >= ts.date()
        while idx_i < len(idx) and idx[idx_i].date() < ts.date():
            idx_i += 1
        if idx_i >= len(idx):
            break
        by_date.setdefault(idx[idx_i], []).append(it)
    return by_date


async def run_configured_backtest(cfg: BacktestConfig) -> BacktestResult:
    # Ensure benchmark is fetched alongside the universe.
    tickers = list(dict.fromkeys([*cfg.universe, cfg.benchmark]))
    prices = await get_price_matrix(tickers, cfg.start, cfg.end)
    if prices.empty:
        raise RuntimeError("No market data returned — check tickers / date range.")

    bench_series = prices[cfg.benchmark].dropna() if cfg.benchmark in prices.columns else pd.Series(dtype=float)

    universe = [t for t in cfg.universe if t in prices.columns]

    # Build target strategy: news-driven if we have data, else momentum.
    start_dt = datetime.fromisoformat(cfg.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(cfg.end).replace(tzinfo=timezone.utc) + timedelta(days=1)

    target_fn = None
    news_used = 0
    if cfg.use_news_signals:
        news = await _load_news_in_range(start_dt, end_dt)
        news_by_date = _news_by_trading_date(news, prices.index)
        news_used = sum(len(v) for v in news_by_date.values())
        if news_used > 0:
            target_fn = news_driven_strategy(
                universe,
                news_by_date,
                benchmark=cfg.benchmark if cfg.benchmark in universe else universe[0],
                max_weight=cfg.max_weight,
                active_share=0.7,
            )
            log.info("Backtest using %d historical news items", news_used)
    if target_fn is None:
        target_fn = momentum_strategy(universe)
        log.info("Backtest falling back to momentum (no news in range)")

    artifacts = run_backtest(
        prices[universe],
        target_fn,
        starting_cash=cfg.starting_cash,
        transaction_cost_bps=cfg.transaction_cost_bps,
        rebalance=cfg.rebalance,
        benchmark_series=bench_series,
        extra_context={"config": cfg.dict(), "news_used": news_used},
    )

    # Shape results for JSON.
    eq = artifacts.equity
    bench_eq = artifacts.bench_equity
    idx = list(eq.index)
    equity_curve = [
        {
            "ts": ts.strftime("%Y-%m-%d"),
            "strategy": float(eq.iloc[i]),
            "benchmark": float(bench_eq.iloc[i]) if (not bench_eq.empty and i < len(bench_eq)) else None,
        }
        for i, ts in enumerate(idx)
    ]
    peak = eq.cummax()
    dd = eq / peak - 1.0
    drawdown_curve = [{"ts": ts.strftime("%Y-%m-%d"), "dd": float(dd.iloc[i])} for i, ts in enumerate(idx)]

    weights_hist = [
        {"ts": ts, **{k: float(v) for k, v in row.items()}}
        for ts, row in artifacts.weights_history.to_dict(orient="index").items()
    ] if not artifacts.weights_history.empty else []

    metrics_model = BacktestMetrics(**artifacts.metrics)

    # Persist
    factory = get_session_factory()
    async with factory() as session:
        row = BacktestRun(
            created_at=datetime.utcnow(),
            config_json=cfg.model_dump_json(),
            metrics_json=metrics_model.model_dump_json(),
            equity_curve_json=json.dumps(equity_curve),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        run_id = row.id

    return BacktestResult(
        id=run_id,
        created_at=datetime.utcnow(),
        config=cfg,
        metrics=metrics_model,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        weights_history=weights_hist,
    )
