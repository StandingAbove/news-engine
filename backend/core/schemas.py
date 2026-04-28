"""Pydantic response/request schemas shared across API routes."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NewsItemOut(BaseModel):
    id: int
    source: str
    title: str
    summary: Optional[str] = None
    url: Optional[str] = None
    published_at: datetime
    tickers: List[str] = []
    sentiment: float = 0.0


class SignalOut(BaseModel):
    ticker: str
    score: float = Field(description="Aggregated directional signal in [-1, 1].")
    num_articles: int
    reason: str = ""


class AllocationOut(BaseModel):
    ticker: str
    weight: float
    value: float


class PortfolioSnapshotOut(BaseModel):
    ts: datetime
    equity: float
    cash: float
    benchmark_equity: float
    weights: Dict[str, float]


class TradeOut(BaseModel):
    ts: datetime
    ticker: str
    side: str
    qty: float
    price: float
    reason: Optional[str] = None


class BacktestConfig(BaseModel):
    universe: List[str] = Field(
        default_factory=lambda: [
            "SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "GLD",
            "XLE", "XLK", "XLF", "XLV", "XLY", "XLI",
        ]
    )
    start: str = "2023-01-01"
    end: str = "2024-12-31"
    starting_cash: float = 1_000_000.0
    rebalance: str = "weekly"  # daily, weekly, monthly
    transaction_cost_bps: float = 2.0
    long_only: bool = True
    max_weight: float = 0.25
    benchmark: str = "SPY"
    # Signal-driven allocator uses simulated news proxy (price momentum + fake events).
    # Real news backtesting still runs when news data is available for the date range.
    use_news_signals: bool = True


class BacktestMetrics(BaseModel):
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    hit_rate: float
    best_month: float
    worst_month: float
    beta: float
    alpha_annual: float
    tracking_error: float
    information_ratio: float
    benchmark_total_return: float
    benchmark_sharpe: float


class BacktestResult(BaseModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    config: BacktestConfig
    metrics: BacktestMetrics
    equity_curve: List[Dict] = []  # [{ts, strategy, benchmark}]
    drawdown_curve: List[Dict] = []
    weights_history: List[Dict] = []
