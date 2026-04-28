"""Paper-trading engine for the live "house" model.

Lifecycle
---------
* On app start: load latest PortfolioSnapshot. If none, initialize with cash.
* Scheduler tick (every few minutes during market hours, else slower):
    1. Refresh latest news → DB.
    2. Compute aggregated signals over last `window_hours`.
    3. Generate target weights (with benchmark blending).
    4. Fetch latest prices, compute portfolio value, apply target.
    5. Persist snapshot + trades.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from ..core.config import get_settings
from ..core.db import NewsItem, PortfolioSnapshot, Trade, get_session_factory
from ..core.logging import get_logger
from ..market.data import get_latest_prices
from ..signals.engine import (
    ScoredNews,
    aggregate_signals,
    blend_with_benchmark,
    signals_to_weights,
    squash_scores,
)

log = get_logger(__name__)


@dataclass
class PaperState:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)  # ticker → qty
    last_prices: Dict[str, float] = field(default_factory=dict)
    benchmark_shares: float = 0.0
    benchmark_start_price: float = 0.0
    started_cash: float = 0.0

    def mark_to_market(self) -> float:
        val = self.cash
        for t, q in self.positions.items():
            val += q * self.last_prices.get(t, 0.0)
        return val

    def benchmark_value(self) -> float:
        bench_px = self.last_prices.get(get_settings().benchmark_ticker, 0.0)
        return self.benchmark_shares * bench_px


class PaperTrader:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.state: Optional[PaperState] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load last snapshot or start fresh."""
        async with self._lock:
            factory = get_session_factory()
            async with factory() as session:
                stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.ts.desc()).limit(1)
                result = await session.execute(stmt)
                last = result.scalar_one_or_none()

            if last is not None:
                # Reconstitute weights → positions at last snapshot using last prices we have.
                weights = json.loads(last.weights_json or "{}")
                # We do NOT have historical prices per-position, so treat last snapshot
                # as accurate equity, reinitialize positions based on current prices.
                tickers = list(weights.keys()) or [self.settings.benchmark_ticker]
                prices = await get_latest_prices(tickers + [self.settings.benchmark_ticker])
                equity = float(last.equity)
                positions: Dict[str, float] = {}
                cash = equity
                for t, w in weights.items():
                    px = prices.get(t, 0.0)
                    if px <= 0:
                        continue
                    dollars = equity * float(w)
                    q = dollars / px
                    positions[t] = q
                    cash -= dollars
                bench_px = prices.get(self.settings.benchmark_ticker, 0.0)
                # If benchmark price is temporarily unavailable (market closed,
                # yfinance hiccup), fall back to the most recent snapshotted
                # benchmark_equity so the SPY line isn't zeroed out until the
                # next tick can refresh. The first successful tick recomputes
                # bench_shares from the live price.
                last_bench_eq = float(last.benchmark_equity or 0.0)
                if bench_px > 0:
                    bench_shares = last_bench_eq / bench_px
                else:
                    bench_shares = 0.0
                    log.warning(
                        "benchmark price unavailable on resume — deferring bench_shares"
                        " (last snapshot benchmark_equity=$%.2f)", last_bench_eq,
                    )
                self.state = PaperState(
                    cash=max(0.0, cash),
                    positions=positions,
                    last_prices=prices,
                    benchmark_shares=bench_shares,
                    benchmark_start_price=bench_px,
                    started_cash=self.settings.paper_starting_cash,
                )
                log.info("Paper trader resumed: equity=$%.2f cash=$%.2f", last.equity, cash)
            else:
                # Fresh start — everything in cash, parallel benchmark.
                prices = await get_latest_prices([self.settings.benchmark_ticker])
                bench_px = prices.get(self.settings.benchmark_ticker, 0.0)
                if bench_px > 0:
                    bench_shares = self.settings.paper_starting_cash / bench_px
                else:
                    # Defer benchmark seeding to the first tick where we can
                    # fetch a live price. Better than zeroing the SPY line.
                    bench_shares = 0.0
                    log.warning(
                        "benchmark price unavailable on fresh start — will seed on first tick",
                    )
                self.state = PaperState(
                    cash=self.settings.paper_starting_cash,
                    last_prices=prices,
                    benchmark_shares=bench_shares,
                    benchmark_start_price=bench_px,
                    started_cash=self.settings.paper_starting_cash,
                )
                await self._persist_snapshot({}, 0.0)
                log.info("Paper trader initialized with $%.0f cash", self.state.cash)

    async def tick(self) -> Dict:
        """Run one rebalance tick. Returns a summary dict."""
        async with self._lock:
            if self.state is None:
                await self.initialize()
            assert self.state is not None

            # 1. Get fresh news-scored items from DB over last 48 hours.
            items = await self._recent_scored_news(hours=48)

            # 2. Aggregate → squash → weights.
            raw = aggregate_signals(items)
            scores = squash_scores(raw)
            target_weights = signals_to_weights(
                scores,
                self.settings.house_universe,
                long_only=True,
                max_weight=0.25,
            )
            target_weights = blend_with_benchmark(
                target_weights, self.settings.benchmark_ticker, active_share=0.7
            )

            # 3. Pull latest prices for everything we might need.
            tickers_needed = list(dict.fromkeys(
                [*self.settings.house_universe, *target_weights.keys(), self.settings.benchmark_ticker]
            ))
            prices = await get_latest_prices(tickers_needed)
            # Merge, preferring fresh ones.
            self.state.last_prices.update(prices)

            # Seed benchmark shares if initialize deferred it (price was not
            # available earlier but is now).
            bench_px = self.state.last_prices.get(self.settings.benchmark_ticker, 0.0)
            if self.state.benchmark_shares == 0.0 and bench_px > 0:
                self.state.benchmark_shares = self.state.started_cash / bench_px
                self.state.benchmark_start_price = bench_px
                log.info("Benchmark seeded on tick: shares=%.4f @ $%.2f", self.state.benchmark_shares, bench_px)

            # 4. Mark-to-market, compute trades to hit targets.
            equity = self.state.mark_to_market()
            trades_made: List[Dict] = []
            new_positions: Dict[str, float] = {}
            cash_after = equity

            for t, w in target_weights.items():
                px = self.state.last_prices.get(t, 0.0)
                if px <= 0:
                    continue
                target_dollars = equity * float(w)
                q = target_dollars / px
                prev_q = self.state.positions.get(t, 0.0)
                delta = q - prev_q
                if abs(delta * px) < 50:  # $50 threshold to avoid noise
                    q = prev_q
                else:
                    trades_made.append({
                        "ticker": t,
                        "side": "BUY" if delta > 0 else "SELL",
                        "qty": float(abs(delta)),
                        "price": float(px),
                        "reason": f"signal_w={w:.3f}",
                    })
                new_positions[t] = q
                cash_after -= q * px

            # Record SELL for any previously-held ticker that dropped out of
            # the target completely — without this the trade log misses
            # liquidations and the allocation donut can't explain where
            # positions went. Value of these dropped positions is already
            # baked into cash_after (we seeded cash_after = equity, which
            # includes all pre-trade position values).
            for t, prev_q in self.state.positions.items():
                if t in target_weights or prev_q <= 0:
                    continue
                px = self.state.last_prices.get(t, 0.0)
                if px <= 0:
                    continue
                trades_made.append({
                    "ticker": t,
                    "side": "SELL",
                    "qty": float(prev_q),
                    "price": float(px),
                    "reason": "dropped_from_target",
                })

            # Apply a 1bp implicit trading cost on turnover.
            turnover = sum(abs(t["qty"] * t["price"]) for t in trades_made)
            cost = turnover * 0.0001
            cash_after = max(0.0, cash_after - cost)

            self.state.positions = new_positions
            # Clamp to cents — prevents snapshot float noise magnifying on the
            # equity chart during flat periods (weekends, holidays).
            self.state.cash = round(cash_after, 2)

            # 5. Persist trades + snapshot.
            await self._persist_trades(trades_made)
            await self._persist_snapshot(target_weights, equity=self.state.mark_to_market())

            return {
                "equity": self.state.mark_to_market(),
                "cash": self.state.cash,
                "benchmark_equity": self.state.benchmark_value(),
                "target_weights": target_weights,
                "trades": trades_made,
                "signals_used": len(items),
            }

    async def status(self) -> Dict:
        async with self._lock:
            if self.state is None:
                await self.initialize()
            assert self.state is not None
            # Refresh prices for held positions + benchmark (fast path).
            tickers = list(self.state.positions.keys()) + [self.settings.benchmark_ticker]
            if tickers:
                prices = await get_latest_prices(list(dict.fromkeys(tickers)))
                self.state.last_prices.update(prices)
            equity = self.state.mark_to_market()
            holdings = [
                {
                    "ticker": t,
                    "qty": float(q),
                    "price": float(self.state.last_prices.get(t, 0.0)),
                    "value": float(q * self.state.last_prices.get(t, 0.0)),
                    "weight": float(q * self.state.last_prices.get(t, 0.0) / equity) if equity > 0 else 0.0,
                }
                for t, q in self.state.positions.items()
                if q > 0
            ]
            return {
                "equity": equity,
                "cash": self.state.cash,
                "started_cash": self.state.started_cash,
                "benchmark_equity": self.state.benchmark_value(),
                "benchmark_ticker": self.settings.benchmark_ticker,
                "pnl": equity - self.state.started_cash,
                "return_pct": (equity / self.state.started_cash - 1.0) if self.state.started_cash > 0 else 0.0,
                "holdings": sorted(holdings, key=lambda x: -x["value"]),
            }

    # --- internals ----------------------------------------------------------

    async def _recent_scored_news(self, hours: int = 48) -> List[ScoredNews]:
        factory = get_session_factory()
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        async with factory() as session:
            stmt = (
                select(NewsItem)
                .where(NewsItem.published_at >= cutoff)
                .order_by(NewsItem.published_at.desc())
                .limit(500)
            )
            result = await session.execute(stmt)
            rows = list(result.scalars())
        out: List[ScoredNews] = []
        for r in rows:
            try:
                tickers = json.loads(r.tickers_json or "[]")
            except Exception:  # noqa: BLE001
                tickers = []
            out.append(
                ScoredNews(
                    ticker=list(tickers),
                    sentiment=float(r.sentiment or 0.0),
                    ts=r.published_at.replace(tzinfo=timezone.utc) if r.published_at.tzinfo is None else r.published_at,
                )
            )
        return out

    async def _persist_trades(self, trades: List[Dict]) -> None:
        if not trades:
            return
        factory = get_session_factory()
        async with factory() as session:
            for t in trades:
                session.add(Trade(
                    ts=datetime.utcnow(),
                    ticker=t["ticker"],
                    side=t["side"],
                    qty=t["qty"],
                    price=t["price"],
                    reason=t.get("reason"),
                ))
            await session.commit()

    async def _persist_snapshot(self, weights: Dict[str, float], equity: float) -> None:
        if self.state is None:
            return
        eq = float(equity if equity > 0 else self.state.mark_to_market())
        factory = get_session_factory()
        async with factory() as session:
            session.add(PortfolioSnapshot(
                ts=datetime.utcnow(),
                # Round to cents everywhere — avoids float garbage like
                # 999999.9999999999 that zooms the equity chart into noise.
                equity=round(eq, 2),
                cash=round(float(self.state.cash), 2),
                benchmark_equity=round(float(self.state.benchmark_value()), 2),
                weights_json=json.dumps({k: float(v) for k, v in weights.items()}),
            ))
            await session.commit()


# Singleton
_trader: Optional[PaperTrader] = None


def get_trader() -> PaperTrader:
    global _trader
    if _trader is None:
        _trader = PaperTrader()
    return _trader
