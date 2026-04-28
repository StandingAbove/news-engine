"""Paper-trading portfolio endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter
from sqlalchemy import select

from ..core.db import PortfolioSnapshot, Trade, get_session_factory
from ..portfolio.paper import get_trader

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("/status")
async def status():
    return await get_trader().status()


@router.post("/tick")
async def tick_now():
    return await get_trader().tick()


@router.get("/equity-curve")
async def equity_curve(hours: int = 24 * 30):
    factory = get_session_factory()
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    async with factory() as session:
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.ts >= cutoff)
            .order_by(PortfolioSnapshot.ts.asc())
        )
        rows = list((await session.execute(stmt)).scalars())
    return [
        {
            "ts": r.ts.isoformat(),
            "equity": float(r.equity),
            "benchmark_equity": float(r.benchmark_equity),
            "cash": float(r.cash),
        }
        for r in rows
    ]


@router.get("/trades")
async def recent_trades(limit: int = 100):
    factory = get_session_factory()
    async with factory() as session:
        stmt = select(Trade).order_by(Trade.ts.desc()).limit(limit)
        rows = list((await session.execute(stmt)).scalars())
    return [
        {
            "ts": r.ts.isoformat(),
            "ticker": r.ticker,
            "side": r.side,
            "qty": float(r.qty),
            "price": float(r.price),
            "reason": r.reason,
        }
        for r in rows
    ]
