"""News endpoints."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import and_, func, select

from ..core.db import NewsItem, get_session_factory
from ..core.schemas import NewsItemOut
from ..news.aggregator import refresh_latest
from ..news.providers import build_providers

router = APIRouter(prefix="/api/news", tags=["news"])


def _to_out(row: NewsItem) -> NewsItemOut:
    try:
        tickers = json.loads(row.tickers_json or "[]")
    except Exception:  # noqa: BLE001
        tickers = []
    return NewsItemOut(
        id=row.id,
        source=row.source,
        title=row.title,
        summary=row.summary,
        url=row.url,
        published_at=row.published_at,
        tickers=tickers,
        sentiment=float(row.sentiment or 0.0),
    )


@router.get("", response_model=List[NewsItemOut])
async def list_news(
    limit: int = Query(50, ge=1, le=500),
    ticker: Optional[str] = Query(None, description="Filter to items mentioning this ticker"),
    hours: Optional[int] = Query(None, ge=1, le=720),
):
    factory = get_session_factory()
    async with factory() as session:
        stmt = select(NewsItem)
        if ticker:
            stmt = stmt.where(NewsItem.tickers_json.like(f"%\"{ticker.upper()}\"%"))
        if hours is not None:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            stmt = stmt.where(NewsItem.published_at >= cutoff)
        stmt = stmt.order_by(NewsItem.published_at.desc()).limit(limit)
        result = await session.execute(stmt)
        rows = list(result.scalars())
    return [_to_out(r) for r in rows]


@router.post("/refresh")
async def refresh_now():
    providers = build_providers()
    inserted = await refresh_latest(providers, limit_per_provider=40)
    return {"inserted": inserted, "providers": [p.name for p in providers if p.enabled()]}


@router.get("/stats")
async def stats():
    factory = get_session_factory()
    async with factory() as session:
        total = (await session.execute(select(func.count(NewsItem.id)))).scalar() or 0
        last24 = (
            await session.execute(
                select(func.count(NewsItem.id)).where(
                    NewsItem.published_at >= datetime.utcnow() - timedelta(hours=24)
                )
            )
        ).scalar() or 0
        by_source = dict(
            (
                await session.execute(
                    select(NewsItem.source, func.count(NewsItem.id)).group_by(NewsItem.source)
                )
            ).all()
        )
    return {"total": total, "last_24h": last24, "by_source": by_source}
