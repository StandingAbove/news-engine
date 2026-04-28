"""Aggregate, enrich, dedupe, and persist news items from all providers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..core.db import NewsItem, get_session_factory
from ..core.logging import get_logger
from .mapping import extract_tickers
from .models import RawNewsItem
from .providers import NewsProvider, fetch_historical_all, fetch_latest_all
from .sentiment import get_sentiment

log = get_logger(__name__)


def dedupe(items: List[RawNewsItem]) -> List[RawNewsItem]:
    seen = set()
    unique: List[RawNewsItem] = []
    for it in items:
        key = it.dedup_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return unique


def enrich(items: List[RawNewsItem]) -> List[RawNewsItem]:
    """Populate tickers and blended sentiment."""
    sent = get_sentiment()
    for it in items:
        # Enrich tickers using the mapping layer, feeding in any provider hints.
        text = f"{it.title}. {it.summary or ''}"
        it.tickers = extract_tickers(text, hint=it.tickers)
        # Blended sentiment — mutate raw to include it for later display.
        it.raw["_sentiment"] = sent.blend(text, it.provider_sentiment)
    return items


async def persist(items: List[RawNewsItem]) -> int:
    """Insert with ON CONFLICT DO NOTHING on (source, external_id). Returns rows inserted."""
    if not items:
        return 0
    factory = get_session_factory()
    inserted = 0
    async with factory() as session:
        for it in items:
            stmt = (
                sqlite_insert(NewsItem)
                .values(
                    source=it.source,
                    external_id=it.external_id,
                    title=it.title,
                    summary=it.summary,
                    url=it.url,
                    published_at=it.published_at.replace(tzinfo=None),
                    tickers_json=json.dumps(it.tickers),
                    sentiment=float(it.raw.get("_sentiment", 0.0)),
                    raw_json=json.dumps(
                        {k: v for k, v in it.raw.items() if k != "_sentiment"},
                        default=str,
                    ),
                )
                .on_conflict_do_nothing(index_elements=["source", "external_id"])
            )
            result = await session.execute(stmt)
            if result.rowcount:
                inserted += 1
        await session.commit()
    return inserted


async def refresh_latest(
    providers: List[NewsProvider],
    *,
    limit_per_provider: int = 40,
    tickers: Optional[List[str]] = None,
) -> int:
    raw = await fetch_latest_all(providers, limit_per_provider=limit_per_provider, tickers=tickers)
    log.info("Pulled %d raw items across providers", len(raw))
    raw = dedupe(raw)
    raw = enrich(raw)
    n = await persist(raw)
    log.info("Persisted %d new news items (deduped=%d)", n, len(raw))
    return n


async def backfill_historical(
    providers: List[NewsProvider],
    *,
    start: datetime,
    end: datetime,
    tickers: Optional[List[str]] = None,
) -> int:
    # Slice into 30-day windows to play nicely with free tiers.
    cur = start
    total = 0
    while cur < end:
        window_end = min(cur + timedelta(days=30), end)
        raw = await fetch_historical_all(
            providers, start=cur, end=window_end, tickers=tickers
        )
        raw = dedupe(raw)
        raw = enrich(raw)
        total += await persist(raw)
        cur = window_end
    return total


async def recent_items(limit: int = 50) -> list[NewsItem]:
    factory = get_session_factory()
    async with factory() as session:
        stmt = select(NewsItem).order_by(NewsItem.published_at.desc()).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars())
