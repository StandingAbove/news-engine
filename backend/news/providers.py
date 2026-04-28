"""HTTP clients for the three news providers.

Every provider returns a list[RawNewsItem] with a consistent shape so the
aggregator can merge them without caring about provider idiosyncrasies.

Design notes
------------
* All providers are async + httpx-based.
* Each provider implements the same `fetch_latest()` / `fetch_historical()`
  interface. Callers can fan out with `asyncio.gather`.
* Failures in one provider MUST NOT take down the aggregator — errors are
  caught and logged, and the provider returns `[]`.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.config import get_settings
from ..core.logging import get_logger
from .models import RawNewsItem

log = get_logger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=8.0)


def _parse_ts(value: str) -> datetime:
    """Accept a variety of ISO-8601 shapes and always return tz-aware UTC."""
    if not value:
        return datetime.now(timezone.utc)
    # Replace trailing Z
    value = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        # NewsAPI.ai frequently uses "2024-05-01T12:34:56"
        try:
            dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class NewsProvider(ABC):
    name: str = "base"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def enabled(self) -> bool:
        return bool(self.api_key)

    async def fetch_latest(self, limit: int = 50, tickers: Optional[List[str]] = None) -> List[RawNewsItem]:
        if not self.enabled():
            log.debug("Provider %s disabled (no key)", self.name)
            return []
        try:
            return await self._fetch_latest(limit=limit, tickers=tickers)
        except Exception:  # noqa: BLE001 - intentional: do not kill aggregator
            log.exception("%s.fetch_latest failed", self.name)
            return []

    async def fetch_historical(
        self,
        start: datetime,
        end: datetime,
        tickers: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[RawNewsItem]:
        if not self.enabled():
            return []
        try:
            return await self._fetch_historical(
                start=start, end=end, tickers=tickers, limit=limit
            )
        except Exception:  # noqa: BLE001
            log.exception("%s.fetch_historical failed", self.name)
            return []

    @abstractmethod
    async def _fetch_latest(self, limit: int, tickers: Optional[List[str]]) -> List[RawNewsItem]: ...

    @abstractmethod
    async def _fetch_historical(
        self, start: datetime, end: datetime, tickers: Optional[List[str]], limit: int
    ) -> List[RawNewsItem]: ...

    # --- utilities ---------------------------------------------------------

    async def _get_with_retry(self, url: str, *, params: dict, client: httpx.AsyncClient) -> dict:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        ):
            with attempt:
                resp = await client.get(url, params=params)
                # Rate-limit? Back off.
                if resp.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "Rate limited", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()
        return {}


class MarketauxProvider(NewsProvider):
    name = "marketaux"
    BASE = "https://api.marketaux.com/v1/news/all"

    async def _fetch_latest(self, limit, tickers):
        params = {
            "api_token": self.api_key,
            "language": "en",
            "limit": min(limit, 3),  # Marketaux free tier: 3/call
        }
        if tickers:
            params["symbols"] = ",".join(tickers[:20])
        return await self._query(params)

    async def _fetch_historical(self, start, end, tickers, limit):
        params = {
            "api_token": self.api_key,
            "language": "en",
            "limit": min(limit, 3),
            "published_after": start.strftime("%Y-%m-%dT%H:%M"),
            "published_before": end.strftime("%Y-%m-%dT%H:%M"),
        }
        if tickers:
            params["symbols"] = ",".join(tickers[:20])
        return await self._query(params)

    async def _query(self, params: dict) -> List[RawNewsItem]:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            data = await self._get_with_retry(self.BASE, params=params, client=client)
        items: List[RawNewsItem] = []
        for row in data.get("data", []) or []:
            syms = [e.get("symbol") for e in (row.get("entities") or []) if e.get("symbol")]
            # Average entity sentiment if present
            sent_vals = [float(e.get("sentiment_score", 0.0) or 0.0) for e in (row.get("entities") or [])]
            provider_sent = sum(sent_vals) / len(sent_vals) if sent_vals else None
            items.append(
                RawNewsItem(
                    source=self.name,
                    external_id=row.get("uuid") or row.get("url", ""),
                    title=row.get("title", ""),
                    summary=row.get("description"),
                    url=row.get("url"),
                    published_at=_parse_ts(row.get("published_at", "")),
                    tickers=[s for s in syms if s],
                    provider_sentiment=provider_sent,
                    raw=row,
                )
            )
        return items


class NewsApiAiProvider(NewsProvider):
    name = "newsapi_ai"
    BASE = "https://eventregistry.org/api/v1/article/getArticles"

    async def _fetch_latest(self, limit, tickers):
        payload = {
            "action": "getArticles",
            "keyword": tickers[0] if tickers else "market OR economy OR finance OR stocks",
            "articlesPage": 1,
            "articlesCount": min(limit, 100),
            "articlesSortBy": "date",
            "articlesSortByAsc": False,
            "articlesArticleBodyLen": -1,
            "resultType": "articles",
            "dataType": ["news"],
            "lang": "eng",
            "apiKey": self.api_key,
        }
        if tickers and len(tickers) > 1:
            payload["keyword"] = tickers
            payload["keywordOper"] = "or"
        return await self._query(payload)

    async def _fetch_historical(self, start, end, tickers, limit):
        payload = {
            "action": "getArticles",
            "keyword": tickers if tickers else "market OR economy OR finance",
            "keywordOper": "or" if tickers else "and",
            "articlesPage": 1,
            "articlesCount": min(limit, 100),
            "articlesSortBy": "date",
            "articlesSortByAsc": True,
            "resultType": "articles",
            "dateStart": start.date().isoformat(),
            "dateEnd": end.date().isoformat(),
            "lang": "eng",
            "apiKey": self.api_key,
        }
        return await self._query(payload)

    async def _query(self, payload: dict) -> List[RawNewsItem]:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
                reraise=True,
            ):
                with attempt:
                    resp = await client.post(self.BASE, json=payload)
                    if resp.status_code == 429:
                        raise httpx.HTTPStatusError(
                            "Rate limited", request=resp.request, response=resp
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    break

        articles = (data.get("articles", {}) or {}).get("results", []) or []
        items: List[RawNewsItem] = []
        for row in articles:
            concepts = row.get("concepts") or []
            # Try to extract ticker-like concept URIs, fall back to empty.
            tickers_found = []
            for c in concepts:
                label = (c.get("label") or {}).get("eng") or ""
                # This API doesn't directly give tickers; rely on downstream mapper.
                if label and label.isupper() and 1 <= len(label) <= 5:
                    tickers_found.append(label)
            items.append(
                RawNewsItem(
                    source=self.name,
                    external_id=row.get("uri") or row.get("url", ""),
                    title=row.get("title", "") or "",
                    summary=row.get("body", "")[:500] if row.get("body") else None,
                    url=row.get("url"),
                    published_at=_parse_ts(row.get("dateTime", "") or row.get("date", "")),
                    tickers=tickers_found,
                    raw=row,
                )
            )
        return items


class EodhdProvider(NewsProvider):
    name = "eodhd"
    BASE = "https://eodhd.com/api/news"

    async def _fetch_latest(self, limit, tickers):
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "limit": min(limit, 100),
            "offset": 0,
        }
        # EODHD takes a single symbol; loop a handful to build a cross-universe feed.
        if tickers:
            items: List[RawNewsItem] = []
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                for sym in tickers[:8]:
                    p = {**params, "s": f"{sym}.US"}
                    try:
                        data = await self._get_with_retry(self.BASE, params=p, client=client)
                    except Exception:  # noqa: BLE001
                        continue
                    items.extend(self._rows_to_items(data, sym))
            return items
        # No tickers: ask for a general stream via a proxy ticker.
        params["s"] = "SPY.US"
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            data = await self._get_with_retry(self.BASE, params=params, client=client)
        return self._rows_to_items(data, "SPY")

    async def _fetch_historical(self, start, end, tickers, limit):
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "limit": min(limit, 100),
            "from": start.date().isoformat(),
            "to": end.date().isoformat(),
        }
        tickers = tickers or ["SPY"]
        items: List[RawNewsItem] = []
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            for sym in tickers[:10]:
                p = {**params, "s": f"{sym}.US"}
                try:
                    data = await self._get_with_retry(self.BASE, params=p, client=client)
                except Exception:  # noqa: BLE001
                    continue
                items.extend(self._rows_to_items(data, sym))
        return items

    @staticmethod
    def _rows_to_items(rows, default_ticker: str) -> List[RawNewsItem]:
        out: List[RawNewsItem] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            syms = row.get("symbols") or [default_ticker]
            # Normalize symbols — EODHD uses "AAPL.US"
            clean = []
            for s in syms:
                if not isinstance(s, str):
                    continue
                clean.append(s.split(".")[0].upper())
            sent = None
            s_obj = row.get("sentiment") or {}
            if isinstance(s_obj, dict):
                try:
                    sent = float(s_obj.get("polarity"))
                except (TypeError, ValueError):
                    sent = None
            out.append(
                RawNewsItem(
                    source="eodhd",
                    external_id=str(row.get("id") or row.get("link") or row.get("date", "")) + "|" + default_ticker,
                    title=row.get("title", "") or "",
                    summary=row.get("content", "")[:500] if row.get("content") else None,
                    url=row.get("link"),
                    published_at=_parse_ts(row.get("date", "")),
                    tickers=clean,
                    provider_sentiment=sent,
                    raw=row,
                )
            )
        return out


def build_providers() -> List[NewsProvider]:
    s = get_settings()
    providers: List[NewsProvider] = [
        MarketauxProvider(s.marketaux_api_key),
        NewsApiAiProvider(s.newsapi_ai_key),
        EodhdProvider(s.eodhd_api_key),
    ]
    enabled = [p for p in providers if p.enabled()]
    log.info("News providers enabled: %s", [p.name for p in enabled])
    return providers


async def fetch_latest_all(
    providers: List[NewsProvider],
    *,
    limit_per_provider: int = 50,
    tickers: Optional[List[str]] = None,
) -> List[RawNewsItem]:
    """Fan out to every enabled provider concurrently."""
    coros = [p.fetch_latest(limit=limit_per_provider, tickers=tickers) for p in providers]
    results = await asyncio.gather(*coros, return_exceptions=False)
    flat: List[RawNewsItem] = []
    for r in results:
        flat.extend(r or [])
    return flat


async def fetch_historical_all(
    providers: List[NewsProvider],
    *,
    start: datetime,
    end: datetime,
    tickers: Optional[List[str]] = None,
    limit_per_provider: int = 100,
) -> List[RawNewsItem]:
    # Avoid silly requests.
    if end <= start:
        return []
    # Cap at 90 days per batch to respect free tiers.
    if (end - start) > timedelta(days=90):
        end = start + timedelta(days=90)
    coros = [
        p.fetch_historical(start=start, end=end, tickers=tickers, limit=limit_per_provider)
        for p in providers
    ]
    results = await asyncio.gather(*coros, return_exceptions=False)
    flat: List[RawNewsItem] = []
    for r in results:
        flat.extend(r or [])
    return flat
