"""Perplexity API client for Bloomberg-sourced financial data.

Per Singh's guidance (May 2026): "Perplexity has, for ~$20/month, access to
an API through which, with some delay, you can query Bloomberg Terminal."

The sonar-pro model has live web search and cites Bloomberg, Reuters, WSJ,
FT, and similar primary sources. Not true real-time (expect minutes to
multi-hour delay depending on the source), but far richer than free APIs for
structured financial news and data.

Setup
-----
1. Subscribe to Perplexity at perplexity.ai/settings/api (~$20/month plan).
2. Add ``PERPLEXITY_API_KEY=<your-key>`` to ``.env``.

Usage
-----
    from backend.market.perplexity_client import PerplexityFinanceClient
    client = PerplexityFinanceClient()

    # Broad market summary
    resp = client.market_summary("2025-04-29")
    print(resp.content)

    # Targeted news for specific tickers
    resp = client.financial_news(["SPY", "NVDA"], "2025-04-29")

    # Sector ETF performance table
    resp = client.sector_performance("2025-04-29")

    # Specific Bloomberg field
    resp = client.bloomberg_data_point("SPY", "30-day implied vol", "2025-04-29")

All responses are disk-cached by (query type, tickers, date) so re-running
the conditioning pipeline does not burn extra API credits.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

_API_BASE = "https://api.perplexity.ai"
_DEFAULT_MODEL = "sonar-pro"
_CACHE_DIR = PROJECT_ROOT / "data" / "perplexity_cache"

_SYSTEM_PROMPT = (
    "You are a financial data assistant with access to Bloomberg Terminal, "
    "Reuters, Wall Street Journal, and Financial Times. "
    "Answer with precise, factual financial information. "
    "Cite your primary source and the publication timestamp for each data point. "
    "Be concise. Use tables where appropriate."
)


@dataclass
class PerplexityResponse:
    """Structured response from the Perplexity API."""
    query: str
    content: str
    citations: List[str] = field(default_factory=list)
    model: str = _DEFAULT_MODEL
    usage: Dict = field(default_factory=dict)
    cached: bool = False

    def __str__(self) -> str:
        lines = [self.content]
        if self.citations:
            lines.append("\nSources:")
            for i, c in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {c}")
        return "\n".join(lines)


class PerplexityFinanceClient:
    """Thin wrapper around Perplexity's /chat/completions endpoint, scoped to
    financial data queries.

    All responses are cached to disk by a stable key derived from the query
    type, tickers, and date. Re-running the conditioning pipeline with the
    same date window burns zero additional API credits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        cache: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "PERPLEXITY_API_KEY is not set. "
                "Add it to .env or pass api_key= directly.\n"
                "Get a key at: https://perplexity.ai/settings/api"
            )
        self.model = model
        self.cache = cache
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        if self.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- core call ----------------------------------------------------------------
    def _call(self, prompt: str, *, cache_key: str = "") -> PerplexityResponse:
        """Send one chat-completion request. Returns a cached response if one
        exists on disk for ``cache_key``."""
        if self.cache and cache_key:
            hit = self._load_cache(cache_key)
            if hit is not None:
                return hit

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for PerplexityFinanceClient: pip install httpx"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])
        usage = data.get("usage", {})

        result = PerplexityResponse(
            query=prompt,
            content=content,
            citations=citations,
            model=self.model,
            usage=usage,
            cached=False,
        )
        if self.cache and cache_key:
            self._save_cache(cache_key, result)
        return result

    # --- domain methods -----------------------------------------------------------
    def market_summary(
        self,
        date: str,
        *,
        universe: Optional[List[str]] = None,
    ) -> PerplexityResponse:
        """Broad market summary for ``date`` (ISO YYYY-MM-DD).

        Covers S&P 500, Nasdaq, Russell 2000, sector rotation, macro events,
        and VIX. If ``universe`` is provided the summary focuses on those
        tickers.
        """
        if universe:
            focus = f"Focus on these instruments: {', '.join(universe)}."
        else:
            focus = "Cover S&P 500 sectors, rates, commodities, and FX."

        prompt = (
            f"Provide a concise financial market summary for {date}. "
            f"{focus} "
            "Include: "
            "(1) major index returns (S&P 500, Nasdaq 100, Russell 2000), "
            "(2) best- and worst-performing sectors with approximate % return, "
            "(3) key macro events or data releases that day, "
            "(4) VIX level and whether it rose or fell, "
            "(5) any notable news that moved the market. "
            "Cite Bloomberg or Reuters for each data point."
        )
        return self._call(prompt, cache_key=f"market_summary_{date}")

    def financial_news(
        self,
        tickers: List[str],
        date: str,
        *,
        max_items: int = 10,
    ) -> PerplexityResponse:
        """Retrieve the most market-moving news for ``tickers`` on or just
        before ``date``.

        Each headline includes source, timestamp, sentiment estimate, and
        estimated price-impact direction — ready to feed into the news
        encoder.
        """
        ticker_str = ", ".join(tickers)
        prompt = (
            f"List the {max_items} most market-moving news items for "
            f"{ticker_str} on or immediately before {date}. "
            "For each item include: "
            "(1) headline, "
            "(2) source name and publication time (as precise as available), "
            "(3) estimated sentiment: positive / negative / neutral, "
            "(4) estimated price-impact direction: up / down / unclear. "
            "Prioritize Bloomberg, Reuters, and WSJ sources."
        )
        key = f"news_{'_'.join(sorted(tickers))}_{date}"
        return self._call(prompt, cache_key=key)

    def sector_performance(self, date: str) -> PerplexityResponse:
        """SPDR sector ETF returns on ``date``, ranked best to worst.

        Also notes the macro theme driving top and bottom performers.
        """
        sectors = "XLE, XLK, XLF, XLV, XLY, XLI, XLP, XLU, XLB, XLRE"
        prompt = (
            f"What were the percentage returns for the SPDR sector ETFs "
            f"({sectors}) on {date}? "
            "Present as a table ranked from best to worst. "
            "For the top and bottom performer, note the macro theme or "
            "news item that drove the move. "
            "Source from Bloomberg or Reuters."
        )
        return self._call(prompt, cache_key=f"sector_perf_{date}")

    def bloomberg_data_point(
        self,
        ticker: str,
        field: str,
        date: str,
    ) -> PerplexityResponse:
        """Query a specific Bloomberg data field for a ticker on a date.

        Examples
        --------
        bloomberg_data_point("SPY", "30-day implied volatility", "2025-04-29")
        bloomberg_data_point("TLT", "yield to maturity", "2025-04-29")
        bloomberg_data_point("GLD", "net ETF flows (1-week)", "2025-04-29")
        """
        prompt = (
            f"What was the Bloomberg {field} for {ticker} on {date}? "
            "Provide the exact numeric value with units, the source, and "
            "the time of the data point. "
            "If the exact value is not available, provide the closest "
            "available data and state the caveat clearly."
        )
        safe_field = field.replace(" ", "_")[:40]
        return self._call(prompt, cache_key=f"bbg_{ticker}_{safe_field}_{date}")

    def macro_indicators(self, date: str) -> PerplexityResponse:
        """Key macro indicators for ``date``: yields, spreads, DXY, oil.

        Useful as additional conditioning context for the time-series model.
        """
        prompt = (
            f"For {date}, provide the following macro indicators "
            "sourced from Bloomberg or Reuters: "
            "(1) US 2-year and 10-year Treasury yields, "
            "(2) 2s10s spread, "
            "(3) investment-grade and high-yield credit spreads (OAS), "
            "(4) DXY (dollar index) level, "
            "(5) WTI crude oil price, "
            "(6) Gold spot price. "
            "Present as a concise table with date, value, and day-over-day change."
        )
        return self._call(prompt, cache_key=f"macro_{date}")

    # --- cache helpers ------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:120]
        return self.cache_dir / f"{safe}.json"

    def _load_cache(self, key: str) -> Optional[PerplexityResponse]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with p.open() as f:
                d = json.load(f)
            return PerplexityResponse(
                query=d.get("query", ""),
                content=d["content"],
                citations=d.get("citations", []),
                model=d.get("model", self.model),
                usage=d.get("usage", {}),
                cached=True,
            )
        except Exception as exc:
            log.debug("cache read failed for %s: %s", key, exc)
            return None

    def _save_cache(self, key: str, resp: PerplexityResponse) -> None:
        p = self._cache_path(key)
        try:
            with p.open("w") as f:
                json.dump(
                    {
                        "query": resp.query,
                        "content": resp.content,
                        "citations": resp.citations,
                        "model": resp.model,
                        "usage": resp.usage,
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            log.debug("cache write failed for %s: %s", key, exc)
