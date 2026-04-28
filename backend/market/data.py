"""Market data abstraction layer.

Historical daily OHLCV: yfinance (free, reliable, fast).
Latest price: yfinance Ticker.fast_info, with a short in-memory cache.

A thin process-local cache avoids hammering yfinance when many UI requests
arrive in the same few seconds.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from ..core.logging import get_logger

log = get_logger(__name__)

# -------- historical cache ----------------------------------------------------
_HIST_TTL_SEC = 60 * 30  # 30 minutes
_hist_cache: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
_hist_lock = threading.Lock()

# -------- quote cache ---------------------------------------------------------
_QUOTE_TTL_SEC = 20
_quote_cache: Dict[str, Tuple[float, float]] = {}
_quote_lock = threading.Lock()


def _yf_download_sync(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Blocking yfinance call — run via `asyncio.to_thread`."""
    # `group_by='ticker'` keeps a multi-index when multiple tickers, but for
    # a single ticker we flatten below.
    df = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    return df


async def get_price_history(
    tickers: List[str], start: str, end: str
) -> Dict[str, pd.Series]:
    """Return adjusted close series per ticker (index: DatetimeIndex, tz-naive)."""
    if not tickers:
        return {}
    tickers = [t.upper() for t in tickers]
    key = (",".join(sorted(tickers)), start, end)

    with _hist_lock:
        hit = _hist_cache.get(key)
        if hit and (time.time() - hit[0]) < _HIST_TTL_SEC:
            df = hit[1]
        else:
            df = None

    if df is None:
        df = await asyncio.to_thread(_yf_download_sync, tickers, start, end)
        if df is None or df.empty:
            log.warning("yfinance returned no data for %s [%s..%s]", tickers, start, end)
            with _hist_lock:
                _hist_cache[key] = (time.time(), pd.DataFrame())
            return {}
        with _hist_lock:
            _hist_cache[key] = (time.time(), df)

    out: Dict[str, pd.Series] = {}
    if len(tickers) == 1:
        t = tickers[0]
        if isinstance(df.columns, pd.MultiIndex):
            try:
                s = df[t]["Close"]
            except KeyError:
                s = df["Close"] if "Close" in df.columns else pd.Series(dtype=float)
        else:
            s = df["Close"] if "Close" in df.columns else pd.Series(dtype=float)
        if not s.empty:
            out[t] = s.dropna()
        return out

    # Multi-ticker frame
    for t in tickers:
        try:
            s = df[t]["Close"]
            if s is not None and not s.dropna().empty:
                out[t] = s.dropna()
        except Exception:  # noqa: BLE001
            continue
    return out


async def get_price_matrix(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame of adjusted-close prices, columns = tickers."""
    series = await get_price_history(tickers, start, end)
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1)
    df.columns = list(series.keys())
    df = df.sort_index().ffill().dropna(how="all")
    return df


def _latest_price_sync(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        # fast_info is fast and avoids the yfinance rate-limit-prone .info
        fi = t.fast_info
        px = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        if px is None:
            hist = t.history(period="1d", interval="1d")
            if not hist.empty:
                px = float(hist["Close"].iloc[-1])
        return float(px) if px is not None else None
    except Exception:  # noqa: BLE001
        return None


async def get_latest_prices(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    tickers = [t.upper() for t in tickers]
    out: Dict[str, float] = {}
    to_fetch: List[str] = []
    now = time.time()
    with _quote_lock:
        for t in tickers:
            hit = _quote_cache.get(t)
            if hit and (now - hit[0]) < _QUOTE_TTL_SEC:
                out[t] = hit[1]
            else:
                to_fetch.append(t)

    if to_fetch:
        # Fetch in parallel but bounded.
        semaphore = asyncio.Semaphore(8)

        async def _one(t: str):
            async with semaphore:
                return t, await asyncio.to_thread(_latest_price_sync, t)

        results = await asyncio.gather(*(_one(t) for t in to_fetch))
        with _quote_lock:
            now = time.time()
            for t, px in results:
                if px is not None:
                    _quote_cache[t] = (now, px)
                    out[t] = px
    return out
