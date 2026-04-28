"""Market-data endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Query

from ..market.data import get_latest_prices, get_price_matrix

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/quote")
async def quote(tickers: str = Query(..., description="Comma-separated tickers")):
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    prices = await get_latest_prices(symbols)
    return prices


@router.get("/history")
async def history(
    tickers: str = Query(...),
    start: str = Query(...),
    end: str = Query(...),
):
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    df = await get_price_matrix(symbols, start, end)
    if df.empty:
        return {"tickers": symbols, "data": []}
    out = []
    for ts, row in df.iterrows():
        entry = {"ts": ts.strftime("%Y-%m-%d")}
        for t in symbols:
            if t in df.columns:
                val = row[t]
                entry[t] = float(val) if val == val else None  # NaN check
        out.append(entry)
    return {"tickers": symbols, "data": out}
