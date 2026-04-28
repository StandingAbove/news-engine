"""Small helper for loading real price series for the "then move to real
data" step in Dr. Singh's progression.

We piggyback on the existing market data service (`backend.market`) when
available, otherwise fall back to a direct yfinance call. The function
returns a plain numpy array so it plugs straight into the forecasting
harness.
"""
from __future__ import annotations

from typing import List

import numpy as np


def load_close_series(ticker: str, period: str = "2y") -> np.ndarray:
    """Fetch adjusted-close prices for `ticker` as a 1-D numpy array."""
    try:
        import yfinance as yf
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("yfinance not installed") from e

    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(f"no data for {ticker}")
    closes = df["Close"].dropna().astype(float).to_numpy()
    if closes.size < 50:
        raise RuntimeError(f"too few points for {ticker}: {closes.size}")
    return closes


def load_many(tickers: List[str], period: str = "2y") -> dict:
    """Load multiple tickers, skipping any that fail."""
    out = {}
    for t in tickers:
        try:
            out[t] = load_close_series(t, period=period)
        except Exception:  # noqa: BLE001
            continue
    return out
