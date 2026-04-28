"""Map entities/keywords in news text to tickers and sectors, including
cross-market spillover (e.g. Hong Kong event → EEM, FXI-like exposure via EEM/EFA).
"""
from __future__ import annotations

import re
from typing import Dict, List, Set


# Human-readable name → ticker. Deliberately conservative; extend as needed.
COMPANY_TO_TICKER: Dict[str, str] = {
    "apple": "AAPL", "apple inc": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL",
    "meta": "META", "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "citigroup": "C",
    "wells fargo": "WFC",
    "berkshire hathaway": "BRK-B",
    "exxon": "XOM", "exxonmobil": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "johnson & johnson": "JNJ",
    "boeing": "BA",
    "intel": "INTC",
    "amd": "AMD",
    "taiwan semiconductor": "TSM", "tsmc": "TSM",
    "alibaba": "BABA",
    "tencent": "TCEHY",
    "samsung": "005930.KS",
    "nestle": "NSRGY", "nestlé": "NSRGY",
}

# Keyword → sector ETF (used when news is about a sector but no company is named).
SECTOR_KEYWORDS: Dict[str, str] = {
    # Tech
    "semiconductor": "XLK", "chip": "XLK", "ai": "XLK", "cloud": "XLK",
    "software": "XLK", "tech": "XLK",
    # Financials
    "bank": "XLF", "banks": "XLF", "financial": "XLF", "rate hike": "XLF",
    # Energy
    "oil": "XLE", "crude": "XLE", "opec": "XLE", "gas prices": "XLE", "refinery": "XLE",
    # Healthcare
    "pharma": "XLV", "drug": "XLV", "fda": "XLV", "clinical trial": "XLV",
    # Consumer
    "retail": "XLY", "consumer": "XLY", "e-commerce": "XLY",
    # Industrial / materials
    "airline": "XLI", "manufacturing": "XLI", "defense": "XLI",
    # Real estate
    "housing": "XLRE", "mortgage": "XLRE", "reit": "XLRE",
    # Utilities / staples
    "utility": "XLU", "utilities": "XLU",
    "grocery": "XLP", "staples": "XLP",
    # Bonds / rates
    "treasury": "TLT", "fed": "TLT", "federal reserve": "TLT", "fomc": "TLT",
    "yield": "TLT", "bond": "TLT",
    # Gold / commodities
    "gold": "GLD", "inflation": "GLD",
}

# Region → ETF (captures the cross-market spillover idea).
REGION_KEYWORDS: Dict[str, str] = {
    "hong kong": "EEM", "china": "EEM", "beijing": "EEM", "shanghai": "EEM",
    "emerging markets": "EEM",
    "europe": "EFA", "eurozone": "EFA", "ecb": "EFA", "germany": "EFA",
    "uk": "EFA", "britain": "EFA", "france": "EFA",
    "japan": "EFA", "tokyo": "EFA", "boj": "EFA",
    "korea": "EEM", "taiwan": "EEM", "india": "EEM",
    "russia": "EEM", "saudi": "EEM", "mideast": "EEM", "middle east": "EEM",
}

# Broad-market keywords → SPY (catch-all for macro events).
BROAD_MARKET_KEYWORDS: List[str] = [
    "s&p", "s&p 500", "stocks", "wall street", "market", "cpi", "ppi", "gdp",
    "jobs report", "unemployment", "earnings season", "nasdaq",
]

_ticker_re = re.compile(r"\b([A-Z]{1,5})\b")


def extract_tickers(text: str, hint: List[str] | None = None) -> List[str]:
    """Pull plausible tickers out of a headline/summary.

    Strategy:
    1. Any explicit symbols the provider already supplied (`hint`).
    2. Company-name matches against COMPANY_TO_TICKER.
    3. Sector-keyword matches → sector ETFs.
    4. Region-keyword matches → regional ETFs.
    5. Fallback to SPY for broad-market keywords.
    """
    if not text:
        return list(dict.fromkeys(hint or []))

    lower = text.lower()
    tickers: List[str] = []
    seen: Set[str] = set()

    def add(sym: str) -> None:
        sym = sym.upper()
        if sym and sym not in seen:
            seen.add(sym)
            tickers.append(sym)

    for h in hint or []:
        if h:
            add(h)

    for name, sym in COMPANY_TO_TICKER.items():
        if name in lower:
            add(sym)

    for kw, sym in SECTOR_KEYWORDS.items():
        if kw in lower:
            add(sym)

    for kw, sym in REGION_KEYWORDS.items():
        if kw in lower:
            add(sym)

    if not tickers:
        for kw in BROAD_MARKET_KEYWORDS:
            if kw in lower:
                add("SPY")
                break

    # As a last resort, pick uppercase tokens that look like tickers
    # (excluding common all-caps words).
    if not tickers:
        stop = {"CEO", "USA", "EU", "UK", "IPO", "AI", "GPT", "ETF", "FOMC", "SEC"}
        for tok in _ticker_re.findall(text):
            if tok in stop:
                continue
            if 2 <= len(tok) <= 5:
                add(tok)
                if len(tickers) >= 2:
                    break

    return tickers
