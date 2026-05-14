"""Multi-channel time series loader for the conditioning pipeline.

Per Singh's guidance: "you should have maybe a 100 channel time series
data, or 50 channels, whatever you can afford. You should know where they
are coming from, what is the source."

The K-channel price series serve as *additional conditioning signals* fed
into the adapter: alongside the vanilla SPY forecast and the news embedding,
each channel contributes a compact set of history summary statistics.
Richer cross-asset context lets the adapter read the macro regime the news
is arriving into.

Channels are pulled from yfinance (free, ~15-min delayed). The default
universe is the HOUSE_UNIVERSE env variable (19 liquid ETFs). Expand toward
~50-100 channels by passing a custom ticker list.

For Bloomberg-quality delayed data, see backend/market/perplexity_client.py.

Integration with the adapter
----------------------------
When ``multi_channel_dim > 0`` in ``AdapterConfig``, the adapter concatenates
a (K * 8) vector of per-channel history features to its input. The adapter
architecture is otherwise unchanged. Use ``--use-multi-channel`` in
``scripts/run_conditioning.py`` to activate this path.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

# --- default universe ---------------------------------------------------------
# Starting point: HOUSE_UNIVERSE from .env (19 liquid ETFs). Expand toward
# ~50-100 by adding more sector ETFs, country ETFs, factor ETFs, rates, etc.
_env_universe = os.environ.get("HOUSE_UNIVERSE", "")
DEFAULT_UNIVERSE: List[str] = (
    [t.strip() for t in _env_universe.split(",") if t.strip()]
    if _env_universe
    else [
        # Broad US market
        "SPY", "QQQ", "DIA", "IWM", "VTI",
        # International
        "EFA", "EEM", "VEA", "VWO",
        # Fixed income
        "TLT", "IEF", "SHY", "HYG", "LQD", "BND",
        # Commodities / real assets
        "GLD", "SLV", "USO", "DBC", "PDBC",
        # US sectors (SPDR XL family)
        "XLE", "XLK", "XLF", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE",
        # Factor / volatility
        "VIXY", "MTUM", "VLUE", "QUAL", "SIZE",
    ]
)

CACHE_DIR = PROJECT_ROOT / "data" / "historical" / "multi_channel"

# Number of per-channel history features (must match CHANNEL_FEATURE_NAMES below).
N_CHANNEL_FEATS = 8


# --- per-channel feature names ------------------------------------------------
CHANNEL_FEATURE_NAMES: List[str] = [
    "ret_1d",         # 1-day return at t-1
    "ret_5d",         # 5-day cumulative return
    "ret_20d",        # 20-day cumulative return
    "vol_5d",         # std of last 5 daily returns
    "vol_20d",        # std of last 20 daily returns
    "drawdown_60d",   # drawdown vs trailing 60-day max
    "trend_slope_60d",# OLS slope on last 60 closes, normalized
    "log_close_z",    # close z-scored over last 252 days
]
assert len(CHANNEL_FEATURE_NAMES) == N_CHANNEL_FEATS


# --- data loader --------------------------------------------------------------
def load_channel(
    ticker: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.Series:
    """Download adjusted-close prices for one ticker via yfinance.

    Returns a tz-naive ascending DatetimeIndex Series (float32). Returns an
    empty Series on failure rather than raising, so the universe loader can
    drop bad tickers silently.
    """
    import yfinance as yf  # lazy — not in all envs

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
        if cache_path.exists():
            log.debug("cache hit: %s", cache_path.name)
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            return s.astype(np.float32)

    log.info("yfinance download: %s [%s → %s]", ticker, start, end)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        log.warning("yfinance failed for %s: %s", ticker, exc)
        return pd.Series(dtype=np.float32)

    if df is None or df.empty:
        log.warning("yfinance returned no data for %s", ticker)
        return pd.Series(dtype=np.float32)

    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df["Close"][ticker]
        except KeyError:
            try:
                s = df["Close"].iloc[:, 0]
            except Exception:
                return pd.Series(dtype=np.float32)
    else:
        if "Close" not in df.columns:
            return pd.Series(dtype=np.float32)
        s = df["Close"]

    s = s.dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    s = s.astype(np.float32)

    if use_cache and not s.empty:
        s.to_csv(cache_path)

    return s


def load_universe(
    tickers: Optional[List[str]] = None,
    start: str = "2018-01-01",
    end: Optional[str] = None,
    *,
    use_cache: bool = True,
    rate_limit_delay: float = 0.3,
) -> Dict[str, pd.Series]:
    """Download close-price series for every ticker in the universe.

    Tickers that fail or return no data are silently dropped. The returned
    dict always contains only valid series.
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    out: Dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        s = load_channel(ticker, start, end, use_cache=use_cache)
        if not s.empty:
            out[ticker] = s
        if i < len(tickers) - 1:
            time.sleep(rate_limit_delay)

    log.info(
        "universe loaded: %d / %d channels  [%s → %s]",
        len(out), len(tickers), start, end,
    )
    return out


def align_universe(
    universe: Dict[str, pd.Series],
    *,
    min_coverage: float = 0.9,
) -> pd.DataFrame:
    """Align all channels to a shared trading-day index.

    Channels with < min_coverage data fraction are dropped. Remaining gaps
    are forward-filled then back-filled. Returns a DataFrame with columns =
    tickers, index = DatetimeIndex, sorted ascending.
    """
    if not universe:
        return pd.DataFrame()

    df = pd.concat(universe, axis=1)
    df.columns = list(universe.keys())
    df = df.sort_index()

    good = [c for c in df.columns if df[c].notna().mean() >= min_coverage]
    dropped = set(df.columns) - set(good)
    if dropped:
        log.warning(
            "dropping %d channels below %.0f%% coverage: %s",
            len(dropped), min_coverage * 100, sorted(dropped),
        )
    df = df[good].ffill().bfill()
    return df


# --- per-channel feature extraction -------------------------------------------
def _channel_feats(closes: np.ndarray) -> np.ndarray:
    """Compute 8 history-summary statistics for one price series.

    Reuses the same feature definitions as ``backend.conditioning.adapter``
    so the adapter sees a consistent representation across SPY and the
    additional channels.
    """
    from .adapter import history_features
    return history_features(closes)


def multi_channel_features(
    aligned: pd.DataFrame,
    t: pd.Timestamp,
    *,
    lookback: int = 60,
) -> np.ndarray:
    """Compute per-channel history features at time t.

    Parameters
    ----------
    aligned
        DataFrame from ``align_universe``, columns = tickers.
    t
        The "today" anchor. Features use closes strictly before t.
    lookback
        Number of trading days of price history to use per channel.

    Returns
    -------
    np.ndarray of shape (K * N_CHANNEL_FEATS,), float32.
        Channels with fewer than 20 history rows before t return zero feats.
    """
    if aligned.empty:
        return np.zeros(0, dtype=np.float32)

    K = len(aligned.columns)
    out = np.zeros((K, N_CHANNEL_FEATS), dtype=np.float32)

    loc = aligned.index.get_indexer([t], method="ffill")[0]
    if loc < 0:
        return out.flatten()

    for k, col in enumerate(aligned.columns):
        start_i = max(0, loc - lookback)
        hist = aligned[col].iloc[start_i:loc].to_numpy(dtype=np.float64)
        if hist.size < 20:
            continue
        out[k] = _channel_feats(hist)

    return out.flatten()


# --- manifest / persistence ---------------------------------------------------
@dataclass
class ChannelManifest:
    """Describes the loaded multi-channel universe and its feature layout."""
    tickers: List[str]
    n_channels: int
    start: str
    end: str
    n_trading_days: int
    feature_dim: int        # = n_channels * N_CHANNEL_FEATS

    def to_text(self) -> str:
        tickers_preview = ", ".join(self.tickers[:6])
        if self.n_channels > 6:
            tickers_preview += f", …+{self.n_channels - 6}"
        return (
            f"channels         : {self.n_channels}  ({tickers_preview})\n"
            f"date range       : {self.start} → {self.end}\n"
            f"trading days     : {self.n_trading_days}\n"
            f"feature dim      : {self.feature_dim}  "
            f"({self.n_channels} × {N_CHANNEL_FEATS} feats)\n"
            f"source           : yfinance  (free, ~15-min delayed)\n"
            f"Bloomberg source : see backend/market/perplexity_client.py"
        )

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or (CACHE_DIR / "manifest.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(
                {
                    "tickers": self.tickers,
                    "n_channels": self.n_channels,
                    "start": self.start,
                    "end": self.end,
                    "n_trading_days": self.n_trading_days,
                    "feature_dim": self.feature_dim,
                },
                f,
                indent=2,
            )
        return p


def build_channel_manifest(aligned: pd.DataFrame, start: str, end: str) -> ChannelManifest:
    tickers = list(aligned.columns)
    return ChannelManifest(
        tickers=tickers,
        n_channels=len(tickers),
        start=start,
        end=end,
        n_trading_days=len(aligned),
        feature_dim=len(tickers) * N_CHANNEL_FEATS,
    )
