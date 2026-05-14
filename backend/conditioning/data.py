"""SPY daily-history loader (FactSet export).

Part 1 of the news-conditioned TimesFM-2 plan: read the FactSet SPY CSV
shipped under ``data/historical/SPY.csv``, parse it into a tidy DataFrame
indexed by trading day, and expose a couple of tiny helpers the rest of the
conditioning pipeline needs:

* ``load_spy()`` — returns a tidy DataFrame, oldest first, tz-naive index.
* ``trading_days_between(start, end)`` — list of trading days inside the
  loaded series, useful for aligning the news side later.
* ``return_window(df, t, H)`` — H-day forward returns starting at ``t``.

The FactSet CSV header is::

    Date,Price,CVol,% Change,Open,Low,High,NAV,Total Return (Gross),
    % Return,Cumulative Return %

Date format is ``MM/DD/YY``; Price is the close. Volume is ``CVol``.
``Total Return (Gross)`` is index-style (1.0 → cumulative gross return).
The CSV ships newest-first; we sort ascending on load.

This module has no dependency on ``backend/forecast/`` or ``backend/news/``
and must stay that way — it is the leaf at the bottom of the conditioning
DAG.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import PROJECT_ROOT

# --- paths --------------------------------------------------------------------
DEFAULT_SPY_PATH = PROJECT_ROOT / "data" / "historical" / "SPY.csv"


# --- core loader --------------------------------------------------------------
def load_spy(
    path: Optional[Path | str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load FactSet SPY daily CSV into a tidy DataFrame.

    Parameters
    ----------
    path
        Override the CSV location. Defaults to ``data/historical/SPY.csv``.
    start, end
        Optional ISO-format date bounds (inclusive). If omitted, the full
        history is returned.

    Returns
    -------
    pandas.DataFrame
        Indexed by tz-naive ``DatetimeIndex`` (sorted ascending), with
        columns ``open``, ``high``, ``low``, ``close``, ``volume``,
        ``pct_change`` (FactSet's daily % change, in *decimal* form), and
        ``total_return`` (FactSet gross-return index, normalized so the
        first row in the loaded slice == 1.0).
    """
    p = Path(path) if path else DEFAULT_SPY_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"SPY history not found at {p}. "
            "Drop the FactSet export at data/historical/SPY.csv."
        )

    raw = pd.read_csv(p, encoding="utf-8-sig", thousands=",")

    # Normalize column names (FactSet's headers have spaces and %).
    rename = {
        "Date": "date",
        "Price": "close",
        "CVol": "volume",
        "% Change": "pct_change",
        "Open": "open",
        "Low": "low",
        "High": "high",
        "NAV": "nav",
        "Total Return (Gross)": "total_return_raw",
        "% Return": "total_pct_return",
        "Cumulative Return %": "cum_return_pct",
    }
    raw = raw.rename(columns=rename)

    # Parse dates. FactSet uses MM/DD/YY — pandas handles that reliably.
    raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%y")

    df = raw[["date", "open", "high", "low", "close", "volume",
              "pct_change", "total_return_raw"]].copy()

    # Coerce types — every numeric column should be float-able.
    for col in ("open", "high", "low", "close", "pct_change", "total_return_raw"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    # Sort ascending (FactSet ships newest-first).
    df = df.sort_values("date").reset_index(drop=True)

    # FactSet stores % change as percent (e.g. -1.31 = -1.31%). Convert to
    # decimal so downstream code never has to think about the unit again.
    df["pct_change"] = df["pct_change"] / 100.0

    # Date-window slice.
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("SPY load returned no rows for the requested window.")

    # Normalize total_return to start at 1.0 within the loaded slice.
    base = df["total_return_raw"].iloc[0]
    if pd.isna(base) or base == 0:
        # First row sometimes has a NaN return; fall back to close-price
        # ratio anchored at 1.0.
        df["total_return"] = df["close"] / df["close"].iloc[0]
    else:
        df["total_return"] = df["total_return_raw"] / base

    df = df.set_index("date").drop(columns=["total_return_raw"])

    return df


# --- small helpers ------------------------------------------------------------
def trading_days_between(df: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    """Trading days actually present in ``df`` between ``start`` and ``end``."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    mask = (df.index >= s) & (df.index <= e)
    return df.index[mask]


def return_window(df: pd.DataFrame, t: pd.Timestamp | str, H: int) -> np.ndarray:
    """H-day forward simple returns starting at ``t`` (close-to-close)."""
    t = pd.Timestamp(t)
    if t not in df.index:
        raise KeyError(f"{t.date()} is not a trading day in the loaded SPY series.")
    i = df.index.get_loc(t)
    end_i = i + H
    if end_i >= len(df):
        raise ValueError(
            f"Not enough forward data: requested {H} days starting {t.date()}, "
            f"but only {len(df) - 1 - i} trading days remain."
        )
    closes = df["close"].iloc[i : end_i + 1].to_numpy()
    return closes[1:] / closes[:-1] - 1.0


# --- train / holdout split ----------------------------------------------------
@dataclass
class Split:
    train: pd.DataFrame
    holdout: pd.DataFrame
    holdout_start: pd.Timestamp

    def to_text(self) -> str:
        return (
            f"train   : {len(self.train):>5d} rows  "
            f"{self.train.index.min().date()} → {self.train.index.max().date()}\n"
            f"holdout : {len(self.holdout):>5d} rows  "
            f"{self.holdout.index.min().date()} → {self.holdout.index.max().date()}\n"
            f"split at : {self.holdout_start.date()}"
        )


def train_holdout_split(
    df: pd.DataFrame,
    holdout_start: str | pd.Timestamp,
) -> Split:
    """Split SPY history into a train period (strictly before ``holdout_start``)
    and a holdout period (``holdout_start`` onward, inclusive).

    Singh's framing: "imagine you are living in last year." The model is fit
    on the train half and replayed forward day-by-day across the holdout half;
    no information from the holdout ever leaks into the fit.
    """
    cutoff = pd.Timestamp(holdout_start)
    train = df[df.index < cutoff]
    holdout = df[df.index >= cutoff]
    if train.empty or holdout.empty:
        raise ValueError(
            f"holdout_start={cutoff.date()} produced an empty split "
            f"(train={len(train)}, holdout={len(holdout)})"
        )
    return Split(train=train, holdout=holdout, holdout_start=cutoff)


# --- on-disk cache ------------------------------------------------------------
# We use pickle for the cache rather than parquet so the package has zero
# extra runtime deps beyond pandas (parquet needs pyarrow/fastparquet, neither
# of which is in requirements.txt yet). For 8k rows the pickle is ~700KB and
# loads in <50ms.
def cache_frame(df: pd.DataFrame, path: Optional[Path | str] = None) -> Path:
    """Persist the loaded SPY frame so downstream parts (alignment, embedding,
    training, eval) don't re-parse the CSV on every run.
    """
    out = Path(path) if path else PROJECT_ROOT / "data" / "historical" / "SPY.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    return out


def load_cached(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Read the cache written by :func:`cache_frame`."""
    p = Path(path) if path else PROJECT_ROOT / "data" / "historical" / "SPY.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"No SPY cache at {p}. Call cache_frame(load_spy()) first."
        )
    return pd.read_pickle(p)


# --- summary dataclass for sanity-checking ------------------------------------
@dataclass
class SPYSummary:
    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    weekend_rows: int
    nan_close_rows: int
    median_daily_return: float
    annualized_return: float
    annualized_vol: float

    def to_text(self) -> str:
        return (
            f"SPY rows           : {self.n_rows:,}\n"
            f"first → last date  : {self.start.date()} → {self.end.date()}\n"
            f"weekend rows       : {self.weekend_rows}  (must be 0)\n"
            f"NaN close rows     : {self.nan_close_rows}  (must be 0)\n"
            f"median daily ret.  : {self.median_daily_return:+.4%}\n"
            f"annualized return  : {self.annualized_return:+.2%}\n"
            f"annualized vol.    : {self.annualized_vol:.2%}"
        )


def summarize(df: pd.DataFrame) -> SPYSummary:
    weekday = df.index.weekday  # 0=Mon..4=Fri
    rets = df["close"].pct_change().dropna()
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std(ddof=0) * np.sqrt(252)
    return SPYSummary(
        n_rows=len(df),
        start=df.index.min(),
        end=df.index.max(),
        weekend_rows=int(((weekday == 5) | (weekday == 6)).sum()),
        nan_close_rows=int(df["close"].isna().sum()),
        median_daily_return=float(rets.median()),
        annualized_return=float(ann_ret),
        annualized_vol=float(ann_vol),
    )
