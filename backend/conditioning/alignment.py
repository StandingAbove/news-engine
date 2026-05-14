"""News ↔ time-series alignment for the conditioning training set.

Per Singh's Apr-28 framing: at training time we walk through the historical
calendar day-by-day. For each "today" t we need three things:

1. **History**  prices observed strictly before t (the model's input context).
2. **News window**  every news item published in some lookback window
   ``[t-K_news, t)``  this is the conditioning signal.
3. **Target**  the actual prices over ``[t, t+H]``  this is what the
   conditioned forecast will be scored against.

This module produces ``AlignedSample`` triples and a generator that walks the
historical calendar producing one sample per trading day. Downstream code
(Part 3 turns the news window into a fixed-size embedding; Parts 5 and 6 use
the (history, news_emb, target) triples for training).

The news side is read straight from the project's existing ``NewsItem`` table
(the same one ``backend/news/aggregator.py`` writes to). The package does not
import any aggregator or provider code  it only reads from the DB.

Decoupling note: this module imports from ``backend.core.db`` (the shared
schema) and ``backend.conditioning.data``. It never touches
``backend/forecast/`` or any v2 code.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import select

from ..core.config import PROJECT_ROOT
from ..core.db import NewsItem, get_session_factory
from .data import load_cached, load_spy


# --- types --------------------------------------------------------------------
@dataclass
class NewsRecord:
    """Subset of NewsItem we need for conditioning."""
    published_at: datetime
    source: str
    title: str
    summary: Optional[str]
    sentiment: float
    tickers: List[str] = field(default_factory=list)


@dataclass
class AlignedSample:
    """One (today, history, news, target) tuple for training."""
    t: pd.Timestamp                         # the "today" we are forecasting from
    history: np.ndarray                     # close prices strictly before t
    history_dates: pd.DatetimeIndex         # corresponding dates
    news: List[NewsRecord]                  # all news in [t - K_news days, t)
    target: np.ndarray                      # close prices on [t, t+H] (length H+1)
    target_dates: pd.DatetimeIndex
    forward_returns: np.ndarray             # length H — close-to-close returns

    def n_news(self) -> int:
        return len(self.news)


# --- news loader --------------------------------------------------------------
async def _fetch_news_rows(
    start: datetime,
    end: datetime,
    ticker_filter: Optional[str] = None,
) -> List[NewsRecord]:
    """Pull every NewsItem with published_at in [start, end) into memory.

    For SPY-only conditioning we don't filter on ticker — broad market news
    is exactly what we want as the conditioning signal. The ``ticker_filter``
    knob exists for when we expand to multi-channel.
    """
    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(NewsItem)
            .where(NewsItem.published_at >= start)
            .where(NewsItem.published_at < end)
            .order_by(NewsItem.published_at.asc())
        )
        result = await session.execute(stmt)
        rows = list(result.scalars())

    out: List[NewsRecord] = []
    for r in rows:
        try:
            tickers = json.loads(r.tickers_json or "[]")
        except (TypeError, ValueError):
            tickers = []
        if ticker_filter is not None and ticker_filter not in tickers:
            continue
        # Normalise tz-naive vs tz-aware (the DB stores UTC, sometimes naive).
        ts = r.published_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append(
            NewsRecord(
                published_at=ts,
                source=r.source or "",
                title=r.title or "",
                summary=r.summary,
                sentiment=float(r.sentiment or 0.0),
                tickers=list(tickers),
            )
        )
    return out


def fetch_news_rows(
    start: datetime,
    end: datetime,
    ticker_filter: Optional[str] = None,
) -> List[NewsRecord]:
    """Sync wrapper around the async fetch — convenient for notebooks/scripts."""
    return asyncio.run(_fetch_news_rows(start, end, ticker_filter))


# --- core walker --------------------------------------------------------------
@dataclass
class AlignmentConfig:
    history_lookback: int = 512   # trading days of price history fed to TimesFM
    news_lookback_days: int = 7   # calendar days of news preceding each "today"
    horizon: int = 20             # trading days the model must forecast
    min_news_per_sample: int = 0  # set >0 to skip days with no news at all

    def __post_init__(self) -> None:
        if self.history_lookback < 32:
            raise ValueError("history_lookback < 32 — TimesFM needs more context")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.news_lookback_days < 1:
            raise ValueError("news_lookback_days must be >= 1")


def walk_aligned(
    df: pd.DataFrame,
    cfg: AlignmentConfig,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    news: Optional[Sequence[NewsRecord]] = None,
) -> Iterator[AlignedSample]:
    """Walk the SPY calendar producing one ``AlignedSample`` per trading day.

    Days where there isn't enough history (first ``history_lookback`` rows) or
    enough forward data (last ``horizon`` rows) are silently skipped — the
    walker only emits days that have a complete training sample.

    ``news`` — if provided, must be a flat list of NewsRecord covering
    ``[df.index.min() - news_lookback_days, df.index.max()]``. The walker
    bisects it per-sample. If ``None``, the walker hits the DB once per
    sample (slower but always correct).
    """
    s = pd.Timestamp(start) if start is not None else df.index.min()
    e = pd.Timestamp(end) if end is not None else df.index.max()

    # Sort news once for fast bisect.
    news_sorted: Optional[List[NewsRecord]] = None
    if news is not None:
        news_sorted = sorted(news, key=lambda n: n.published_at)
    news_ts = (
        np.array([n.published_at.timestamp() for n in news_sorted])
        if news_sorted is not None
        else None
    )

    closes = df["close"].to_numpy()
    dates = df.index

    for i, t in enumerate(dates):
        if t < s or t > e:
            continue
        if i < cfg.history_lookback:
            continue
        if i + cfg.horizon >= len(df):
            continue

        history = closes[i - cfg.history_lookback : i]
        history_dates = dates[i - cfg.history_lookback : i]
        target = closes[i : i + cfg.horizon + 1]
        target_dates = dates[i : i + cfg.horizon + 1]
        forward_returns = target[1:] / target[:-1] - 1.0

        # News window: [t - news_lookback_days, t)
        news_start = (t - pd.Timedelta(days=cfg.news_lookback_days)).to_pydatetime().replace(tzinfo=timezone.utc)
        news_end = t.to_pydatetime().replace(tzinfo=timezone.utc)

        if news_sorted is not None and news_ts is not None:
            lo = int(np.searchsorted(news_ts, news_start.timestamp(), side="left"))
            hi = int(np.searchsorted(news_ts, news_end.timestamp(), side="left"))
            news_window = news_sorted[lo:hi]
        else:
            news_window = fetch_news_rows(news_start, news_end)

        if len(news_window) < cfg.min_news_per_sample:
            continue

        yield AlignedSample(
            t=t,
            history=history,
            history_dates=history_dates,
            news=list(news_window),
            target=target,
            target_dates=target_dates,
            forward_returns=forward_returns,
        )


# --- on-disk dataset ----------------------------------------------------------
@dataclass
class DatasetManifest:
    n_samples: int
    history_lookback: int
    horizon: int
    news_lookback_days: int
    first_t: pd.Timestamp
    last_t: pd.Timestamp
    median_news_per_sample: float
    zero_news_samples: int
    path: Path


def build_dataset(
    cfg: AlignmentConfig,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    out_path: Optional[Path | str] = None,
    news_override: Optional[Sequence[NewsRecord]] = None,
) -> DatasetManifest:
    """Materialize the full aligned dataset to a single ``.npz`` file.

    Stored arrays
    -------------
    * ``t``                       (N,)   timestamps as int64 ns since epoch
    * ``history``                 (N, L) close prices  (L = history_lookback)
    * ``target``                  (N, H+1) close prices on [t, t+H]
    * ``forward_returns``         (N, H) close-to-close returns
    * ``news_count``              (N,)   how many news items fell in the window
    * ``news_titles``             list-of-list[str], pickled separately as .json
    * ``news_sentiment_mean``     (N,)   mean VADER sentiment over the window
    * ``news_sentiment_std``      (N,)   std VADER sentiment
    * ``news_pos_share``          (N,)   share of items with sentiment > 0.1
    * ``news_neg_share``          (N,)   share of items with sentiment < -0.1
    """
    df = load_cached() if (PROJECT_ROOT / "data" / "historical" / "SPY.pkl").exists() else load_spy()

    # Pre-fetch news once so walk_aligned doesn't hit the DB N times.
    if news_override is not None:
        all_news = list(news_override)
    else:
        s = pd.Timestamp(start) if start else df.index.min()
        e = pd.Timestamp(end) if end else df.index.max()
        # widen by news_lookback_days so the earliest sample has its full window
        s_news = (s - pd.Timedelta(days=cfg.news_lookback_days)).to_pydatetime().replace(tzinfo=timezone.utc)
        e_news = (e + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)
        all_news = fetch_news_rows(s_news, e_news)

    samples: List[AlignedSample] = list(
        walk_aligned(df, cfg, start=start, end=end, news=all_news)
    )

    if not samples:
        raise RuntimeError("walk_aligned produced 0 samples — check the date window")

    N = len(samples)
    L = cfg.history_lookback
    H = cfg.horizon

    t_arr = np.array([s.t.value for s in samples], dtype=np.int64)
    history_arr = np.stack([s.history for s in samples]).astype(np.float32)
    target_arr = np.stack([s.target for s in samples]).astype(np.float32)
    fwd_arr = np.stack([s.forward_returns for s in samples]).astype(np.float32)

    news_count = np.array([s.n_news() for s in samples], dtype=np.int32)
    sent = [np.array([n.sentiment for n in s.news], dtype=np.float32) for s in samples]
    sent_mean = np.array([float(a.mean()) if a.size else 0.0 for a in sent], dtype=np.float32)
    sent_std = np.array([float(a.std(ddof=0)) if a.size else 0.0 for a in sent], dtype=np.float32)
    pos_share = np.array(
        [float((a > 0.1).mean()) if a.size else 0.0 for a in sent], dtype=np.float32
    )
    neg_share = np.array(
        [float((a < -0.1).mean()) if a.size else 0.0 for a in sent], dtype=np.float32
    )

    out = Path(out_path) if out_path else PROJECT_ROOT / "data" / "conditioning" / "aligned.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t=t_arr,
        history=history_arr,
        target=target_arr,
        forward_returns=fwd_arr,
        news_count=news_count,
        news_sentiment_mean=sent_mean,
        news_sentiment_std=sent_std,
        news_pos_share=pos_share,
        news_neg_share=neg_share,
        history_lookback=L,
        horizon=H,
        news_lookback_days=cfg.news_lookback_days,
    )

    # Stash the actual headlines as JSON next to the .npz. We keep them out of
    # the .npz (which is best for fixed-size numerics) so Part 3 can choose
    # a richer text encoder later.
    titles_path = out.with_suffix(".titles.jsonl")
    with titles_path.open("w") as f:
        for s in samples:
            row = {
                "t": s.t.isoformat(),
                "n": s.n_news(),
                "items": [
                    {
                        "ts": n.published_at.isoformat(),
                        "src": n.source,
                        "title": n.title,
                        "sent": n.sentiment,
                        "tickers": n.tickers,
                    }
                    for n in s.news
                ],
            }
            f.write(json.dumps(row) + "\n")

    return DatasetManifest(
        n_samples=N,
        history_lookback=L,
        horizon=H,
        news_lookback_days=cfg.news_lookback_days,
        first_t=samples[0].t,
        last_t=samples[-1].t,
        median_news_per_sample=float(np.median(news_count)),
        zero_news_samples=int((news_count == 0).sum()),
        path=out,
    )


def load_dataset(path: Optional[Path | str] = None) -> dict:
    """Load the .npz produced by :func:`build_dataset`.

    Returns a plain dict with numpy arrays + scalar metadata; we don't wrap it
    in a class so torch DataLoader code stays uncluttered.
    """
    p = Path(path) if path else PROJECT_ROOT / "data" / "conditioning" / "aligned.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"No aligned dataset at {p}. Run build_dataset(...) first."
        )
    npz = np.load(p, allow_pickle=False)
    out = {k: npz[k] for k in npz.files}
    # Convert int64 ns timestamps back to a DatetimeIndex for convenience.
    out["t_index"] = pd.to_datetime(out["t"], unit="ns")
    return out
