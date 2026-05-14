"""Per-day news encoder.

Part 3 of the news-conditioned plan: turn the variable-length list of news
items inside each sample's lookback window into a *fixed-size* vector that
the conditioning adapter (Part 5) can consume.

Two encoders are provided behind a uniform :class:`NewsEncoder` protocol:

* ``StatsEncoder`` — 16-dim vector of cheap aggregate statistics
  (volume, sentiment, ticker mix, source mix, recency-weighted sentiment).
  No model dependency. Always available. This is the default for the first
  training run.

* ``TextEncoder``  (optional, lazy import) — pools a sentence-transformer
  embedding (default: ``sentence-transformers/all-MiniLM-L6-v2``) over the
  headlines + summaries. 384-dim, downsampled with PCA if requested.
  Activated only if ``sentence-transformers`` is installed and the user
  passes ``encoder="text"``.

The point of the protocol is that Part 5 (the adapter) takes ``encoder.dim``
as a hyperparameter and doesn't care which encoder produced the vector.
Swapping ``StatsEncoder`` for ``TextEncoder`` later is a one-line config flip.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Protocol, Sequence

import numpy as np

from .alignment import NewsRecord


# --- protocol -----------------------------------------------------------------
class NewsEncoder(Protocol):
    """Each encoder maps a *list of NewsRecord* + a "today" timestamp to one
    fixed-size float vector. Empty news lists must still produce a valid (zero-
    vector) embedding — the adapter sees those days too."""

    @property
    def dim(self) -> int: ...

    def encode(self, news: Sequence[NewsRecord], t: datetime) -> np.ndarray: ...

    def encode_batch(
        self, news_per_sample: Sequence[Sequence[NewsRecord]], t_per_sample: Sequence[datetime]
    ) -> np.ndarray:
        """Default: just loop. Subclasses with a real batched implementation
        can override (e.g., the text encoder runs the transformer once)."""
        return np.stack([self.encode(n, t) for n, t in zip(news_per_sample, t_per_sample)])


# --- helpers ------------------------------------------------------------------
def _ensure_aware(t: datetime) -> datetime:
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _hours_before(t_news: datetime, t_today: datetime) -> float:
    return max(0.0, (t_today - t_news).total_seconds() / 3600.0)


# --- StatsEncoder -------------------------------------------------------------
# 16-dim stats vector. Order is fixed and exposed via ``feature_names`` so the
# adapter checkpoints stay interpretable.
_STATS_FEATURE_NAMES: List[str] = [
    "log1p_count",                # 1. log(1 + n)
    "sentiment_mean",             # 2. mean VADER over the window
    "sentiment_std",              # 3. std VADER
    "sentiment_min",              # 4. most-bearish item
    "sentiment_max",              # 5. most-bullish item
    "pos_share",                  # 6. share with sentiment > +0.1
    "neg_share",                  # 7. share with sentiment < -0.1
    "neutral_share",              # 8. share with |sentiment| <= 0.1
    "recent_sentiment_mean",      # 9. mean over last 24h, if any
    "recent_count",               # 10. log(1 + items in last 24h)
    "decay_weighted_sentiment",   # 11. exp-decay-weighted sentiment, τ=24h
    "spy_share",                  # 12. share of items tagged SPY
    "macro_share",                # 13. share tagged SPY/QQQ/TLT/EEM/EFA
    "earnings_share",             # 14. heuristic: title contains "earnings"
    "fed_share",                  # 15. heuristic: title contains "fed"/"rate"
    "n_unique_sources",           # 16. log(1 + unique source count)
]

_MACRO_TICKERS = {"SPY", "QQQ", "TLT", "EEM", "EFA", "GLD", "DIA"}


@dataclass
class StatsEncoder:
    """Cheap, deterministic, always-available encoder.

    Returns a 16-dim float32 vector. Empty input → zero vector.
    """
    decay_tau_hours: float = 24.0

    @property
    def dim(self) -> int:
        return len(_STATS_FEATURE_NAMES)

    @property
    def feature_names(self) -> List[str]:
        return list(_STATS_FEATURE_NAMES)

    def encode(self, news: Sequence[NewsRecord], t: datetime) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        if not news:
            return v
        t_aware = _ensure_aware(t)

        sents = np.array([n.sentiment for n in news], dtype=np.float32)
        ages_h = np.array(
            [_hours_before(_ensure_aware(n.published_at), t_aware) for n in news],
            dtype=np.float32,
        )

        v[0] = float(np.log1p(len(news)))
        v[1] = float(sents.mean())
        v[2] = float(sents.std(ddof=0))
        v[3] = float(sents.min())
        v[4] = float(sents.max())
        v[5] = float((sents > 0.1).mean())
        v[6] = float((sents < -0.1).mean())
        v[7] = float((np.abs(sents) <= 0.1).mean())

        # last 24h slice
        recent_mask = ages_h <= 24.0
        if recent_mask.any():
            v[8] = float(sents[recent_mask].mean())
            v[9] = float(np.log1p(int(recent_mask.sum())))
        # exp decay weighted
        w = np.exp(-ages_h / max(1e-3, self.decay_tau_hours))
        if w.sum() > 0:
            v[10] = float((sents * w).sum() / w.sum())

        # ticker / topic shares
        spy_hits = sum(1 for n in news if "SPY" in n.tickers)
        macro_hits = sum(
            1 for n in news if any(t_ in _MACRO_TICKERS for t_ in n.tickers)
        )
        earnings_hits = sum(1 for n in news if "earnings" in (n.title or "").lower())
        fed_hits = sum(
            1 for n in news
            if any(k in (n.title or "").lower() for k in ("fed", "rate", "fomc"))
        )
        v[11] = spy_hits / len(news)
        v[12] = macro_hits / len(news)
        v[13] = earnings_hits / len(news)
        v[14] = fed_hits / len(news)

        sources = {n.source for n in news if n.source}
        v[15] = float(np.log1p(len(sources)))
        return v

    def encode_batch(
        self, news_per_sample: Sequence[Sequence[NewsRecord]], t_per_sample: Sequence[datetime]
    ) -> np.ndarray:
        # No transformer to batch through — just loop.
        return np.stack([self.encode(n, t) for n, t in zip(news_per_sample, t_per_sample)])


# --- TextEncoder (optional) ---------------------------------------------------
class TextEncoder:
    """Sentence-transformer encoder. Lazy-imports the dependency so the
    package still works in environments without it.

    Pooling is mean over the per-headline embeddings inside the lookback
    window. Empty windows → zero vector.

    Use only if you've installed `sentence-transformers` (which itself pulls
    in torch + transformers).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "TextEncoder requires `sentence-transformers`. "
                "Install with `pip install sentence-transformers`."
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, news: Sequence[NewsRecord], t: datetime) -> np.ndarray:
        if not news:
            return np.zeros(self.dim, dtype=np.float32)
        texts = [
            (n.title or "") + (" — " + n.summary if n.summary else "")
            for n in news
        ]
        emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32).mean(axis=0)

    def encode_batch(
        self, news_per_sample: Sequence[Sequence[NewsRecord]], t_per_sample: Sequence[datetime]
    ) -> np.ndarray:
        # Batch all headlines through one transformer call, then pool per
        # sample using offsets. Much faster than the default loop.
        flat_texts: List[str] = []
        offsets: List[int] = [0]
        for items in news_per_sample:
            flat_texts.extend(
                (n.title or "") + (" — " + n.summary if n.summary else "")
                for n in items
            )
            offsets.append(len(flat_texts))
        if not flat_texts:
            return np.zeros((len(news_per_sample), self.dim), dtype=np.float32)
        flat_emb = self._model.encode(
            flat_texts, normalize_embeddings=True, show_progress_bar=False
        )
        flat_emb = np.asarray(flat_emb, dtype=np.float32)
        out = np.zeros((len(news_per_sample), self.dim), dtype=np.float32)
        for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            if b > a:
                out[i] = flat_emb[a:b].mean(axis=0)
        return out


# --- factory ------------------------------------------------------------------
def make_encoder(kind: str = "stats", **kwargs) -> NewsEncoder:
    if kind == "stats":
        return StatsEncoder(**kwargs)
    if kind == "text":
        return TextEncoder(**kwargs)  # type: ignore[return-value]
    raise ValueError(f"unknown encoder kind: {kind!r}. choose 'stats' or 'text'.")
