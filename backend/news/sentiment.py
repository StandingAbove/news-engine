"""Finance-tuned sentiment scoring.

We use VADER with an extended finance lexicon. VADER is deterministic, CPU-only,
and more than good enough for the demo. If a provider supplies a sentiment value
we blend it with VADER to produce a final score in [-1, 1].
"""
from __future__ import annotations

from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Finance-specific lexicon — biased so "beat"/"miss"/"bankruptcy" land where you'd expect.
_FINANCE_LEXICON = {
    # positives
    "beat": 2.5, "beats": 2.5, "surge": 2.8, "surges": 2.8, "soars": 3.0, "soar": 3.0,
    "rally": 2.2, "rallies": 2.2, "outperform": 2.5, "upgraded": 2.4, "upgrade": 2.4,
    "breakthrough": 2.5, "record-high": 2.8, "record": 1.2, "tops": 1.8, "topped": 1.8,
    "profit": 1.6, "profits": 1.6, "guidance-raised": 2.6, "raised": 1.2,
    "bullish": 2.5, "dividend-hike": 2.0, "buyback": 2.0, "beat-estimates": 2.8,
    "strong": 1.4, "stronger": 1.4, "robust": 1.6, "accelerate": 1.6,
    # negatives
    "miss": -2.5, "misses": -2.5, "missed": -2.5, "plunge": -3.0, "plunges": -3.0,
    "crash": -3.2, "crashes": -3.2, "downgrade": -2.4, "downgraded": -2.4,
    "bankruptcy": -3.8, "insolvent": -3.5, "default": -3.0, "defaults": -3.0,
    "probe": -1.8, "investigation": -1.8, "fraud": -3.5, "scandal": -2.8,
    "lawsuit": -2.0, "sued": -2.0, "fine": -1.5, "fined": -1.5,
    "layoffs": -2.2, "layoff": -2.2, "recession": -2.8, "contraction": -2.0,
    "bearish": -2.5, "sell-off": -2.6, "selloff": -2.6, "tumble": -2.4,
    "slump": -2.2, "slashed": -2.0, "warning": -1.8, "warns": -1.8, "warned": -1.8,
    "probe": -1.8, "tariff": -1.2, "tariffs": -1.2, "sanctions": -1.8,
    # geopolitics (mildly negative by default)
    "war": -2.2, "conflict": -1.6, "invasion": -2.6, "unrest": -1.6,
}


class FinanceSentiment:
    """Thin wrapper around VADER with an extended financial lexicon."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()
        self._analyzer.lexicon.update(_FINANCE_LEXICON)

    def score(self, text: Optional[str]) -> float:
        """Return compound VADER sentiment in [-1, 1]."""
        if not text:
            return 0.0
        return float(self._analyzer.polarity_scores(text)["compound"])

    def blend(self, text: str, provider_sentiment: Optional[float]) -> float:
        """Combine our own score with a provider-supplied one when present.

        Blending rule: 60% own, 40% provider if provider is in [-1, 1].
        """
        import math

        own = self.score(text)
        if provider_sentiment is None:
            return own
        try:
            p = float(provider_sentiment)
        except (TypeError, ValueError):
            return own
        if math.isnan(p) or math.isinf(p):
            return own
        p = max(-1.0, min(1.0, p))
        return 0.6 * own + 0.4 * p


_singleton: Optional[FinanceSentiment] = None


def get_sentiment() -> FinanceSentiment:
    global _singleton
    if _singleton is None:
        _singleton = FinanceSentiment()
    return _singleton
