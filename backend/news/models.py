"""Internal dataclasses used while ingesting news."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RawNewsItem:
    """Provider-agnostic news representation."""

    source: str
    external_id: str
    title: str
    summary: Optional[str]
    url: Optional[str]
    published_at: datetime
    tickers: List[str] = field(default_factory=list)
    # Optional provider-supplied sentiment score in [-1, 1].
    provider_sentiment: Optional[float] = None
    raw: dict = field(default_factory=dict)

    def dedup_key(self) -> str:
        # Title + rounded timestamp as fallback uniqueness for cross-provider dedup.
        return f"{self.title.strip().lower()[:160]}|{self.published_at.date().isoformat()}"
