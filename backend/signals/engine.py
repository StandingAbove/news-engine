"""News → allocation signal engine.

Turns a batch of scored news items into a dict[ticker] → score in [-1, 1],
then into a target allocation vector over a universe.

Algorithm
---------
1. For each news item, distribute its sentiment across its tickers (equal
   split), with half-life decay based on age.
2. Aggregate per-ticker to produce a raw score.
3. Squash via tanh to keep values in [-1, 1].
4. Translate scores into target weights over a universe, respecting:
     * long_only clamp (default True)
     * per-name `max_weight`
     * a cash buffer sized by the fraction of universe with no signal
     * normalization to sum = 1.0 over allocated portion
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScoredNews:
    ticker: List[str]
    sentiment: float
    ts: datetime


def _half_life_weight(age_hours: float, half_life_hours: float = 24.0) -> float:
    if age_hours <= 0:
        return 1.0
    return 0.5 ** (age_hours / half_life_hours)


def aggregate_signals(
    items: Iterable[ScoredNews],
    *,
    now: Optional[datetime] = None,
    half_life_hours: float = 24.0,
) -> Dict[str, float]:
    """Return ticker → raw aggregated score (pre-squash)."""
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=timezone.utc) if (now and now.tzinfo is None) else (now or datetime.now(timezone.utc))
    agg: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for it in items:
        if not it.ticker:
            continue
        ts = it.ts if it.ts.tzinfo else it.ts.replace(tzinfo=timezone.utc)
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        decay = _half_life_weight(age_h, half_life_hours)
        per = it.sentiment * decay / max(1, len(it.ticker))
        for t in it.ticker:
            agg[t] = agg.get(t, 0.0) + per
            counts[t] = counts.get(t, 0) + 1

    # Slight confidence boost for tickers mentioned many times (log-scaled).
    for t in list(agg.keys()):
        agg[t] *= 1.0 + 0.25 * math.log1p(counts[t])

    return agg


def squash_scores(raw: Dict[str, float]) -> Dict[str, float]:
    return {t: math.tanh(v) for t, v in raw.items()}


def signals_to_weights(
    scores: Dict[str, float],
    universe: List[str],
    *,
    long_only: bool = True,
    max_weight: float = 0.25,
    min_abs_signal: float = 0.05,
) -> Dict[str, float]:
    """Convert per-ticker signals to target weights over `universe`.

    * Universe constituents with no signal default to a uniform "cash-like"
      passive allocation (split with the benchmark) to avoid empty portfolios.
    * Per-name weights capped by `max_weight`, then re-normalized.
    * If `long_only` and every signal is negative, we hold cash.
    """
    if not universe:
        return {}

    filtered = {t: s for t, s in scores.items() if t in universe and abs(s) >= min_abs_signal}

    if long_only:
        filtered = {t: max(0.0, s) for t, s in filtered.items()}
        filtered = {t: s for t, s in filtered.items() if s > 0}

    if not filtered:
        # No actionable signals — passive equal-weight across universe.
        n = len(universe)
        w = 1.0 / n
        return {t: w for t in universe}

    # Convert scores to proportional weights.
    total = sum(abs(s) for s in filtered.values())
    raw_weights = {t: (abs(s) / total) for t, s in filtered.items()}

    # Apply max-weight cap, re-distribute excess to non-capped holdings.
    capped = _apply_cap(raw_weights, max_weight)

    # Assign 0 to universe tickers without a signal (pure active).
    out = {t: 0.0 for t in universe}
    out.update(capped)
    return out


def _apply_cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap each weight at `cap`, redistributing excess into items'
    remaining headroom (water-filling).

    The algorithm never exceeds the cap. If the total active allocation
    can't reach 1.0 under the cap constraint, the remainder is treated as
    an implicit cash buffer (sum of returned weights < 1).
    """
    if not weights:
        return weights
    if cap <= 0:
        return {t: 0.0 for t in weights}

    w = dict(weights)

    for _ in range(16):
        over = {t: (v - cap) for t, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(over.values())
        for t in over:
            w[t] = cap
        # Headroom of all currently positive items that are below the cap.
        headroom = {t: (cap - w[t]) for t, v in w.items() if w[t] < cap - 1e-12 and w[t] > 0}
        total_headroom = sum(headroom.values())
        if total_headroom <= 0:
            break  # leftover excess becomes cash
        distributable = min(excess, total_headroom)
        for t, room in headroom.items():
            share = distributable * (room / total_headroom)
            w[t] = min(cap, w[t] + share)
        # Any undistributed excess is intentionally dropped → cash.

    # If somehow we exceeded 1 through numerical drift, renormalize down.
    s = sum(w.values())
    if s > 1.0 + 1e-9:
        w = {t: v / s for t, v in w.items()}
    return w


def blend_with_benchmark(
    active_weights: Dict[str, float],
    benchmark: str,
    *,
    active_share: float = 0.7,
) -> Dict[str, float]:
    """Blend active bets with benchmark exposure for risk control.

    Final weights = active_share * active + (1 - active_share) * 100% benchmark.
    """
    if not active_weights:
        return {benchmark: 1.0}
    out = {t: v * active_share for t, v in active_weights.items()}
    out[benchmark] = out.get(benchmark, 0.0) + (1.0 - active_share)
    # Re-normalize just in case.
    s = sum(out.values())
    if s > 0:
        out = {t: v / s for t, v in out.items()}
    return out


def fraction_active(weights: Dict[str, float], benchmark: str) -> float:
    """Diagnostic: how concentrated the portfolio is away from the benchmark."""
    if not weights:
        return 0.0
    bench_w = weights.get(benchmark, 0.0)
    return 1.0 - bench_w
