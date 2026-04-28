"""Toy harness for sub-component (b) — text feed synthesis.

The full pipeline is already unit-tested, but Dr. Singh's progression asks
for each sub-component to be driven on toy data in isolation, end-to-end,
before anything real goes through it. This module provides the "toy input
+ expected output" bundle that drives that.

What's toy about it
-------------------
* Headlines are handwritten — no provider calls, no network.
* Expected sentiment direction (+ / 0 / -) is asserted in advance, so the
  harness can report accuracy, not just "it returned something".
* Expected ticker tags are asserted per headline.
* A tiny end-to-end aggregation asserts that the sign of the aggregated
  score per ticker matches intuition (positive earnings on AAPL → AAPL up).

Usage
-----
    python -m backend.news.toy_eval   # runs the harness and prints a report.

Or from Python:
    from backend.news.toy import run_eval
    print(run_eval())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple


@dataclass
class ToyHeadline:
    text: str
    expected_sentiment_sign: int        # -1, 0, +1
    expected_tickers: List[str] = field(default_factory=list)
    note: str = ""


# 18 headlines, handwritten. Mix of direct company mentions, sector
# keywords, and regional spillover — the three code paths in mapping.py.
TOY_HEADLINES: List[ToyHeadline] = [
    ToyHeadline(
        "Apple reports record earnings, topping Wall Street estimates",
        +1, ["AAPL"], "direct + positive"
    ),
    ToyHeadline(
        "Tesla misses delivery targets, shares plunge in after-hours trading",
        -1, ["TSLA"], "direct + negative"
    ),
    ToyHeadline(
        "Nvidia rallies as data-center demand surges on AI spending",
        +1, ["NVDA"], "direct + sector (AI)"
    ),
    ToyHeadline(
        "Fed signals rate pause at next FOMC meeting",
        0, ["TLT"], "macro → bonds"
    ),
    ToyHeadline(
        "Oil prices surge after OPEC cuts production",
        +1, ["XLE"], "sector keyword"
    ),
    ToyHeadline(
        "Major news breaks in Hong Kong overnight",
        0, ["EEM"], "region spillover"
    ),
    ToyHeadline(
        "Microsoft upgraded to buy, analysts cite cloud momentum",
        +1, ["MSFT"], "direct + positive"
    ),
    ToyHeadline(
        "Boeing faces new FAA investigation over safety concerns",
        -1, ["BA"], "direct + negative"
    ),
    ToyHeadline(
        "ECB holds rates as eurozone growth slows",
        0, ["EFA"], "region spillover (ECB → EFA)"
    ),
    ToyHeadline(
        "Semiconductor shortage eases, chip makers see outlook brighten",
        +1, ["XLK"], "sector keyword"
    ),
    ToyHeadline(
        "Amazon warns on Q3 guidance, stock falls 6%",
        -1, ["AMZN"], "direct + negative"
    ),
    ToyHeadline(
        "Gold hits record high on inflation worries",
        +1, ["GLD"], "sector keyword"
    ),
    ToyHeadline(
        "Pfizer announces breakthrough in clinical trial",
        +1, ["PFE"], "direct + positive"
    ),
    ToyHeadline(
        "Bank of America beats estimates, raises dividend",
        +1, ["BAC"], "direct + positive"
    ),
    ToyHeadline(
        "JPMorgan probe deepens over trading desk misconduct",
        -1, ["JPM"], "direct + negative"
    ),
    ToyHeadline(
        "Tokyo market tumbles on weak Japan GDP print",
        -1, ["EFA"], "region spillover"
    ),
    ToyHeadline(
        "Routine weekly market wrap: volumes average, rates steady",
        0, [], "neutral / no tag"
    ),
    ToyHeadline(
        "Nvidia downgraded amid AI bubble concerns, shares sell off",
        -1, ["NVDA"], "direct + negative override of sector keyword"
    ),
]


def _sign(x: float) -> int:
    if x > 0.1:
        return +1
    if x < -0.1:
        return -1
    return 0


def _assert_tickers_subset(expected: List[str], actual: List[str]) -> bool:
    return set(expected).issubset(set(actual))


def run_eval() -> Dict[str, object]:
    """Run the full text-feed pipeline on TOY_HEADLINES and return a report.

    Returns a dict:
      {
        "headlines": [ per-row result ],
        "sentiment_accuracy": float,
        "ticker_recall": float,
        "aggregate_scores": { ticker: score },
        "aggregate_sign_correct": float,   # fraction of tickers whose
                                           # aggregate sign matches intuition
      }
    """
    from .sentiment import get_sentiment
    from .mapping import extract_tickers
    from ..signals.engine import ScoredNews, aggregate_signals, squash_scores

    fs = get_sentiment()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows: List[Dict[str, object]] = []
    correct_sentiment = 0
    correct_tickers = 0
    scored: List[ScoredNews] = []

    # Per-ticker intuition: what sign should aggregate end up at?
    expected_sign_by_ticker: Dict[str, int] = {}

    for i, th in enumerate(TOY_HEADLINES):
        s = fs.score(th.text)
        got_sign = _sign(s)
        sent_ok = got_sign == th.expected_sentiment_sign
        correct_sentiment += int(sent_ok)

        tickers = extract_tickers(th.text)
        tick_ok = _assert_tickers_subset(th.expected_tickers, tickers) if th.expected_tickers else True
        correct_tickers += int(tick_ok)

        for t in th.expected_tickers:
            # Accumulate expected aggregate direction per ticker — if any
            # expected sign is nonzero, use that; ties stay zero.
            expected_sign_by_ticker.setdefault(t, 0)
            expected_sign_by_ticker[t] += th.expected_sentiment_sign

        rows.append({
            "text": th.text,
            "sentiment": round(s, 3),
            "sentiment_sign_ok": sent_ok,
            "expected_sign": th.expected_sentiment_sign,
            "tickers_found": tickers,
            "tickers_ok": tick_ok,
            "note": th.note,
        })

        # Space headlines 1h apart so decay is nontrivial in aggregation.
        scored.append(ScoredNews(
            ticker=[t for t in tickers if t],
            sentiment=s,
            ts=now - timedelta(hours=len(TOY_HEADLINES) - i),
        ))

    raw = aggregate_signals(scored, now=now, half_life_hours=48)
    squashed = squash_scores(raw)

    # Aggregate-sign accuracy.
    agg_ok_n = 0
    agg_total = 0
    for t, expected in expected_sign_by_ticker.items():
        if expected == 0:
            continue
        agg_total += 1
        got = _sign(squashed.get(t, 0.0))
        expected_s = 1 if expected > 0 else -1
        if got == expected_s:
            agg_ok_n += 1

    return {
        "headlines": rows,
        "sentiment_accuracy": correct_sentiment / len(TOY_HEADLINES),
        "ticker_recall": correct_tickers / len(TOY_HEADLINES),
        "aggregate_scores": squashed,
        "aggregate_sign_correct": (agg_ok_n / agg_total) if agg_total else float("nan"),
    }


def print_report(report: Dict[str, object]) -> None:
    print(f"headlines evaluated:      {len(report['headlines'])}")
    print(f"sentiment direction acc:  {report['sentiment_accuracy']:.1%}")
    print(f"ticker-tag recall:        {report['ticker_recall']:.1%}")
    print(f"aggregate-sign accuracy:  {report['aggregate_sign_correct']:.1%}")
    print()
    print("per-headline results (sign mismatches marked *):")
    for r in report["headlines"]:
        flag = "" if r["sentiment_sign_ok"] and r["tickers_ok"] else "*"
        print(f"  {flag:1} [{r['sentiment']:+.2f}] tickers={r['tickers_found']!r}  {r['text']}")
    print("\naggregate (per-ticker tanh-squashed scores):")
    for t, v in sorted(report["aggregate_scores"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {t:<6} {v:+.3f}")
