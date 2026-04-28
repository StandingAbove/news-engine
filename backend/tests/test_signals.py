"""Unit tests for the signal engine + news mapping layer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.news.mapping import extract_tickers
from backend.signals.engine import (
    ScoredNews,
    _apply_cap,
    aggregate_signals,
    blend_with_benchmark,
    signals_to_weights,
    squash_scores,
)


def test_aggregate_signals_weights_by_recency():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    items = [
        ScoredNews(ticker=["AAPL"], sentiment=1.0, ts=now),
        ScoredNews(ticker=["AAPL"], sentiment=1.0, ts=now - timedelta(hours=48)),
    ]
    raw = aggregate_signals(items, now=now, half_life_hours=24)
    # First item: weight 1. Second item: weight 0.25 (half-life 24, age 48).
    # Plus log-scaled count bump (2 items → 1 + 0.25*log(3)).
    assert raw["AAPL"] > 1.0 and raw["AAPL"] < 2.0


def test_aggregate_signals_splits_ticker_share():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    items = [
        ScoredNews(ticker=["AAPL", "MSFT"], sentiment=1.0, ts=now),
    ]
    raw = aggregate_signals(items, now=now)
    # Each ticker gets half of the sentiment.
    assert raw["AAPL"] == pytest.approx(raw["MSFT"], abs=1e-10)
    assert raw["AAPL"] < 1.0


def test_squash_scores_bounds():
    s = {"A": 2.0, "B": -3.0, "C": 0.0}
    out = squash_scores(s)
    assert all(-1.0 <= v <= 1.0 for v in out.values())
    assert out["A"] > 0 and out["B"] < 0 and out["C"] == 0


def test_signals_to_weights_long_only_passive_when_all_negative():
    universe = ["AAPL", "MSFT", "GOOGL"]
    scores = {"AAPL": -0.5, "MSFT": -0.2, "GOOGL": -0.9}
    w = signals_to_weights(scores, universe, long_only=True)
    # All scores rejected → passive equal weight fallback.
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    assert all(v == pytest.approx(1 / 3, abs=1e-9) for v in w.values())


def test_signals_to_weights_respects_max_weight():
    # 5-ticker universe with cap=0.25 can sum to 1 (5 * 0.25 = 1.25 >= 1).
    universe = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    scores = {"AAPL": 0.95, "MSFT": 0.1, "GOOGL": 0.1, "NVDA": 0.1, "META": 0.1}
    w = signals_to_weights(scores, universe, long_only=True, max_weight=0.25)
    assert w["AAPL"] <= 0.25 + 1e-9
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0


def test_signals_to_weights_cap_holds_cash_when_impossible():
    # 3 items at cap 0.25 cannot reach 1.0 — remainder is held as cash
    # (reflected as sum of weights < 1). The hard cap must not be violated.
    universe = ["AAPL", "MSFT", "GOOGL"]
    scores = {"AAPL": 0.95, "MSFT": 0.1, "GOOGL": 0.1}
    w = signals_to_weights(scores, universe, long_only=True, max_weight=0.25)
    assert max(w.values()) <= 0.25 + 1e-9
    assert sum(w.values()) <= 0.75 + 1e-9


def test_signals_to_weights_ignores_tiny_signals():
    universe = ["AAPL", "MSFT"]
    scores = {"AAPL": 0.01, "MSFT": 0.8}  # AAPL below min_abs_signal
    w = signals_to_weights(scores, universe, min_abs_signal=0.05, max_weight=1.0)
    assert w["AAPL"] == 0.0
    # MSFT gets the full active allocation; AAPL remains 0.
    assert w["MSFT"] == pytest.approx(1.0, abs=1e-9)


def test_apply_cap_redistributes_excess():
    w = {"A": 0.6, "B": 0.3, "C": 0.1}
    out = _apply_cap(w, 0.4)
    assert out["A"] <= 0.4 + 1e-9
    assert pytest.approx(sum(out.values()), abs=1e-9) == 1.0


def test_blend_with_benchmark_sums_to_one():
    active = {"AAPL": 0.5, "MSFT": 0.5}
    w = blend_with_benchmark(active, "SPY", active_share=0.7)
    assert "SPY" in w
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    assert w["SPY"] == pytest.approx(0.3, abs=1e-9)


# -------- mapping tests --------

def test_extract_tickers_company_names():
    t = extract_tickers("Apple reports record earnings, Tesla misses estimates")
    assert "AAPL" in t and "TSLA" in t


def test_extract_tickers_sector_keywords():
    t = extract_tickers("Oil prices surge after OPEC cuts production")
    assert "XLE" in t


def test_extract_tickers_region_spillover_hong_kong():
    # The exact cross-market case from the meeting.
    t = extract_tickers("Major news breaks in Hong Kong overnight")
    assert "EEM" in t


def test_extract_tickers_fed_maps_to_tlt():
    t = extract_tickers("Fed signals rate pause at next FOMC meeting")
    assert "TLT" in t


def test_extract_tickers_respects_hint():
    t = extract_tickers("Some generic market story", hint=["NVDA"])
    assert "NVDA" in t


def test_extract_tickers_empty_text_returns_hint_only():
    t = extract_tickers("", hint=["AAPL"])
    assert t == ["AAPL"]
