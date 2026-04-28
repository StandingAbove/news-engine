"""Tests for the text-feed toy harness.

These tests assert that the pipeline clears sensible accuracy floors on the
hand-written TOY_HEADLINES set. They're intentionally floor-based (not
exact-match) because the underlying sentiment model is VADER — it's good
enough as a placeholder but not perfect, and we don't want the test to fight
small lexicon changes.
"""
from __future__ import annotations

from backend.news.toy import run_eval, TOY_HEADLINES


def test_toy_headlines_is_nonempty():
    assert len(TOY_HEADLINES) >= 15


def test_sentiment_direction_accuracy_floor():
    rep = run_eval()
    # VADER + finance lexicon should clear 70% direction accuracy on these.
    assert rep["sentiment_accuracy"] >= 0.70, rep


def test_ticker_recall_floor():
    rep = run_eval()
    # Every headline with an expected ticker should resolve to at least one
    # of the expected tickers in extract_tickers().
    assert rep["ticker_recall"] >= 0.80, rep


def test_aggregate_signs_match_intuition():
    rep = run_eval()
    # Aggregate-sign accuracy over tickers with a clear directional prior
    # should be at least 2/3 — very loose floor; we only want to catch
    # regressions where the aggregator inverts signs or drops tickers.
    agg = rep["aggregate_sign_correct"]
    assert agg >= 0.66, rep
