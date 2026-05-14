"""Part 1 unit tests — SPY loader correctness.

Asserts the FactSet ingest produces a clean, monotonic, holiday-aware
trading-day series with returns that reconcile against close prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.conditioning.data import (
    DEFAULT_SPY_PATH,
    load_spy,
    return_window,
    summarize,
    trading_days_between,
)


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    if not DEFAULT_SPY_PATH.exists():
        pytest.skip("data/historical/SPY.csv not present")
    return load_spy()


def test_loader_returns_nonempty_sorted(spy: pd.DataFrame):
    assert len(spy) > 5_000  # we have ~33 years of daily data
    assert spy.index.is_monotonic_increasing
    assert not spy.index.has_duplicates


def test_no_weekend_rows(spy: pd.DataFrame):
    weekday = spy.index.weekday
    assert int(((weekday == 5) | (weekday == 6)).sum()) == 0


def test_required_columns_present(spy: pd.DataFrame):
    for col in ("open", "high", "low", "close", "volume", "pct_change", "total_return"):
        assert col in spy.columns


def test_close_prices_strictly_positive(spy: pd.DataFrame):
    assert (spy["close"] > 0).all()


def test_pct_change_in_decimal(spy: pd.DataFrame):
    """FactSet ships % change as percent. Our loader converts to decimal so a
    typical daily move is well under 1.0."""
    # Drop the first row (NaN) and any extreme outliers; we just want to
    # verify the unit is decimal not percent.
    pc = spy["pct_change"].dropna()
    assert pc.abs().max() < 0.5, "pct_change > 50% — likely still in percent"
    assert pc.abs().median() < 0.02


def test_pct_change_reconciles_with_close(spy: pd.DataFrame):
    """FactSet's % Change column should equal close.pct_change() to ~1bp."""
    derived = spy["close"].pct_change()
    diff = (spy["pct_change"] - derived).abs()
    # Compare on rows where both are present.
    both = pd.concat([spy["pct_change"], derived], axis=1).dropna()
    diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    # Allow 1bp slack — FactSet rounds to 2dp on % Change.
    assert (diff < 0.0001).mean() > 0.95


def test_total_return_starts_at_one(spy: pd.DataFrame):
    assert spy["total_return"].iloc[0] == pytest.approx(1.0)


def test_total_return_monotone_ish_long_run(spy: pd.DataFrame):
    """Over 30+ years, end > 5x start — the loader hasn't accidentally
    inverted or normalized away the long-run drift."""
    if (spy.index[-1] - spy.index[0]).days < 365 * 20:
        pytest.skip("series too short for long-run drift assertion")
    assert spy["total_return"].iloc[-1] > 5.0


def test_summary_passes_basic_gates(spy: pd.DataFrame):
    s = summarize(spy)
    assert s.weekend_rows == 0
    assert s.nan_close_rows == 0
    assert 0.05 < s.annualized_return < 0.40
    assert 0.08 < s.annualized_vol < 0.40


def test_trading_days_between_excludes_weekends(spy: pd.DataFrame):
    days = trading_days_between(spy, "2024-01-01", "2024-01-31")
    weekday = days.weekday
    assert int(((weekday == 5) | (weekday == 6)).sum()) == 0
    # January 2024 has 21 trading days (Mon-Fri minus MLK day).
    assert 19 <= len(days) <= 22


def test_return_window_length_matches_H(spy: pd.DataFrame):
    pivot = spy.index[100]  # safely far from the end
    fwd = return_window(spy, pivot, H=5)
    assert fwd.shape == (5,)
    assert np.isfinite(fwd).all()


def test_return_window_raises_on_non_trading_day(spy: pd.DataFrame):
    # 2024-01-01 was a market-closed holiday.
    with pytest.raises(KeyError):
        return_window(spy, "2024-01-01", H=5)


def test_return_window_raises_when_insufficient_forward(spy: pd.DataFrame):
    last = spy.index[-1]
    with pytest.raises(ValueError):
        return_window(spy, last, H=5)


def test_start_end_window(spy: pd.DataFrame):
    df = load_spy(start="2023-01-01", end="2023-12-31")
    assert df.index.min() >= pd.Timestamp("2023-01-01")
    assert df.index.max() <= pd.Timestamp("2023-12-31")
    # ~250 trading days a year.
    assert 240 <= len(df) <= 260
    # total_return is renormalized to start at 1.0 within the slice.
    assert df["total_return"].iloc[0] == pytest.approx(1.0)
