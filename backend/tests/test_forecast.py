"""Unit tests for sub-component (a) — forecasting models + harness.

These tests exercise the shared interface on toy data. Heavy-dep models
(statsmodels, torch, timesfm) are skipped with a clear reason when the
dependency isn't installed, matching how the harness reports availability.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.forecast import metrics as mx
from backend.forecast import toy
from backend.forecast import models as M
from backend.forecast.base import as_array
from backend.forecast.harness import evaluate


# ------------------------ metrics ------------------------

def test_mae_rmse_mape_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.7])
    assert mx.mae(y, p) == pytest.approx(0.175, abs=1e-6)
    assert mx.rmse(y, p) == pytest.approx(
        float(np.sqrt(np.mean((y - p) ** 2))), abs=1e-6
    )
    assert 0.0 < mx.mape(y, p) < 0.2


def test_mape_skips_zero_truth():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([0.1, 0.2, 0.3])
    # No non-zero truth values → undefined.
    assert np.isnan(mx.mape(y, p))


def test_directional_accuracy_picks_up_direction():
    history = np.array([10.0])
    y = np.array([11.0, 12.0, 13.0])   # all up
    p_all_up = np.array([10.5, 11.0, 11.5])  # all up
    p_all_down = np.array([9.5, 9.0, 8.5])   # all down
    assert mx.directional_accuracy(y, p_all_up, 10.0) == pytest.approx(1.0)
    assert mx.directional_accuracy(y, p_all_down, 10.0) == pytest.approx(0.0)


def test_mase_below_one_for_good_model():
    history = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_true = np.array([9.0, 10.0])
    y_good = np.array([9.1, 10.0])    # near-perfect
    y_bad = np.array([0.0, 0.0])
    assert mx.mase(y_true, y_good, history) < 1.0
    assert mx.mase(y_true, y_bad, history) > 1.0


# --------------------- as_array helper -------------------

def test_as_array_fills_non_finite():
    arr = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    out = as_array(arr)
    assert np.all(np.isfinite(out))
    assert out[0] == 1.0
    assert out[-1] == 5.0


# --------------------- naive models ----------------------

def test_naive_last_is_flat():
    m = M.NaiveLast()
    m.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    fc = m.forecast(horizon=4)
    assert fc.point.shape == (4,)
    assert np.allclose(fc.point, 5.0)


def test_naive_drift_follows_trend():
    m = M.NaiveDrift()
    m.fit(np.arange(10, dtype=float))   # slope = 1
    fc = m.forecast(horizon=3)
    assert np.allclose(fc.point, [10.0, 11.0, 12.0])


def test_naive_mean_is_mean():
    m = M.NaiveMean()
    m.fit(np.array([1.0, 3.0, 5.0]))
    fc = m.forecast(horizon=2)
    assert np.allclose(fc.point, 3.0)


def test_naive_last_is_always_available():
    assert M.NaiveLast.is_available()
    assert M.NaiveMean.is_available()
    assert M.NaiveDrift.is_available()


# --------------------- EWMA ------------------------------

def test_ewma_is_flat_and_stable():
    m = M.EWMAForecaster(alpha=0.3)
    m.fit(np.array([1.0] * 20 + [10.0]))
    fc = m.forecast(4)
    # Should land between 1 and 10, biased toward 1 since alpha=0.3 and 20 ones.
    assert 1.0 <= float(fc.point[0]) <= 10.0
    assert np.allclose(fc.point, fc.point[0])  # flat


# --------------------- harness ---------------------------

def test_harness_runs_naive_models_on_all_toy_series():
    series = toy.all_series(200)
    df = evaluate(series, ["naive_last", "naive_drift", "naive_mean", "ewma"], horizon=16)
    assert len(df) == len(series) * 4
    # Every run should succeed (these models have no heavy deps).
    statuses = df["status"].value_counts().to_dict()
    assert statuses.get("ok", 0) == len(df), f"unexpected statuses: {statuses}"
    # MAE must be non-negative and finite.
    assert df["mae"].ge(0).all()
    assert np.all(np.isfinite(df["mae"]))


def test_naive_drift_beats_naive_last_on_trend():
    series = {"trend": np.arange(200, dtype=float)}
    df = evaluate(series, ["naive_last", "naive_drift"], horizon=16)
    ok = df[df["status"] == "ok"].set_index("model")
    assert ok.loc["naive_drift", "mae"] < ok.loc["naive_last", "mae"]


def test_harness_reports_unknown_model_as_error():
    series = {"s": np.arange(50, dtype=float)}
    df = evaluate(series, ["does_not_exist"], horizon=8)
    assert df.iloc[0]["status"] == "error"


def test_harness_skips_when_deps_missing(monkeypatch):
    # Force statsmodels-based models to report unavailable.
    monkeypatch.setattr(M.ARIMAForecaster, "is_available", staticmethod(lambda: False))
    series = {"s": np.arange(50, dtype=float)}
    df = evaluate(series, ["arima"], horizon=8)
    assert df.iloc[0]["status"] == "skipped"


# --------------------- optional heavy-dep models ---------

@pytest.mark.skipif(not M.ARIMAForecaster.is_available(), reason="statsmodels not installed")
def test_arima_on_ar1_series():
    s = toy.ar1(200)
    df = evaluate({"ar1": s}, ["arima", "naive_last"], horizon=16)
    ok = df[df["status"] == "ok"].set_index("model")
    assert "arima" in ok.index  # fit succeeded
    # ARIMA should be finite and not blow up.
    assert np.isfinite(ok.loc["arima", "mae"])


@pytest.mark.skipif(not M.MLPForecaster.is_available(), reason="torch not installed")
def test_mlp_runs_on_sine():
    s = toy.sine(200)
    m = M.MLPForecaster(lookback=16, epochs=40)  # small for speed
    m.fit(s)
    fc = m.forecast(horizon=8)
    assert fc.point.shape == (8,)
    assert np.all(np.isfinite(fc.point))


@pytest.mark.skipif(not M.LSTMForecaster.is_available(), reason="torch not installed")
def test_lstm_runs_on_sine():
    s = toy.sine(200)
    m = M.LSTMForecaster(lookback=16, epochs=40)
    m.fit(s)
    fc = m.forecast(horizon=8)
    assert fc.point.shape == (8,)
    assert np.all(np.isfinite(fc.point))


@pytest.mark.skipif(not M.TimesFMForecaster.is_available(), reason="timesfm not installed")
def test_timesfm_interface_wired():
    # Sanity check: the wrapper constructs and accepts our interface.
    # We intentionally don't actually load the 800MB checkpoint in unit
    # tests — that's for the CLI eval. We just check that availability
    # and dry-construction work.
    f = M.TimesFMForecaster()
    assert f.name == "timesfm"
    # is_available should be True here per the skipif.
    assert M.TimesFMForecaster.is_available()
