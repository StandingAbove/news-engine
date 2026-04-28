"""Toy data generators.

Dr. Singh, Feb-6: "Feed them through some data, see that you get some
mechanism operational — where the data is toy. Then move to some real data."

Each generator returns a deterministic numpy array. The harness splits it
into train/test windows before handing to any model, so the generator
itself doesn't need to know about splits.

Series catalog
--------------
* `sine(n)`                 — pure sinusoid; every model should nail it.
* `sine_plus_trend(n)`      — sinusoid riding a linear drift.
* `ar1(n, phi)`             — AR(1); ARIMA should dominate.
* `random_walk(n)`          — naive-last should be hard to beat.
* `regime_switch(n)`        — piecewise-stationary; stresses foundation models.
* `noisy_price(n)`          — log-returns with fat tails; proxy for real data.

`all_series(n)` returns a dict of every toy series at length `n`, ready for
the harness.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def sine(n: int = 400, period: int = 24, amplitude: float = 1.0, seed: int = 0) -> np.ndarray:
    _ = seed  # deterministic
    t = np.arange(n, dtype=np.float64)
    return amplitude * np.sin(2 * np.pi * t / period)


def sine_plus_trend(
    n: int = 400, period: int = 24, amplitude: float = 1.0, slope: float = 0.01, seed: int = 0
) -> np.ndarray:
    _ = seed
    t = np.arange(n, dtype=np.float64)
    return amplitude * np.sin(2 * np.pi * t / period) + slope * t


def ar1(n: int = 400, phi: float = 0.85, sigma: float = 1.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float64)
    eps = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def random_walk(n: int = 400, sigma: float = 1.0, start: float = 100.0, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=n)
    return start + np.cumsum(steps)


def regime_switch(
    n: int = 400, regimes: int = 3, sigma: float = 0.5, seed: int = 3
) -> np.ndarray:
    """Piecewise-stationary series — each segment has its own mean.

    Good for seeing whether a model just chases the last mean or actually
    picks up level shifts.
    """
    rng = np.random.default_rng(seed)
    segment = n // regimes
    out = []
    mu = 0.0
    for r in range(regimes):
        mu = mu + rng.normal(0.0, 3.0)
        length = segment if r < regimes - 1 else n - len(out) * 1 - segment * (regimes - 1)
        length = max(length, segment)
        out.append(mu + rng.normal(0.0, sigma, size=length))
    return np.concatenate(out)[:n]


def noisy_price(
    n: int = 400, start: float = 100.0, drift: float = 0.0003, sigma: float = 0.012, seed: int = 4
) -> np.ndarray:
    """Log-normal price path — a crude stand-in for real equity returns."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, sigma, size=n)
    # Add occasional fat-tail shocks.
    shocks = rng.choice([0, 1], size=n, p=[0.98, 0.02]) * rng.normal(0.0, 4 * sigma, size=n)
    return start * np.exp(np.cumsum(rets + shocks))


def all_series(n: int = 400) -> Dict[str, np.ndarray]:
    return {
        "sine":              sine(n),
        "sine_plus_trend":   sine_plus_trend(n),
        "ar1":               ar1(n),
        "random_walk":       random_walk(n),
        "regime_switch":     regime_switch(n),
        "noisy_price":       noisy_price(n),
    }
