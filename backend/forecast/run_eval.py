"""CLI entry point for sub-component (a) evaluation.

Usage
-----
    python -m backend.forecast.run_eval                # toy data only
    python -m backend.forecast.run_eval --real SPY,QQQ # add real tickers
    python -m backend.forecast.run_eval --models naive_last,arima,timesfm

The output mirrors Dr. Singh's toy-then-real progression: first a block of
toy series is printed with per-model rankings, then (optionally) real tickers.
Models with missing deps are printed as "skipped", not silently dropped, so
the state of the environment is explicit.
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd

from . import models as M
from . import toy
from .harness import evaluate, print_report


def _parse_csv(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Forecasting sub-component eval")
    p.add_argument(
        "--models", default=None,
        help="comma-separated model names; default = all available",
    )
    p.add_argument(
        "--horizon", type=int, default=32,
        help="forecast horizon in steps (default 32)",
    )
    p.add_argument(
        "--n", type=int, default=400,
        help="toy-series length (default 400)",
    )
    p.add_argument(
        "--real", default="",
        help="comma-separated tickers to add after the toy block (e.g. SPY,QQQ)",
    )
    p.add_argument(
        "--real-period", default="2y",
        help="yfinance period for --real (default 2y)",
    )
    p.add_argument(
        "--skip-toy", action="store_true",
        help="skip the toy block and only run --real",
    )
    args = p.parse_args(argv)

    models = _parse_csv(args.models) or M.list_available()
    print(f"models: {models}")
    print(f"horizon: {args.horizon}\n")

    frames: list[pd.DataFrame] = []

    if not args.skip_toy:
        print("--- TOY ---")
        toy_series = toy.all_series(args.n)
        toy_df = evaluate(toy_series, models, horizon=args.horizon)
        print_report(toy_df)
        frames.append(toy_df.assign(dataset="toy"))

    tickers = _parse_csv(args.real)
    if tickers:
        print("\n--- REAL ---")
        from .real_data import load_many
        real_series = load_many(tickers, period=args.real_period)
        if not real_series:
            print("(no real data could be fetched)")
        else:
            real_df = evaluate(real_series, models, horizon=args.horizon)
            print_report(real_df)
            frames.append(real_df.assign(dataset="real"))

    if frames:
        full = pd.concat(frames, ignore_index=True)
        # Print a flat leaderboard across everything at the end.
        ok = full[full["status"] == "ok"]
        if not ok.empty:
            print("\n--- OVERALL ---")
            lb = ok.groupby("model")[["mae", "rmse", "mape", "mase", "dir_acc"]].mean().sort_values("mae")
            print(lb.to_string(float_format=lambda v: f"{v:0.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
