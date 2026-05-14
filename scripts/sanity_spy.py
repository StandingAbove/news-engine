"""Quick sanity check on the FactSet SPY history loader.

Run from the repo root:

    python scripts/sanity_spy.py
    python scripts/sanity_spy.py --start 2020-01-01

Prints a summary block. Exits non-zero if anything looks wrong.
"""
from __future__ import annotations

import argparse
import sys

from backend.conditioning.data import load_spy, summarize, return_window


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    df = load_spy(start=args.start, end=args.end)
    s = summarize(df)
    print(s.to_text())

    # Sanity gates — if any of these fail, fail the script loudly.
    if s.weekend_rows:
        print("FAIL: weekend rows present.", file=sys.stderr)
        return 1
    if s.nan_close_rows:
        print("FAIL: NaN closes present.", file=sys.stderr)
        return 1
    if not (-0.0005 < s.median_daily_return < 0.001):
        print(f"FAIL: median daily return {s.median_daily_return:+.4%} "
              "is outside [-0.05%, +0.10%] — suspicious.", file=sys.stderr)
        return 1
    if not (0.05 < s.annualized_return < 0.40):
        print(f"FAIL: annualized return {s.annualized_return:+.2%} outside [5%, 40%].",
              file=sys.stderr)
        return 1
    if not (0.08 < s.annualized_vol < 0.40):
        print(f"FAIL: annualized vol {s.annualized_vol:.2%} outside [8%, 40%].",
              file=sys.stderr)
        return 1

    # Spot-check forward-return helper on the most recent date that has 5 days
    # of forward data.
    pivot = df.index[-10]
    fwd5 = return_window(df, pivot, H=5)
    print(f"\n5-day forward return from {pivot.date()}: "
          f"{(1 + fwd5).prod() - 1:+.4%}  (per-day: {[f'{r:+.4%}' for r in fwd5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
