"""CLI entry point for the text-feed sub-component's toy eval."""
from __future__ import annotations

import sys

from .toy import run_eval, print_report


def main() -> int:
    rep = run_eval()
    print_report(rep)
    # Non-zero exit if any accuracy metric is below a soft floor.
    ok = (
        rep["sentiment_accuracy"] >= 0.70
        and rep["ticker_recall"] >= 0.80
        and (rep["aggregate_sign_correct"] != rep["aggregate_sign_correct"]  # NaN = no ground truth, treat as ok
             or rep["aggregate_sign_correct"] >= 0.70)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
