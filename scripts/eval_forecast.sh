#!/usr/bin/env bash
# Stand up sub-component (a) — forecasting models — on toy data first,
# then optionally on real tickers.
#
# Usage: pass flags through as-is to the Python CLI:
#   ./scripts/eval_forecast.sh                                  # toy only
#   ./scripts/eval_forecast.sh --real SPY,QQQ                   # toy + real
#   ./scripts/eval_forecast.sh --real SPY,QQQ --skip-toy        # real only
#   ./scripts/eval_forecast.sh --models naive_last,arima,timesfm --horizon 20
#   ./scripts/eval_forecast.sh --help
#
# Heavy models (torch, timesfm) are skipped cleanly if their deps aren't
# installed — the harness reports per-model status.

set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer the venv's Python; this script works under `(base) (.venv)` prompts
# where stock `python` may resolve to conda.
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY="python"
else
  PY="python"
fi

# Pass every arg through untouched. `"$@"` under `set -u` is safe even empty.
exec "$PY" -m backend.forecast.run_eval "$@"
