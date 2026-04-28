#!/usr/bin/env bash
# Stand up sub-component (b) — text-feed synthesis — on toy headlines,
# reporting per-stage accuracy (sentiment direction, ticker recall,
# aggregate sign agreement).

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY="python"
else
  PY="python"
fi

exec "$PY" -m backend.news.toy_eval "$@"
