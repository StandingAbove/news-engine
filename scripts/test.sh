#!/usr/bin/env bash
# Run the unit tests using the venv's Python explicitly.
#
# Why this exists: if conda base is active alongside .venv (`(.venv) (base)`
# in the shell prompt), the `pytest` binary on PATH will often resolve to
# the conda one, which uses the wrong interpreter and misses venv-installed
# deps. Calling `python -m pytest` forces the venv's interpreter.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "No .venv directory found. Create one with:" >&2
  echo "    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

# Use the venv interpreter directly — avoids any PATH surprises.
PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No Python in $PY. Did venv creation fail?" >&2
  exit 1
fi

exec "$PY" -m pytest "$@"
