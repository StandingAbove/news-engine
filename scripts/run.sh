#!/usr/bin/env bash
# One-shot launcher for News-Engine.
# Creates a venv, installs deps, starts the API + frontend, and opens a
# Cloudflare quick tunnel so anyone with the printed URL can reach the dashboard.
#
# Usage:  ./scripts/run.sh          # local only
#         ./scripts/run.sh --share  # local + public Cloudflare tunnel
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$ROOT"

SHARE=0
if [[ "${1:-}" == "--share" || "${SHARE:-0}" == "1" ]]; then
  SHARE=1
fi

PYTHON_BIN="${PYTHON:-python3}"
VENV="$ROOT/.venv"

echo "▶ News-Engine launcher"

# ---- venv ----
if [[ ! -d "$VENV" ]]; then
  echo "▶ Creating virtualenv at .venv"
  "$PYTHON_BIN" -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# ---- env ----
if [[ ! -f "$ROOT/.env" ]]; then
  echo "⚠ .env missing — copying .env.example. Edit it to add API keys."
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

mkdir -p "$ROOT/data"

# ---- server ----
HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8000}"
echo "▶ Starting server on http://$HOST:$PORT"

# Kill background jobs on exit
PIDS=()
cleanup() {
  echo
  echo "▶ Shutting down…"
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

uvicorn backend.main:app --host "$HOST" --port "$PORT" --log-level info &
PIDS+=($!)

# ---- optional Cloudflare tunnel ----
if [[ "$SHARE" == "1" ]]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo
    echo "⚠ cloudflared is not installed."
    echo "  macOS:   brew install cloudflared"
    echo "  Linux:   https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"
    echo "  Windows: winget install --id Cloudflare.cloudflared"
    echo
    echo "Running locally only. Share http://localhost:$PORT on same network."
  else
    echo "▶ Opening Cloudflare quick tunnel (public URL)…"
    # Give uvicorn a moment to bind.
    sleep 2
    cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate 2>&1 | tee /tmp/news-engine-tunnel.log &
    PIDS+=($!)
    # Tail for the generated URL and print it prominently.
    (
      for i in {1..30}; do
        URL=$(grep -oE "https://[a-zA-Z0-9.-]+\.trycloudflare\.com" /tmp/news-engine-tunnel.log | head -1 || true)
        if [[ -n "$URL" ]]; then
          echo
          echo "============================================================"
          echo "  🌐 Public link for your professor:"
          echo "     $URL"
          echo "============================================================"
          echo
          break
        fi
        sleep 1
      done
    ) &
  fi
fi

wait
