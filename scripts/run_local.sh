#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "No .venv found. Create one with: python -m venv .venv"
  echo "Then activate: source .venv/bin/activate"
  exit 1
fi

PYTHON=".venv/bin/python"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Tip: activate the venv first: source .venv/bin/activate"
fi

echo "Installing requirements..."
"$PYTHON" -m pip install -q -r requirements.txt
echo "Installing SDK (editable)..."
"$PYTHON" -m pip install -q -e sdk

if [[ ! -f .env ]]; then
  echo "Warning: .env missing. Copy .env.example to .env and set DATABASE_URL."
fi

echo "Creating database tables..."
"$PYTHON" -c "from config.database import Base, engine; Base.metadata.create_all(bind=engine)"

echo "Starting API on http://0.0.0.0:8000 ..."
exec "$PYTHON" -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
