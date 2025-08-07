#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found" >&2
  exit 1
fi

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt -r requirements-test.txt
pre-commit install

pytest tests/ || true

echo "Development environment ready using $($PYTHON_BIN --version)"
