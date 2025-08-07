#!/usr/bin/env bash
set -euo pipefail

echo "Running CI test suite locally..."

pre-commit run --all-files
pytest tests/ --cov=src --cov-report=term-missing
mypy src/ --ignore-missing-imports
bandit -r src/ || true
pip list --outdated

echo "CI test suite complete!"
