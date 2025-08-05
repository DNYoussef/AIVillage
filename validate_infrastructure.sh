#!/usr/bin/env bash
set -euo pipefail

echo "Validating development infrastructure..."

python --version || { echo "Python not found"; exit 1; }
pre-commit --version || { echo "pre-commit not installed"; exit 1; }
pre-commit run --all-files || { echo "Pre-commit hooks failed"; exit 1; }
pytest tests/ || { echo "Tests failed"; exit 1; }

for script in scripts/*.sh; do
    [ -x "$script" ] || { echo "$script not executable"; exit 1; }
done

echo "✅ Infrastructure validation complete!"
