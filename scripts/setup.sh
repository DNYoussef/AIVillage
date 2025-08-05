#!/usr/bin/env bash
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dir"

pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
