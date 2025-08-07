#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
pre-commit install
