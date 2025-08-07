#!/usr/bin/env bash
# Setup script for local development and CI
# Installs all dependencies required for running tests
set -e

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
