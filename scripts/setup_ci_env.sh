#!/bin/bash
# CI Environment Setup Script
set -e

echo "Setting up CI environment..."

# Set Python path
export PYTHONPATH="${PYTHONPATH}:.:src:packages:core:infrastructure"

# Set test environment variables
export AIVILLAGE_ENV=testing
export DB_PASSWORD=${DB_PASSWORD:-test_password}
export REDIS_PASSWORD=${REDIS_PASSWORD:-test_redis}
export JWT_SECRET=${JWT_SECRET:-test_jwt_secret_key_minimum_32_characters}

# Disable pip version warnings
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Set encoding for UTF-8 support
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Cache directories
export PIP_CACHE_DIR=${HOME}/.cache/pip
export PYTEST_CACHE_DIR=${HOME}/.cache/pytest

echo "Environment setup complete"
echo "PYTHONPATH: $PYTHONPATH"
echo "Working directory: $(pwd)"
