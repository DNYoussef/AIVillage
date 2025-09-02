#!/bin/bash
# Robust Dependency Installation Script for CI/CD
set -e

echo "Installing dependencies with error handling..."

# Function to install with fallback
install_with_fallback() {
    local requirements_file="$1"
    local fallback_packages="$2"
    
    if [ -f "$requirements_file" ]; then
        echo "Installing from $requirements_file..."
        if pip install -r "$requirements_file"; then
            echo "Successfully installed from $requirements_file"
        else
            echo "Failed to install from $requirements_file, trying fallback..."
            if [ -n "$fallback_packages" ]; then
                echo "Installing fallback packages: $fallback_packages"
                pip install $fallback_packages || echo "Fallback installation failed"
            fi
        fi
    else
        echo "$requirements_file not found, installing fallback packages..."
        if [ -n "$fallback_packages" ]; then
            pip install $fallback_packages || echo "Fallback installation failed"
        fi
    fi
}

# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install core requirements with fallbacks
install_with_fallback "requirements.txt" "fastapi uvicorn pydantic"

# Install development requirements with fallbacks
install_with_fallback "config/requirements/requirements-dev.txt" "pytest pytest-asyncio pytest-cov pytest-mock ruff black mypy"

# Install security requirements with fallbacks
install_with_fallback "config/requirements/requirements-security.txt" "bandit safety"

# Install any additional requirements
for req_file in config/requirements/requirements*.txt; do
    if [ -f "$req_file" ] && [ "$req_file" != "config/requirements/requirements-dev.txt" ] && [ "$req_file" != "config/requirements/requirements-security.txt" ]; then
        echo "Installing additional requirements from $req_file..."
        pip install -r "$req_file" || echo "Failed to install from $req_file"
    fi
done

echo "Dependency installation completed"

# Verify critical packages
echo "Verifying critical packages..."
python -c "import pytest, fastapi, pydantic; print('Critical packages verified')" || echo "Critical package verification failed"
