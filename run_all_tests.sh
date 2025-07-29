#!/bin/bash
# Run all tests with proper configuration

echo "Running All Tests..."
export PYTHONPATH="${PYTHONPATH}:.:agent_forge"

# Run pytest with all tests
pytest tests/ \
    -v \
    --tb=short \
    --maxfail=10 \
    --cov=. \
    --cov-report=html \
    --cov-report=term-missing \
    -W ignore::DeprecationWarning \
    -W ignore::PendingDeprecationWarning

# Open coverage report if available
if [ -f htmlcov/index.html ]; then
    echo "Coverage report generated at htmlcov/index.html"
fi
