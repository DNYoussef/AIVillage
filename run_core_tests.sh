#!/bin/bash
# Run core module tests with proper configuration

echo "Running Core Module Tests..."
export PYTHONPATH="${PYTHONPATH}:.:agent_forge"

# Run tests by category
echo -e "\n=== Communication Tests ==="
pytest tests/core/test_communication.py -v

echo -e "\n=== Evidence Pack Tests ==="
pytest tests/core/test_evidencepack.py -v

echo -e "\n=== Message Tests ==="
pytest tests/test_message.py -v

echo -e "\n=== Integration Tests ==="
pytest tests/test_integration.py -v --maxfail=5

# Generate coverage report
echo -e "\n=== Coverage Report ==="
pytest tests/core/ --cov=core --cov-report=term-missing