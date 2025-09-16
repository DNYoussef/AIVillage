# Makefile for Quiet-STaR Test Suite
# Provides convenient commands for test execution and quality assurance

.PHONY: help install test test-quick test-unit test-integration test-performance
.PHONY: test-property test-contract coverage report clean lint format check-deps
.PHONY: test-debug test-gpu benchmark ci-test

# Default target
help:
	@echo "Quiet-STaR Test Suite Commands"
	@echo "=============================="
	@echo ""
	@echo "Installation:"
	@echo "  install       Install test dependencies"
	@echo "  check-deps    Check if dependencies are installed"
	@echo ""
	@echo "Test Execution:"
	@echo "  test          Run comprehensive test suite (default)"
	@echo "  test-quick    Run quick tests (no slow tests)"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance benchmarks"
	@echo "  test-property Run property-based tests"
	@echo "  test-contract Run contract tests"
	@echo "  test-debug    Run tests in debug mode"
	@echo "  test-gpu      Run GPU-specific tests"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  coverage      Generate coverage report"
	@echo "  report        Generate comprehensive HTML report"
	@echo "  lint          Run code linting"
	@echo "  format        Format code with black and isort"
	@echo "  benchmark     Run performance benchmarks"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci-test       Run tests for CI environment"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Clean generated files"

# Installation
install:
	@echo "Installing test dependencies..."
	pip install -r requirements-test.txt
	@echo "✅ Dependencies installed"

check-deps:
	@echo "Checking dependencies..."
	@python -c "import pytest, torch, numpy; print('✅ Core dependencies available')"

# Test execution targets
test: check-deps
	@echo "Running comprehensive test suite..."
	python run_tests.py --mode comprehensive
	@echo "✅ Comprehensive tests completed"

test-quick: check-deps
	@echo "Running quick tests..."
	python run_tests.py --mode quick
	@echo "✅ Quick tests completed"

test-unit: check-deps
	@echo "Running unit tests..."
	python run_tests.py --mode unit
	@echo "✅ Unit tests completed"

test-integration: check-deps
	@echo "Running integration tests..."
	python run_tests.py --mode integration
	@echo "✅ Integration tests completed"

test-performance: check-deps
	@echo "Running performance tests..."
	python run_tests.py --mode performance
	@echo "✅ Performance tests completed"

test-property: check-deps
	@echo "Running property-based tests..."
	python run_tests.py --mode property
	@echo "✅ Property tests completed"

test-contract: check-deps
	@echo "Running contract tests..."
	python run_tests.py --mode contract
	@echo "✅ Contract tests completed"

test-debug: check-deps
	@echo "Running tests in debug mode..."
	python run_tests.py --mode debug --no-parallel
	@echo "✅ Debug tests completed"

test-gpu: check-deps
	@echo "Running GPU tests..."
	pytest -m gpu -v
	@echo "✅ GPU tests completed"

# Quality assurance
coverage: check-deps
	@echo "Generating coverage report..."
	pytest --cov=quiet_star --cov-report=html --cov-report=term-missing
	@echo "✅ Coverage report generated in htmlcov/"

report: test
	@echo "Generating comprehensive HTML report..."
	pytest --html=report.html --self-contained-html
	@echo "✅ HTML report generated: report.html"

lint:
	@echo "Running code linting..."
	flake8 test_*.py conftest.py run_tests.py --max-line-length=100
	@echo "✅ Linting completed"

format:
	@echo "Formatting code..."
	black test_*.py conftest.py run_tests.py --line-length=100
	isort test_*.py conftest.py run_tests.py
	@echo "✅ Code formatting completed"

benchmark: check-deps
	@echo "Running performance benchmarks..."
	pytest -m performance --benchmark-only --benchmark-sort=mean
	@echo "✅ Benchmarks completed"

# CI/CD targets
ci-test: check-deps
	@echo "Running CI test suite..."
	python run_tests.py --mode comprehensive --fail-fast --report-format xml
	@echo "✅ CI tests completed"

# Maintenance
clean:
	@echo "Cleaning generated files..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test-results.xml
	rm -rf report.html
	rm -rf test_report.json
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	@echo "✅ Cleanup completed"

# Advanced targets
test-all-modes: test-unit test-integration test-performance test-property test-contract
	@echo "✅ All test modes completed"

test-with-profiling: check-deps
	@echo "Running tests with memory profiling..."
	pytest --profile-mem
	@echo "✅ Profiled tests completed"

test-stress: check-deps
	@echo "Running stress tests..."
	pytest test_quietstar.py::TestQuietSTaRPerformance --count=10
	@echo "✅ Stress tests completed"

validate-quality: test coverage lint
	@echo "Running quality validation..."
	@echo "✅ Quality validation completed"

# Help for specific test categories
help-markers:
	@echo "Available pytest markers:"
	@echo "  unit         - Unit tests for individual components"
	@echo "  integration  - Integration tests for component interaction"
	@echo "  performance  - Performance and benchmark tests"
	@echo "  property     - Property-based tests for algorithm invariants"
	@echo "  contract     - Contract tests for input/output validation"
	@echo "  slow         - Tests that take more than 10 seconds"
	@echo "  gpu          - Tests that require GPU acceleration"

# Development workflow
dev-setup: install
	@echo "Setting up development environment..."
	pre-commit install
	@echo "✅ Development environment ready"

dev-test: format lint test-quick
	@echo "✅ Development test cycle completed"

# Docker support (if needed)
docker-test:
	@echo "Running tests in Docker..."
	docker build -t quietstar-tests .
	docker run --rm quietstar-tests make test
	@echo "✅ Docker tests completed"

# Performance monitoring
perf-monitor:
	@echo "Running performance monitoring..."
	python -m pytest test_quietstar.py::TestQuietSTaRPerformance -v --tb=short
	@echo "✅ Performance monitoring completed"

# Generate test matrix
test-matrix:
	@echo "Running test matrix..."
	@echo "Python $(python --version)"
	@echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
	@make test-quick
	@echo "✅ Test matrix completed"