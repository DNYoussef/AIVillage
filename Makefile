.PHONY: help install lint test docs clean dev-setup validate-all fix-all

help:  ## Show this help message
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies for development
	python -m pip install --upgrade pip
	pip install -e ".[dev,test,docs,security]"

dev-setup:  ## Set up complete development environment
	$(MAKE) install
	pre-commit install
	@echo "✅ Development environment ready!"

lint:  ## Run all linters and code quality checks
	@echo "Running code quality checks..."
	black src/ tests/ --check --verbose
	ruff check src/ tests/ --output-format=github
	isort src/ tests/ --check-only --diff --profile black
	mypy src/ --ignore-missing-imports --show-error-codes

security:  ## Run security scans
	@echo "Running security scans..."
	bandit -r src/ -f txt
	safety check
	pip-audit

test:  ## Run all tests
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:  ## Run unit tests only
	pytest tests/unit -v --cov=src --cov-report=term-missing

test-integration:  ## Run integration tests only
	pytest tests/integration -v --timeout=300

fix:  ## Auto-fix code style and formatting issues
	@echo "Auto-fixing code issues..."
	black src/ tests/
	ruff check src/ tests/ --fix --unsafe-fixes
	isort src/ tests/ --profile black

validate-all: lint security test  ## Run comprehensive validation
	@echo "🎉 All validations passed!"

clean:  ## Clean build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage 2>/dev/null || true

build:  ## Build the package
	python -m build
	twine check dist/*