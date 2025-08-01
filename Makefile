
.PHONY: help clean install install-dev setup lint fmt test test-fast build dev-up dev-down benchmark deploy-staging deploy-prod status

# Configuration
PYTHON := python
POETRY := poetry
PYTEST_ARGS := --tb=short --cov=src --cov-report=xml --cov-report=term-missing
DOCKER_COMPOSE := docker compose

# Help target - shows available commands
help: ## Show this help message
	@echo "AIVillage Development Makefile"
	@echo "==============================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Cleanup targets
clean: ## Clean generated files and directories
	@echo "🧹 Cleaning generated files..."
	@rm -rf wandb/ forge_output/ test_output/ coverage.xml .coverage
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Installation targets
install: ## Install production dependencies
	@echo "📦 Installing dependencies..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) install --no-dev; \
	else \
		$(PYTHON) -m pip install -e .; \
	fi
	@echo "✅ Installation complete"

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) install --with dev; \
	else \
		$(PYTHON) -m pip install -e .[dev]; \
		$(PYTHON) -m pip install -r requirements-dev.txt 2>/dev/null || true; \
	fi
	@echo "✅ Development installation complete"

setup: install-dev ## Set up development environment
	@echo "🛠️ Setting up development environment..."
	@$(PYTHON) tools/scripts/setup_dev_env.py
	@echo "✅ Development environment ready"

# Code quality targets
lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) run black --check src/ tests/ tools/; \
		$(POETRY) run ruff check src/ tests/ tools/; \
		$(POETRY) run mypy src/ --ignore-missing-imports; \
	else \
		$(PYTHON) -m black --check src/ tests/ tools/; \
		$(PYTHON) -m ruff check src/ tests/ tools/; \
		$(PYTHON) -m mypy src/ --ignore-missing-imports; \
	fi
	@echo "✅ Linting complete"

fmt: ## Format code
	@echo "🎨 Formatting code..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) run black src/ tests/ tools/; \
		$(POETRY) run ruff check --fix src/ tests/ tools/; \
	else \
		$(PYTHON) -m black src/ tests/ tools/; \
		$(PYTHON) -m ruff check --fix src/ tests/ tools/; \
	fi
	@echo "✅ Formatting complete"

# Testing targets
test: ## Run full test suite
	@echo "🧪 Running test suite..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) run pytest tests/ $(PYTEST_ARGS); \
	else \
		$(PYTHON) -m pytest tests/ $(PYTEST_ARGS); \
	fi
	@echo "✅ Tests complete"

test-fast: ## Run tests without coverage
	@echo "⚡ Running fast tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) run pytest tests/ -x --tb=short; \
	else \
		$(PYTHON) -m pytest tests/ -x --tb=short; \
	fi
	@echo "✅ Fast tests complete"

# Build targets
build: ## Build distribution packages
	@echo "🏗️ Building packages..."
	@if command -v poetry >/dev/null 2>&1; then \
		$(POETRY) build; \
	else \
		$(PYTHON) -m build; \
	fi
	@echo "✅ Build complete"

# Development environment targets
dev-up: ## Start development services
	@echo "🚀 Starting development services..."
	@$(DOCKER_COMPOSE) up --build -d
	@echo "✅ Development services started"

dev-down: ## Stop development services
	@echo "🛑 Stopping development services..."
	@$(DOCKER_COMPOSE) down
	@echo "✅ Development services stopped"

# Benchmarking targets
benchmark: ## Run benchmarks
	@echo "📊 Running benchmarks..."
	@$(PYTHON) tools/scripts/run_benchmarks.py --categories all
	@echo "✅ Benchmarks complete"

benchmark-compression: ## Run compression benchmarks only
	@echo "📊 Running compression benchmarks..."
	@$(PYTHON) tools/scripts/run_benchmarks.py --categories compression
	@echo "✅ Compression benchmarks complete"

benchmark-quick: ## Run quick benchmarks
	@echo "⚡ Running quick benchmarks..."
	@$(PYTHON) tools/scripts/run_benchmarks.py --categories compression agents --quick
	@echo "✅ Quick benchmarks complete"

# Deployment targets
deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	@$(PYTHON) tools/scripts/deploy_production.py --environment staging
	@echo "✅ Staging deployment complete"

deploy-prod: ## Deploy to production environment
	@echo "🚀 Deploying to production..."
	@$(PYTHON) tools/scripts/deploy_production.py --environment production
	@echo "✅ Production deployment complete"

status: ## Check deployment status
	@echo "📊 Checking deployment status..."
	@$(PYTHON) tools/scripts/deploy_production.py --environment staging --action status
	@$(PYTHON) tools/scripts/deploy_production.py --environment production --action status

# Composite targets
ci: lint test ## Run CI pipeline (lint + test)
	@echo "✅ CI pipeline complete"

pre-commit: fmt lint test-fast ## Run pre-commit checks
	@echo "✅ Pre-commit checks complete"

all: clean install-dev lint test build ## Run full pipeline
	@echo "✅ Full pipeline complete"
