# AIVillage Makefile
# Common development and CI/CD commands

.PHONY: help install dev-install clean format lint test security ci deploy

# ============================================
# Help & Documentation
# ============================================
help: ## Show this help message
	@echo "AIVillage Development Commands"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage: make <command>"

# ============================================
# Setup & Installation
# ============================================
install: ## Install production dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Production dependencies installed"

dev-install: install ## Install all dependencies (prod + dev + test)
	pip install -r requirements-dev.txt -r requirements-test.txt
	pre-commit install
	@echo "✅ Development environment ready"

clean: ## Clean build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/ .coverage* coverage.xml
	@echo "✅ Cleaned build artifacts and caches"

# ============================================
# Code Quality
# ============================================
format: ## Format code with Ruff and Black
	@echo "🎨 Formatting code..."
	ruff check . --fix --select I  # Fix imports first
	ruff format .  # Format with Ruff
	black . --line-length=120  # Format with Black
	@echo "✅ Code formatted"

lint: ## Run linting checks
	@echo "📝 Running linting checks..."
	ruff check . --select E,W,F,I,UP,B,C4,SIM,RUF
	@echo "✅ Linting passed"

lint-fix: ## Fix linting issues automatically
	@echo "🔧 Fixing linting issues..."
	ruff check . --fix --unsafe-fixes
	@echo "✅ Linting issues fixed"

lint-report: ## Generate linting report
	@echo "📊 Generating linting report..."
	ruff check . --output-format=json > lint_report.json
	ruff check . --statistics
	@echo "✅ Report saved to lint_report.json"

# ============================================
# Type Checking
# ============================================
type-check: ## Run MyPy type checking
	@echo "🔍 Running type checking..."
	mypy . --ignore-missing-imports --no-strict-optional \
		--exclude 'deprecated|archive|experimental|tests'
	@echo "✅ Type checking completed"

# ============================================
# Security
# ============================================
security: ## Run security scans
	@echo "🔒 Running security scans..."
	@echo "Checking for hardcoded secrets..."
	ruff check . --select S --output-format=concise
	@echo ""
	@echo "Running Bandit security scan..."
	-bandit -r src/ packages/ -ll -f json -o security_report.json 2>/dev/null
	-bandit -r src/ packages/ -ll
	@echo ""
	@echo "Checking dependency vulnerabilities..."
	-safety check --json
	@echo "✅ Security scan completed"

# ============================================
# Testing
# ============================================
test: ## Run all tests
	@echo "🧪 Running tests..."
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v --tb=short --timeout=60

test-integration: ## Run integration tests
	@echo "🔗 Running integration tests..."
	pytest tests/integration/ -v --tb=short --timeout=120

test-coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=src --cov=packages \
		--cov-report=html --cov-report=term-missing \
		--cov-fail-under=60
	@echo "✅ Coverage report generated in htmlcov/"

test-fast: ## Run fast tests only
	@echo "⚡ Running fast tests..."
	pytest tests/unit/ -v --tb=short -m "not slow" -n auto

# ============================================
# Continuous Integration
# ============================================
ci-pre-flight: ## Run pre-flight checks (fast fail)
	@echo "🚨 Running pre-flight checks..."
	ruff check . --select E9,F63,F7,F82,F821,F823 --output-format=concise
	@grep -r "TODO\|FIXME\|XXX" src/production/ packages/core/ --include="*.py" 2>/dev/null && \
		(echo "❌ Production code cannot contain TODOs!" && exit 1) || \
		echo "✅ No TODOs in production"
	@echo "✅ Pre-flight checks passed"

ci-local: format lint type-check test-fast ## Run local CI checks
	@echo "✅ Local CI checks passed"

ci: ci-pre-flight lint security type-check test-coverage ## Run full CI pipeline
	@echo "✅ Full CI pipeline passed"

# ============================================
# Development Helpers
# ============================================
watch: ## Watch for changes and run tests
	@echo "👀 Watching for changes..."
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command="make test-fast" \
		.

serve: ## Run development server
	@echo "🚀 Starting development server..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

shell: ## Start interactive Python shell
	@echo "🐍 Starting Python shell..."
	python -i -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'packages')"

# ============================================
# Documentation
# ============================================
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	pdoc --html --output-dir docs/api src/ packages/ --force
	@echo "✅ Documentation generated in docs/api/"

# ============================================
# Deployment
# ============================================
build: clean ## Build distribution packages
	@echo "📦 Building packages..."
	python -m build
	@echo "✅ Packages built in dist/"

docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t aivillage:latest .
	@echo "✅ Docker image built"

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm -p 8000:8000 aivillage:latest

deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	# Add staging deployment commands here
	@echo "✅ Deployed to staging"

deploy-production: ## Deploy to production (requires confirmation)
	@echo "⚠️  Production deployment - are you sure? [y/N]"
	@read -r REPLY; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		echo "🚀 Deploying to production..."; \
		# Add production deployment commands here
		echo "✅ Deployed to production"; \
	else \
		echo "❌ Production deployment cancelled"; \
	fi

# ============================================
# Maintenance
# ============================================
update-deps: ## Update dependencies to latest versions
	@echo "📦 Updating dependencies..."
	pip-compile --upgrade requirements.in -o requirements.txt
	pip-compile --upgrade requirements-dev.in -o requirements-dev.txt
	pip-compile --upgrade requirements-test.in -o requirements-test.txt
	@echo "✅ Dependencies updated"

check-deps: ## Check for outdated dependencies
	@echo "🔍 Checking dependencies..."
	pip list --outdated
	safety check

pre-commit-update: ## Update pre-commit hooks
	@echo "🔄 Updating pre-commit hooks..."
	pre-commit autoupdate
	@echo "✅ Pre-commit hooks updated"

# ============================================
# Git Helpers
# ============================================
git-clean: ## Clean untracked files (dry run)
	@echo "🧹 Files that would be removed:"
	git clean -n -d -x -e .env -e venv/ -e .vscode/

git-clean-force: ## Clean untracked files (force)
	@echo "🧹 Cleaning untracked files..."
	git clean -f -d -x -e .env -e venv/ -e .vscode/
	@echo "✅ Untracked files removed"