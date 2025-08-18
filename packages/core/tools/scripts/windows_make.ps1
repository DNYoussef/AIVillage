# Windows PowerShell equivalent of Makefile commands
# Usage: .\tools\scripts\windows_make.ps1 <command>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$PYTHON = "python"
$POETRY = "poetry"
$PYTEST_ARGS = "--tb=short --cov=src --cov-report=xml --cov-report=term-missing"

function Show-Help {
    Write-Host "AIVillage Development Commands (Windows)" -ForegroundColor Cyan
    Write-Host "=======================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Environment:" -ForegroundColor Green
    Write-Host "  setup           Set up development environment"
    Write-Host "  install         Install production dependencies"
    Write-Host "  install-dev     Install development dependencies"
    Write-Host ""
    Write-Host "Code Quality:" -ForegroundColor Green
    Write-Host "  fmt             Format code"
    Write-Host "  lint            Run linting checks"
    Write-Host ""
    Write-Host "Testing:" -ForegroundColor Green
    Write-Host "  test            Run full test suite"
    Write-Host "  test-fast       Run tests without coverage"
    Write-Host ""
    Write-Host "Development:" -ForegroundColor Green
    Write-Host "  dev-up          Start development services"
    Write-Host "  dev-down        Stop development services"
    Write-Host "  build           Build distribution packages"
    Write-Host ""
    Write-Host "Benchmarking:" -ForegroundColor Green
    Write-Host "  benchmark       Run all benchmarks"
    Write-Host "  benchmark-compression  Run compression benchmarks"
    Write-Host "  benchmark-quick Run quick benchmarks"
    Write-Host ""
    Write-Host "Deployment:" -ForegroundColor Green
    Write-Host "  deploy-staging  Deploy to staging"
    Write-Host "  deploy-prod     Deploy to production"
    Write-Host "  status          Check deployment status"
    Write-Host ""
    Write-Host "Utility:" -ForegroundColor Green
    Write-Host "  clean           Clean generated files"
    Write-Host "  help            Show this help"
}

function Invoke-Clean {
    Write-Host "🧹 Cleaning generated files..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force -ErrorAction Ignore wandb/, forge_output/, test_output/, coverage.xml, .coverage
    Remove-Item -Recurse -Force -ErrorAction Ignore __pycache__, .pytest_cache, .mypy_cache, .ruff_cache
    Get-ChildItem -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction Ignore
    Get-ChildItem -Recurse -Name "*.pyc" | Remove-Item -Force -ErrorAction Ignore
    Write-Host "✅ Cleanup complete" -ForegroundColor Green
}

function Invoke-Install {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry install --no-dev
    } else {
        & $PYTHON -m pip install -e .
    }
    Write-Host "✅ Installation complete" -ForegroundColor Green
}

function Invoke-InstallDev {
    Write-Host "📦 Installing development dependencies..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry install --with dev
    } else {
        & $PYTHON -m pip install -e ".[dev]"
        & $PYTHON -m pip install -r requirements-dev.txt -ErrorAction SilentlyContinue
    }
    Write-Host "✅ Development installation complete" -ForegroundColor Green
}

function Invoke-Setup {
    Write-Host "🛠️ Setting up development environment..." -ForegroundColor Yellow
    Invoke-InstallDev
    & $PYTHON tools/scripts/setup_dev_env.py
    Write-Host "✅ Development environment ready" -ForegroundColor Green
}

function Invoke-Lint {
    Write-Host "🔍 Running linting checks..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry run black --check src/ tests/ tools/
        & poetry run ruff check src/ tests/ tools/
        & poetry run mypy src/ --ignore-missing-imports
    } else {
        & $PYTHON -m black --check src/ tests/ tools/
        & $PYTHON -m ruff check src/ tests/ tools/
        & $PYTHON -m mypy src/ --ignore-missing-imports
    }
    Write-Host "✅ Linting complete" -ForegroundColor Green
}

function Invoke-Format {
    Write-Host "🎨 Formatting code..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry run black src/ tests/ tools/
        & poetry run ruff check --fix src/ tests/ tools/
    } else {
        & $PYTHON -m black src/ tests/ tools/
        & $PYTHON -m ruff check --fix src/ tests/ tools/
    }
    Write-Host "✅ Formatting complete" -ForegroundColor Green
}

function Invoke-Test {
    Write-Host "🧪 Running test suite..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry run pytest tests/ $PYTEST_ARGS.Split(' ')
    } else {
        & $PYTHON -m pytest tests/ $PYTEST_ARGS.Split(' ')
    }
    Write-Host "✅ Tests complete" -ForegroundColor Green
}

function Invoke-TestFast {
    Write-Host "⚡ Running fast tests..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry run pytest tests/ -x --tb=short
    } else {
        & $PYTHON -m pytest tests/ -x --tb=short
    }
    Write-Host "✅ Fast tests complete" -ForegroundColor Green
}

function Invoke-Build {
    Write-Host "🏗️ Building packages..." -ForegroundColor Yellow
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        & poetry build
    } else {
        & $PYTHON -m build
    }
    Write-Host "✅ Build complete" -ForegroundColor Green
}

function Invoke-DevUp {
    Write-Host "🚀 Starting development services..." -ForegroundColor Yellow
    & docker compose up --build -d
    Write-Host "✅ Development services started" -ForegroundColor Green
}

function Invoke-DevDown {
    Write-Host "🛑 Stopping development services..." -ForegroundColor Yellow
    & docker compose down
    Write-Host "✅ Development services stopped" -ForegroundColor Green
}

function Invoke-Benchmark {
    Write-Host "📊 Running benchmarks..." -ForegroundColor Yellow
    & $PYTHON tools/scripts/run_benchmarks.py --categories all
    Write-Host "✅ Benchmarks complete" -ForegroundColor Green
}

function Invoke-BenchmarkCompression {
    Write-Host "📊 Running compression benchmarks..." -ForegroundColor Yellow
    & $PYTHON tools/scripts/run_benchmarks.py --categories compression
    Write-Host "✅ Compression benchmarks complete" -ForegroundColor Green
}

function Invoke-BenchmarkQuick {
    Write-Host "⚡ Running quick benchmarks..." -ForegroundColor Yellow
    & $PYTHON tools/scripts/run_benchmarks.py --categories compression agents --quick
    Write-Host "✅ Quick benchmarks complete" -ForegroundColor Green
}

function Invoke-DeployStaging {
    Write-Host "🚀 Deploying to staging..." -ForegroundColor Yellow
    & $PYTHON tools/scripts/deploy_production.py --environment staging
    Write-Host "✅ Staging deployment complete" -ForegroundColor Green
}

function Invoke-DeployProd {
    Write-Host "🚀 Deploying to production..." -ForegroundColor Yellow
    & $PYTHON tools/scripts/deploy_production.py --environment production
    Write-Host "✅ Production deployment complete" -ForegroundColor Green
}

function Invoke-Status {
    Write-Host "📊 Checking deployment status..." -ForegroundColor Yellow
    & $PYTHON tools/scripts/deploy_production.py --environment staging --action status
    & $PYTHON tools/scripts/deploy_production.py --environment production --action status
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "clean" { Invoke-Clean }
    "install" { Invoke-Install }
    "install-dev" { Invoke-InstallDev }
    "setup" { Invoke-Setup }
    "lint" { Invoke-Lint }
    "fmt" { Invoke-Format }
    "format" { Invoke-Format }
    "test" { Invoke-Test }
    "test-fast" { Invoke-TestFast }
    "build" { Invoke-Build }
    "dev-up" { Invoke-DevUp }
    "dev-down" { Invoke-DevDown }
    "benchmark" { Invoke-Benchmark }
    "benchmark-compression" { Invoke-BenchmarkCompression }
    "benchmark-quick" { Invoke-BenchmarkQuick }
    "deploy-staging" { Invoke-DeployStaging }
    "deploy-prod" { Invoke-DeployProd }
    "status" { Invoke-Status }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run '.\tools\scripts\windows_make.ps1 help' for available commands" -ForegroundColor Yellow
    }
}
