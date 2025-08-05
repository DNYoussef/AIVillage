.PHONY: help install test lint format clean build all

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linters"
	@echo "  make format   - Format code"
	@echo "  make clean    - Clean generated files"
	@echo "  make build    - Build package"
	@echo "  make all      - Run everything"

install:
	pip install -r requirements.txt
	pip install -r requirements/requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

lint:
	pre-commit run --all-files

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/

build:
	python -m build

all: clean install lint test build
