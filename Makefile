.PHONY: install format lint test ci lint-fix lint-report security-check type-check

install:
	pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
	pre-commit install

format:
	ruff format .
	black . --line-length=120

lint:
	ruff check . --fix

lint-fix:
	ruff check . --fix --unsafe-fixes

lint-report:
	ruff check . --output-format=json > lint_report_$(shell date +%Y%m%d_%H%M%S).json

security-check:
	ruff check . --select S --output-format=concise

type-check:
	mypy . --ignore-missing-imports --no-strict-optional

test:
	pytest

ci: format lint security-check type-check test
