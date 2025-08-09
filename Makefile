.PHONY: install format lint test ci lint-fix lint-report security-check

install:
	pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
	pre-commit install

format:
	black .
	isort .

lint:
	python tools/linting/lint.py . --output summary

lint-fix:
	python tools/linting/lint.py . --fix --output summary

lint-report:
	python tools/linting/lint.py . --output both --output-file lint_report_$(shell date +%Y%m%d_%H%M%S)

security-check:
	ruff check . --select S

test:
	pytest

ci: format lint test
