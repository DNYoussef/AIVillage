.PHONY: install format lint test ci

install:
	pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
	pre-commit install

format:
	black .

lint:
	ruff check .

test:
	pytest

ci: format lint test
