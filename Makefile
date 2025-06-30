.PHONY: dev-up dev-down fmt lint test

IMAGE=atlantis-dev

dev-up:
	docker compose up --build -d

dev-down:
	docker compose down

fmt:
	poetry run black .

lint:
	poetry run ruff .

test:
	poetry run pytest -q --cov=.
