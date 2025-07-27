
.PHONY: dev-up dev-down fmt lint test

POETRY = poetry

IMAGE=atlantis-dev

dev-up:
	docker compose up --build -d

dev-down:
	docker compose down

fmt:
	$(POETRY) run black .

lint:
	$(POETRY) run ruff check .

test:
	$(POETRY) run pytest -q --cov=.

install:
	$(POETRY) install

install-dev:
	$(POETRY) install --with dev
