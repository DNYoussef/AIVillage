# Contributing to AIVillage

## Development Setup
1. Clone the repository and create a virtual environment.
2. Run `./scripts/setup.sh` to install dependencies and set up pre-commit hooks.

## Local Workflow
- `make format` – format code with Black
- `make lint` – run Ruff linting
- `make test` – run the pytest suite
- `make ci` – run formatting, linting, and tests

## Continuous Integration
GitHub Actions runs `make ci` on pushes and pull requests using Python 3.10.

## Common Issues
- Missing dependencies: rerun `./scripts/setup.sh`
- Pre-commit failures: run `pre-commit run --all-files` locally and commit fixes
