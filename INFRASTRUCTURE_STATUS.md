# Development Infrastructure Status

This document lists key configuration files and scripts in the repository as of this audit.

## Configuration Files

- `.pre-commit-config.yaml` – **broken**: pinned to `python3.10` and missing default language version.
- `pyproject.toml` – **outdated**: restricts to Python 3.10+, tool configs inconsistent.
- `setup.py` – **present**: needs Python version flexibility review.
- `Makefile` – **outdated**: custom targets, lacks standardized commands.
- `.github/workflows/` – **varied**: multiple workflows; many use older GitHub action versions.
- `requirements.txt` – **present**: contains both core and dev dependencies.
- `requirements/requirements-dev.txt` – **present**: development dependencies.
- `requirements/requirements-test.txt` – **present**: testing dependencies.
- Shell scripts in `scripts/` – **mixed**: many lack shebang or safe flags.

## Python Version Requirements

Current configurations often require **Python 3.10+**, which causes issues on environments with only `python3` available. Goal is to support **Python 3.8+**.

## Dependencies

Development tools expected:
- `pre-commit`
- `black`
- `isort`
- `flake8`
- `mypy`
- `pytest`

These should be installed via `requirements-dev.txt` and `requirements-test.txt`.
