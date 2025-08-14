# Codex Runbook

## Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Lint / Typecheck / Format
ruff check .
ruff format --check .
mypy .

## Unit / Integration Tests
pytest -q
pytest -q tests/p2p/test_dual_path.py -q
pytest -q tests/test_orchestrator_integration.py -q

## Project Conventions
- Use existing module structure under src/.
- No new runtime deps without updating requirements.txt.
- Keep public APIs backward compatible unless the task says otherwise.
- Prefer minimal diffs; include unit tests with each change.

## PR Template
Title: <concise change>
Body:
- Summary
- Implementation notes
- Tradeoffs
- Tests added
- Local run logs (lint/type/tests)
Definition of Done: File exists with the exact sections above; all commands run locally without modification.

Run to verify:
