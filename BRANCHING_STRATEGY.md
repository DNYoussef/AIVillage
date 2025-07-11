# Branching Strategy

The project now uses a simplified trunk-based workflow.

## Branches

- `main`: single integration branch for all development and releases.
- `feature/*`: short-lived branches created from `main` for new features or refactoring.
- `hotfix/*`: urgent bug fix branches created from `main` and merged back after verification.

Release tags are created from `main` when preparing a new version. Long‑lived `develop` or `release` branches are no longer used.

## Workflow

1. Create a `feature/<description>` branch from `main`.
2. Commit your changes and open a pull request targeting `main`.
3. Ensure tests pass and the PR references any related issues.
4. After approval, merge the PR and delete the branch.

Hotfix branches follow the same process but use the `hotfix/` prefix.
