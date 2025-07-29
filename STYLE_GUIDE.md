# AIVillage Style Guide

This document defines the comprehensive coding standards and linting configuration for the AIVillage project. All code contributions must adhere to these guidelines.

## Table of Contents

1. [Python Style Guide](#python-style-guide)
2. [JavaScript/TypeScript Style Guide](#javascripttypescript-style-guide)
3. [Pre-commit Hooks](#pre-commit-hooks)
4. [CI/CD Linting](#cicd-linting)
5. [Editor Configuration](#editor-configuration)

## Python Style Guide

### Code Formatting

We use **Black** and **Ruff** for Python code formatting and linting.

#### Black Configuration
- Line length: 88 characters
- Target Python versions: 3.10, 3.11, 3.12
- Format: `.py` and `.pyi` files
- Excludes: `.eggs`, `.git`, `.mypy_cache`, `.ruff_cache`, `.venv`, `_build`, `buck-out`, `build`, `dist`, `node_modules`, `models`, `download_env`

#### Ruff Configuration

**Target Version:** Python 3.10+
**Line Length:** 88 characters

**Enabled Rules:**
- `E`, `W` - pycodestyle errors and warnings
- `F` - pyflakes
- `I` - isort (import sorting)
- `N` - pep8-naming
- `D` - pydocstyle (Google convention)
- `UP` - pyupgrade
- `S` - flake8-bandit (security)
- `B` - flake8-bugbear
- `A` - flake8-builtins
- `C4` - flake8-comprehensions
- `DTZ` - flake8-datetimez
- `T10` - flake8-debugger
- `EM` - flake8-errmsg
- `FA` - flake8-future-annotations
- `ISC` - flake8-implicit-str-concat
- `ICN` - flake8-import-conventions
- `G` - flake8-logging-format
- `PIE` - flake8-pie
- `PT` - flake8-pytest-style
- `Q` - flake8-quotes
- `RSE` - flake8-raise
- `RET` - flake8-return
- `SLF` - flake8-self
- `SIM` - flake8-simplify
- `TID` - flake8-tidy-imports
- `TCH` - flake8-type-checking
- `ARG` - flake8-unused-arguments
- `PTH` - flake8-use-pathlib
- `ERA` - eradicate
- `PD` - pandas-vet
- `PGH` - pygrep-hooks
- `PL` - pylint
- `TRY` - tryceratops
- `FLY` - flynt
- `NPY` - numpy
- `PERF` - perflint
- `RUF` - ruff-specific rules

**Ignored Rules:**
- `D100-D105` - Missing docstrings (module, class, method, function, package, magic method)
- `ANN101`, `ANN102` - Missing type annotation for self/cls
- `ANN401` - Dynamically typed expressions (Any)
- `S101` - Use of assert
- `COM812` - Trailing comma missing
- `ISC001` - Implicit string concatenation
- `T201`, `T203` - print/pprint found
- `PD011` - Use .to_numpy() instead of .values
- `RET504` - Unnecessary variable assignment
- `PLR0913` - Too many arguments to function call
- `PLR0915` - Too many statements

**Per-file Ignores:**
- `tests/**/*.py`: Ignore all docstring, annotation, security checks, and magic value comparisons
- `stubs/**/*.py`: Ignore docstrings, annotations, unused arguments, and commented code
- `**/conftest.py`, `setup.py`: Ignore docstrings and annotations

### Type Checking (MyPy)

**Configuration:**
- Python version: 3.10
- Strict type checking enabled with exceptions
- `warn_return_any`: true
- `warn_unused_configs`: true
- `check_untyped_defs`: true
- `no_implicit_optional`: true
- `warn_redundant_casts`: true
- `warn_unused_ignores`: true
- `warn_no_return`: true
- `warn_unreachable`: true
- `strict_equality`: true
- `ignore_missing_imports`: true

**Ignored Modules:**
- All test and stub modules
- External libraries: langroid, qdrant_client, neo4j, faiss, chromadb, ollama, bitsandbytes, transformers, accelerate, peft, triton, xformers, sentence_transformers, gym, mcts, llama_cpp, grokfast, sleep_and_dream

### Security Scanning (Bandit)

**Configuration:**
- Excluded directories: tests, stubs, models, download_env
- Skipped checks: B101 (assert_used), B601 (shell_injection)

### Test Coverage

**Coverage Configuration:**
- Source: Project root
- Omitted: tests/*, stubs/*, models/*, download_env/*, .claude/*, setup.py, conftest.py
- Excluded lines: pragma: no cover, debug blocks, abstract methods, protocol definitions

### Import Ordering

**Known First-Party Modules:** agents, communications, core, agent_forge, utils, ingestion
**Force Sort Within Sections:** true

### Local Development Commands

```bash
# Format code with Black
make fmt
# or
poetry run black .

# Lint code with Ruff
make lint
# or
poetry run ruff check .

# Run tests with coverage
make test
# or
poetry run pytest -q --cov=.
```

## JavaScript/TypeScript Style Guide

### Monorepo Structure

The JavaScript/TypeScript code is organized as a monorepo using Turbo and pnpm workspaces.

**Package Manager:** pnpm@8.0.0
**Build System:** Turbo

### Available Scripts

```bash
# Development
pnpm dev         # Run all packages in dev mode
pnpm build       # Build all packages
pnpm lint        # Lint all packages
pnpm type-check  # Type check all packages
pnpm clean       # Clean build artifacts
pnpm test        # Run tests
```

### Package-Specific Configuration

#### UI Kit Package (@aivillage/ui-kit)
- **Framework:** React Native Web
- **Type Checking:** TypeScript with strict mode
- **Linting:** ESLint
- **Build:** TypeScript compiler

#### Web App (@aivillage/web)
- **Framework:** Next.js 14
- **Styling:** Tailwind CSS
- **Linting:** Next.js ESLint configuration
- **Type Checking:** TypeScript with no emit
- **Development Port:** 8081

### TypeScript Configuration

- **Version:** ^5.0.0
- **Strict mode** should be enabled
- **No emit** for type checking only
- All packages should have proper type definitions

### Code Quality Tools

- **ESLint:** ^8.0.0
- **Prettier:** ^3.0.0
- **TypeScript:** ^5.0.0

## Pre-commit Hooks

Pre-commit hooks are configured to run automatically before each commit.

### General Hooks
1. **trailing-whitespace** - Remove trailing whitespace
2. **end-of-file-fixer** - Ensure files end with newline
3. **check-yaml** - Validate YAML syntax (allows multiple documents)
4. **check-json** - Validate JSON syntax
5. **check-added-large-files** - Prevent large files (max 10MB)
6. **check-merge-conflict** - Check for merge conflict markers
7. **debug-statements** - Check for debugger statements
8. **mixed-line-ending** - Standardize line endings

### Python-Specific Hooks
1. **Ruff** - Lint and auto-fix Python code
   - Targets: `agent_forge/`, `tests/`, `scripts/`, `run_*.py`
2. **Ruff Format** - Format Python code
   - Same target paths as Ruff
3. **MyPy** - Type check Python code
   - Targets: `agent_forge/`, `run_*.py`
   - Additional dependencies: torch, numpy, types-requests, types-PyYAML, pydantic
4. **Bandit** - Security scanning
   - Excludes: tests directory

### Local Custom Hooks
1. **pytest-check** - Run tests on modified files
2. **automation-style-guide** - Enforce style guide for automation scripts
3. **no-misleading-stubs** - Check for misleading stub implementations
4. **documentation-accuracy** - Verify documentation matches implementation

### Pre-commit CI Configuration
- **Auto-fix enabled** - Automatically fixes and commits formatting issues
- **Auto-update** - Weekly updates for hook versions
- **Works on all PRs** - Ensures consistent code quality

## CI/CD Linting

### GitHub Actions Workflows

#### Compression Pipeline Tests
Runs comprehensive tests and linting for compression code:

1. **Code Quality Checks:**
   - Ruff linting
   - Black formatting verification
   - Flake8 static analysis

2. **Test Execution:**
   - Unit tests with coverage
   - Integration tests
   - Basic benchmarks

3. **Paths Monitored:**
   - `agent_forge/compression/**`
   - `tests/compression/**`
   - `config/compression.yaml`
   - `notebooks/compression_benchmarks.ipynb`

#### Workflow Features
- **Python Version:** 3.11
- **Caching:** pip dependencies cached
- **Parallel Testing:** Unit and integration tests run in parallel
- **Coverage Upload:** Test coverage uploaded to Codecov
- **Artifact Storage:** Test logs and benchmark results archived

### Automated Fixes

Several workflows automatically commit fixes:
1. **Pre-commit CI** - Commits formatting fixes
2. **Performance Regression** - Updates baseline metrics
3. **Stats Update** - Updates repository statistics

## Editor Configuration

### Recommended VS Code Extensions
1. **Python** - ms-python.python
2. **Pylance** - ms-python.vscode-pylance
3. **Black Formatter** - ms-python.black-formatter
4. **Ruff** - charliermarsh.ruff
5. **ESLint** - dbaeumer.vscode-eslint
6. **Prettier** - esbenp.prettier-vscode

### VS Code Settings (settings.json)
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll.ruff": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

## Enforcement

### Local Development
1. Install pre-commit hooks: `pre-commit install`
2. Run hooks manually: `pre-commit run --all-files`
3. Use Makefile commands: `make fmt`, `make lint`, `make test`

### Pull Requests
1. Pre-commit CI runs automatically
2. GitHub Actions run linting and tests
3. All checks must pass before merging

### Continuous Improvement
- Style guide enforcement script: `scripts/enforce_style_guide.py`
- Automated stub auditing: `scripts/audit_stubs.py`
- Documentation alignment: `scripts/align_documentation.py`

## Summary

This style guide ensures consistent, high-quality code across the AIVillage project. All contributors must:

1. Configure their editors with the recommended settings
2. Install and use pre-commit hooks
3. Run linting and formatting before committing
4. Ensure all CI checks pass
5. Follow language-specific conventions and best practices

For questions or suggestions about this style guide, please open an issue or submit a pull request.
