# AIVillage Codebase Cleanup Plan

## Overview
This plan outlines the comprehensive cleanup process for the AIVillage codebase based on the STYLE_GUIDE.md standards and available automation scripts.

## Current State Analysis

### Python Codebase
- **Main code locations:**
  - `agent_forge/` - Core agent framework code
  - `tests/` - Test files
  - `scripts/` - Automation and utility scripts (60+ Python files)
  - Various `run_*.py` files in root

- **Key directories to clean:**
  - `agent_forge/` (including subdirectories: adas, bakedquietiot, compression, core, etc.)
  - `scripts/` (60+ automation scripts)
  - `tests/` (test files)
  - Root level `run_*.py` files

### JavaScript/TypeScript Codebase
- **Monorepo structure (`aivillage-monorepo/`):**
  - `apps/web/` - Next.js web application
  - `packages/ui-kit/` - React Native UI components
  - Uses Turbo and pnpm for build management

### Available Automation Tools
1. **enforce_style_guide.py** - Comprehensive style checking for Python scripts
2. **check_quality_gates.py** - Quality gate checker for production/experimental separation
3. **Pre-commit hooks** - Automated formatting and linting
4. **Makefile** - Contains fmt, lint, and test commands

## Cleanup Strategy

### Phase 1: Setup and Preparation
1. **Ensure all dependencies are installed**
   - Install Python dev dependencies: `poetry install --with dev`
   - Install pre-commit hooks: `pre-commit install`
   - Install Node.js dependencies in monorepo: `cd aivillage-monorepo && pnpm install`

2. **Create backup branch**
   - Create a backup branch before making extensive changes
   - This allows safe rollback if needed

### Phase 2: Python Cleanup (Priority: High)

#### Step 1: Run Black Formatter
```bash
# Format all Python files with Black
poetry run black .
```

#### Step 2: Run Ruff Linter with Auto-fix
```bash
# Run Ruff with auto-fix on all Python code
poetry run ruff check . --fix
```

#### Step 3: Run Style Guide Enforcement
```bash
# Run custom style guide enforcement on scripts
python scripts/enforce_style_guide.py --fix --report style_report.txt
```

#### Step 4: Type Checking with MyPy
```bash
# Run MyPy type checking
poetry run mypy agent_forge/ scripts/
```

#### Step 5: Security Scanning with Bandit
```bash
# Run Bandit security scanner
poetry run bandit -r agent_forge/ scripts/ --exclude tests
```

### Phase 3: JavaScript/TypeScript Cleanup (Priority: Medium)

#### Step 1: Run ESLint
```bash
cd aivillage-monorepo
pnpm lint
```

#### Step 2: Run Type Checking
```bash
cd aivillage-monorepo
pnpm type-check
```

#### Step 3: Format with Prettier
```bash
cd aivillage-monorepo
pnpm prettier --write .
```

### Phase 4: Pre-commit Hook Validation

#### Step 1: Run All Pre-commit Hooks
```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files
```

#### Step 2: Fix Any Hook Failures
- Address any issues reported by pre-commit hooks
- Re-run until all hooks pass

### Phase 5: Test Validation

#### Step 1: Run Python Tests
```bash
# Run pytest with coverage
poetry run pytest -q --cov=.
```

#### Step 2: Run JavaScript Tests
```bash
cd aivillage-monorepo
pnpm test
```

### Phase 6: Manual Review and Fixes

#### Areas Requiring Manual Attention:
1. **Docstring Addition**
   - Add missing module, class, and function docstrings
   - Follow Google docstring convention

2. **Import Organization**
   - Ensure imports follow the order: standard library, third-party, local
   - Use isort rules defined in ruff

3. **Type Hints**
   - Add type hints to function signatures
   - Fix any mypy errors that require manual intervention

4. **TODO/FIXME Cleanup**
   - Review and address TODO/FIXME comments
   - Move experimental code out of production directories

5. **File Structure**
   - Ensure all Python files have proper shebang (`#!/usr/bin/env python3`)
   - Add `__init__.py` files where missing

### Phase 7: Quality Gate Verification

#### Run Quality Gate Checks
```bash
python scripts/check_quality_gates.py
```

### Phase 8: Documentation and Reporting

1. **Generate Cleanup Report**
   - Document all changes made
   - List any remaining issues that need attention
   - Create metrics on improvements (lines fixed, issues resolved, etc.)

2. **Update Documentation**
   - Update README files if needed
   - Ensure code documentation matches implementation

## Execution Order

1. **Day 1: Setup and Python Core Cleanup**
   - Setup and preparation
   - Black formatting
   - Ruff auto-fixes
   - Initial commit of auto-fixes

2. **Day 2: Python Deep Cleanup**
   - Style guide enforcement
   - MyPy type checking fixes
   - Bandit security fixes
   - Manual docstring addition

3. **Day 3: JavaScript/TypeScript and Integration**
   - JavaScript/TypeScript cleanup
   - Pre-commit hook validation
   - Test suite validation

4. **Day 4: Final Review and Documentation**
   - Quality gate verification
   - Final manual reviews
   - Generate reports
   - Create PR with all changes

## Success Criteria

1. **All automated tools pass:**
   - Black formatting check passes
   - Ruff linting passes with no errors
   - MyPy type checking passes
   - Bandit security scan passes
   - ESLint passes in monorepo
   - All pre-commit hooks pass

2. **Code quality metrics:**
   - 100% of Python files have proper shebang
   - 100% of functions/classes have docstrings
   - No TODO/FIXME in production code
   - All imports properly organized
   - Type hints added where feasible

3. **Tests pass:**
   - All Python tests pass
   - All JavaScript tests pass
   - Test coverage maintained or improved

## Risk Mitigation

1. **Backup Strategy**
   - Create feature branch for cleanup
   - Commit after each major phase
   - Keep original branch untouched

2. **Incremental Approach**
   - Fix auto-fixable issues first
   - Manual fixes in separate commits
   - Test after each phase

3. **Review Process**
   - Self-review after each phase
   - Run tests frequently
   - Document any breaking changes

## Estimated Timeline

- **Total Duration:** 4-5 days
- **Automated Cleanup:** 1-2 days
- **Manual Fixes:** 2-3 days
- **Testing & Documentation:** 1 day

## Next Steps

1. Create cleanup branch: `git checkout -b feature/comprehensive-cleanup`
2. Start with Phase 1: Setup and Preparation
3. Execute phases sequentially
4. Create detailed PR with cleanup summary
