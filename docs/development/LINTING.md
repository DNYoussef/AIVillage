# AIVillage Linting & Code Quality Guide

## Overview

This project uses a comprehensive linting system to maintain high code quality, security, and consistency across the entire codebase. We've implemented a unified linting orchestrator that coordinates multiple tools to provide thorough analysis and automated fixes.

## Quick Start

### Basic Commands

```bash
# Run full linting check
python lint.py .

# Apply safe automatic fixes
python lint.py . --fix

# Generate comprehensive reports
python lint.py . --output both --output-file my_report

# Check specific directory
python lint.py src/core

# Security-only scan
ruff check . --select S
```

### Make Targets (if available)

```bash
make lint          # Run linting check
make lint-fix      # Apply automatic fixes
make lint-report   # Generate timestamped report
make security-check # Run security scan only
make format        # Format code with Black + isort
```

## Unified Linting System

### Tools Integrated

1. **Ruff** (Primary Linter)
   - Comprehensive rule set (E,W,F,I,N,D,UP,ANN,S,etc.)
   - Fast execution (~1-2 seconds for entire codebase)
   - Auto-fix capabilities for most issues

2. **Black** (Code Formatter)
   - Consistent code formatting
   - Line length: 88 characters
   - Automatic quote normalization

3. **isort** (Import Sorter)
   - Organizes and sorts imports
   - Black-compatible profile
   - First-party package recognition

4. **mypy** (Type Checker)
   - Static type analysis
   - Non-blocking in CI (warnings only)
   - Gradual typing support

5. **flake8** (Style Checker)
   - Additional style validation
   - Complementary to Ruff rules
   - Legacy compatibility

### Configuration Files

- **`pyproject.toml`**: Primary configuration (Ruff, Black, isort, mypy)
- **`.flake8`**: Flake8-specific rules
- **`.isort.cfg`**: Import sorting configuration
- **`.pre-commit-config.yaml`**: Git hooks configuration
- **`lint.py`**: Unified linting orchestrator

## Code Quality Standards

### Line Length
- **Standard**: 88 characters (Black default)
- **Rationale**: Balance between readability and screen real estate
- **Tools**: All tools configured consistently

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import requests
from transformers import AutoModel

# Local imports (sorted)
from src.core.utils import logger
from src.agents.base import BaseAgent
```

### Type Annotations
```python
# Required for new code
def process_data(
    input_path: Path,
    options: Dict[str, Any],
    timeout: Optional[int] = None
) -> Tuple[bool, str]:
    """Process data with specified options."""
    ...
```

### Documentation Standards
```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of function purpose.

    Longer description explaining behavior, assumptions, and important details.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Dictionary containing processing results.

    Raises:
        ValueError: If param2 is negative.

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['status'])
        True
    """
```

## Security Guidelines

### Critical Security Rules (Always Fix)

1. **S101**: Use of assert (disabled for tests)
2. **S102**: Use of exec()
3. **S103**: Use of eval()
4. **S105**: Hardcoded passwords
5. **S106**: Hardcoded password arguments
6. **S107**: Hardcoded password defaults
7. **S108**: Hardcoded temporary files

### Security Best Practices

```python
# ❌ Bad: Hardcoded credentials
password = "secret123"
api_key = "abc123"

# ✅ Good: Environment variables
password = os.environ.get("PASSWORD")
api_key = os.environ.get("API_KEY")

# ❌ Bad: Insecure random
import random
token = random.randint(1000, 9999)

# ✅ Good: Cryptographically secure
import secrets
token = secrets.token_urlsafe(32)
```

## Pre-commit Hooks

### Installed Hooks

1. **Basic Checks**
   - Trailing whitespace removal
   - End-of-file fixing
   - YAML/JSON validation
   - Large file detection
   - Merge conflict detection
   - Private key detection

2. **Code Quality**
   - Ruff linting (auto-fix enabled)
   - Ruff formatting
   - Black formatting
   - isort import sorting

3. **Security**
   - Security scan (critical issues only)
   - Private key detection

4. **Custom**
   - Unified linter (on pre-push)

### Hook Configuration

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate

# Skip hooks (emergency only)
git commit --no-verify
```

## CI/CD Integration

### GitHub Actions Workflow

The CI pipeline includes three jobs:

1. **Lint Job**
   - Runs unified linter
   - Blocks merge if critical issues found
   - Generates lint reports on failure
   - Uploads artifacts for debugging

2. **Security Job**
   - Dedicated security scan
   - Blocks merge if security issues found
   - Focuses on critical security rules

3. **Test Job**
   - Runs after lint and security pass
   - Executes test suite with coverage
   - Uploads coverage reports

### Failure Handling

- **Linting failures**: Block merge, generate reports
- **Security failures**: Always block merge
- **Test failures**: Block merge, upload coverage

## Issue Categories & Priorities

### 🔴 Critical (Block Merge)
- **F-series**: Logic errors, undefined variables
- **S-series**: Security vulnerabilities
- **E9-series**: Syntax errors

### 🟡 High Priority (Fix Soon)
- **E-series**: Style violations
- **W-series**: Warnings
- **I-series**: Import issues
- **ANN-series**: Missing type annotations

### 🟢 Medium Priority (Fix Eventually)
- **D-series**: Documentation issues
- **UP-series**: Upgrade opportunities
- **SIM-series**: Simplification suggestions

### ⚪ Low Priority (Optional)
- **N-series**: Naming conventions
- **PLR-series**: Refactoring suggestions

## Auto-Fix Capabilities

### Safe Auto-Fixes (Always Apply)
```bash
# Format code
python lint.py . --fix --select E,W,F541,I,Q

# Apply Black formatting
black .

# Sort imports
isort .
```

### Moderate Risk (Review First)
```bash
# Modernize code
ruff check . --fix --select UP,SIM,PIE

# Fix specific issues
ruff check . --fix --select F401,F841  # Remove unused imports/variables
```

### Manual Review Required
- Security issues (S-series)
- Type annotations (ANN-series)
- Complex refactoring (PLR-series)
- Documentation (D-series)

## Developer Workflow

### Before Committing
1. **Run linter**: `python lint.py . --output summary`
2. **Apply safe fixes**: `python lint.py . --fix`
3. **Review changes**: Ensure fixes don't break logic
4. **Commit changes**: Pre-commit hooks will run automatically

### Before Creating PR
1. **Full security scan**: `ruff check . --select S`
2. **Generate report**: `python lint.py . --output both --output-file pr_report`
3. **Address critical issues**: Fix all blocking issues
4. **Test changes**: Ensure tests still pass

### Handling Failures

#### Pre-commit Hook Failures
```bash
# See what failed
git commit -v

# Fix issues and try again
python lint.py . --fix
git add -u
git commit
```

#### CI/CD Failures
1. **Download lint report** from GitHub Actions artifacts
2. **Review critical issues** in the report
3. **Apply fixes** locally
4. **Push updates** to trigger new CI run

## IDE Integration

### VS Code Settings
```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.linting.ruffArgs": ["--line-length", "88"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm Settings
1. Enable Ruff plugin
2. Configure Black as formatter
3. Set line length to 88
4. Enable format on save
5. Configure import optimization

## Troubleshooting

### Common Issues

1. **"ruff not found"**
   ```bash
   pip install ruff
   ```

2. **"Line too long" errors**
   - Check line length settings in all config files
   - Ensure all tools use 88 characters

3. **Import order issues**
   ```bash
   isort .
   ```

4. **Pre-commit hook failures**
   ```bash
   pre-commit clean
   pre-commit install
   ```

5. **Unicode encoding errors**
   - Use ASCII characters in commit messages
   - Check file encoding (should be UTF-8)

### Performance Issues

1. **Slow linting**
   - Use `--select` to run specific rules only
   - Enable parallel execution: `python lint.py . --parallel`

2. **Large codebase**
   - Exclude unnecessary directories in config
   - Use incremental linting on changed files

### Getting Help

1. **Check configuration**: `python lint.py --help`
2. **Review reports**: Generated lint reports contain detailed issue information
3. **Consult documentation**: Ruff documentation for specific rule explanations
4. **Team consultation**: Discuss complex issues with team members

## Best Practices

### For New Code
- Write type annotations from the start
- Include docstrings for public functions
- Follow security guidelines
- Test changes thoroughly

### For Legacy Code
- Fix critical issues first (F, S series)
- Apply safe auto-fixes
- Gradually add type annotations
- Improve documentation incrementally

### For Team Collaboration
- Don't disable hooks without team agreement
- Communicate configuration changes
- Share lint reports for complex issues
- Review security implications of changes

## Metrics & Monitoring

### Quality Metrics Tracked
- Total issues by category
- Auto-fix success rate
- Pre-commit hook compliance
- Security issue trends
- Coverage improvements

### Regular Monitoring
- Weekly quality reports
- Security scan summaries
- CI/CD success rates
- Developer productivity metrics

---

## Configuration Reference

### Environment Variables
```bash
export RUFF_CACHE_DIR=~/.cache/ruff
export BLACK_CACHE_DIR=~/.cache/black
export MYPY_CACHE_DIR=~/.cache/mypy
```

### Tool Versions
- Ruff: v0.12.3+
- Black: v24.8.0+
- isort: v5.13.2+
- mypy: Latest stable
- flake8: v7.0.0+

### Excluded Directories
- `.git/`, `__pycache__/`, `build/`, `dist/`
- `venv/`, `env/`, `.mypy_cache/`
- `archived/`, `deprecated/`
- `.github/workflows/archived/`
- `scripts/archived/`

---

*Last updated: 2025-08-09*
*Unified Linting System Version: 1.0*
