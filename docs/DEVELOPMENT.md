# AIVillage Development Guide

## Pre-commit Setup

### Quick Start
```bash
# Install pre-commit hooks (one time)
pre-commit install

# Run manually before committing (optional)
pre-commit run --all-files
```

## Manual Linting (Alternative to Pre-commit)

### Quick Check
```bash
# Fix common issues automatically
ruff check packages/ --fix
black packages/
isort packages/

# Check specific file
ruff check path/to/file.py --fix
```

### Full Check
```bash
# Run all pre-commit hooks manually
pre-commit run --all-files
```

## Our Tooling Configuration

### Ruff (Primary Linter)
- **Version**: 0.0.263 (compatible with our setup)
- **Rules**: Core only (E, F, I, UP) - focused on essential issues
- **Auto-fixes**: Import sorting, syntax errors, modern Python upgrades
- **Config**: `pyproject.toml` [tool.ruff] section

### Black (Code Formatter)
- **Line length**: 120 characters
- **Target**: Python 3.11+ syntax
- **Auto-formats**: Code style, indentation, quotes

### Pre-commit Hooks
- **Simplified setup**: 8 essential hooks only
- **Fast execution**: Focused on critical issues
- **Windows compatible**: LF line endings enforced

## Troubleshooting

### Line Ending Issues
```bash
# If you see "fixed mixed line endings"
git add --renormalize .
git commit -m "fix: Normalize line endings"
```

### Pre-commit Failures
```bash
# Reset and clean pre-commit cache
pre-commit clean
pre-commit install

# Validate configuration
pre-commit validate-config
```

### Ruff Configuration Errors
```bash
# Check what ruff settings are being used
ruff check --show-settings packages/

# Test ruff manually
ruff check packages/ --fix --verbose
```

### Bypassing Hooks (Emergency Only)
```bash
# Skip pre-commit hooks (use sparingly)
git commit --no-verify -m "emergency commit"

# Skip specific hooks
SKIP=ruff,black git commit -m "skip specific hooks"
```

## Supported File Types

- **Python**: `.py` files in `packages/`, `src/`
- **Configuration**: `.toml`, `.yaml`, `.json` files
- **Documentation**: `.md` files (basic checks only)
- **Excluded**: `deprecated/`, `archive/`, `experimental/` directories

## Development Workflow

1. **Make changes** to code
2. **Pre-commit automatically runs** on `git commit`
3. **Fix any issues** reported by hooks
4. **Commit again** - should pass cleanly
5. **Push changes**

If pre-commit fails:
- Review the changes it made automatically
- Stage the fixes with `git add .`
- Commit again

## Configuration Files

- **`.pre-commit-config.yaml`**: Hook configuration (streamlined for compatibility)
- **`pyproject.toml`**: Ruff, Black, MyPy settings
- **`.gitattributes`**: Line ending enforcement
- **`.secrets.baseline`**: Security scanning baseline
