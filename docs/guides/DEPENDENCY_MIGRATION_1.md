# Dependency Migration Guide

## Overview

All requirements files have been consolidated into a modern `pyproject.toml` configuration. This provides better dependency management, optional dependencies, and improved compatibility.

## Changes Made

### Archived Files
- `requirements.txt` → `archive/requirements_backup_YYYYMMDD/`
- `requirements_production.txt` → `archive/requirements_backup_YYYYMMDD/`
- `requirements_security_audit.txt` → `archive/requirements_backup_YYYYMMDD/`
- `requirements-dev.txt` → `archive/requirements_backup_YYYYMMDD/`
- `requirements-test.txt` → `archive/requirements_backup_YYYYMMDD/`
- `requirements_consolidated.txt` → `archive/requirements_backup_YYYYMMDD/`
- `requirements_development.txt` → `archive/requirements_backup_YYYYMMDD/`

### New Structure

The new `pyproject.toml` provides:

1. **Core dependencies** (always installed)
2. **Optional dependency groups**:
   - `dev` - Development tools, testing, linting
   - `test` - Testing framework and mocking tools
   - `experimental` - Experimental ML features, advanced models
   - `security` - Security scanning and enhanced authentication
   - `production` - Production server and monitoring tools

## Installation Commands

### Basic Installation
```bash
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

### Testing Installation
```bash
pip install -e ".[test]"
```

### Production Installation
```bash
pip install -e ".[production]"
```

### Multiple Groups
```bash
pip install -e ".[dev,test,security]"
```

### All Optional Dependencies
```bash
pip install -e ".[dev,test,experimental,security,production]"
```

## Key Improvements

### Security Enhancements
- All packages updated to latest secure versions
- Platform-specific markers for compatibility
- Clear separation of security-focused dependencies

### Dependency Resolution
- Removed version conflicts between requirements files
- Standardized version specifications
- Added platform markers for ARM64 compatibility

### Development Experience
- Cleaner dependency groups
- Better tooling integration
- Consistent formatting and linting configuration

### Production Optimization
- Minimal core dependencies
- Optional heavy ML dependencies
- Production-specific monitoring tools

## Migration Checklist

- [x] Consolidate all requirements files into pyproject.toml
- [x] Archive old requirements files
- [x] Update version specifications for security
- [x] Add platform markers for compatibility
- [x] Organize dependencies into logical groups
- [ ] Update CI/CD workflows to use pyproject.toml
- [ ] Update Docker files to use new installation commands
- [ ] Update documentation to reference new installation method

## Docker Updates Required

Update Dockerfile commands from:
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
```

To:
```dockerfile
COPY pyproject.toml .
RUN pip install -e ".[production]"
```

## CI/CD Updates Required

Update workflow files to use:
```yaml
- name: Install dependencies
  run: pip install -e ".[dev,test]"
```

## Component-Specific Requirements

Component-specific requirements files in subdirectories (like `agent_forge/requirements.txt`) are preserved for modular development but should eventually be migrated to pyproject.toml optional dependencies.
