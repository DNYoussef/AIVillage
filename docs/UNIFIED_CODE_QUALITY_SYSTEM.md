# Unified Code Quality System

This document describes the unified code quality system implemented for AIVillage, which consolidates multiple linting and formatting tools into a streamlined, efficient workflow.

## Overview

The unified system replaces the previous overlapping configuration of 7+ tools with 3 core tools:
- **Ruff**: Primary linter and formatter (replaces flake8, isort, bandit for most cases)
- **Black**: Secondary formatter for specific formatting needs
- **MyPy**: Type checking

## Configuration

All configuration is centralized in `pyproject.toml`:

### Ruff Configuration
- **Target Version**: Python 3.12 (matching actual environment)
- **Line Length**: 120 characters
- **Rule Selection**: Comprehensive set including security, import sorting, and code quality
- **Auto-fixing**: Enabled by default

### Black Configuration
- **Line Length**: 120 characters (consistent with Ruff)
- **Target Version**: Python 3.12
- **Exclude Patterns**: Aligned with Ruff for consistency

### MyPy Configuration
- **Python Version**: 3.12
- **Explicit Package Bases**: Enabled to handle project structure
- **Strict Mode**: Disabled initially for gradual adoption

## Usage

### Command Line
```bash
# Format code
ruff format .
black . --line-length=120

# Lint code (with auto-fixing)
ruff check . --fix

# Security check
ruff check . --select S --output-format=concise

# Type check
mypy . --ignore-missing-imports --no-strict-optional
```

### Makefile Commands
```bash
# Format all code
make format

# Lint with auto-fixing
make lint

# Generate lint report
make lint-report

# Security scan
make security-check

# Type checking
make type-check

# Complete CI pipeline
make ci
```

### Pre-commit Hooks
The system includes optimized pre-commit hooks:
- Essential file quality checks (trailing whitespace, file endings, etc.)
- Ruff linting and formatting
- Black formatting for specific needs
- MyPy type checking (optional, excludes test files)

## Key Improvements

### Performance
- **Faster execution**: Ruff is significantly faster than the previous tool chain
- **Parallel processing**: Tools run independently without conflicts
- **Reduced overhead**: Fewer tool invocations and configurations

### Consistency
- **Unified configuration**: Single source of truth in pyproject.toml
- **Consistent exclude patterns**: Same patterns across all tools
- **Aligned formatting**: 120-character line length across all formatters

### Maintainability
- **Fewer dependencies**: Reduced from 8+ tools to 4 core tools
- **Simplified debugging**: Clear separation of concerns
- **Version alignment**: All tools target Python 3.12

## File Structure

### Removed Files
- `.flake8` - Functionality moved to pyproject.toml
- `.bandit` - Security rules moved to Ruff configuration
- `.isort.cfg` - Import sorting moved to Ruff configuration

### Updated Files
- `pyproject.toml` - Centralized configuration
- `.pre-commit-config.yaml` - Optimized hook chain
- `Makefile` - Unified commands
- `requirements-dev.txt` - Simplified dependencies

## Migration Notes

### For Developers
1. **Update your environment**: Run `pip install -r requirements-dev.txt`
2. **Reinstall pre-commit**: Run `pre-commit install`
3. **Use new commands**: Switch to unified Makefile targets
4. **Configuration location**: All settings are now in pyproject.toml

### For CI/CD
1. **Update build scripts**: Use new Makefile targets
2. **Simplified workflow**: Fewer tool invocations required
3. **Better error reporting**: JSON output available for automated processing

## Troubleshooting

### Common Issues
1. **Syntax errors**: Some files may need manual fixing before formatting
2. **MyPy module conflicts**: Use `--explicit-package-bases` flag
3. **Windows path issues**: Use forward slashes in exclude patterns

### Performance Tips
1. **Use ruff format first**: Faster than Black for bulk formatting
2. **Run security checks separately**: Use `--select S` for targeted scans
3. **Exclude large directories**: Update pyproject.toml exclude patterns

## Future Enhancements

### Planned Improvements
1. **Gradual MyPy strictness**: Incrementally enable strict mode
2. **Custom rule development**: Add project-specific linting rules
3. **Integration with IDEs**: Enhanced editor integration
4. **Automated fixes**: Expand auto-fixing capabilities

### Monitoring
- Track linting performance and coverage
- Monitor code quality metrics
- Regular tool version updates and optimization

## Status

✅ **Phase 1 Complete**: Tool consolidation and configuration unification
✅ **Phase 2 Complete**: Pre-commit hooks and Makefile optimization
✅ **Phase 3 Complete**: Testing and documentation

The unified system is now production-ready and provides a solid foundation for maintaining code quality across the AIVillage codebase.
