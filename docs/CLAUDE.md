# AIVillage Project Knowledge Base and Coding Style Guide

## Project Overview

AIVillage is a sophisticated multi-agent AI system with self-evolution capabilities. The project provides a prototype core infrastructure for:
- **Compression Pipeline**: Advanced model compression with 4x-100x+ compression ratios
- **RAG System**: Retrieval-augmented generation with ~1.19 ms/query baseline latency
- **Agent Orchestration**: 18 specialized agent types with KPI-based evolution
- **P2P Networking**: Mesh networking for distributed computing
- **Mobile Support**: Framework for mobile AI deployment (in development)

### Current Status (August 2025)
- **Overall Progress**: ~35% towards Atlantis Vision
- **Code Quality Score**: 85%
- **Critical Issues**: 16 fixed, ongoing quality improvements
- **Testing Coverage**: Comprehensive test suite with 164 test files

## AIVillage Coding Style Guide

### Code Formatting Standards

#### Python Version
- **Target**: Python 3.10+ (configured in pyproject.toml)
- **Minimum**: Python 3.8 (for compatibility)

#### Line Length and Formatting
- **Line Length**: 120 characters (configured for Black, isort, and flake8)
- **Formatter**: Black with line-length=120
- **Import Sorting**: isort with profile=black, line-length=120
- **Indentation**: 4 spaces (no tabs)

#### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import torch
from transformers import AutoModel

# Local imports (sorted by module)
from agent_forge.cli import main
from agents.base import BaseAgent
from communications.protocol import MessageProtocol
from core.utils import logger
from ingestion.pipeline import DataPipeline
from utils.helpers import format_output
```

### Linting Rules and Standards (Updated August 2025)

#### Ruff Configuration (Primary Linter)
The project uses **Ruff v0.0.263** with streamlined rule sets for compatibility:
- **pycodestyle** (E): Critical style violations
- **pyflakes** (F): Logical errors and undefined names
- **isort** (I): Import sorting and organization
- **pyupgrade** (UP): Modern Python syntax upgrades

#### Streamlined Pre-commit Hooks
Essential quality checks that run on every commit:
1. **trailing-whitespace**: Remove trailing spaces
2. **end-of-file-fixer**: Ensure files end with newline
3. **check-merge-conflict**: Detect merge conflicts
4. **check-added-large-files**: Prevent large files (>5MB)
5. **detect-private-key**: Security check for private keys
6. **check-toml**: Validate TOML syntax
7. **check-ast**: Validate Python syntax
8. **debug-statements**: Check for debug statements
9. **ruff**: Core linting with auto-fix
10. **black**: Code formatting (120 char line length)
11. **isort**: Import sorting with black profile
12. **detect-secrets**: Security baseline scanning

#### Compatibility Configuration
- **Target Python**: 3.11 (compatible with old ruff version)
- **Line Endings**: LF only (Windows compatible via .gitattributes)
- **Excluded Paths**: `deprecated/`, `archive/`, `experimental/`
- **Auto-fix Enabled**: Ruff automatically fixes import issues

#### Manual Linting Commands
```bash
# Quick fix
ruff check packages/ --fix

# Full format
black packages/ && isort packages/

# Test hooks
pre-commit run --all-files
```

### Type Annotations
```python
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path

def process_data(
    input_path: Path,
    output_dir: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """Process data with specified options."""
    ...
```

### Documentation Standards

#### Docstring Format (Google Style)
```python
def complex_function(
    param1: str,
    param2: int,
    optional_param: Optional[float] = None
) -> Dict[str, Any]:
    """Brief description of function purpose.

    Longer description if needed, explaining the function's behavior,
    assumptions, and any important details.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.
        optional_param: Description of optional parameter. Defaults to None.

    Returns:
        Dictionary containing:
            - 'status': Success status (bool)
            - 'result': Processed result
            - 'metadata': Additional information

    Raises:
        ValueError: If param2 is negative.
        FileNotFoundError: If input file doesn't exist.

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['status'])
        True
    """
```

### Security Standards

#### Bandit Configuration
- **Excluded directories**: tests/, experimental/, deprecated/
- **Skipped checks**:
  - B101: assert_used (allowed for tests)
  - B601: paramiko_calls (if needed)

#### Security Best Practices
1. Never hardcode credentials or API keys
2. Use environment variables for sensitive data
3. Validate all user inputs
4. Use cryptography library for encryption needs
5. Regular dependency updates for security patches

### Testing Standards

#### Test Organization
```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests
├── performance/       # Performance benchmarks
├── mobile/           # Mobile compatibility tests
└── conftest.py       # Shared pytest fixtures
```

#### Test Naming Convention
```python
def test_function_name_describes_scenario():
    """Test that function handles specific scenario correctly."""
    ...

def test_error_raised_when_invalid_input():
    """Test that appropriate error is raised for invalid input."""
    ...
```

### CI/CD Integration

#### GitHub Actions Workflow
All PRs must pass:
1. **Linting**: Black, Ruff, isort, flake8
2. **Type Checking**: MyPy (non-blocking initially)
3. **Security**: Bandit scan
4. **Tests**: Unit and integration tests
5. **Coverage**: Code coverage reporting

#### Local Development Commands
```bash
# Format code
make format  # or: black src/ tests/ && isort src/ tests/

# Run linting
make lint    # or: pre-commit run --all-files

# Run tests
make test    # or: pytest tests/ -v

# Full check
make all     # clean, install, lint, test, build
```

### File and Directory Standards

#### Naming Conventions
- **Files**: lowercase_with_underscores.py
- **Classes**: PascalCase
- **Functions/Variables**: lowercase_with_underscores
- **Constants**: UPPERCASE_WITH_UNDERSCORES
- **Private**: _leading_underscore

#### Module Organization
```python
# 1. Module docstring
"""Module for handling agent communications."""

# 2. Imports (organized by isort)
import logging
from typing import Optional

# 3. Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# 4. Module-level variables
logger = logging.getLogger(__name__)

# 5. Classes and functions
class AgentCommunicator:
    """Handles inter-agent communication."""
    ...
```

### Error Handling

#### Best Practices
```python
# Good: Specific error handling
try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid value, using default: {e}")
    result = default_value

# Bad: Generic catch-all
try:
    result = risky_operation()
except Exception:
    pass  # Never suppress errors silently
```

### Performance Considerations

1. **Profiling**: Use cProfile for performance analysis
2. **Benchmarking**: Add benchmarks in tests/performance/
3. **Memory**: Monitor memory usage with psutil
4. **Async**: Use async/await for I/O-bound operations

### Compression Pipeline Development Guidelines

When working on compression features:
1. Always benchmark before/after changes
2. Document compression ratios and performance metrics
3. Test with various model sizes (small, medium, large)
4. Ensure backwards compatibility
5. Update COMPRESSION_EVOLUTION.md with significant changes

### Development Workflow

1. **Before Coding**:
   - Read relevant documentation in docs/
   - Check existing patterns in similar files
   - Review test cases for the component

2. **While Coding**:
   - Follow the style guide strictly
   - Write tests alongside code
   - Add appropriate type hints
   - Document complex logic

3. **Before Committing**:
   - Run `make format` to auto-format
   - Run `make lint` to check style
   - Run `make test` to ensure tests pass
   - Update documentation if needed

4. **PR Checklist**:
   - [ ] Code follows style guide
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No hardcoded secrets
   - [ ] Benchmarks run (if performance-related)

### Common Patterns and Best Practices

#### Configuration Management
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with validation."""
    api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"
```

#### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning about potential issues")
logger.error("Error that needs attention")
logger.critical("Critical error requiring immediate action")
```

#### Resource Management
```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    """Properly manage resources."""
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)
```

## Quick Reference

### Essential Commands
```bash
# Development setup
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install

# Code quality
black src/ tests/ --check  # Check formatting
black src/ tests/          # Apply formatting
ruff check src/           # Run linting
mypy src/                 # Type checking

# Testing
pytest                    # Run all tests
pytest tests/unit/ -v     # Run unit tests
pytest --cov=src         # With coverage

# Pre-commit
pre-commit run --all-files  # Run all checks
```

### File Locations
- **Source Code**: src/
- **Tests**: tests/
- **Documentation**: docs/
- **Scripts**: scripts/
- **Configuration**: pyproject.toml, .pre-commit-config.yaml

---

This guide ensures consistent, high-quality code across the AIVillage project. When in doubt, refer to existing code patterns and ask for clarification in PR reviews.
