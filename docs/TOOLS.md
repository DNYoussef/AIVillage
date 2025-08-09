# AIVillage Development Tools Installation Guide

## Overview

This guide provides step-by-step instructions for installing and configuring all development tools required for the AIVillage project. The setup includes linting tools, formatters, type checkers, and automation systems.

## Prerequisites

### Required Software
- **Python 3.9+** (Recommended: Python 3.10 or 3.11)
- **Git** (Latest stable version)
- **pip** (Usually comes with Python)

### Operating System Support
- ✅ **Linux**: Full support (Ubuntu 20.04+, CentOS 8+)
- ✅ **macOS**: Full support (macOS 11+)
- ⚠️ **Windows**: Full support with PowerShell/WSL recommended

## Core Tool Installation

### Option 1: Requirements File Installation (Recommended)

```bash
# Install all development dependencies
pip install -r requirements-dev.txt

# Verify installation
python -c "import ruff, black, isort, mypy, flake8; print('All tools installed')"
```

### Option 2: Individual Tool Installation

```bash
# Core linting tools
pip install ruff>=0.12.0
pip install black>=24.8.0
pip install isort>=5.13.0
pip install mypy>=1.8.0
pip install flake8>=7.0.0

# Pre-commit framework
pip install pre-commit>=3.6.0

# Additional quality tools
pip install bandit>=1.7.0    # Security scanner
pip install safety>=3.0.0    # Vulnerability scanner
pip install pytest>=8.0.0    # Testing framework
```

## Tool Configuration

### 1. Ruff (Primary Linter)

**Installation:**
```bash
pip install ruff
# Or via conda
conda install -c conda-forge ruff
```

**Verification:**
```bash
ruff --version
# Expected: ruff 0.12.3 or later
```

**Configuration**: Already configured in `pyproject.toml`

### 2. Black (Code Formatter)

**Installation:**
```bash
pip install black
```

**Verification:**
```bash
black --version  
# Expected: black, 24.8.0 or later
```

**Configuration**: Already configured in `pyproject.toml`

### 3. isort (Import Sorter)

**Installation:**
```bash
pip install isort
```

**Verification:**
```bash
isort --version
# Expected: 5.13.2 or later
```

**Configuration**: Already configured in `.isort.cfg`

### 4. mypy (Type Checker)

**Installation:**
```bash
pip install mypy
```

**Verification:**
```bash
mypy --version
# Expected: mypy 1.8.0 or later
```

**Configuration**: Already configured in `pyproject.toml`

### 5. flake8 (Style Checker)

**Installation:**
```bash
pip install flake8
```

**Verification:**
```bash
flake8 --version
# Expected: 7.0.0 or later
```

**Configuration**: Already configured in `.flake8`

## Unified Linter Setup

### Installation Verification

```bash
# Test the unified linter
python lint.py --help

# Run a quick test
python lint.py src/core --output summary
```

### Expected Output
```
[TOOLS] Available Tools:
  [OK] ruff
  [OK] black  
  [OK] isort
  [OK] mypy
  [OK] flake8

[PYTHON] Running Python linting on: src/core
Found XX Python files
...
```

## Pre-commit Hooks Setup

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# (Optional) Install pre-push hooks
pre-commit install --hook-type pre-push
```

### Verification

```bash
# Test hooks manually
pre-commit run --all-files

# Check hook status
pre-commit run --help
```

### Configuration

The `.pre-commit-config.yaml` file is already configured with:
- Basic file checks
- Ruff linting with auto-fix
- Black formatting
- isort import sorting
- Security scanning
- Custom unified linter

## IDE Integration

### Visual Studio Code

**Install Extensions:**
1. Python (Microsoft)
2. Ruff (Astral Software)  
3. Black Formatter (Microsoft)
4. isort (Microsoft)

**Settings Configuration (`settings.json`):**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.flake8Enabled": false,
    "python.linting.pylintEnabled": false,
    
    "python.formatting.provider": "none",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.codeActionsOnSave": {
            "source.organizeImports": true,
            "source.fixAll.ruff": true
        }
    },
    
    "ruff.args": ["--line-length", "88"],
    "black-formatter.args": ["--line-length", "88"],
    "isort.args": ["--profile", "black", "--line-length", "88"],
    
    "files.exclude": {
        "**/__pycache__": true,
        "**/.mypy_cache": true,
        "**/build": true,
        "**/dist": true
    }
}
```

### PyCharm/IntelliJ

**Plugin Installation:**
1. Go to Settings → Plugins
2. Install "Ruff" plugin
3. Install "Black" plugin (if available)

**Configuration:**
1. **File → Settings → Tools → External Tools**
2. Add new tool:
   - Name: Unified Linter
   - Program: `python`
   - Arguments: `lint.py $FilePath$ --output summary`
   - Working Directory: `$ProjectFileDir$`

3. **Code Style → Python**
   - Set line length to 88
   - Configure import organization

### Vim/Neovim

**Plugin Installation (using vim-plug):**
```vim
Plug 'psf/black', { 'branch': 'stable' }
Plug 'fisadev/vim-isort'
Plug 'dense-analysis/ale'  " For ruff integration
```

**Configuration (.vimrc):**
```vim
" Black configuration
autocmd BufWritePre *.py execute ':Black'

" ALE configuration for ruff
let g:ale_linters = {'python': ['ruff', 'mypy']}
let g:ale_fixers = {'python': ['ruff', 'black', 'isort']}
let g:ale_fix_on_save = 1
```

## CI/CD Integration

### GitHub Actions

The workflow is already configured in `.github/workflows/ci.yml` with:
- Unified linter integration
- Security scanning
- Artifact upload for lint reports
- Multi-job pipeline (lint → security → test)

### GitLab CI (Optional)

Create `.gitlab-ci.yml`:
```yaml
stages:
  - lint
  - security  
  - test

lint:
  stage: lint
  image: python:3.10
  script:
    - pip install ruff black isort mypy flake8
    - python lint.py . --output summary
  artifacts:
    when: on_failure
    reports:
      junit: lint_report.xml
    paths:
      - lint_report.json
      - lint_report.txt

security:
  stage: security
  image: python:3.10
  script:
    - pip install ruff
    - ruff check . --select S
  allow_failure: false
```

### Jenkins (Optional)

Create `Jenkinsfile`:
```groovy
pipeline {
    agent any
    stages {
        stage('Lint') {
            steps {
                sh 'pip install ruff black isort mypy flake8'
                sh 'python lint.py . --output summary'
            }
            post {
                failure {
                    archiveArtifacts artifacts: 'lint_report.*'
                }
            }
        }
    }
}
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Module Not Found

**Problem:** `ModuleNotFoundError: No module named 'ruff'`

**Solution:**
```bash
# Check Python installation
python --version

# Check pip installation  
pip --version

# Install in correct environment
pip install ruff

# Verify installation
python -c "import ruff; print(ruff.__version__)"
```

#### 2. Permission Denied (Linux/macOS)

**Problem:** Permission errors during installation

**Solution:**
```bash
# Use user installation
pip install --user ruff black isort mypy flake8

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install ruff black isort mypy flake8
```

#### 3. Windows Path Issues

**Problem:** Tools not found in PATH

**Solution:**
```powershell
# Add Python Scripts to PATH
$env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python310\Scripts"

# Or use Python module execution
python -m ruff check .
python -m black .
python -m isort .
```

#### 4. Pre-commit Hook Failures

**Problem:** Hooks fail to execute

**Solution:**
```bash
# Reinstall pre-commit
pip uninstall pre-commit
pip install pre-commit

# Clean and reinstall hooks
pre-commit clean
pre-commit install

# Update to latest versions
pre-commit autoupdate
```

### Performance Optimization

#### 1. Slow Linting

**Solutions:**
- Use `--select` for specific rules only
- Exclude large directories in config
- Enable parallel execution
- Use SSD storage for cache directories

#### 2. Memory Issues

**Solutions:**
```bash
# Increase cache sizes
export RUFF_CACHE_DIR=~/.cache/ruff
export BLACK_CACHE_DIR=~/.cache/black

# Process files in batches
python lint.py src/ --no-parallel
```

### Environment-Specific Issues

#### Docker Environment

```dockerfile
FROM python:3.10-slim

# Install linting tools
RUN pip install ruff black isort mypy flake8 pre-commit

# Copy configuration files
COPY pyproject.toml .flake8 .isort.cfg ./

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "lint.py", ".", "--output", "summary"]
```

#### Virtual Environment Issues

```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/macOS
# fresh_env\Scripts\activate   # Windows

# Install requirements
pip install --upgrade pip
pip install -r requirements-dev.txt

# Verify installation
python lint.py --help
```

## Version Management

### Tool Version Matrix

| Tool | Minimum | Recommended | Latest Tested |
|------|---------|-------------|---------------|
| ruff | 0.10.0 | 0.12.3 | 0.12.3 |
| black | 23.0.0 | 24.8.0 | 24.8.0 |
| isort | 5.10.0 | 5.13.2 | 5.13.2 |
| mypy | 1.5.0 | 1.8.0 | 1.8.0 |
| flake8 | 6.0.0 | 7.0.0 | 7.0.0 |

### Update Strategy

```bash
# Check current versions
pip list | grep -E "(ruff|black|isort|mypy|flake8)"

# Update all tools
pip install --upgrade ruff black isort mypy flake8

# Test after updates
python lint.py src/core --output summary

# Update pre-commit hooks
pre-commit autoupdate
```

## Maintenance

### Regular Tasks

**Weekly:**
- Update tool versions
- Run comprehensive linting
- Review security scan results

**Monthly:**
- Update pre-commit hook versions
- Review and update configurations
- Analyze linting metrics

### Health Checks

```bash
# Verify all tools work
python -c "
import ruff, black, isort, mypy, flake8
print('✅ All tools imported successfully')
"

# Test unified linter
python lint.py . --output summary | head -20

# Check pre-commit status
pre-commit run --all-files --verbose
```

## Getting Help

### Documentation Links
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)

### Community Support
- [Ruff GitHub Issues](https://github.com/astral-sh/ruff/issues)
- [Black GitHub Issues](https://github.com/psf/black/issues)
- [Pre-commit Documentation](https://pre-commit.com/)

### Project Support
- Check `LINTING.md` for project-specific guidelines
- Review lint reports for detailed issue information
- Consult team members for complex configuration issues

---

*Last updated: 2025-08-09*
*Tool Installation Guide Version: 1.0*