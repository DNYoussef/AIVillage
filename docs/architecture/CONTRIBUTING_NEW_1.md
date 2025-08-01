# Contributing to AIVillage

Welcome to AIVillage! We're excited about your interest in contributing to our self-evolving AI infrastructure platform. This guide covers everything you need to know about developing, testing, and contributing to the project.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Repository Structure](#repository-structure)
3. [Development Workflow](#development-workflow)
4. [Testing Guidelines](#testing-guidelines)
5. [Code Quality Standards](#code-quality-standards)
6. [Component-Specific Guidelines](#component-specific-guidelines)
7. [Submitting Changes](#submitting-changes)

## Development Environment Setup

### Prerequisites

 - Python 3.10+ (recommended: 3.11+)
- Git
- Docker (for microservices development)
- Poetry (optional, for dependency management)

### Initial Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/AIVillage.git
   cd AIVillage
   ```

2. **Virtual Environment**
   ```bash
   python -m venv new_env
   source new_env/bin/activate  # Linux/Mac
   # or
   new_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Development Mode Setup**
   ```bash
   export AIVILLAGE_DEV_MODE=true
   ```

5. **Pre-commit Hooks** (Optional but recommended)
   ```bash
   pre-commit install
   ```

## Repository Structure

### Production Components (`production/`)
- **compression/**: Model compression algorithms (BitNet, VPTQ, SeedLM)
- **evolution/**: Evolutionary optimization and model merging
- **rag/**: Retrieval-augmented generation system
- **geometry/**: Geometric analysis of model weight spaces

### Core Infrastructure
- **agent_forge/**: Model deployment and serving infrastructure
- **communications/**: Mesh networking and credit system
- **tools/**: Development and deployment utilities
- **configs/**: Configuration management

### Experimental Components (`experimental/`)
- **agents/**: Multi-agent system (King, Sage, Magi)
- **services/**: Microservices architecture
- **training/**: Advanced training pipelines

## Development Workflow

### Branch Strategy

1. **Main Branch**: Production-ready code
2. **Feature Branches**: `feature/your-feature-name`
3. **Bug Fixes**: `fix/issue-description`
4. **Experimental**: `experimental/component-name`

### Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop with Quality Gates**
   ```bash
   # Run quality checks frequently
   python run_quality_checks.py

   # Run relevant tests
   pytest tests/unit/production/
   pytest tests/integration/
   ```

3. **Pre-commit Validation**
   ```bash
   # Format code
   python quick_format_fix.py

   # Run comprehensive tests
   python run_comprehensive_tests.py
   ```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── production/         # Production component tests
│   └── experimental/       # Experimental component tests
├── integration/            # Integration tests
├── performance/            # Performance benchmarks
└── security/              # Security tests
```

### Running Tests

```bash
# All tests
pytest

# Production tests only
pytest tests/unit/production/

# Specific component
pytest tests/unit/production/compression/

# With coverage
pytest --cov=production --cov-report=html

# Performance tests
python run_coverage_analysis.py
```

### Test Requirements

- **Production Code**: 80%+ test coverage required
- **Experimental Code**: Tests encouraged but not mandatory
- **Integration Tests**: Required for API changes
- **Performance Tests**: Required for optimization features

## Code Quality Standards

### Automated Quality Gates

Our CI/CD pipeline enforces:

- **Formatting**: Black, isort
- **Linting**: flake8, pylint
- **Type Checking**: mypy
- **Security**: bandit
- **Test Coverage**: 80%+ for production code

### Manual Quality Checks

```bash
# Run all quality checks
python run_quality_checks.py

# Fix common issues
python comprehensive_code_quality_analysis.py

# Memory optimization
python memory_optimizer.py

# Security audit
python apply_security_patches.py
```

## Component-Specific Guidelines

### Production Components

**Compression System**
```bash
# Test compression algorithms
python production/compression/model_compression/model_compression.py
pytest tests/unit/production/compression/
```

**Evolution System**
```bash
# Test evolutionary algorithms
python production/evolution/evomerge/evolutionary_tournament.py
pytest tests/unit/production/evolution/
```

**RAG System**
```bash
# Test RAG pipeline
python production/rag/rag_system/main.py
pytest tests/unit/production/rag/
```

### Experimental Components

**Agent System**
```bash
# Test agent specialization
python experimental/agents/agents/king/init.py
python experimental/agents/agents/sage/config.py
python experimental/agents/agents/magi/core/magi_agent.py
```

**Microservices**
```bash
# Start development services
python experimental/services/services/gateway/app.py
python experimental/services/services/twin/app.py

# Test service integration
python experimental/services/services/wave_bridge/test_integration.py
```

### Core Infrastructure

**Agent Forge**
```bash
# Deploy models
python agent_forge/deploy_agent.py

# Test deployment pipeline
pytest tests/integration/agent_forge/
```

**Communications**
```bash
# Test mesh networking
python communications/test_credits_standalone.py
python communications/mcp_client.py
```

## Submitting Changes

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement Changes**
   - Follow component-specific guidelines
   - Maintain test coverage requirements
   - Update documentation as needed

3. **Quality Validation**
   ```bash
   # Run comprehensive validation
   python run_comprehensive_tests.py
   python run_quality_checks.py

   # Check security
   python apply_security_patches.py
   ```

4. **Submit Pull Request**
   - Clear description of changes
   - Reference related issues
   - Include test results
   - Update documentation

### PR Requirements

**Production Components**
- 80%+ test coverage
- No security vulnerabilities
- Performance benchmarks
- API documentation updates

**Experimental Components**
- Basic functionality tests
- Clear experimental warnings
- Development documentation

Thank you for contributing to AIVillage! Your efforts help advance the state of autonomous AI infrastructure and benefit the entire community.

---

**For additional support**: Check our comprehensive documentation in the `docs/` directory or reach out through GitHub issues.
