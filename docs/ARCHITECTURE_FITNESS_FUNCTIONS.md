# AIVillage Architecture Fitness Functions

A comprehensive system for automated architectural quality assurance based on connascence principles and architectural best practices.

## Overview

This system provides automated verification of architectural constraints and quality attributes through:

- **Fitness Functions**: Executable tests that verify architectural properties
- **Connascence Detection**: Automated detection of coupling violations
- **Dependency Analysis**: Circular dependency detection and layer validation
- **Quality Metrics**: Coupling, cohesion, and complexity measurements
- **Continuous Monitoring**: Integration with CI/CD pipelines
- **Health Dashboard**: Real-time architectural health monitoring

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Architecture Fitness Functions           │
├─────────────────────────────────────────────────────────────┤
│  📋 Test Suite              │  🔍 Analysis Engine           │
│  ├── Circular Dependencies  │  ├── Dependency Graph Builder │
│  ├── Connascence Rules     │  ├── Coupling Calculator      │
│  ├── Module Size Limits    │  ├── Complexity Analyzer      │
│  ├── Complexity Limits     │  └── Technical Debt Assessor  │
│  ├── Parameter Limits      │                                │
│  ├── Layering Rules        │  ⚙️  Configuration System      │
│  ├── Global State Rules    │  ├── Architecture Rules       │
│  └── Security Patterns     │  ├── Quality Thresholds       │
├─────────────────────────────┼─────────────────────────────────┤
│  🚀 CI/CD Integration       │  📊 Health Dashboard           │
│  ├── GitHub Actions        │  ├── Metrics Visualization    │
│  ├── Pre-commit Hooks      │  ├── Trend Analysis          │
│  ├── Quality Gates         │  ├── Violation Reports       │
│  └── Automated Reporting   │  └── Recommendations         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Fitness Functions Test Suite
Located in `tests/architecture/test_fitness_functions.py`

**Circular Dependencies**
- Detects circular dependencies at module and package levels
- Uses NetworkX to build and analyze dependency graphs
- Provides clear violation reporting with dependency chains

**Connascence Violations**
- **Name Connascence**: Variable names used across contexts
- **Type Connascence**: Magic numbers and hardcoded values
- **Position Connascence**: Parameter order dependencies
- **Algorithm Connascence**: Duplicated complex logic

**Size and Complexity Limits**
- File size limits (default: 500 lines)
- Function complexity limits (default: 10 cyclomatic complexity)
- Parameter count limits (default: 3 parameters)

**Architectural Layering**
- Enforces dependency rules between layers
- Prevents lower layers from depending on higher layers
- Validates allowed/forbidden dependencies

**Global State Detection**
- Identifies global singletons and mutable state
- Enforces immutability principles

**Security Anti-patterns**
- Detects dangerous patterns (eval, exec, pickle.loads)
- Enforces secure coding practices

### 2. Architectural Analysis Engine
Located in `scripts/architectural_analysis.py`

**Dependency Analysis**
- Builds comprehensive dependency graphs
- Calculates coupling metrics (efferent, afferent, instability)
- Measures abstractness and distance from main sequence

**Connascence Detection**
- Automated detection of all connascence types
- Severity scoring based on locality and impact
- Hotspot identification across modules

**Architectural Drift Detection**
- Detects violations of architectural rules
- Monitors complexity and size drift
- Identifies technical debt accumulation

**Quality Metrics**
- Maintainability index calculation
- Technical debt ratio assessment
- Architectural health scoring

### 3. Configuration System
Located in `config/architecture_rules.yaml`

**Customizable Rules**
```yaml
# File and Module Constraints
max_file_lines: 500
max_function_complexity: 10
max_function_parameters: 3

# Coupling Thresholds
max_coupling_threshold: 0.3
min_cohesion_threshold: 0.7

# Layer Dependencies
allowed_dependencies:
  core: [common, legacy]
  agents: [core, common]
  rag: [core, common]

# Security Rules
forbidden_patterns:
  - "eval("
  - "exec("
  - "pickle.loads"
```

### 4. CI/CD Integration
Located in `scripts/ci_integration.py`

**GitHub Actions Workflow**
- Automated architecture validation on pull requests
- Quality gate enforcement
- Automated report generation and PR comments

**Pre-commit Hooks**
- Local validation before commits
- Fast feedback loop for developers

**Quality Gates**
- Configurable pass/fail criteria
- Integration with build pipelines
- Blocking deployments on critical violations

### 5. Health Dashboard
Located in `scripts/architecture_dashboard.py`

**Real-time Monitoring**
- Architecture health score
- Trend analysis over time
- Interactive visualizations

**Violation Management**
- Detailed violation reports
- Priority matrix for improvements
- Actionable recommendations

## Installation and Setup

### Quick Setup
```bash
# Install dependencies and setup tools
python scripts/setup_architecture_tools.py

# Or manually install requirements
pip install -r config/requirements-architecture.txt
```

### Manual Setup
```bash
# Create directories
mkdir -p tests/architecture config scripts reports/architecture reports/ci

# Install Python dependencies
pip install pytest networkx matplotlib seaborn pandas numpy radon pyyaml jinja2 streamlit plotly

# Install optional tools
pip install flake8 mypy bandit vulture coverage pytest-cov
```

## Usage

### Running Fitness Functions

**All Tests**
```bash
python -m pytest tests/architecture/test_fitness_functions.py -v
```

**Specific Test Categories**
```bash
# Circular dependencies only
python -m pytest tests/architecture/test_fitness_functions.py::test_no_circular_dependencies -v

# Connascence violations only
python -m pytest tests/architecture/test_fitness_functions.py::test_connascence_of_name_locality -v

# Size limits only
python -m pytest tests/architecture/test_fitness_functions.py::test_file_size_limits -v
```

### Running Architectural Analysis

**Full Analysis with Visualizations**
```bash
python scripts/architectural_analysis.py --output-dir reports/architecture --format both --visualizations
```

**JSON Report Only**
```bash
python scripts/architectural_analysis.py --format json
```

### Running Complete Suite

**Full Suite with Dashboard**
```bash
python scripts/run_architecture_suite.py --full --dashboard
```

**CI Mode (for automated environments)**
```bash
python scripts/run_architecture_suite.py --ci --fail-on-violations
```

### Starting Dashboard Only

```bash
# Using Streamlit directly
streamlit run scripts/architecture_dashboard.py --server.port 8501

# Using Python
python scripts/architecture_dashboard.py --port 8501
```

## CI/CD Integration

### GitHub Actions
```bash
# Generate workflow file
python scripts/ci_integration.py --mode github-actions
```

This creates `.github/workflows/architecture-qa.yml`:

```yaml
name: Architecture Quality Assurance
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  architecture-qa:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r config/requirements-architecture.txt
      - run: python scripts/ci_integration.py --mode ci
      - uses: actions/upload-artifact@v3
        with:
          name: architecture-reports
          path: reports/
```

### Pre-commit Hooks
```bash
# Generate pre-commit configuration
python scripts/ci_integration.py --mode pre-commit

# Install pre-commit
pip install pre-commit
pre-commit install
```

## Configuration

### Architecture Rules (`config/architecture_rules.yaml`)

**File and Module Constraints**
- `max_file_lines`: Maximum lines per file (default: 500)
- `max_function_complexity`: Maximum cyclomatic complexity (default: 10)
- `max_function_parameters`: Maximum parameters per function (default: 3)

**Coupling and Cohesion**
- `max_coupling_threshold`: Maximum acceptable instability (default: 0.3)
- `min_cohesion_threshold`: Minimum required cohesion (default: 0.7)

**Layer Dependencies**
```yaml
allowed_dependencies:
  core: [common, legacy]          # Core can depend on common and legacy
  agents: [core, common]          # Agents can depend on core and common
  rag: [core, common]             # RAG can depend on core and common
  p2p: [core, common]             # P2P can depend on core and common
  fog: [core, p2p, common]        # Fog can depend on core, P2P, and common
  edge: [core, p2p, fog, common]  # Edge is highest layer
```

**Security Rules**
```yaml
forbidden_patterns:
  - "eval("           # Dynamic code execution
  - "exec("           # Dynamic code execution
  - "pickle.loads"    # Insecure deserialization
  - "__import__"      # Dynamic imports
  - "globals("        # Global state access
```

### Quality Thresholds
```yaml
quality_thresholds:
  maintainability_index: 70     # Minimum maintainability score
  technical_debt_ratio: 5       # Maximum tech debt percentage
  halstead_difficulty: 15       # Maximum Halstead difficulty
```

## Connascence Principles

The system enforces Martin Fowler's connascence principles:

### Weak Forms (Acceptable across modules)
- **Connascence of Name**: Shared names/identifiers
- **Connascence of Type**: Shared data types
- **Connascence of Meaning**: Shared interpretation of values

### Strong Forms (Should be local only)
- **Connascence of Position**: Order dependency
- **Connascence of Algorithm**: Duplicate algorithms
- **Connascence of Execution Order**: Timing dependencies

### Locality Rules
- Same function: All forms acceptable
- Same class: Name and type acceptable
- Same module: Name only acceptable
- Different modules: Minimal name connascence only

## Quality Gates

The system enforces these quality gates:

1. **No Circular Dependencies**: Zero tolerance for circular imports
2. **Coupling Threshold**: Instability < 0.3 for critical modules
3. **No Critical Connascence**: No critical severity violations
4. **Technical Debt Acceptable**: < 10 high-risk debt items
5. **No Critical Drift**: No severe architectural violations

## Dashboard Features

### Overview Section
- Architecture health score (0-100)
- Quality gates status
- Key metrics summary
- Trend indicators

### Trends Analysis
- Historical metrics tracking
- Performance over time
- Regression detection
- Improvement trends

### Dependencies
- Dependency graph visualization
- Coupling metrics distribution
- Package relationships
- Circular dependency reports

### Violations
- Connascence violations by type and severity
- Technical debt by category and risk
- Detailed violation listings
- File-level impact analysis

### Recommendations
- Prioritized improvement suggestions
- Effort vs impact analysis
- Priority matrix visualization
- Actionable next steps

## Best Practices

### For Developers

**Before Committing**
```bash
# Run fitness functions locally
python -m pytest tests/architecture/ -x

# Run quick analysis
python scripts/architectural_analysis.py --format json
```

**Code Organization**
- Keep files under 500 lines
- Limit function complexity to 10
- Use maximum 3 parameters per function
- Follow layer dependency rules

**Avoiding Violations**
- Use dependency injection instead of global state
- Prefer composition over inheritance
- Minimize cross-module coupling
- Use explicit interfaces between layers

### For Teams

**Architecture Reviews**
- Include fitness function results in PR reviews
- Require clean architecture status for merges
- Regular architecture health meetings
- Trend analysis in retrospectives

**Continuous Improvement**
- Weekly dashboard reviews
- Monthly architecture debt reduction sprints
- Quarterly architecture rule updates
- Annual architecture fitness assessment

## Troubleshooting

### Common Issues

**ImportError: Missing Dependencies**
```bash
pip install -r config/requirements-architecture.txt
```

**No Reports Generated**
```bash
# Ensure directories exist
mkdir -p reports/architecture reports/ci

# Check permissions
ls -la reports/
```

**Fitness Functions Failing**
```bash
# Run with detailed output
python -m pytest tests/architecture/test_fitness_functions.py -v -s

# Check specific failures
python -m pytest tests/architecture/test_fitness_functions.py::test_no_circular_dependencies -v
```

**Dashboard Not Loading**
```bash
# Install Streamlit
pip install streamlit plotly

# Check if reports exist
ls -la reports/architecture/

# Run analysis first
python scripts/architectural_analysis.py --format json
```

### Performance Issues

**Large Codebase Analysis**
- Use `--fail-fast` for quicker feedback
- Run specific test categories only
- Consider excluding test directories
- Increase timeout limits for CI

**Memory Usage**
- Process files in batches
- Use generators for large file lists
- Clear NetworkX graphs after analysis
- Limit visualization complexity

## Extension Points

### Custom Fitness Functions
Add new tests to `test_fitness_functions.py`:

```python
def test_custom_architectural_rule(self):
    """Test custom architectural constraint"""
    violations = []
    # Implementation here
    if violations:
        raise ArchitecturalViolation(f"Found {len(violations)} violations")
```

### Custom Metrics
Extend `architectural_analysis.py`:

```python
def calculate_custom_metric(self, file_path: Path) -> float:
    """Calculate custom architectural metric"""
    # Implementation here
    return metric_value
```

### Custom Dashboard Sections
Add to `architecture_dashboard.py`:

```python
def create_custom_section(self):
    """Create custom dashboard section"""
    st.header("🎯 Custom Analysis")
    # Implementation here
```

## References

- [Connascence Theory by Martin Fowler](https://martinfowler.com/bliki/Connascence.html)
- [Architecture Decision Records](https://adr.github.io/)
- [Martin's Package Metrics](https://www.objectmentor.com/resources/articles/Principles_and_Patterns.pdf)
- [Clean Architecture by Robert Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

## License

This architectural fitness function system is part of the AIVillage project and follows the same licensing terms.

---

For questions, issues, or contributions, please refer to the main AIVillage documentation and contribution guidelines.
