# Connascence Linting Tools Guide

This guide covers the comprehensive connascence-based linting tools implemented for the AIVillage project. These tools help identify and reduce coupling in your codebase by detecting various forms of connascence as defined by Meilir Page-Jones.

## Overview

Connascence is a way of categorizing the different types of coupling that can exist in software. Understanding and managing connascence helps create more maintainable, flexible, and robust code.

### Forms of Connascence

**Static Connascence** (detected at compile/parse time):
- **Connascence of Name (CoN)**: Multiple components must agree on a name
- **Connascence of Type (CoT)**: Multiple components must agree on a type
- **Connascence of Meaning (CoM)**: Multiple components must agree on the meaning of values
- **Connascence of Position (CoP)**: Multiple components must agree on the order of values
- **Connascence of Algorithm (CoA)**: Multiple components must agree on an algorithm

**Dynamic Connascence** (detected at runtime):
- **Connascence of Execution (CoE)**: Order of execution matters
- **Connascence of Timing (CoTi)**: Timing of execution matters
- **Connascence of Value (CoV)**: Multiple components must agree on values
- **Connascence of Identity (CoI)**: Multiple components must reference the same entity

## Tools Overview

### 1. Connascence Violation Detector (`check_connascence.py`)

Detects specific connascence violations and provides actionable recommendations.

#### Usage

```bash
# Analyze current directory
python scripts/check_connascence.py .

# Analyze specific directory with JSON output
python scripts/check_connascence.py src/ --format json

# Only show high and critical severity issues
python scripts/check_connascence.py . --severity high

# Save report to file
python scripts/check_connascence.py . --output connascence_report.txt

# Verbose output with timing information
python scripts/check_connascence.py . --verbose
```

#### What It Detects

- **Positional Parameter Violations**: Functions with >3 positional parameters
- **Magic Literals**: Hardcoded numbers and strings that should be constants
- **God Objects**: Classes with >20 methods or >500 lines
- **Algorithm Duplication**: Similar function implementations
- **Global Variable Usage**: Excessive use of global state
- **Timing Dependencies**: Sleep-based synchronization

#### Example Output

```
CONNASCENCE ANALYSIS REPORT
============================

Total violations: 45
Files analyzed: 23

Severity breakdown:
  Critical:   3
  High:      15
  Medium:    20
  Low:        7

Violation types:
  connascence_of_meaning    : 20
  connascence_of_position   : 12
  god_object               :  3
  connascence_of_algorithm :  5
  connascence_of_timing    :  3
  connascence_of_identity  :  2
```

### 2. Coupling Metrics Analyzer (`coupling_metrics.py`)

Provides quantitative metrics for tracking coupling improvements over time.

#### Usage

```bash
# Analyze current directory
python scripts/coupling_metrics.py .

# Save baseline for future comparisons
python scripts/coupling_metrics.py . --save-baseline

# Compare with baseline
python scripts/coupling_metrics.py . --baseline coupling_baseline.json

# JSON output for CI integration
python scripts/coupling_metrics.py . --format json
```

#### Metrics Tracked

- **Positional Parameter Ratio**: Percentage of functions with >3 positional parameters
- **Magic Literal Density**: Count of magic numbers/strings per 100 lines of code
- **God Class Ratio**: Percentage of classes that are "God Objects"
- **Global Usage Count**: Number of global variable references
- **Coupling Score**: Overall coupling score (0-100, lower is better)
- **Maintainability Index**: Microsoft-style maintainability score

#### Example Output

```
COUPLING METRICS REPORT
=======================

Total files analyzed: 45
Total lines of code: 12,345
Average coupling score: 23.4/100 (lower is better)

CONNASCENCE METRICS
-------------------
Positional parameter violations: 23
Positional parameter ratio: 15.2%
Magic literal count: 145
Magic literal density: 1.17 per 100 LOC
God classes detected: 3
God class ratio: 6.7%
Global usage instances: 8

MOST COUPLED FILES
------------------
  45.2: src/legacy/monolith.py
  38.7: src/utils/helpers.py
  32.1: src/core/processor.py
```

### 3. Anti-Pattern Detector (`detect_anti_patterns.py`)

Identifies common anti-patterns that indicate poor software design.

#### Usage

```bash
# Analyze current directory
python scripts/detect_anti_patterns.py .

# Filter by severity
python scripts/detect_anti_patterns.py . --severity high

# Filter by specific anti-pattern
python scripts/detect_anti_patterns.py . --pattern god_object

# JSON output
python scripts/detect_anti_patterns.py . --format json
```

#### Anti-Patterns Detected

- **God Object**: Classes with too many responsibilities
- **God Method**: Methods that are too complex or long
- **Copy-Paste Programming**: Duplicate code blocks
- **Database-as-IPC**: Using databases for inter-process communication
- **Sequential Coupling**: APIs requiring specific call sequences
- **Long Parameter Lists**: Functions with too many parameters
- **Magic Number/String Abuse**: Excessive use of magic literals
- **Feature Envy**: Methods using other objects more than their own
- **Data Class**: Classes that are just data containers with business logic

#### Example Output

```
ANTI-PATTERN DETECTION REPORT
==============================

Total anti-patterns detected: 28
Files analyzed: 45

Severity breakdown:
  Critical:  3
  High:     12
  Medium:   10
  Low:       3

Anti-pattern types detected:
  god_object              : 3
  god_method              : 8
  copy_paste_programming  : 5
  long_parameter_list     : 7
  magic_number_abuse      : 5

REFACTORING PRIORITIES
----------------------
• CRITICAL: Break down God Objects using Extract Class refactoring (3 instances)
• HIGH: Split complex methods into smaller, focused functions (8 instances)
• HIGH: Extract duplicate code into shared utilities (5 instances)
```

## Integration with Development Workflow

### Pre-commit Hooks

The tools are integrated with pre-commit hooks for automatic checking:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run connascence-check --all-files
pre-commit run anti-pattern-detection --all-files

# Run coupling metrics manually (not in automatic hooks)
pre-commit run coupling-metrics --hook-stage manual
```

### CI/CD Integration

#### GitHub Actions Example

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  connascence-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Check connascence violations
        run: python scripts/check_connascence.py . --severity high --format json --output connascence.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: connascence-report
          path: connascence.json

      - name: Check coupling metrics
        run: |
          python scripts/coupling_metrics.py . --format json --output coupling.json
          # Fail if coupling score is too high
          python -c "
          import json
          with open('coupling.json') as f:
              data = json.load(f)
          if data['average_coupling_score'] > 50:
              exit(1)
          "
```

### Ruff Integration

The enhanced ruff configuration includes connascence-aware rules:

```toml
[tool.ruff.lint]
select = [
    # Connascence of Name (CoN)
    "N",     # pep8-naming
    "I",     # isort

    # Connascence of Type (CoT)
    "ANN",   # flake8-annotations

    # Connascence of Meaning (CoM)
    "PLR2004", # Magic value used in comparison

    # Connascence of Position (CoP)
    "ARG",   # flake8-unused-arguments
    "PLR0913", # Too many arguments

    # Connascence of Algorithm (CoA)
    "C90",   # mccabe
    "PLR0912", # Too many branches
    "SIM",   # flake8-simplify

    # And more...
]
```

## Best Practices and Recommendations

### Reducing Connascence

1. **Prefer Weak Forms Over Strong Forms**
   - Static connascence is weaker than dynamic
   - Name connascence is weaker than position connascence
   - Local connascence is weaker than distant connascence

2. **Specific Refactoring Strategies**

   **For Connascence of Position:**
   ```python
   # Bad
   def create_user(name, email, age, address, phone):
       pass

   # Good
   @dataclass
   class UserInfo:
       name: str
       email: str
       age: int
       address: str
       phone: str

   def create_user(user_info: UserInfo):
       pass
   ```

   **For Connascence of Meaning:**
   ```python
   # Bad
   if status == 1:  # Magic number
       process_active()
   elif status == 2:
       process_inactive()

   # Good
   class Status(Enum):
       ACTIVE = 1
       INACTIVE = 2

   if status == Status.ACTIVE:
       process_active()
   elif status == Status.INACTIVE:
       process_inactive()
   ```

   **For God Objects:**
   ```python
   # Bad
   class UserManager:
       def create_user(self): pass
       def delete_user(self): pass
       def send_email(self): pass
       def validate_password(self): pass
       def generate_report(self): pass
       # ... 20+ more methods

   # Good
   class UserRepository:
       def create_user(self): pass
       def delete_user(self): pass

   class EmailService:
       def send_email(self): pass

   class PasswordValidator:
       def validate_password(self): pass

   class ReportGenerator:
       def generate_report(self): pass
   ```

### Monitoring and Improvement

1. **Set Thresholds**
   - Coupling score should be < 30
   - Positional parameter ratio should be < 20%
   - Magic literal density should be < 2 per 100 LOC
   - God class ratio should be < 5%

2. **Track Progress**
   - Use baseline comparisons to track improvements
   - Set up alerts for regression in coupling metrics
   - Review worst-coupled files regularly

3. **Team Guidelines**
   - Review connascence reports in code reviews
   - Address critical and high-severity violations first
   - Refactor in small, focused commits

## Troubleshooting

### Common Issues

1. **Unicode Encoding Errors (Windows)**
   - The tools handle encoding automatically
   - For CI, ensure UTF-8 encoding: `PYTHONIOENCODING=utf-8`

2. **False Positives**
   - Magic number detection may flag acceptable constants
   - Adjust thresholds or add exceptions as needed
   - Some patterns may be necessary for compatibility

3. **Performance on Large Codebases**
   - Use directory filtering to focus on important areas
   - Run on specific modules rather than entire codebase
   - Consider parallel processing for very large projects

### Configuration

All tools support exclusion patterns:

```bash
# Exclude test files and deprecated code
python scripts/check_connascence.py . --exclude "tests/*" --exclude "deprecated/*"
```

### Extending the Tools

The tools are designed to be extensible. You can:

1. Add new connascence detectors by extending the AST visitors
2. Add new anti-pattern detectors by implementing pattern-specific logic
3. Customize metrics calculation for project-specific needs

## Integration with AIVillage Architecture

These tools are specifically tuned for the AIVillage project architecture:

- **Agent-based Architecture**: Detects coupling between agents
- **Microservices Patterns**: Identifies service boundary violations
- **Event-driven Architecture**: Detects timing and execution dependencies
- **Multi-layer Architecture**: Applies different rules to different layers

## Conclusion

Regular use of these connascence-based linting tools will help maintain a high-quality, maintainable codebase. Focus on addressing critical and high-severity violations first, and use the metrics to track improvements over time.

For questions or suggestions about these tools, please refer to the project documentation or raise an issue in the repository.
