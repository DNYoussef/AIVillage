# Quality Gates Configuration and CI/CD Integration

## Overview

This document describes the architectural quality gates system integrated into the AIVillage CI/CD pipeline. The quality gates prevent architectural regression by enforcing standards on every commit and pull request.

## Quality Gates Configuration

### 1. Coupling Score Gate
- **Threshold**: ≤ 12.0 per file
- **Description**: Maximum allowed coupling score across modules
- **Failure Impact**: Blocks merge if exceeded
- **Remediation**: Implement dependency injection, extract interfaces

### 2. Cyclomatic Complexity Gate
- **Threshold**: ≤ 15 per function (critical threshold)
- **Description**: Maximum cyclomatic complexity for any function
- **Failure Impact**: Blocks merge if exceeded
- **Remediation**: Break down complex functions using Extract Method

### 3. God Object Prevention
- **Threshold**: ≤ 500 lines per class
- **Description**: Prevents classes from becoming too large
- **Failure Impact**: Blocks merge if new God Objects introduced
- **Remediation**: Apply Single Responsibility Principle, extract classes

### 4. Magic Literal Density
- **Threshold**: ≤ 20% per file
- **Description**: Percentage of magic literals in conditionals and logic
- **Failure Impact**: Blocks merge if density exceeds threshold
- **Remediation**: Replace with named constants, enums, configuration

### 5. Connascence Violations
- **Threshold**: 0 cross-module violations
- **Description**: Strong connascence must stay local within classes/functions
- **Failure Impact**: Blocks merge for any cross-module violations
- **Remediation**: Refactor to weaker connascence forms

### 6. Anti-pattern Detection
- **Threshold**: 0 new anti-patterns
- **Description**: Prevents introduction of architectural anti-patterns
- **Failure Impact**: Blocks merge for new anti-patterns
- **Remediation**: Follow documented refactoring patterns

## CI/CD Integration Points

### GitHub Actions Workflow

The architectural quality workflow (`.github/workflows/architectural-quality.yml`) runs on:
- Every push to main/develop branches
- Every pull request to main/develop
- Daily scheduled runs for trend analysis

### Pre-commit Hooks

The following pre-commit hooks enforce quality at commit time:

```yaml
# Quick checks on every commit
- connascence-check (severity: high)
- anti-pattern-detection (severity: high)

# Push-time comprehensive checks
- architectural-fitness (on push)
- god-object-check (threshold: 500)
- magic-literal-check (threshold: 20%)
```

### Quality Report Generation

For every PR, the system generates:
1. **Comprehensive Quality Report** - Posted as PR comment
2. **Quality Metrics Dashboard** - Available in workflow artifacts
3. **Trend Analysis** - Compared against historical data
4. **Actionable Recommendations** - Specific refactoring guidance

## Quality Levels and Scoring

### Overall Quality Score (0-100)

The overall score is calculated using weighted metrics:

- **Coupling Score** (25%): `max(0, 100 - (coupling * 5))`
- **Cyclomatic Complexity** (20%): `max(0, 100 - (complexity * 3))`
- **God Objects** (15%): `100 if count == 0 else max(0, 100 - (count * 10))`
- **Magic Literal Density** (15%): `max(0, 100 - density)`
- **Connascence Violations** (15%): `100 if count == 0 else max(0, 100 - (count * 10))`
- **Anti-patterns** (10%): `100 if count == 0 else max(0, 100 - (count * 10))`

### Quality Levels

- **🟢 Excellent (90-100)**: Best practices followed, minimal technical debt
- **🔵 Good (80-89)**: Minor issues, generally well-structured
- **🟡 Acceptable (70-79)**: Some technical debt, needs attention
- **🟠 Poor (60-69)**: Significant issues, refactoring recommended
- **🔴 Critical (0-59)**: Major problems, immediate action required

## Developer Workflow

### 1. Local Development
```bash
# Run quality checks locally before committing
python scripts/architectural_analysis.py --pre-commit
python scripts/coupling_metrics.py --threshold 12.0
python scripts/ci/god-object-detector.py --threshold 500
```

### 2. Pre-commit Validation
```bash
# Automatic pre-commit hooks run on git commit
git commit -m "feat: implement new feature"
# Hooks validate: connascence, anti-patterns, basic quality
```

### 3. Pre-push Validation
```bash
# Comprehensive checks run on git push
git push origin feature-branch
# Hooks validate: full architectural fitness, God objects, magic literals
```

### 4. Pull Request Process
1. **Automated Quality Analysis** - Runs on PR creation
2. **Quality Report Posted** - Detailed metrics and recommendations
3. **Quality Gate Evaluation** - Pass/fail decision
4. **Manual Review** - If quality gate passes
5. **Merge Protection** - Only allowed if quality gate passes

## Quality Improvement Guide

### Reducing Coupling (25% weight)

**Bad Example:**
```python
# Direct import coupling
from external_service import DatabaseManager
class UserService:
    def __init__(self):
        self.db = DatabaseManager()  # Tight coupling
```

**Good Example:**
```python
# Dependency injection
from abc import ABC, abstractmethod

class DatabaseInterface(ABC):
    @abstractmethod
    def save_user(self, user): pass

class UserService:
    def __init__(self, database: DatabaseInterface):
        self.db = database  # Loose coupling
```

### Lowering Complexity (20% weight)

**Bad Example:**
```python
def process_user_data(user_data):
    if user_data:
        if user_data.get('email'):
            if '@' in user_data['email']:
                if user_data.get('age'):
                    if user_data['age'] > 18:
                        if user_data.get('country'):
                            # ... nested complexity continues
                            return True
    return False
```

**Good Example:**
```python
def process_user_data(user_data):
    if not user_data:
        return False

    if not _is_valid_email(user_data.get('email')):
        return False

    if not _is_adult(user_data.get('age')):
        return False

    return _has_valid_country(user_data.get('country'))

def _is_valid_email(email):
    return email and '@' in email

def _is_adult(age):
    return age and age > 18

def _has_valid_country(country):
    return bool(country)
```

### Eliminating God Objects (15% weight)

**Bad Example:**
```python
class UserManager:  # 500+ lines
    def create_user(self): pass
    def update_user(self): pass
    def delete_user(self): pass
    def send_email(self): pass
    def generate_report(self): pass
    def validate_permissions(self): pass
    def log_activity(self): pass
    # ... 50+ more methods
```

**Good Example:**
```python
class UserService:
    def __init__(self, email_service, reporting_service, audit_service):
        self.email_service = email_service
        self.reporting_service = reporting_service
        self.audit_service = audit_service

    def create_user(self): pass
    def update_user(self): pass
    def delete_user(self): pass

class EmailService:
    def send_notification(self): pass

class ReportingService:
    def generate_user_report(self): pass

class AuditService:
    def log_user_activity(self): pass
```

### Removing Magic Literals (15% weight)

**Bad Example:**
```python
def process_batch():
    if batch_size > 100:  # Magic number
        return "error"    # Magic string

    timeout = 30         # Magic number
    retry_count = 3      # Magic number
```

**Good Example:**
```python
class BatchConfig:
    MAX_BATCH_SIZE = 100
    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_RETRY_ATTEMPTS = 3

class BatchResult(Enum):
    ERROR = "error"
    SUCCESS = "success"

def process_batch():
    if batch_size > BatchConfig.MAX_BATCH_SIZE:
        return BatchResult.ERROR

    timeout = BatchConfig.DEFAULT_TIMEOUT_SECONDS
    retry_count = BatchConfig.MAX_RETRY_ATTEMPTS
```

### Fixing Connascence Issues (15% weight)

**Bad Example (Connascence of Position):**
```python
# Callers must remember parameter order
create_user("Alice", True, "US", 25)
```

**Good Example (Connascence of Name):**
```python
# Self-documenting with keyword arguments
create_user(name="Alice", email_verified=True, country="US", age=25)
```

### Addressing Anti-patterns (10% weight)

**Bad Example (Big Ball of Mud):**
```python
# Everything in one massive module with no clear structure
# functions.py (2000+ lines)
def user_stuff(): pass
def email_things(): pass
def database_operations(): pass
def random_utilities(): pass
```

**Good Example (Modular Design):**
```python
# Clear module boundaries
# domain/user.py
class User: pass

# services/user_service.py
class UserService: pass

# services/email_service.py
class EmailService: pass

# infrastructure/database.py
class UserRepository: pass
```

## Monitoring and Alerts

### Real-time Monitoring

The system provides continuous monitoring through:

1. **Quality History Database** - Tracks metrics over time
2. **Trend Analysis** - Identifies improving/declining patterns
3. **Regression Alerts** - Immediate notification of quality drops
4. **Dashboard Metrics** - Real-time architectural health visualization

### Alert Types

- **🚨 Critical Regression**: Quality score drops >10 points
- **⚠️ Threshold Violation**: Individual metric exceeds threshold
- **📈 Trend Alert**: Consistent quality decline over 5+ commits
- **🔄 Dependency Alert**: New circular dependencies detected

### Escalation Process

1. **Level 1**: Automated PR comment with recommendations
2. **Level 2**: Quality gate blocks merge, manual review required
3. **Level 3**: Architecture team notification for critical regressions
4. **Level 4**: Engineering manager escalation for persistent violations

## Tools and Scripts

### Core Quality Scripts

- `scripts/ci/quality-gate.py` - Main quality evaluation
- `scripts/ci/quality-report-generator.py` - PR report generation
- `scripts/ci/god-object-detector.py` - God object detection
- `scripts/ci/magic-literal-detector.py` - Magic literal analysis
- `scripts/ci/update-quality-history.py` - Historical tracking

### Development Tools

- `scripts/architectural_analysis.py` - Comprehensive analysis
- `scripts/coupling_metrics.py` - Coupling analysis
- `scripts/detect_anti_patterns.py` - Anti-pattern detection
- `scripts/check_connascence.py` - Connascence violation detection

### Dashboard and Reporting

- `scripts/architecture_dashboard.py` - Interactive Streamlit dashboard
- Quality history database tracking
- Automated trend analysis and reporting

## Configuration Management

### Architecture Rules File

Location: `config/architecture_rules.yaml`

```yaml
quality_thresholds:
  max_coupling_score: 12.0
  max_complexity: 15
  max_file_lines: 500
  max_magic_literal_density: 20.0

allowed_dependencies:
  domain: []
  application: ["domain"]
  infrastructure: ["domain", "application"]
  presentation: ["domain", "application"]

forbidden_patterns:
  - "import \\*"
  - "eval\\("
  - "exec\\("
```

### Pre-commit Configuration

Location: `.pre-commit-config.yaml`

Key hooks for architectural quality:
- `connascence-check`
- `coupling-metrics`
- `anti-pattern-detection`
- `architectural-fitness`
- `god-object-check`
- `magic-literal-check`

## Best Practices

### For Developers

1. **Run Local Checks First** - Before committing
2. **Understand Violations** - Read quality reports carefully
3. **Refactor Incrementally** - Small, focused improvements
4. **Follow Recommendations** - Use provided guidance
5. **Monitor Trends** - Track personal quality metrics

### For Team Leads

1. **Review Quality Reports** - In PR reviews
2. **Track Team Metrics** - Monitor overall trends
3. **Provide Training** - On architectural patterns
4. **Enforce Standards** - Don't bypass quality gates
5. **Celebrate Improvements** - Recognize quality work

### For Architects

1. **Define Clear Standards** - Update architecture rules
2. **Monitor System Health** - Review dashboard regularly
3. **Guide Refactoring** - Prioritize improvements
4. **Evolve Standards** - Update thresholds as needed
5. **Share Knowledge** - Educate on architectural principles

## Troubleshooting

### Common Issues

**Q: Quality gate failed but I don't understand why**
A: Check the detailed PR comment for specific violations and recommendations

**Q: Pre-commit hooks are too slow**
A: Most hooks run only on changed files; use `--no-verify` only in emergencies

**Q: False positive in God object detection**
A: Check if the class truly has single responsibility; consider extracting methods

**Q: Magic literal detector flagging valid constants**
A: Move constants to a dedicated configuration class or enum

**Q: Coupling score higher than expected**
A: Look for direct imports; implement interfaces and dependency injection

### Getting Help

1. **Check Documentation** - This guide and inline comments
2. **Review Examples** - Good/bad patterns in this document
3. **Use Dashboard** - Visual analysis of issues
4. **Ask Architecture Team** - For complex refactoring guidance
5. **Create GitHub Issue** - For tool bugs or improvements

## Continuous Improvement

The quality gates system evolves based on:

- **Team Feedback** - Adjusting thresholds and rules
- **Industry Best Practices** - Incorporating new patterns
- **Tool Updates** - Improving detection accuracy
- **Metrics Analysis** - Optimizing score calculations
- **Developer Experience** - Streamlining workflows

Regular reviews ensure the system remains effective while supporting developer productivity and architectural excellence.
