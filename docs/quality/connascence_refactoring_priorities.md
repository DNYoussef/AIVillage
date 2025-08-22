# AIVillage Connascence Analysis - Refactoring Priorities

## Code Quality Analysis Report

### Summary
- **Overall Quality Score**: 6.5/10 (Good foundation, significant debt)
- **Files Analyzed**: 564
- **Issues Found**: 44,561 total violations
- **Technical Debt Estimate**: 1,200-1,500 hours

### Critical Issues

#### 1. God Objects (90 instances) - CRITICAL PRIORITY

**Most Severe Cases:**
- `ArchitecturalAnalyzer` (scripts/architectural_analysis.py:95) - 35 methods, ~982 lines
- `BaseAgentTemplate` (packages/agents/core/base_agent_template.py:233) - 5 methods, ~845 lines
- `MultiTenantSystem` (packages/core/security/multi_tenant_system.py:134) - 20 methods, ~1066 lines

**Refactoring Strategy:**
```python
# Current God Object Pattern
class MultiTenantSystem:  # 1066 lines, 20 methods
    def create_tenant(self): ...
    def manage_quotas(self): ...
    def handle_isolation(self): ...
    def manage_encryption(self): ...
    # ... 16 more methods

# Recommended Decomposition
class TenantCreationService:
    def create_tenant(self): ...
    def validate_tenant_data(self): ...

class QuotaManagementService:
    def manage_quotas(self): ...
    def check_quota_limits(self): ...

class IsolationManager:
    def handle_isolation(self): ...
    def enforce_boundaries(self): ...

class TenantEncryptionService:
    def manage_encryption(self): ...
    def rotate_keys(self): ...

class MultiTenantSystem:  # Coordinator only
    def __init__(self):
        self.creation = TenantCreationService()
        self.quotas = QuotaManagementService()
        self.isolation = IsolationManager()
        self.encryption = TenantEncryptionService()
```

#### 2. Magic Literals (37,060 instances) - HIGH PRIORITY

**Security Module Example** (packages/core/security/):
```python
# Current problematic patterns
if status_code == 200:  # Magic number
    print("✓ ALLOWED")  # Magic string
if user.role == 2:  # Magic number meaning
    return {"error": "Access denied"}  # Magic string

# Recommended constants module
class SecurityConstants:
    class HTTPStatus:
        OK = 200
        CREATED = 201
        FORBIDDEN = 403
        UNAUTHORIZED = 401

    class UserRoles:
        GUEST = 1
        USER = 2
        ADMIN = 3
        SUPER_ADMIN = 4

    class Messages:
        ACCESS_ALLOWED = "✓ ALLOWED"
        ACCESS_DENIED = "✗ DENIED"
        ACCESS_FORBIDDEN = "Access denied"

# Usage
if status_code == SecurityConstants.HTTPStatus.OK:
    print(SecurityConstants.Messages.ACCESS_ALLOWED)
```

#### 3. Positional Parameter Violations (441 instances) - HIGH PRIORITY

**Current Pattern:**
```python
def _ref(key, value, input, known, output):  # 5 positional params
def create_user(name, email, role, tenant, active):  # 5 positional params
```

**Recommended Patterns:**
```python
# Option 1: Data Classes
@dataclass
class UserCreationRequest:
    name: str
    email: str
    role: str
    tenant_id: str
    is_active: bool = True

def create_user(request: UserCreationRequest) -> UserCreationResponse:
    # Implementation

# Option 2: Keyword-only parameters
def create_user(
    *,
    name: str,
    email: str,
    role: str,
    tenant_id: str,
    is_active: bool = True
) -> UserCreationResponse:
    # Implementation
```

### Code Smells

#### Feature Envy (1,114 instances)
**Problem:** Methods using other objects more than their own
```python
# Current problematic pattern
class ReportGenerator:
    def generate_metrics_report(self, analyzer):
        data = analyzer.get_coupling_metrics()
        trends = analyzer.get_trend_data()
        violations = analyzer.get_violations()
        summary = analyzer.get_summary()
        # Method heavily depends on analyzer

# Recommended pattern
class MetricsReportService:
    def __init__(self, metrics_provider: MetricsProvider):
        self.provider = metrics_provider

    def generate_report(self) -> Report:
        # Use injected dependency, not external object
```

#### Database-as-IPC (718 instances)
**Problem:** Using database operations for inter-process communication
```python
# Current anti-pattern
def send_agent_message(self, target_agent, message):
    self.db.execute(
        "INSERT INTO agent_messages (target, message, status) VALUES (?, ?, 'pending')",
        (target_agent, message)
    )

# Recommended patterns
class AgentMessageBus:
    async def send_message(self, target: str, message: Message):
        await self.message_queue.publish(f"agent.{target}", message)

class AgentEventSystem:
    def emit_event(self, event: AgentEvent):
        for handler in self.event_handlers:
            handler.handle(event)
```

#### Copy-Paste Programming (286 instances)
**Problem:** Duplicate code blocks across files
```python
# Extract common patterns like:
def validate_input_data(data, schema):
    """Common validation logic used in 15+ places"""
    if not isinstance(data, dict):
        raise ValueError("Data must be dictionary")
    # ... validation logic

def calculate_hash(file_path, chunk_size=8192):
    """Common file hashing used in 8+ places"""
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()
```

### Refactoring Opportunities

#### 1. Security Module Cleanup (CRITICAL)
**File:** `packages/core/security/multi_tenant_system.py`
**Issues:** 158 violations, mostly magic literals
**Priority:** Critical (security-sensitive code)

**Immediate Actions:**
1. Extract all magic HTTP status codes to constants
2. Replace hardcoded tenant types with enums
3. Extract configuration values to config files
4. Split God Object into focused services

#### 2. Agent Template Refactoring (HIGH)
**File:** `packages/agents/core/base_agent_template.py`
**Issues:** 845 lines, single responsibility violation
**Priority:** High (affects all 23 agents)

**Immediate Actions:**
1. Extract memory management to separate service
2. Extract communication handling to dedicated class
3. Extract reflection/journaling to separate component
4. Create focused agent base with composition

#### 3. Legacy Code Modernization (MEDIUM)
**Directory:** `packages/core/legacy/`
**Issues:** 12 God Objects, 323 high-priority violations
**Priority:** Medium (isolated impact)

**Immediate Actions:**
1. Prioritize most-used legacy components
2. Extract common utilities first
3. Modernize configuration management
4. Replace embedded SQL with ORM

### Positive Findings

1. **Overall Architecture**: Sound modular structure
2. **Type Hints**: Good usage of modern Python typing
3. **Documentation**: Comprehensive docstrings
4. **Testing Structure**: Well-organized test directories
5. **Dependency Management**: Clean package structure

### Recommended Development Workflow

#### Phase 1: Foundation (Weeks 1-2)
1. **Create Constants Module**
   ```python
   # packages/core/constants/__init__.py
   from .http_constants import HTTPStatus
   from .security_constants import SecurityRoles, SecurityMessages
   from .system_constants import SystemLimits, ConfigDefaults
   ```

2. **Extract Common Utilities**
   ```python
   # packages/core/utils/validation.py
   # packages/core/utils/hashing.py
   # packages/core/utils/file_operations.py
   ```

3. **Implement Quality Gates**
   ```yaml
   # .pre-commit-config.yaml
   - id: connascence-checker
     args: ['--severity', 'high', '--max-violations', '10']
   ```

#### Phase 2: Critical Refactoring (Weeks 3-6)
1. Refactor top 5 God Objects
2. Replace magic literals in security modules
3. Fix positional parameter violations in public APIs
4. Extract duplicate algorithms

#### Phase 3: Systematic Cleanup (Weeks 7-12)
1. Replace database-as-IPC patterns
2. Implement proper messaging systems
3. Modernize legacy components
4. Add architectural fitness functions

### CI/CD Integration

#### Quality Gates
```python
# .github/workflows/quality-check.yml
- name: Connascence Analysis
  run: |
    python scripts/check_connascence.py . --severity high
    if [ $? -eq 1 ]; then
      echo "❌ Critical connascence violations found"
      exit 1
    fi

- name: Anti-pattern Detection
  run: |
    python scripts/detect_anti_patterns.py . --pattern god_object
    if [ $? -eq 1 ]; then
      echo "❌ New God Objects detected"
      exit 1
    fi
```

#### Metrics Tracking
```python
# Daily metrics collection
python scripts/coupling_metrics.py . --save-baseline --format json
# Upload to metrics dashboard
```

### Expected Outcomes

#### After Phase 1 (2 weeks):
- 90% reduction in magic literal violations
- Established quality gates
- Common utilities extracted

#### After Phase 2 (6 weeks):
- 80% reduction in God Objects
- API usability improvements
- Security module hardening

#### After Phase 3 (12 weeks):
- Overall coupling score < 20/100
- God class ratio < 3%
- Architectural fitness functions active

### Success Metrics

1. **Connascence Metrics**
   - Magic literal density: < 20 per 100 LOC (currently 38.16)
   - Positional parameter ratio: < 10% (currently 10.7%)
   - God class ratio: < 3% (currently 5.7%)

2. **Coupling Metrics**
   - Average coupling score: < 20/100 (currently 15.9)
   - No files > 35 coupling score (currently 5 files)

3. **Quality Gates**
   - Zero critical connascence violations
   - Zero new God Objects
   - All security modules under coupling threshold

The codebase shows good architectural foundation with systematic quality issues that can be resolved through focused refactoring efforts. Priority should be on security modules and agent templates due to their critical role and wide impact.
