# Behaviors Discovered Through Testing

This document captures important system behaviors that were discovered or clarified during the comprehensive test infrastructure repair. These insights represent the "hidden API contracts" and implementation details that tests revealed.

## Error Handling System

### API Contract: ErrorCode Enum
**Discovery**: Tests revealed that ErrorCode enum must include specific values for proper error categorization.
**Location**: `core/error_handling.py`

The ErrorCode enum requires these exact values for the system to function correctly:
- `VALIDATION_ERROR` - Input validation failures
- `PROCESSING_ERROR` - General processing failures
- `DEPENDENCY_ERROR` - External dependency issues
- `CONFIGURATION_ERROR` - Configuration-related problems

**Testing Impact**: Without these exact enum values, error handling tests fail because the system expects these specific categorizations.

```python
# Required for tests to pass
class ErrorCode(Enum):
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"
```

### ErrorHandler Constructor Signature
**Discovery**: The ErrorHandler constructor signature was undocumented but tests revealed the expected interface.
**Location**: `core/error_handling.py:ErrorHandler.__init__`

**Previous assumption**: Constructor took no parameters
**Actual requirement**: Constructor expects optional ContextManager parameter

```python
# Discovered signature from test failures
def __init__(self, context_manager: Optional[ContextManager] = None):
    self.context_manager = context_manager
```

**Testing Impact**: Tests that didn't provide the correct constructor parameters failed during initialization.

## HypeRAG System

### Guardian Approval Bypass Behavior
**Discovery**: When `require_guardian_approval=False`, the system preserves the original status rather than resetting it.
**Location**: `mcp_servers/hyperag/lora/registry.py`

**Previous assumption**: Bypassing Guardian would always approve entries
**Actual behavior**: Original status is preserved to maintain test isolation

```python
# Discovered behavior in guardian bypass
if not require_guardian_approval:
    # Preserve original status if no Guardian (for testing)
    # Only set signed_at for approved adapters
    if entry.status == "approved":
        entry.signed_at = datetime.now(timezone.utc).isoformat()
```

**Testing Impact**: This behavior is critical for testing scenarios where Guardian is disabled but we need to test different approval states.

### LoRA Adapter Registry State Management
**Discovery**: The adapter registry maintains internal state that can leak between tests if not properly reset.
**Location**: `mcp_servers/hyperag/lora/registry.py`

**Discovered behavior**:
- Registry caches adapter entries across method calls
- Status changes are persistent within the same registry instance
- Tests must use fresh registry instances or explicit state reset

**Testing implications**: Test isolation requires careful management of registry instances.

## Time Handling System-Wide Changes

### Timezone-Aware DateTime Requirements
**Discovery**: The entire codebase was migrated from `datetime.utcnow()` to `datetime.now(timezone.utc)` during test repair.
**Affected files**: 20+ files across the codebase

**Previous pattern** (deprecated):
```python
timestamp = datetime.utcnow().isoformat()
```

**New required pattern**:
```python
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).isoformat()
```

**Testing Impact**: Tests failed with deprecation warnings until all datetime usage was modernized. This change affects:
- Error logging timestamps
- Registry entry creation times
- Audit trail generation
- Test execution timestamps

## Dependency Stubbing Patterns

### Optional Dependency Handling
**Discovery**: The test system uses sophisticated dependency stubbing rather than skipping tests.
**Location**: `conftest.py`

**Key discoveries**:
1. Heavy dependencies (torch, sklearn, faiss) are stubbed with minimal interfaces
2. Stubs provide just enough functionality for tests to run
3. Tests can run in environments missing production dependencies

### Successful Stubbing Patterns

**Torch/PyTorch stubbing**:
```python
# Minimal interface that allows tests to pass
torch_mod = _ensure_module("torch", {
    "Tensor": object,
    "randn": lambda *a, **k: 0,
})
# CUDA availability stub
cuda_mod.is_available = lambda: False
```

**Scikit-learn stubbing**:
```python
# Function stubs that return expected types
_ensure_module("sklearn.model_selection", {
    "train_test_split": lambda *args, **kwargs: ([], [], [], [])
})
```

### Numba JIT Decorator Behavior
**Discovery**: The numba `jit` decorator can be stubbed to return functions unchanged.
**Implementation**:

```python
def jit(*args, **kwargs):
    """Stub jit decorator that returns the function unchanged."""
    if len(args) == 1 and callable(args[0]):
        return args[0]  # Function was passed directly
    return lambda func: func  # Decorator with arguments
```

**Testing Impact**: This allows code using `@jit` decorators to run in test environments without numba installed.

## Performance Characteristics Discovered

### Test Suite Performance Profile
**Discovered during monitoring implementation**:

- **Average test duration**: ~0.1-0.3s per test
- **Slowest modules**: Compression tests (due to tensor operations)
- **Fastest modules**: Core utility tests
- **Total suite time**: ~30-60s for full run

### Memory Usage Patterns
**Discovery**: Some tests have significant memory requirements that affect CI environments.

**Memory-intensive components**:
- HypeRAG retrieval tests (large vector operations)
- Compression pipeline tests (tensor allocations)
- EvoMerge tests (when they can run - model loading)

## Configuration System Behaviors

### Environment Variable Precedence
**Discovery**: Configuration loading follows specific precedence rules that tests must respect.

**Precedence order** (highest to lowest):
1. Explicit test configuration
2. Environment variables
3. Configuration files
4. Default values

**Testing implications**: Tests must either mock environment variables or use isolated configuration contexts.

### Default Configuration Assumptions
**Discovery**: Many components assume default configurations exist and are valid.

**Components with default config dependencies**:
- Alert manager (expects alert_config.yaml)
- Monitoring system (expects monitoring directory)
- Guardian system (expects policies.yaml)

## Database/Storage Behaviors

### SQLite Connection Handling
**Discovery**: Tests revealed that SQLite connections must be explicitly closed to prevent resource leaks.

**Required pattern**:
```python
# Always use context managers for DB connections
with sqlite3.connect(db_path) as conn:
    # Database operations
    pass  # Connection auto-closed
```

### File System State Persistence
**Discovery**: Some components persist state to the filesystem that affects test isolation.

**Persistent state locations**:
- `monitoring/test_history.json` - Test monitoring history
- `monitoring/alerts.log` - Alert history
- Various cache directories

**Testing requirement**: Tests must clean up persistent state or use temporary directories.

## Integration Points Discovered

### Internal API Dependencies
**Discovery**: Components have undocumented internal API dependencies that tests revealed.

**Key dependency chains**:
1. **Monitoring → Test Results**: Monitoring system expects specific JSON report format
2. **AlertManager → Monitoring**: Alert system depends on monitoring data structures
3. **Registry → Guardian**: LoRA registry has optional Guardian integration
4. **Configuration → Environment**: All systems depend on consistent environment setup

### Event System Behaviors
**Discovery**: Some components use implicit event/callback systems that tests must account for.

**Callback registration patterns**:
- Test monitoring automatically triggers on pytest completion
- Alert system hooks into monitoring events
- Dashboard updates trigger on test result changes

## Testing Best Practices Derived

### Patterns That Work Well

1. **Dependency Stubbing**: Prefer stubs over test skipping
2. **State Isolation**: Use fresh instances or explicit state reset
3. **Async Context Management**: Proper async/await patterns for monitoring
4. **Configuration Isolation**: Test-specific configuration contexts
5. **Resource Cleanup**: Explicit cleanup of files, connections, state

### Patterns That Cause Problems

1. **Shared State**: Global variables that persist between tests
2. **Real External Dependencies**: Network calls, real databases
3. **Hardcoded Paths**: Absolute paths that don't work in CI
4. **Time-Dependent Logic**: Code that depends on specific timestamps
5. **Missing Error Handling**: Components that fail hard instead of gracefully

## Architecture Insights

### Modular Design Validation
**Discovery**: The test repair process validated that the architecture is reasonably modular.

**Evidence of good modularity**:
- Components can be tested in isolation
- Dependencies can be stubbed effectively
- Configuration is externalized
- Core functionality is separable from integrations

### Areas Needing Architecture Improvements
**Discovery**: Tests revealed architectural debt that should be addressed.

**Improvement opportunities**:
1. **Dependency Injection**: Some components have hard-coded dependencies
2. **Interface Segregation**: Some interfaces are too broad for testing
3. **Configuration Management**: Inconsistent configuration patterns
4. **Error Handling**: Inconsistent error handling across components

---

## Using This Document

### For Developers
- Review this document when working on components mentioned
- Use the discovered patterns as implementation guidance
- Reference the testing implications when writing new tests

### For Test Authors
- Follow the successful patterns documented here
- Avoid the problematic patterns identified
- Use the stubbing examples as templates

### For Architecture Decisions
- Consider the modularity insights when planning changes
- Address the improvement opportunities in future development
- Maintain the successful architectural patterns

---

*Last Updated: 2025-01-23*
*Next Review: 2025-04-23*

**Contributing**: When you discover new behaviors during testing, add them to this document following the established format. Include the location, discovery context, and testing implications.
