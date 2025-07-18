# Error Handling Standardization Migration Plan

## Current State Analysis

We have identified **4 different AIVillageException implementations** across the codebase:

### 1. Root `exceptions.py` (Most Comprehensive)
- **Location**: `exceptions.py:8-14`
- **Features**: Context support, structured logging
- **Usage**: Main project exceptions

### 2. `agents/utils/exceptions.py` (Basic)
- **Location**: `agents/utils/exceptions.py:1-5`
- **Features**: Simple message-only implementation
- **Usage**: Agent-specific errors

### 3. `communications/protocol.py` (Fallback)
- **Location**: `communications/protocol.py:24-26`
- **Features**: Minimal fallback implementation
- **Usage**: Communication layer errors

### 4. `rag_system/error_handling/error_handler.py` (Handler-focused)
- **Location**: `rag_system/error_handling/error_handler.py:6-8`
- **Features**: Basic exception class
- **Usage**: RAG system errors

## Unified Error Handling System Design

### Core Components

#### 1. Exception Hierarchy (`core/error_handling.py`)
```python
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from datetime import datetime
import logging

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    SYSTEM = "system"
    NETWORK = "network"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SECURITY = "security"

class AIVillageException(Exception):
    """Unified exception class for AI Village with comprehensive error context."""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None,
        parent_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_id = error_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.component = component
        self.operation = operation
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.parent_error = parent_error

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_id": self.error_id,
            "message": str(self),
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "parent_error": str(self.parent_error) if self.parent_error else None
        }
```

#### 2. Error Handling Decorator
```python
def with_error_handling(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    attempt_recovery: bool = False,
    max_retries: int = 3
):
    """Decorator for standardized error handling across components."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AIVillageException as e:
                # Already our exception, just log and re-raise
                logger.error(f"AIVillageException in {component}.{operation}", extra=e.to_dict())
                raise
            except Exception as e:
                # Convert to AIVillageException
                error = AIVillageException(
                    message=str(e),
                    component=component,
                    operation=operation,
                    severity=severity,
                    category=category,
                    parent_error=e
                )
                logger.error(f"Exception in {component}.{operation}", extra=error.to_dict())

                if attempt_recovery and max_retries > 0:
                    # Implement retry logic
                    pass

                raise error

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AIVillageException as e:
                logger.error(f"AIVillageException in {component}.{operation}", extra=e.to_dict())
                raise
            except Exception as e:
                error = AIVillageException(
                    message=str(e),
                    component=component,
                    operation=operation,
                    severity=severity,
                    category=category,
                    parent_error=e
                )
                logger.error(f"Exception in {component}.{operation}", extra=error.to_dict())
                raise error

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

#### 3. Component-specific Loggers
```python
def get_component_logger(component: str) -> logging.Logger:
    """Get a logger configured for a specific component."""
    logger = logging.getLogger(f"aivillage.{component}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

## Migration Steps

### Phase 1: Foundation (Days 1-3)

#### Step 1.1: Create Unified Exception System
- [ ] Create `core/error_handling.py` with unified AIVillageException
- [ ] Implement error handling decorators
- [ ] Add component-specific logging utilities

#### Step 1.2: Remove Duplicate Implementations
- [ ] Remove `agents/utils/exceptions.py`
- [ ] Remove fallback in `communications/protocol.py:24-26`
- [ ] Remove `rag_system/error_handling/error_handler.py:6-8`

#### Step 1.3: Update Import Statements
Create migration script to update all imports:

```bash
#!/bin/bash
# migrate_error_imports.sh

# Update agents/utils/exceptions.py imports
find . -name "*.py" -exec sed -i 's/from agents.utils.exceptions import AIVillageException/from core.error_handling import AIVillageException/g' {} \;

# Update communications/protocol.py imports
find . -name "*.py" -exec sed -i 's/from communications.protocol import AIVillageException/from core.error_handling import AIVillageException/g' {} \;

# Update rag_system/error_handling imports
find . -name "*.py" -exec sed -i 's/from rag_system.error_handling.error_handler import AIVillageException/from core.error_handling import AIVillageException/g' {} \;
```

### Phase 2: Agent Integration (Days 4-7)

#### Step 2.1: Update Agent Classes
Update all agent classes to use the new error handling system:

```python
# Example: agents/king/king_agent.py
from core.error_handling import (
    with_error_handling,
    AIVillageException,
    ErrorSeverity,
    ErrorCategory,
    get_component_logger
)

class KingAgent:
    def __init__(self):
        self.logger = get_component_logger("Agent.King")

    @with_error_handling(
        component="Agent.King",
        operation="process_task",
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.BUSINESS_LOGIC,
        attempt_recovery=True
    )
    async def process_task(self, task):
        # Implementation
        pass
```

#### Step 2.2: Update Error Contexts
Replace generic exceptions with contextualized ones:

```python
# OLD
raise AIVillageException("Task failed")

# NEW
raise AIVillageException(
    "Task processing failed",
    component="Agent.King",
    operation="process_task",
    severity=ErrorSeverity.HIGH,
    category=ErrorCategory.BUSINESS_LOGIC,
    context={"task_id": task.id, "agent_id": self.id}
)
```

### Phase 3: Service Integration (Days 8-10)

#### Step 3.1: Update Service Error Handlers
Update all service-level error handling:

```python
# services/gateway/app.py
from core.error_handling import get_component_logger, AIVillageException

logger = get_component_logger("Service.Gateway")

@app.exception_handler(AIVillageException)
async def aivillage_exception_handler(request, exc):
    logger.error(f"Gateway error: {exc}", extra=exc.to_dict())
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "error_id": exc.error_id,
            "timestamp": exc.timestamp.isoformat()
        }
    )
```

## Validation Checklist

### Before Migration
- [ ] All 4 AIVillageException implementations identified
- [ ] 129 try/except blocks catalogued
- [ ] Import dependencies mapped

### During Migration
- [ ] No breaking changes to existing API
- [ ] Backward compatibility maintained
- [ ] All tests pass

### After Migration
- [ ] Single AIVillageException implementation
- [ ] All 129 try/except blocks use standardized patterns
- [ ] Error recovery mechanisms in place
- [ ] Centralized error logging active
- [ ] Zero duplicate exception classes

## Success Metrics

### Quantitative
- **Exception Classes**: 4 → 1 (75% reduction)
- **Import Statements**: Standardized across all modules
- **Error Context**: 100% of exceptions include component/operation context
- **Logging Coverage**: 100% of exceptions logged with structured data

### Qualitative
- **Debugging**: Easier error tracking with error_id and context
- **Monitoring**: Centralized error monitoring and alerting
- **Recovery**: Standardized retry and recovery mechanisms
- **Documentation**: Comprehensive error handling documentation

## Risk Mitigation

### Risk 1: Breaking Changes
- **Mitigation**: Gradual migration with backward compatibility layer
- **Timeline**: 3-day grace period for dependent systems

### Risk 2: Performance Impact
- **Mitigation**: Lazy logging initialization, async-safe decorators
- **Testing**: Performance regression tests before deployment

### Risk 3: Missing Error Context
- **Mitigation**: Comprehensive audit of all exception raising sites
- **Validation**: Automated checks for required context fields

## Next Steps

1. **Switch to Code Mode** to implement the unified error handling system
2. **Create core/error_handling.py** with the new exception hierarchy
3. **Execute migration scripts** to update all imports
4. **Update critical agents** (King, Sage, Magi) first
5. **Validate with integration tests**

This migration plan ensures zero-downtime transition to a unified, comprehensive error handling system that will significantly improve debugging, monitoring, and system reliability.
