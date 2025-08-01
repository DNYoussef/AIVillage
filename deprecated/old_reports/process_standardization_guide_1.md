# Process Method Standardization Guide

This guide provides instructions for migrating existing `process_*` methods to use the standardized `BaseProcessHandler` framework, eliminating code duplication and ensuring consistent error handling across the AIVillage codebase.

## Overview

The codebase currently has 19+ variants of `process_query`, `process_task`, `process_result` methods across different agents. The new standardization framework provides:

- **Consistent Error Handling**: Standardized exception handling and logging
- **Performance Metrics**: Built-in timing and success rate tracking
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Timeout Management**: Configurable processing timeouts
- **Input Validation**: Standardized input validation patterns
- **Result Standardization**: Uniform result format across all processors

## Migration Steps

### Step 1: Import the Base Framework

```python
from agents.base import (
    QueryProcessor,
    TaskProcessor,
    MessageProcessor,
    ProcessConfig,
    ProcessResult,
    standardized_process
)
```

### Step 2: Choose Migration Approach

#### Option A: Inherit from Base Classes (Recommended)

For new implementations or major refactoring:

```python
class EnhancedQueryProcessor(QueryProcessor):
    def __init__(self, rag_system, cognitive_nexus):
        config = ProcessConfig(
            timeout_seconds=30.0,
            retry_attempts=2,
            enable_logging=True
        )
        super().__init__("EnhancedQueryProcessor", config)
        self.rag_system = rag_system
        self.cognitive_nexus = cognitive_nexus

    async def _process_query(self, query: str, **kwargs) -> str:
        # Your existing logic here
        rag_result = await self.rag_system.query(query)
        cognitive_context = await self.cognitive_nexus.process(query)

        return f"Enhanced: {rag_result} + {cognitive_context}"
```

#### Option B: Use Factory Functions (Quick Migration)

For existing code with minimal changes:

```python
from agents.base import create_query_processor

async def existing_process_query(query: str) -> str:
    # Your existing implementation
    return f"Processed: {query}"

# Create standardized processor
processor = create_query_processor(
    "ExistingQueryProcessor",
    existing_process_query,
    ProcessConfig(timeout_seconds=10.0)
)

# Use in your class
class MyAgent:
    def __init__(self):
        self.query_processor = processor

    async def process_query(self, query: str) -> str:
        result = await self.query_processor.process(query)
        return result.data
```

#### Option C: Use Decorator (Minimal Changes)

For existing methods with decorator wrapping:

```python
class MyAgent:
    @standardized_process("query", config=ProcessConfig(timeout_seconds=15.0))
    async def process_query(self, query: str) -> str:
        # Your existing implementation unchanged
        return await self.rag_system.query(query)
```

## Migration Examples

### Example 1: Sage Query Processor

**Before (agents/sage/query_processing.py):**
```python
class QueryProcessor:
    async def process_query(self, query: str) -> str:
        try:
            results = await asyncio.gather(
                self.activate_latent_space(query),
                self.query_cognitive_nexus(query),
                self.apply_advanced_reasoning({"content": query}),
                self.query_rag(query),
            )
            # ... processing logic
            return enhanced_query
        except Exception as e:
            logger.error(f"Error processing query: {e!s}")
            return query
```

**After:**
```python
from agents.base import QueryProcessor, ProcessConfig

class SageQueryProcessor(QueryProcessor):
    def __init__(self, rag_system, latent_space_activation, cognitive_nexus):
        config = ProcessConfig(
            timeout_seconds=30.0,
            retry_attempts=1,
            enable_logging=True,
            enable_metrics=True
        )
        super().__init__("SageQueryProcessor", config)
        self.rag_system = rag_system
        self.latent_space_activation = latent_space_activation
        self.cognitive_nexus = cognitive_nexus

    async def _process_query(self, query: str, **kwargs) -> str:
        results = await asyncio.gather(
            self.activate_latent_space(query),
            self.query_cognitive_nexus(query),
            self.apply_advanced_reasoning({"content": query}),
            self.query_rag(query),
        )
        activated_knowledge, cognitive_context, reasoning_result, rag_result = results

        enhanced_query = f"""
        Original Query: {query}
        Activated Knowledge: {activated_knowledge}
        Cognitive Context: {cognitive_context}
        Advanced Reasoning: {reasoning_result}
        RAG Result: {rag_result}
        """

        return enhanced_query
```

### Example 2: King Task Processing

**Before (agents/king/king_agent.py):**
```python
async def process_task(self, task: LangroidTask) -> dict[str, Any]:
    try:
        # Task validation
        if not task or not task.content:
            return {"error": "Invalid task"}

        # Processing logic
        result = await self._internal_process(task)
        return {"result": result}
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        return {"error": str(e)}
```

**After:**
```python
from agents.base import TaskProcessor, ProcessConfig

class KingTaskProcessor(TaskProcessor):
    def __init__(self):
        config = ProcessConfig(
            timeout_seconds=60.0,
            retry_attempts=2,
            validation_enabled=True
        )
        super().__init__("KingTaskProcessor", config)

    async def _validate_input(self, input_data):
        await super()._validate_input(input_data)
        if isinstance(input_data, dict):
            if not input_data.get("content"):
                raise ValueError("Task content is required")

    async def _process_task(self, task: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        result = await self._internal_process(task)
        return {"result": result}
```

### Example 3: Batch Processing

**Before (agents/task_management/unified_task_manager.py):**
```python
async def process_task_batch(self):
    results = []
    for task in self.task_queue:
        try:
            result = await self.process_single_task(task)
            results.append(result)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            results.append(None)
    return results
```

**After:**
```python
from agents.base import BatchProcessor, ProcessConfig

class TaskBatchProcessor(BatchProcessor):
    def __init__(self):
        config = ProcessConfig(
            timeout_seconds=120.0,
            retry_attempts=1
        )
        super().__init__(
            "TaskBatchProcessor",
            batch_size=5,
            parallel_processing=True,
            config=config
        )

    async def _process_single_item(self, task: Any, **kwargs) -> Any:
        return await self.process_single_task(task)
```

## Configuration Options

### ProcessConfig Parameters

```python
config = ProcessConfig(
    timeout_seconds=30.0,        # Processing timeout (None = no timeout)
    retry_attempts=2,            # Number of retry attempts on failure
    retry_delay_seconds=1.0,     # Delay between retries
    enable_logging=True,         # Enable automatic logging
    enable_metrics=True,         # Track performance metrics
    validation_enabled=True      # Enable input validation
)
```

### Performance Monitoring

All processors automatically track metrics:

```python
processor = SageQueryProcessor(...)

# Process queries
await processor.process("What is AI?")
await processor.process("Explain machine learning")

# Check performance metrics
metrics = processor.metrics
print(f"Success rate: {processor.success_rate:.1f}%")
print(f"Average processing time: {metrics['avg_processing_time']:.2f}ms")
print(f"Total processed: {metrics['total_processed']}")
```

## Benefits After Migration

### 1. Consistent Error Handling
- All exceptions are caught and logged uniformly
- Standardized error messages and stack traces
- Graceful degradation with meaningful error responses

### 2. Performance Monitoring
- Automatic timing measurements for all operations
- Success/failure rate tracking
- Performance trend analysis capabilities

### 3. Reliability Improvements
- Configurable timeout prevention of hanging operations
- Retry logic for transient failures
- Input validation to catch errors early

### 4. Maintainability
- Single source of truth for processing patterns
- Reduced code duplication across 19+ process methods
- Easier testing with standardized interfaces

### 5. Observability
- Structured logging with consistent format
- Metrics collection for monitoring dashboards
- Debugging information for troubleshooting

## Migration Checklist

### Phase 1: Critical Path (High Priority)
- [ ] `agents/unified_base_agent.py` - Core agent processing
- [ ] `agents/sage/query_processing.py` - RAG query processing
- [ ] `agents/king/king_agent.py` - Main coordination agent
- [ ] `rag_system/core/pipeline.py` - RAG pipeline processing

### Phase 2: Agent Systems (Medium Priority)
- [ ] `agents/sage/reasoning_agent.py`
- [ ] `agents/king/response_generation_agent.py`
- [ ] `agents/task_management/unified_task_manager.py`
- [ ] `agents/orchestration.py`

### Phase 3: Supporting Components (Low Priority)
- [ ] Input processing modules in `agents/king/input/`
- [ ] Communication protocol handlers
- [ ] Specialized processing utilities

## Testing Strategy

### Unit Tests
```python
import pytest
from agents.base import ProcessConfig, ProcessStatus

@pytest.mark.asyncio
async def test_query_processor():
    processor = SageQueryProcessor(mock_rag, mock_cognitive)

    result = await processor.process("test query")

    assert result.status == ProcessStatus.COMPLETED
    assert result.data is not None
    assert result.processing_time_ms > 0

@pytest.mark.asyncio
async def test_timeout_handling():
    config = ProcessConfig(timeout_seconds=0.1)
    processor = SlowQueryProcessor(config=config)

    result = await processor.process("slow query")

    assert result.status == ProcessStatus.TIMEOUT
    assert result.error is not None
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_end_to_end_processing():
    agent = SageAgent()

    # Test successful processing
    response = await agent.process_query("What is AI?")
    assert "Enhanced:" in response

    # Test error handling
    response = await agent.process_query("")
    assert "error" in response.lower()
```

## Rollback Plan

If issues arise during migration:

1. **Keep Original Methods**: Temporarily maintain original implementations alongside new ones
2. **Feature Flags**: Use configuration to switch between old and new implementations
3. **Gradual Rollout**: Migrate one processor type at a time
4. **Monitoring**: Watch error rates and performance metrics during transition

## Support and Resources

- **Code Examples**: See `examples/process_migration/` for complete examples
- **Base Classes**: `agents/base/process_handler.py` for implementation details
- **Migration Scripts**: `scripts/migrate_process_methods.py` for automated assistance
- **Testing**: `tests/base/test_process_handlers.py` for test examples

## Next Steps

1. **Start with High Priority**: Begin with `unified_base_agent.py` and core pipeline
2. **Test Thoroughly**: Ensure existing functionality is preserved
3. **Monitor Performance**: Watch for any regression in processing speed
4. **Gather Feedback**: Collect input from team members using migrated components
5. **Iterate**: Refine the base classes based on real-world usage

This standardization will significantly improve code maintainability, reduce duplication, and provide better error handling and monitoring across the entire AIVillage platform.
