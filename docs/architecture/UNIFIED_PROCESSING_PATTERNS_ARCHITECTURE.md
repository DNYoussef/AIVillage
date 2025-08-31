# Unified Processing Patterns Architecture

## Executive Summary

This document defines the unified architecture for implementing abstract base class methods across the AIVillage agent system. The analysis identifies three key abstract base classes requiring consistent implementation patterns: `UnifiedBaseAgent._process_task()`, `BaseAnalytics` methods, and `ProcessingInterface` methods.

## Current Abstract Method Analysis

### 1. UnifiedBaseAgent Abstract Methods

**Primary Abstract Method:**
- `_process_task(self, task: LangroidTask) -> dict[str, Any]` (Line 176-181)
  - **Purpose**: Core task processing logic for agent specializations
  - **Input**: LangroidTask with content and type
  - **Output**: Dictionary with processing results
  - **Current Status**: Raises NotImplementedError

**Key Integration Points:**
- Integrates with 5-layer architecture (Quality Assurance, Foundational, Continuous Learning, Agent Architecture, Decision Making)
- Must handle error propagation through `with_error_handling` decorator
- Required for agent factory pattern via `create_agent()`

### 2. BaseAnalytics Abstract Methods

**Primary Abstract Methods:**
- `generate_analytics_report(self) -> dict[str, Any]` (Lines 34-41)
  - **Purpose**: Generate comprehensive analytics reports
  - **Input**: Instance metrics stored in `self.metrics`
  - **Output**: Dictionary with analytics data
  - **Current Status**: NotImplementedError with roadmap reference

**Supporting Methods Requiring Implementation:**
- `save(self, path: str) -> NoReturn` (Line 43-44)
- `load(self, path: str) -> NoReturn` (Line 46-47)

### 3. ProcessingInterface Abstract Methods

**Core Processing Methods:**
- `initialize(self) -> bool` (Lines 148-150)
- `shutdown(self) -> bool` (Lines 152-154)
- `process(self, input_data: T, **kwargs) -> ProcessResult[U]` (Lines 156-167)
- `validate_input(self, input_data: T) -> bool` (Lines 168-177)
- `estimate_processing_time(self, input_data: T) -> float | None` (Lines 179-188)

**Key Features:**
- Generic typing with TypeVar T, U for input/output
- Built-in queue processing, caching, batch processing
- Comprehensive metrics collection
- Health check capabilities

## Unified Implementation Patterns

### 1. Error Handling Architecture

#### Exception Hierarchy Design
```python
# Unified Exception Hierarchy
AIVillageException (base)
├── ProcessingError (processing failures)
├── ValidationError (input validation)
├── InitializationError (setup failures)
├── CommunicationError (agent messaging)
├── ExternalServiceError (RAG, LLM failures)
└── EvolutionError (self-modification failures)
```

#### Standard Error Context Pattern
```python
# Consistent error context creation
def create_error_context(operation: str, **metadata) -> ErrorContext:
    return ErrorContext(
        component=f"{self.__class__.__name__}",
        operation=operation,
        metadata={
            "agent_id": getattr(self, "agent_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            **metadata
        }
    )
```

#### Retry and Recovery Pattern
```python
@with_error_handling(retries=3, context={"component": "Agent", "method": "process_task"})
async def process_implementation(self, input_data: Any) -> ProcessResult:
    # Implementation with automatic retry on transient failures
    pass
```

### 2. Input Validation and Sanitization Patterns

#### Standard Validation Pipeline
```python
class ValidationPipeline:
    def __init__(self):
        self.validators = [
            TypeValidator(),
            SizeValidator(),
            ContentValidator(),
            SecurityValidator()
        ]
    
    async def validate(self, data: Any, schema: dict) -> ValidationResult:
        for validator in self.validators:
            result = await validator.validate(data, schema)
            if not result.is_valid:
                return result
        return ValidationResult.success()
```

#### Input Sanitization Template
```python
async def sanitize_input(self, input_data: Any) -> Any:
    """Standard input sanitization across all processors"""
    
    # 1. Remove malicious content
    if isinstance(input_data, str):
        input_data = self._sanitize_string(input_data)
    
    # 2. Validate size limits
    if self._exceeds_size_limit(input_data):
        raise ValidationError("Input exceeds size limit")
    
    # 3. Apply domain-specific sanitization
    return await self._domain_specific_sanitization(input_data)
```

### 3. Async Processing with Cancellation Support

#### Cancellation-Aware Processing Pattern
```python
import asyncio
from typing import Optional

class CancellableProcessor:
    def __init__(self):
        self._cancellation_token: Optional[asyncio.Event] = None
        self._active_tasks: set[asyncio.Task] = set()
    
    async def process_with_cancellation(self, input_data: Any) -> ProcessResult:
        self._cancellation_token = asyncio.Event()
        
        try:
            # Create processing task
            task = asyncio.create_task(self._internal_process(input_data))
            self._active_tasks.add(task)
            
            # Wait for completion or cancellation
            done, pending = await asyncio.wait(
                [task, asyncio.create_task(self._cancellation_token.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if self._cancellation_token.is_set():
                task.cancel()
                return ProcessResult.cancelled()
            
            return await task
            
        finally:
            self._active_tasks.discard(task)
    
    def cancel_processing(self):
        if self._cancellation_token:
            self._cancellation_token.set()
```

### 4. Memory Management and Cleanup Patterns

#### Resource Lifecycle Management
```python
class ResourceManager:
    def __init__(self):
        self._resources: dict[str, Any] = {}
        self._cleanup_handlers: list[callable] = []
    
    async def __aenter__(self):
        await self.initialize_resources()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup_resources()
    
    async def cleanup_resources(self):
        for handler in reversed(self._cleanup_handlers):
            try:
                await handler()
            except Exception as e:
                logger.warning(f"Cleanup handler failed: {e}")
```

#### Memory-Aware Processing
```python
import psutil
from typing import Protocol

class MemoryAware(Protocol):
    def get_memory_usage(self) -> dict[str, float]:
        process = psutil.Process()
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    async def process_with_memory_monitoring(self, input_data: Any):
        initial_memory = self.get_memory_usage()
        
        try:
            result = await self._process_internal(input_data)
            return result
        finally:
            final_memory = self.get_memory_usage()
            memory_delta = final_memory["rss_mb"] - initial_memory["rss_mb"]
            
            if memory_delta > 100:  # 100MB threshold
                logger.warning(f"High memory usage detected: {memory_delta:.2f}MB")
                await self._trigger_garbage_collection()
```

## Implementation Templates

### 1. UnifiedBaseAgent._process_task() Template

```python
async def _process_task(self, task: LangroidTask) -> dict[str, Any]:
    """
    Template implementation for agent-specific task processing
    
    This template provides the standard pattern for implementing
    the _process_task method in agent specializations.
    """
    
    # 1. Input validation
    if not task or not task.content:
        return {
            "status": "error",
            "error": "Invalid task: missing content",
            "error_type": "ValidationError"
        }
    
    # 2. Task-specific processing with memory management
    async with ResourceManager() as resources:
        try:
            # 3. Pre-processing (domain-specific)
            processed_input = await self._preprocess_task_input(task)
            
            # 4. Core processing logic (implement in subclass)
            result = await self._execute_core_logic(processed_input)
            
            # 5. Post-processing and validation
            validated_result = await self._validate_result(result)
            
            # 6. Success response
            return {
                "status": "success",
                "result": validated_result,
                "metadata": {
                    "agent_type": self.__class__.__name__,
                    "processing_time_ms": self._get_processing_time(),
                    "task_id": getattr(task, 'id', None)
                }
            }
            
        except ValidationError as e:
            return {"status": "error", "error": str(e), "error_type": "ValidationError"}
        except ProcessingError as e:
            return {"status": "error", "error": str(e), "error_type": "ProcessingError"}
        except Exception as e:
            logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
            return {"status": "error", "error": "Internal processing error", "error_type": "InternalError"}

    # Abstract methods for subclass implementation
    async def _preprocess_task_input(self, task: LangroidTask) -> Any:
        """Preprocess task input - implement in subclass"""
        return task.content
    
    async def _execute_core_logic(self, processed_input: Any) -> Any:
        """Execute core processing logic - implement in subclass"""
        raise NotImplementedError("Subclasses must implement _execute_core_logic")
    
    async def _validate_result(self, result: Any) -> Any:
        """Validate processing result - implement in subclass"""
        return result
```

### 2. BaseAnalytics Implementation Template

```python
class StandardAnalytics(BaseAnalytics):
    """Standard implementation template for analytics generation"""
    
    def __init__(self):
        super().__init__()
        self.report_generators = {
            "performance": self._generate_performance_report,
            "usage": self._generate_usage_report,
            "errors": self._generate_error_report,
            "trends": self._generate_trend_report
        }
    
    def generate_analytics_report(self) -> dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "comprehensive",
            "metrics_summary": {},
            "sections": {}
        }
        
        # Generate each report section
        for section_name, generator in self.report_generators.items():
            try:
                section_data = generator()
                report["sections"][section_name] = section_data
                
                # Add to summary
                if isinstance(section_data, dict) and "summary" in section_data:
                    report["metrics_summary"][section_name] = section_data["summary"]
                    
            except Exception as e:
                logger.error(f"Failed to generate {section_name} report: {e}")
                report["sections"][section_name] = {"error": str(e)}
        
        return report
    
    def _generate_performance_report(self) -> dict[str, Any]:
        """Generate performance analytics"""
        if not self.metrics:
            return {"summary": "No performance data available"}
        
        performance_data = {}
        for metric_name, values in self.metrics.items():
            if values:
                performance_data[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else None
                }
        
        return {
            "summary": f"Tracking {len(performance_data)} performance metrics",
            "metrics": performance_data
        }
    
    def save(self, path: str) -> None:
        """Save analytics state to file"""
        import json
        
        save_data = {
            "metrics": self.metrics,
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"Analytics saved to {path}")
        except Exception as e:
            raise ProcessingError(f"Failed to save analytics to {path}: {e}")
    
    def load(self, path: str) -> None:
        """Load analytics state from file"""
        import json
        
        try:
            with open(path, 'r') as f:
                save_data = json.load(f)
            
            self.metrics = save_data.get("metrics", {})
            logger.info(f"Analytics loaded from {path}")
            
        except FileNotFoundError:
            logger.warning(f"Analytics file not found: {path}")
        except Exception as e:
            raise ProcessingError(f"Failed to load analytics from {path}: {e}")
```

### 3. ProcessingInterface Implementation Template

```python
class StandardProcessor(ProcessingInterface[T, U]):
    """Standard implementation template for processing interface"""
    
    def __init__(self, processor_id: str, config: ProcessConfig = None):
        super().__init__(processor_id, config)
        
        # Add standard capabilities
        self.add_capability(ProcessorCapability.VALIDATION)
        self.add_capability(ProcessorCapability.QUALITY_CONTROL)
        self.add_capability(ProcessorCapability.CACHING)
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize processor with standard setup"""
        try:
            logger.info(f"Initializing processor: {self.processor_id}")
            
            # 1. Validate configuration
            if not self._validate_configuration():
                return False
            
            # 2. Initialize domain-specific resources
            await self._initialize_domain_resources()
            
            # 3. Start background workers
            if self.has_capability(ProcessorCapability.BATCH_PROCESSING):
                asyncio.create_task(self.start_processing_worker())
            
            # 4. Set status
            self.set_status(ProcessorStatus.IDLE)
            self._initialized = True
            
            logger.info(f"Processor {self.processor_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize processor {self.processor_id}: {e}")
            self.set_status(ProcessorStatus.ERROR)
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown processor gracefully"""
        try:
            logger.info(f"Shutting down processor: {self.processor_id}")
            
            # 1. Set shutting down status
            self.set_status(ProcessorStatus.SHUTTING_DOWN)
            
            # 2. Cancel active tasks
            for task_id, task in list(self._processing_tasks.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # 3. Clear cache
            self.clear_cache()
            
            # 4. Cleanup domain resources
            await self._cleanup_domain_resources()
            
            logger.info(f"Processor {self.processor_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown processor {self.processor_id}: {e}")
            return False
    
    async def process(self, input_data: T, **kwargs) -> ProcessResult[U]:
        """Process input with standard pipeline"""
        
        if not self._initialized:
            return ProcessResult.failed("Processor not initialized")
        
        self.set_status(ProcessorStatus.PROCESSING)
        
        try:
            # 1. Validate input
            if not await self.validate_input(input_data):
                return ProcessResult.failed("Input validation failed")
            
            # 2. Check cache
            cache_key = self._generate_cache_key(input_data, kwargs)
            cached_result = await self.get_cached_result(cache_key)
            if cached_result is not None:
                self.metrics.cache_hit_rate += 1
                return ProcessResult.success(cached_result)
            
            # 3. Execute processing
            result = await self._execute_processing(input_data, **kwargs)
            
            # 4. Cache result
            if result.is_success:
                await self.cache_result(cache_key, result.data)
            
            # 5. Update metrics
            self._update_processing_metrics(result.is_success)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessResult.failed(str(e))
            
        finally:
            self.set_status(ProcessorStatus.IDLE)
    
    async def validate_input(self, input_data: T) -> bool:
        """Validate input data"""
        try:
            # 1. Type validation
            if not self._validate_type(input_data):
                return False
            
            # 2. Size validation
            if not self._validate_size(input_data):
                return False
            
            # 3. Content validation
            if not await self._validate_content(input_data):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    async def estimate_processing_time(self, input_data: T) -> float | None:
        """Estimate processing time based on input characteristics"""
        
        if self.metrics.average_processing_time_ms == 0:
            return None  # No historical data
        
        try:
            # Simple estimation based on input size and historical averages
            base_time = self.metrics.average_processing_time_ms / 1000  # Convert to seconds
            size_factor = self._calculate_size_factor(input_data)
            
            estimated_time = base_time * size_factor
            return max(0.1, estimated_time)  # Minimum 0.1 second
            
        except Exception:
            return None
    
    # Abstract methods for subclass implementation
    async def _initialize_domain_resources(self) -> None:
        """Initialize domain-specific resources - implement in subclass"""
        pass
    
    async def _cleanup_domain_resources(self) -> None:
        """Cleanup domain-specific resources - implement in subclass"""
        pass
    
    async def _execute_processing(self, input_data: T, **kwargs) -> ProcessResult[U]:
        """Execute core processing logic - implement in subclass"""
        raise NotImplementedError("Subclasses must implement _execute_processing")
```

## Integration Architecture

### 1. Langroid Integration Patterns

#### Task Routing Integration
```python
class LangroidCompatibleAgent(UnifiedBaseAgent):
    """Integration layer for Langroid compatibility"""
    
    async def _process_task(self, task: LangroidTask) -> dict[str, Any]:
        """Langroid-compatible task processing"""
        
        # Convert Langroid task to internal format
        internal_task = self._convert_langroid_task(task)
        
        # Process using standard patterns
        result = await self._process_internal_task(internal_task)
        
        # Convert result back to Langroid format
        return self._convert_to_langroid_result(result)
    
    def _convert_langroid_task(self, langroid_task: LangroidTask) -> InternalTask:
        """Convert Langroid task to internal task format"""
        return InternalTask(
            id=getattr(langroid_task, 'id', None),
            content=langroid_task.content,
            task_type=getattr(langroid_task, 'type', 'general'),
            metadata=getattr(langroid_task, 'metadata', {})
        )
```

### 2. Agent System Compatibility

#### Communication Protocol Integration
```python
class CommunicationIntegrator:
    """Integrates processing interface with agent communication"""
    
    def __init__(self, processor: ProcessingInterface, 
                 communication_protocol: StandardCommunicationProtocol):
        self.processor = processor
        self.protocol = communication_protocol
    
    async def process_agent_message(self, message: Message) -> Message:
        """Process agent message through processing interface"""
        
        # Extract processing request
        request = ProcessingRequest(
            request_id=message.id,
            input_data=message.content,
            processing_type=message.type.value,
            metadata={"sender": message.sender, "receiver": message.receiver}
        )
        
        # Process through interface
        response = await self.processor._process_request(request)
        
        # Convert back to agent message
        return Message(
            type=MessageType.RESPONSE,
            sender=message.receiver,
            receiver=message.sender,
            content=response.output_data,
            parent_id=message.id
        )
```

## Quality Criteria for Implementations

### 1. Correctness Criteria
- ✅ All abstract methods implemented
- ✅ Input validation performs type, size, and content checks
- ✅ Error handling uses unified exception hierarchy
- ✅ Async operations support proper cancellation
- ✅ Memory cleanup occurs in all code paths

### 2. Performance Criteria
- ✅ Processing time estimation accuracy within 20%
- ✅ Memory usage growth < 10% per operation
- ✅ Cache hit rate > 80% for repeated operations
- ✅ Error recovery time < 1 second for transient failures

### 3. Reliability Criteria
- ✅ Graceful degradation on resource constraints
- ✅ No data corruption during failures
- ✅ Consistent behavior across agent types
- ✅ Backward compatibility with existing implementations

### 4. Maintainability Criteria
- ✅ Clear separation of concerns
- ✅ Minimal coupling between components
- ✅ Comprehensive logging at appropriate levels
- ✅ Testable interfaces with dependency injection

## Implementation Validation Checklist

### Pre-Implementation
- [ ] Domain requirements analyzed
- [ ] Abstract method contracts understood
- [ ] Error scenarios identified
- [ ] Resource requirements estimated

### During Implementation
- [ ] Template patterns followed
- [ ] Error handling implemented
- [ ] Input validation added
- [ ] Memory management considered
- [ ] Async patterns used correctly

### Post-Implementation
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests validate compatibility
- [ ] Performance benchmarks meet criteria
- [ ] Error handling tested under failure conditions
- [ ] Memory usage verified under load

---

**Architectural Decision Record (ADR)**
- **Status**: Proposed
- **Date**: 2025-08-31
- **Architect**: Claude Code Architecture Agent
- **Context**: Unifying abstract method implementations across agent system
- **Decision**: Standardize on template-based implementation patterns with unified error handling
- **Consequences**: Improved consistency, testability, and maintainability across all agent implementations