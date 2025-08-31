# Implementation Integration Guide

## Quick Start for Development Agents

This guide provides practical steps for implementing the unified processing patterns across all AIVillage agents and processors.

## 🚀 Immediate Implementation Steps

### 1. Choose Your Abstract Method

**UnifiedBaseAgent._process_task()**
```bash
# Copy template from: docs/templates/unified_processing_templates.py
# Line: 366-465 (UnifiedBaseAgentTemplate)

# Example implementation:
class MySpecializedAgent(UnifiedBaseAgentTemplate):
    async def _execute_core_logic(self, processed_input: Any) -> Any:
        # Your domain-specific logic here
        return {"processed": processed_input, "agent_type": "specialized"}
```

**BaseAnalytics Methods**
```bash
# Copy template from: docs/templates/unified_processing_templates.py  
# Line: 567-758 (StandardAnalytics)

# Example implementation:
class MyAnalytics(StandardAnalytics):
    def __init__(self):
        super().__init__()
        self.add_custom_generators()
```

**ProcessingInterface Methods**
```bash
# Copy template from: docs/templates/unified_processing_templates.py
# Line: 767-1094 (StandardProcessor)

# Example implementation:  
class MyProcessor(StandardProcessor):
    async def _execute_processing(self, input_data: Any, **kwargs) -> ProcessResult:
        # Your processing logic here
        return ProcessResult.success(processed_data)
```

### 2. Import Required Dependencies

Add to your agent file:
```python
from docs.templates.unified_processing_templates import (
    # Error handling
    AIVillageException, ProcessingError, ValidationError,
    ErrorContext,
    
    # Resource management
    ResourceManager, MemoryAwareProcessor,
    
    # Validation
    ValidationPipeline, ValidationResult,
    
    # Results
    ProcessResult, ProcessStatus,
    
    # Templates (choose one)
    UnifiedBaseAgentTemplate,      # For agent _process_task()
    StandardAnalytics,             # For analytics methods
    StandardProcessor,             # For processing interface
)
```

### 3. Replace Your Abstract Method

**Before (raises NotImplementedError):**
```python
async def _process_task(self, task: LangroidTask) -> dict[str, Any]:
    raise NotImplementedError("Subclasses must implement _process_task method")
```

**After (using template):**
```python
class MyAgent(UnifiedBaseAgentTemplate):
    async def _execute_core_logic(self, processed_input: Any) -> Any:
        # Your existing logic, adapted
        result = self.my_existing_processing_method(processed_input)
        return result
        
    def _get_task_validation_schema(self) -> dict[str, Any]:
        # Define your input validation rules
        return {
            "type": str,
            "max_length": 5000,
            "required_fields": ["content", "task_type"]
        }
```

## 🔧 Integration Patterns by Use Case

### Agent Specialization Pattern

```python
# For agents that need specific task processing
class ResearchAgent(UnifiedBaseAgentTemplate):
    def __init__(self):
        super().__init__()
        self.research_tools = ResearchToolkit()
        self.validator.add_validator(ResearchValidator())
    
    async def _execute_core_logic(self, processed_input: Any) -> Any:
        # Research-specific processing
        research_query = self._extract_research_query(processed_input)
        findings = await self.research_tools.search(research_query)
        analysis = await self.research_tools.analyze(findings)
        
        return {
            "research_findings": findings,
            "analysis": analysis,
            "confidence_score": self._calculate_confidence(analysis)
        }
    
    def _get_task_validation_schema(self) -> dict[str, Any]:
        return {
            "type": str,
            "max_length": 2000,
            "required_patterns": [r"research|analyze|investigate"]
        }
```

### Analytics Enhancement Pattern

```python
# For agents that need comprehensive analytics
class PerformanceAnalyticsAgent(StandardAnalytics):
    def __init__(self):
        super().__init__()
        # Add custom report generators
        self.report_generators.update({
            "bottleneck_analysis": self._generate_bottleneck_report,
            "capacity_planning": self._generate_capacity_report,
            "anomaly_detection": self._generate_anomaly_report
        })
    
    def _generate_bottleneck_report(self) -> dict[str, Any]:
        # Custom bottleneck analysis
        bottlenecks = self._identify_bottlenecks()
        return {
            "summary": f"Found {len(bottlenecks)} potential bottlenecks",
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_recommendations(bottlenecks)
        }
```

### Processing Pipeline Pattern

```python
# For agents that need complex processing workflows
class MultiStageProcessor(StandardProcessor):
    def __init__(self, processor_id: str):
        super().__init__(processor_id)
        self.stages = [
            PreprocessingStage(),
            AnalysisStage(), 
            SynthesisStage(),
            ValidationStage()
        ]
    
    async def _execute_processing(self, input_data: Any, **kwargs) -> ProcessResult:
        # Multi-stage processing pipeline
        current_data = input_data
        stage_results = []
        
        for i, stage in enumerate(self.stages):
            try:
                stage_result = await stage.process(current_data, **kwargs)
                stage_results.append(stage_result)
                current_data = stage_result.data
                
            except Exception as e:
                return ProcessResult.failed(f"Stage {i} failed: {e}")
        
        return ProcessResult.success({
            "final_result": current_data,
            "stage_results": stage_results,
            "pipeline_metadata": self._generate_pipeline_metadata()
        })
```

## 🔌 Backward Compatibility Integration

### Existing Agent Wrapper

```python
# Wrap existing agents without breaking changes
class BackwardCompatibleWrapper(UnifiedBaseAgentTemplate):
    def __init__(self, existing_agent):
        super().__init__()
        self.legacy_agent = existing_agent
    
    async def _execute_core_logic(self, processed_input: Any) -> Any:
        # Bridge to existing agent implementation
        if hasattr(self.legacy_agent, 'process'):
            return await self.legacy_agent.process(processed_input)
        elif hasattr(self.legacy_agent, 'execute'):
            return await self.legacy_agent.execute(processed_input)
        else:
            # Fallback to legacy pattern
            return {"legacy_result": str(processed_input)}
    
    async def _preprocess_task_input(self, task: Any) -> Any:
        # Convert new task format to legacy format
        if hasattr(task, 'content'):
            return task.content
        return task
```

### Gradual Migration Strategy

```python
# Migrate existing agents gradually
class MigrationHelper:
    @staticmethod
    def wrap_existing_agent(agent_class):
        """Decorator to wrap existing agents with new templates"""
        
        class WrappedAgent(UnifiedBaseAgentTemplate):
            def __init__(self, *args, **kwargs):
                self.original_agent = agent_class(*args, **kwargs)
                super().__init__()
            
            async def _execute_core_logic(self, processed_input: Any) -> Any:
                # Delegate to original agent
                if hasattr(self.original_agent, '_process_task'):
                    # Create mock task for legacy interface
                    mock_task = type('MockTask', (), {'content': processed_input})()
                    return await self.original_agent._process_task(mock_task)
                else:
                    return {"migrated": processed_input}
        
        return WrappedAgent

# Usage:
@MigrationHelper.wrap_existing_agent
class ExistingAgent:
    # Your existing agent code unchanged
    pass
```

## 🧪 Testing Integration

### Unit Testing Template

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestUnifiedAgent:
    @pytest.fixture
    def agent(self):
        # Create your agent using the template
        return YourSpecializedAgent()
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, agent):
        # Test successful processing
        mock_task = MagicMock()
        mock_task.content = "test task"
        mock_task.id = "test-123"
        
        result = await agent._process_task(mock_task)
        
        assert result["status"] == "success"
        assert "result" in result
        assert "metadata" in result
    
    @pytest.mark.asyncio  
    async def test_process_task_validation_failure(self, agent):
        # Test validation failure
        mock_task = MagicMock()
        mock_task.content = ""  # Invalid empty content
        
        result = await agent._process_task(mock_task)
        
        assert result["status"] == "failed"
        assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_memory_management(self, agent):
        # Test memory usage doesn't grow excessively
        initial_memory = agent.processor.get_memory_usage()
        
        # Process multiple tasks
        for i in range(100):
            mock_task = MagicMock()
            mock_task.content = f"test task {i}"
            mock_task.id = f"test-{i}"
            await agent._process_task(mock_task)
        
        final_memory = agent.processor.get_memory_usage()
        memory_growth = final_memory["rss_mb"] - initial_memory["rss_mb"]
        
        assert memory_growth < 100  # Less than 100MB growth
```

### Integration Testing Template

```python
@pytest.mark.integration
class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_langroid_compatibility(self):
        # Test integration with Langroid tasks
        from agents.utils.task import Task as LangroidTask
        
        agent = YourSpecializedAgent()
        langroid_task = LangroidTask(agent, "integration test task")
        
        result = await agent._process_task(langroid_task)
        
        assert result["status"] == "success"
        assert "metadata" in result
        assert result["metadata"]["agent_type"] == "YourSpecializedAgent"
    
    @pytest.mark.asyncio
    async def test_communication_integration(self):
        # Test integration with communication protocol
        protocol = StandardCommunicationProtocol()
        agent = YourSpecializedAgent()
        
        # Test message handling
        message = Message(
            type=MessageType.TASK,
            sender="test_sender",
            receiver=agent.name,
            content={"task": "test integration"}
        )
        
        await agent.handle_message(message)
        # Verify message was processed correctly
```

## 📊 Monitoring and Observability

### Metrics Collection Template

```python
class MonitoringMixin:
    """Add monitoring capabilities to any agent"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_tracker = PerformanceTracker()
    
    async def _process_task_with_monitoring(self, task: Any) -> dict[str, Any]:
        # Wrap task processing with monitoring
        with self.performance_tracker.track_operation("task_processing"):
            self.metrics_collector.increment("tasks_received")
            
            try:
                result = await super()._process_task(task)
                
                if result["status"] == "success":
                    self.metrics_collector.increment("tasks_successful")
                else:
                    self.metrics_collector.increment("tasks_failed")
                
                return result
                
            except Exception as e:
                self.metrics_collector.increment("tasks_error")
                self.metrics_collector.record_error(str(e))
                raise

# Usage:
class MonitoredAgent(MonitoringMixin, UnifiedBaseAgentTemplate):
    async def _process_task(self, task: Any) -> dict[str, Any]:
        return await self._process_task_with_monitoring(task)
```

## ⚡ Performance Optimization

### Caching Integration

```python
class CachedProcessor(StandardProcessor):
    def __init__(self, processor_id: str):
        super().__init__(processor_id)
        self.add_capability("advanced_caching")
        self.cache_strategies = {
            "lru": LRUCache(maxsize=1000),
            "ttl": TTLCache(maxsize=500, ttl=3600),  # 1 hour
            "adaptive": AdaptiveCache()
        }
    
    def _generate_cache_key(self, input_data: Any, kwargs: dict) -> str:
        # Smart cache key generation
        import hashlib
        
        # Include relevant context for better cache hits
        context = {
            "input_hash": hashlib.md5(str(input_data).encode()).hexdigest(),
            "kwargs_hash": hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest(),
            "processor_id": self.processor_id
        }
        
        return f"{context['processor_id']}:{context['input_hash']}:{context['kwargs_hash']}"
```

### Async Optimization

```python
class OptimizedAsyncProcessor(StandardProcessor):
    def __init__(self, processor_id: str):
        super().__init__(processor_id)
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
        self.task_pool = asyncio.Queue(maxsize=100)
    
    async def process_batch_optimized(self, input_batch: list, batch_size: int = 10):
        # Optimized batch processing with controlled concurrency
        async def process_single_with_semaphore(item):
            async with self.semaphore:
                return await self.process(item)
        
        # Process in chunks to avoid overwhelming the system
        results = []
        for i in range(0, len(input_batch), batch_size):
            chunk = input_batch[i:i + batch_size]
            
            # Process chunk concurrently
            chunk_tasks = [process_single_with_semaphore(item) for item in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            results.extend(chunk_results)
        
        return results
```

## 🔍 Debugging and Troubleshooting

### Debug Mode Integration

```python
class DebuggableAgent(UnifiedBaseAgentTemplate):
    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode
        self.debug_traces = []
    
    async def _process_task(self, task: Any) -> dict[str, Any]:
        if self.debug_mode:
            debug_context = {
                "task_id": getattr(task, 'id', 'unknown'),
                "task_content": str(task.content)[:100] + "..." if len(str(task.content)) > 100 else str(task.content),
                "timestamp": datetime.now().isoformat(),
                "memory_before": self.processor.get_memory_usage()
            }
            
            try:
                result = await super()._process_task(task)
                debug_context["status"] = result["status"]
                debug_context["memory_after"] = self.processor.get_memory_usage()
                
                self.debug_traces.append(debug_context)
                return result
                
            except Exception as e:
                debug_context["error"] = str(e)
                debug_context["memory_after"] = self.processor.get_memory_usage()
                self.debug_traces.append(debug_context)
                raise
        else:
            return await super()._process_task(task)
    
    def get_debug_report(self) -> dict[str, Any]:
        """Generate debug report with traces"""
        if not self.debug_mode:
            return {"error": "Debug mode not enabled"}
        
        return {
            "total_traces": len(self.debug_traces),
            "traces": self.debug_traces[-10:],  # Last 10 traces
            "memory_trend": self._analyze_memory_trend(),
            "error_summary": self._summarize_errors()
        }
```

## ✅ Implementation Checklist

### Pre-Implementation
- [ ] Identified which abstract method(s) to implement
- [ ] Reviewed template code and patterns
- [ ] Planned integration with existing codebase
- [ ] Set up test environment

### During Implementation
- [ ] Copied appropriate template class
- [ ] Implemented required abstract methods
- [ ] Added domain-specific validation schema
- [ ] Integrated error handling patterns
- [ ] Added logging and metrics collection
- [ ] Implemented resource cleanup

### Post-Implementation
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests validate compatibility
- [ ] Performance benchmarks meet requirements
- [ ] Memory usage verified under load
- [ ] Error handling tested under failure conditions
- [ ] Documentation updated

### Production Readiness
- [ ] Monitoring and alerting configured
- [ ] Logging properly configured
- [ ] Error rates within acceptable limits
- [ ] Performance meets SLA requirements
- [ ] Backward compatibility verified
- [ ] Rollback plan prepared

---

## 🤝 Getting Help

### Common Issues and Solutions

**Issue: Template import errors**
```bash
# Solution: Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/AIVillage"

# Or use relative imports
from ..templates.unified_processing_templates import UnifiedBaseAgentTemplate
```

**Issue: Langroid task compatibility**
```bash
# Solution: Use compatibility wrapper
def convert_langroid_task(langroid_task):
    return {
        "content": langroid_task.content,
        "id": getattr(langroid_task, 'id', None),
        "type": getattr(langroid_task, 'type', 'general')
    }
```

**Issue: Memory usage growth**
```bash
# Solution: Implement proper cleanup
async def cleanup_after_processing(self):
    # Clear caches periodically
    if len(self._cache) > 1000:
        self._cache.clear()
    
    # Trigger garbage collection
    import gc
    gc.collect()
```

### Support Resources

- **Architecture Documentation**: `/docs/architecture/UNIFIED_PROCESSING_PATTERNS_ARCHITECTURE.md`
- **Code Templates**: `/docs/templates/unified_processing_templates.py`
- **Test Examples**: `/tests/unit/test_unified_base_agent.py`
- **Integration Examples**: This guide

For specific implementation questions, create detailed examples using the templates and refer to the comprehensive architecture documentation.