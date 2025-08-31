"""
Unified Processing Templates
============================

Concrete code templates for implementing abstract base class methods
following the unified processing patterns architecture.

These templates ensure consistent implementation across all AIVillage
agents and processors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import psutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, Protocol, TypeVar

# Type variables for generic implementations
T = TypeVar("T")
U = TypeVar("U")

logger = logging.getLogger(__name__)


# ============================================================================
# Error Handling Templates
# ============================================================================

class AIVillageException(Exception):
    """Base exception for all AIVillage errors"""
    
    def __init__(self, message: str, error_type: str = "UnknownError", context: dict[str, Any] = None):
        self.message = message
        self.error_type = error_type
        self.context = context or {}
        super().__init__(message)


class ProcessingError(AIVillageException):
    """Processing-related errors"""
    pass


class ValidationError(AIVillageException):
    """Input validation errors"""
    pass


class InitializationError(AIVillageException):
    """Initialization errors"""
    pass


class CommunicationError(AIVillageException):
    """Agent communication errors"""
    pass


class ExternalServiceError(AIVillageException):
    """External service errors (RAG, LLM, etc.)"""
    pass


class EvolutionError(AIVillageException):
    """Agent evolution errors"""
    pass


class ErrorContext:
    """Standard error context for logging and debugging"""
    
    def __init__(self, component: str, operation: str, metadata: dict[str, Any] = None):
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


# ============================================================================
# Resource Management Templates
# ============================================================================

class ResourceManager:
    """Template for managing resources with proper cleanup"""
    
    def __init__(self):
        self._resources: dict[str, Any] = {}
        self._cleanup_handlers: list[callable] = []
    
    async def __aenter__(self):
        await self.initialize_resources()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup_resources()
    
    def register_cleanup(self, handler: callable):
        """Register a cleanup handler"""
        self._cleanup_handlers.append(handler)
    
    def add_resource(self, name: str, resource: Any):
        """Add a managed resource"""
        self._resources[name] = resource
    
    def get_resource(self, name: str) -> Any:
        """Get a managed resource"""
        return self._resources.get(name)
    
    async def initialize_resources(self):
        """Override in subclasses for custom initialization"""
        pass
    
    async def cleanup_resources(self):
        """Execute all cleanup handlers in reverse order"""
        for handler in reversed(self._cleanup_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.warning(f"Cleanup handler failed: {e}")


class MemoryAwareProcessor:
    """Template for memory-aware processing"""
    
    def __init__(self):
        self.memory_threshold_mb = 500  # 500MB threshold
        self.memory_history: list[dict[str, float]] = []
    
    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {"error": str(e)}
    
    def record_memory_usage(self):
        """Record current memory usage"""
        usage = self.get_memory_usage()
        usage["timestamp"] = datetime.now().isoformat()
        self.memory_history.append(usage)
        
        # Keep only last 100 records
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
    
    async def process_with_memory_monitoring(self, input_data: Any, processor_func: callable):
        """Process data with memory monitoring"""
        initial_memory = self.get_memory_usage()
        self.record_memory_usage()
        
        try:
            result = await processor_func(input_data)
            return result
        
        finally:
            final_memory = self.get_memory_usage()
            memory_delta = final_memory.get("rss_mb", 0) - initial_memory.get("rss_mb", 0)
            
            if memory_delta > 100:  # 100MB increase
                logger.warning(f"High memory usage detected: {memory_delta:.2f}MB increase")
                await self._trigger_memory_cleanup()
    
    async def _trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures"""
        import gc
        gc.collect()
        logger.info("Memory cleanup triggered")


# ============================================================================
# Validation Templates
# ============================================================================

class ValidationResult:
    """Standard validation result"""
    
    def __init__(self, is_valid: bool, errors: list[str] = None, warnings: list[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    @classmethod
    def success(cls, warnings: list[str] = None):
        return cls(True, warnings=warnings)
    
    @classmethod
    def failure(cls, errors: list[str], warnings: list[str] = None):
        return cls(False, errors, warnings)


class InputValidator(ABC):
    """Base class for input validators"""
    
    @abstractmethod
    async def validate(self, data: Any, schema: dict = None) -> ValidationResult:
        """Validate input data"""
        pass


class TypeValidator(InputValidator):
    """Validates input types"""
    
    async def validate(self, data: Any, schema: dict = None) -> ValidationResult:
        if not schema or "type" not in schema:
            return ValidationResult.success()
        
        expected_type = schema["type"]
        if not isinstance(data, expected_type):
            return ValidationResult.failure([
                f"Expected type {expected_type.__name__}, got {type(data).__name__}"
            ])
        
        return ValidationResult.success()


class SizeValidator(InputValidator):
    """Validates input size limits"""
    
    async def validate(self, data: Any, schema: dict = None) -> ValidationResult:
        if not schema:
            return ValidationResult.success()
        
        errors = []
        
        # String length validation
        if isinstance(data, str) and "max_length" in schema:
            max_length = schema["max_length"]
            if len(data) > max_length:
                errors.append(f"String length {len(data)} exceeds maximum {max_length}")
        
        # List/dict size validation
        if hasattr(data, "__len__") and "max_items" in schema:
            max_items = schema["max_items"]
            if len(data) > max_items:
                errors.append(f"Item count {len(data)} exceeds maximum {max_items}")
        
        return ValidationResult.failure(errors) if errors else ValidationResult.success()


class SecurityValidator(InputValidator):
    """Validates input for security issues"""
    
    DANGEROUS_PATTERNS = [
        "<script",
        "javascript:",
        "eval(",
        "exec(",
        "__import__",
        "subprocess",
    ]
    
    async def validate(self, data: Any, schema: dict = None) -> ValidationResult:
        if isinstance(data, str):
            data_lower = data.lower()
            dangerous_found = []
            
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in data_lower:
                    dangerous_found.append(pattern)
            
            if dangerous_found:
                return ValidationResult.failure([
                    f"Dangerous patterns detected: {', '.join(dangerous_found)}"
                ])
        
        return ValidationResult.success()


class ValidationPipeline:
    """Pipeline for running multiple validators"""
    
    def __init__(self):
        self.validators = [
            TypeValidator(),
            SizeValidator(),
            SecurityValidator()
        ]
    
    def add_validator(self, validator: InputValidator):
        """Add a custom validator to the pipeline"""
        self.validators.append(validator)
    
    async def validate(self, data: Any, schema: dict = None) -> ValidationResult:
        """Run all validators in sequence"""
        all_errors = []
        all_warnings = []
        
        for validator in self.validators:
            try:
                result = await validator.validate(data, schema)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                
                # Stop on first critical error
                if result.errors:
                    break
                    
            except Exception as e:
                all_errors.append(f"Validator {validator.__class__.__name__} failed: {e}")
        
        if all_errors:
            return ValidationResult.failure(all_errors, all_warnings)
        else:
            return ValidationResult.success(all_warnings)


# ============================================================================
# Processing Result Templates
# ============================================================================

class ProcessStatus:
    """Standard processing status values"""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    PENDING = "pending"


class ProcessResult(Generic[T]):
    """Standard processing result container"""
    
    def __init__(self, status: str, data: T = None, error: str = None, 
                 metadata: dict[str, Any] = None):
        self.status = status
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    @property
    def is_success(self) -> bool:
        return self.status == ProcessStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        return self.status == ProcessStatus.FAILED
    
    @property
    def is_cancelled(self) -> bool:
        return self.status == ProcessStatus.CANCELLED
    
    @classmethod
    def success(cls, data: T = None, metadata: dict[str, Any] = None):
        return cls(ProcessStatus.SUCCESS, data, metadata=metadata)
    
    @classmethod
    def failed(cls, error: str, metadata: dict[str, Any] = None):
        return cls(ProcessStatus.FAILED, error=error, metadata=metadata)
    
    @classmethod
    def cancelled(cls, metadata: dict[str, Any] = None):
        return cls(ProcessStatus.CANCELLED, metadata=metadata)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# UnifiedBaseAgent._process_task() Template Implementation
# ============================================================================

class UnifiedBaseAgentTemplate:
    """Template for implementing UnifiedBaseAgent._process_task()"""
    
    def __init__(self):
        self.processor = MemoryAwareProcessor()
        self.validator = ValidationPipeline()
        self.processing_history: list[dict[str, Any]] = []
    
    async def _process_task(self, task: Any) -> dict[str, Any]:
        """
        Template implementation for agent-specific task processing
        
        This template provides the standard pattern for implementing
        the _process_task method in agent specializations.
        """
        
        # Create error context for debugging
        error_context = ErrorContext(
            component=self.__class__.__name__,
            operation="_process_task",
            metadata={"task_id": getattr(task, 'id', 'unknown')}
        )
        
        # 1. Input validation
        if not task or not hasattr(task, 'content') or not task.content:
            return {
                "status": ProcessStatus.FAILED,
                "error": "Invalid task: missing content",
                "error_type": "ValidationError",
                "context": error_context.to_dict()
            }
        
        # 2. Task-specific processing with memory management
        async with ResourceManager() as resources:
            try:
                start_time = datetime.now()
                
                # 3. Validate task input
                validation_schema = self._get_task_validation_schema()
                validation_result = await self.validator.validate(task.content, validation_schema)
                
                if not validation_result.is_valid:
                    return {
                        "status": ProcessStatus.FAILED,
                        "error": f"Task validation failed: {', '.join(validation_result.errors)}",
                        "error_type": "ValidationError",
                        "context": error_context.to_dict()
                    }
                
                # 4. Pre-processing (domain-specific)
                processed_input = await self._preprocess_task_input(task)
                
                # 5. Core processing logic with memory monitoring
                result = await self.processor.process_with_memory_monitoring(
                    processed_input,
                    self._execute_core_logic
                )
                
                # 6. Post-processing and validation
                validated_result = await self._validate_result(result)
                
                # 7. Record processing metrics
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._record_processing_metrics(task, processing_time, True)
                
                # 8. Success response
                return {
                    "status": ProcessStatus.SUCCESS,
                    "result": validated_result,
                    "metadata": {
                        "agent_type": self.__class__.__name__,
                        "processing_time_ms": processing_time,
                        "task_id": getattr(task, 'id', None),
                        "memory_usage": self.processor.get_memory_usage()
                    }
                }
                
            except ValidationError as e:
                return {
                    "status": ProcessStatus.FAILED,
                    "error": str(e),
                    "error_type": "ValidationError",
                    "context": error_context.to_dict()
                }
            except ProcessingError as e:
                return {
                    "status": ProcessStatus.FAILED,
                    "error": str(e),
                    "error_type": "ProcessingError",
                    "context": error_context.to_dict()
                }
            except Exception as e:
                logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
                return {
                    "status": ProcessStatus.FAILED,
                    "error": "Internal processing error",
                    "error_type": "InternalError",
                    "context": error_context.to_dict()
                }
    
    # Abstract methods for subclass implementation
    def _get_task_validation_schema(self) -> dict[str, Any]:
        """Get validation schema for tasks - override in subclass"""
        return {
            "type": str,
            "max_length": 10000
        }
    
    async def _preprocess_task_input(self, task: Any) -> Any:
        """Preprocess task input - implement in subclass"""
        return task.content
    
    async def _execute_core_logic(self, processed_input: Any) -> Any:
        """Execute core processing logic - implement in subclass"""
        raise NotImplementedError("Subclasses must implement _execute_core_logic")
    
    async def _validate_result(self, result: Any) -> Any:
        """Validate processing result - implement in subclass"""
        return result
    
    def _record_processing_metrics(self, task: Any, processing_time_ms: float, success: bool):
        """Record processing metrics for analysis"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "task_id": getattr(task, 'id', None),
            "processing_time_ms": processing_time_ms,
            "success": success,
            "memory_usage": self.processor.get_memory_usage()
        }
        
        self.processing_history.append(metrics)
        
        # Keep only recent history
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-1000:]


# ============================================================================
# BaseAnalytics Implementation Template
# ============================================================================

class StandardAnalytics:
    """Standard implementation template for analytics generation"""
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
        self.report_generators = {
            "performance": self._generate_performance_report,
            "usage": self._generate_usage_report,
            "errors": self._generate_error_report,
            "trends": self._generate_trend_report
        }
    
    def record_metric(self, metric: str, value: float) -> None:
        """Record a metric value"""
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)
        logger.debug(f"Recorded {metric}: {value}")
    
    def generate_analytics_report(self) -> dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "comprehensive",
            "metrics_summary": {},
            "sections": {},
            "data_quality": self._assess_data_quality()
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
                    "latest": values[-1] if values else None,
                    "trend": self._calculate_trend(values)
                }
        
        return {
            "summary": f"Tracking {len(performance_data)} performance metrics",
            "metrics": performance_data,
            "overall_health": self._calculate_overall_health(performance_data)
        }
    
    def _generate_usage_report(self) -> dict[str, Any]:
        """Generate usage analytics"""
        total_operations = sum(len(values) for values in self.metrics.values())
        
        if total_operations == 0:
            return {"summary": "No usage data available"}
        
        return {
            "summary": f"Total operations: {total_operations}",
            "operations_by_metric": {
                metric: len(values) for metric, values in self.metrics.items()
            },
            "activity_level": self._assess_activity_level(total_operations)
        }
    
    def _generate_error_report(self) -> dict[str, Any]:
        """Generate error analytics"""
        error_metrics = {k: v for k, v in self.metrics.items() if "error" in k.lower()}
        
        if not error_metrics:
            return {"summary": "No error data available"}
        
        total_errors = sum(len(values) for values in error_metrics.values())
        
        return {
            "summary": f"Total errors: {total_errors}",
            "errors_by_type": {metric: len(values) for metric, values in error_metrics.items()},
            "error_rate": self._calculate_error_rate()
        }
    
    def _generate_trend_report(self) -> dict[str, Any]:
        """Generate trend analysis"""
        trends = {}
        
        for metric_name, values in self.metrics.items():
            if len(values) >= 2:
                trends[metric_name] = self._calculate_trend(values)
        
        return {
            "summary": f"Trend analysis for {len(trends)} metrics",
            "trends": trends
        }
    
    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for values"""
        if len(values) < 2:
            return "insufficient_data"
        
        recent_avg = sum(values[-5:]) / len(values[-5:])  # Last 5 values
        older_avg = sum(values[:5]) / len(values[:5]) if len(values) > 5 else sum(values[:-5]) / len(values[:-5])
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _assess_data_quality(self) -> dict[str, Any]:
        """Assess quality of collected data"""
        if not self.metrics:
            return {"status": "no_data", "score": 0}
        
        total_points = len(self.metrics) * 100
        completeness = min(100, (sum(len(v) for v in self.metrics.values()) / max(1, len(self.metrics))) * 10)
        
        return {
            "status": "good" if completeness > 70 else "poor",
            "score": completeness,
            "metrics_count": len(self.metrics),
            "data_points_count": sum(len(v) for v in self.metrics.values())
        }
    
    def _calculate_overall_health(self, performance_data: dict) -> str:
        """Calculate overall system health from performance data"""
        if not performance_data:
            return "unknown"
        
        # Simple health assessment based on trends
        increasing_count = sum(1 for data in performance_data.values() 
                             if data.get("trend") == "increasing")
        decreasing_count = sum(1 for data in performance_data.values() 
                             if data.get("trend") == "decreasing")
        
        if increasing_count > decreasing_count:
            return "improving"
        elif decreasing_count > increasing_count:
            return "degrading"
        else:
            return "stable"
    
    def _assess_activity_level(self, total_operations: int) -> str:
        """Assess activity level based on operation count"""
        if total_operations > 1000:
            return "high"
        elif total_operations > 100:
            return "medium"
        else:
            return "low"
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        total_ops = sum(len(values) for values in self.metrics.values() if "error" not in values)
        total_errors = sum(len(values) for values in self.metrics.values() if "error" in values)
        
        if total_ops == 0:
            return 0.0
        
        return (total_errors / (total_ops + total_errors)) * 100
    
    def save(self, path: str) -> None:
        """Save analytics state to file"""
        save_data = {
            "metrics": self.metrics,
            "saved_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path_obj, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"Analytics saved to {path}")
            
        except Exception as e:
            raise ProcessingError(f"Failed to save analytics to {path}: {e}")
    
    def load(self, path: str) -> None:
        """Load analytics state from file"""
        try:
            with open(path, 'r') as f:
                save_data = json.load(f)
            
            self.metrics = save_data.get("metrics", {})
            logger.info(f"Analytics loaded from {path}")
            
        except FileNotFoundError:
            logger.warning(f"Analytics file not found: {path}")
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON in analytics file {path}: {e}")
        except Exception as e:
            raise ProcessingError(f"Failed to load analytics from {path}: {e}")


# ============================================================================
# ProcessingInterface Implementation Template
# ============================================================================

class StandardProcessor:
    """Standard implementation template for processing interface"""
    
    def __init__(self, processor_id: str, config: dict = None):
        self.processor_id = processor_id
        self.config = config or {}
        self.status = "idle"
        self.capabilities = set()
        self.metrics = {
            "total_processed": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "average_processing_time_ms": 0.0
        }
        
        self.processing_queue = asyncio.Queue()
        self._cache = {}
        self._processing_tasks = {}
        self._initialized = False
        
        # Resource management
        self.resource_manager = ResourceManager()
        self.validator = ValidationPipeline()
        self.memory_monitor = MemoryAwareProcessor()
        
        # Add standard capabilities
        self.capabilities.add("validation")
        self.capabilities.add("caching")
        self.capabilities.add("batch_processing")
    
    async def initialize(self) -> bool:
        """Initialize processor with standard setup"""
        try:
            logger.info(f"Initializing processor: {self.processor_id}")
            
            # 1. Validate configuration
            if not self._validate_configuration():
                return False
            
            # 2. Initialize resource manager
            async with self.resource_manager:
                # 3. Initialize domain-specific resources
                await self._initialize_domain_resources()
                
                # 4. Start background workers if needed
                if "batch_processing" in self.capabilities:
                    asyncio.create_task(self.start_processing_worker())
                
                # 5. Set status
                self.status = "idle"
                self._initialized = True
                
                logger.info(f"Processor {self.processor_id} initialized successfully")
                return True
            
        except Exception as e:
            logger.error(f"Failed to initialize processor {self.processor_id}: {e}")
            self.status = "error"
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown processor gracefully"""
        try:
            logger.info(f"Shutting down processor: {self.processor_id}")
            
            # 1. Set shutting down status
            self.status = "shutting_down"
            
            # 2. Cancel active tasks
            for task_id, task in list(self._processing_tasks.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # 3. Clear cache
            self._cache.clear()
            
            # 4. Cleanup resources
            await self.resource_manager.cleanup_resources()
            
            # 5. Save final metrics
            await self._save_final_metrics()
            
            logger.info(f"Processor {self.processor_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown processor {self.processor_id}: {e}")
            return False
    
    async def process(self, input_data: Any, **kwargs) -> ProcessResult:
        """Process input with standard pipeline"""
        
        if not self._initialized:
            return ProcessResult.failed("Processor not initialized")
        
        self.status = "processing"
        
        try:
            # 1. Validate input
            validation_result = await self.validate_input(input_data)
            if not validation_result.is_valid:
                return ProcessResult.failed(f"Input validation failed: {', '.join(validation_result.errors)}")
            
            # 2. Check cache
            cache_key = self._generate_cache_key(input_data, kwargs)
            cached_result = self.get_cached_result(cache_key)
            if cached_result is not None:
                return ProcessResult.success(cached_result)
            
            # 3. Execute processing with memory monitoring
            result = await self.memory_monitor.process_with_memory_monitoring(
                input_data,
                lambda data: self._execute_processing(data, **kwargs)
            )
            
            # 4. Cache successful results
            if result.is_success:
                self.cache_result(cache_key, result.data)
            
            # 5. Update metrics
            self._update_processing_metrics(result.is_success)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed in {self.processor_id}: {e}")
            self._update_processing_metrics(False)
            return ProcessResult.failed(str(e))
            
        finally:
            self.status = "idle"
    
    async def validate_input(self, input_data: Any) -> ValidationResult:
        """Validate input data using validation pipeline"""
        try:
            schema = self._get_input_validation_schema()
            return await self.validator.validate(input_data, schema)
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return ValidationResult.failure([f"Validation error: {e}"])
    
    async def estimate_processing_time(self, input_data: Any) -> Optional[float]:
        """Estimate processing time based on input characteristics and history"""
        
        if self.metrics["average_processing_time_ms"] == 0:
            return None  # No historical data
        
        try:
            # Simple estimation based on input size and historical averages
            base_time = self.metrics["average_processing_time_ms"] / 1000  # Convert to seconds
            size_factor = self._calculate_size_factor(input_data)
            
            estimated_time = base_time * size_factor
            return max(0.1, estimated_time)  # Minimum 0.1 second
            
        except Exception:
            return None
    
    async def health_check(self) -> dict[str, Any]:
        """Comprehensive processor health check"""
        return {
            "processor_id": self.processor_id,
            "status": self.status,
            "initialized": self._initialized,
            "capabilities": list(self.capabilities),
            "metrics": {
                "total_processed": self.metrics["total_processed"],
                "success_rate": self._calculate_success_rate(),
                "average_processing_time_ms": self.metrics["average_processing_time_ms"],
            },
            "resources": {
                "queue_size": self.processing_queue.qsize(),
                "active_tasks": len(self._processing_tasks),
                "cache_size": len(self._cache),
            },
            "memory_usage": self.memory_monitor.get_memory_usage(),
            "last_check": datetime.now().isoformat()
        }
    
    # Utility methods
    def cache_result(self, key: str, result: Any, ttl_seconds: Optional[float] = None):
        """Cache processing result"""
        if "caching" in self.capabilities:
            self._cache[key] = {
                "result": result,
                "timestamp": datetime.now(),
                "ttl_seconds": ttl_seconds
            }
    
    def get_cached_result(self, key: str) -> Any:
        """Get cached result if available and not expired"""
        if "caching" not in self.capabilities or key not in self._cache:
            return None
        
        cache_entry = self._cache[key]
        
        # Check TTL if specified
        if cache_entry["ttl_seconds"]:
            age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            if age > cache_entry["ttl_seconds"]:
                del self._cache[key]
                return None
        
        return cache_entry["result"]
    
    # Abstract/customizable methods
    def _validate_configuration(self) -> bool:
        """Validate processor configuration - override in subclass"""
        return True
    
    async def _initialize_domain_resources(self):
        """Initialize domain-specific resources - override in subclass"""
        pass
    
    async def _execute_processing(self, input_data: Any, **kwargs) -> ProcessResult:
        """Execute core processing logic - implement in subclass"""
        raise NotImplementedError("Subclasses must implement _execute_processing")
    
    def _get_input_validation_schema(self) -> dict[str, Any]:
        """Get input validation schema - override in subclass"""
        return {"type": str, "max_length": 10000}
    
    def _generate_cache_key(self, input_data: Any, kwargs: dict) -> str:
        """Generate cache key for input - override for custom caching"""
        import hashlib
        key_data = f"{input_data}{kwargs}".encode()
        return hashlib.md5(key_data).hexdigest()
    
    def _calculate_size_factor(self, input_data: Any) -> float:
        """Calculate size factor for processing time estimation"""
        try:
            if hasattr(input_data, "__len__"):
                size = len(input_data)
            else:
                size = 1
            
            # Logarithmic scaling
            import math
            return max(1.0, math.log(size + 1) / math.log(1000))
            
        except:
            return 1.0
    
    def _update_processing_metrics(self, success: bool):
        """Update processing metrics"""
        self.metrics["total_processed"] += 1
        
        if success:
            self.metrics["successful_processes"] += 1
        else:
            self.metrics["failed_processes"] += 1
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.metrics["total_processed"]
        if total == 0:
            return 100.0
        
        return (self.metrics["successful_processes"] / total) * 100
    
    async def _save_final_metrics(self):
        """Save final metrics before shutdown"""
        try:
            metrics_file = f"{self.processor_id}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    "processor_id": self.processor_id,
                    "metrics": self.metrics,
                    "shutdown_time": datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save final metrics: {e}")
    
    async def start_processing_worker(self):
        """Background worker for processing queued requests"""
        while self.status != "shutting_down":
            try:
                # Get next request with timeout
                request = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Create processing task
                task = asyncio.create_task(self._process_queued_request(request))
                self._processing_tasks[request.get("id", "unknown")] = task
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_queued_request(self, request: dict):
        """Process a queued request"""
        try:
            result = await self.process(request.get("input_data"), **request.get("kwargs", {}))
            
            # Handle callback if provided
            if "callback" in request:
                callback = request["callback"]
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
                    
        except Exception as e:
            logger.error(f"Failed to process queued request: {e}")
        finally:
            # Cleanup
            request_id = request.get("id", "unknown")
            self._processing_tasks.pop(request_id, None)


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Example usage of the templates
    
    class ExampleAgent(UnifiedBaseAgentTemplate):
        """Example agent implementation using the template"""
        
        async def _execute_core_logic(self, processed_input: Any) -> Any:
            # Simulate some processing
            await asyncio.sleep(0.1)
            return f"Processed: {processed_input}"
    
    class ExampleProcessor(StandardProcessor):
        """Example processor implementation using the template"""
        
        async def _execute_processing(self, input_data: Any, **kwargs) -> ProcessResult:
            # Simulate processing
            result = f"Processed: {input_data}"
            return ProcessResult.success(result)
    
    # Demonstrate the templates
    async def main():
        # Test agent template
        agent = ExampleAgent()
        
        class MockTask:
            def __init__(self, content):
                self.content = content
                self.id = "test-123"
        
        task = MockTask("test task content")
        result = await agent._process_task(task)
        print(f"Agent result: {result}")
        
        # Test processor template
        processor = ExampleProcessor("test-processor")
        await processor.initialize()
        
        proc_result = await processor.process("test input")
        print(f"Processor result: {proc_result.to_dict()}")
        
        await processor.shutdown()
        
        # Test analytics template
        analytics = StandardAnalytics()
        analytics.record_metric("response_time", 150.0)
        analytics.record_metric("response_time", 200.0)
        analytics.record_metric("error_count", 1.0)
        
        report = analytics.generate_analytics_report()
        print(f"Analytics report: {json.dumps(report, indent=2)}")
    
    # Run the example
    asyncio.run(main())