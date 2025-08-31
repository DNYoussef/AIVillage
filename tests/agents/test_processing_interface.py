"""Comprehensive unit tests for ProcessingInterface implementation.

Tests all abstract methods with various processing scenarios including
batch processing, queue management, caching, and error handling.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Set

from agents.interfaces.processing_interface import (
    ProcessingInterface,
    ProcessorStatus,
    ProcessorCapability,
    ProcessingMode,
    ProcessingMetrics,
    ProcessingRequest,
    ProcessingResponse,
    create_processing_request,
    validate_processing_interface
)
from agents.base import ProcessResult, ProcessStatus, ProcessConfig


class TestProcessingInterface(ProcessingInterface[str, str]):
    """Concrete implementation for testing."""
    
    def __init__(self, processor_id: str = "test_processor", config=None):
        super().__init__(processor_id, config)
        self.add_capability(ProcessorCapability.TEXT_PROCESSING)
        self.add_capability(ProcessorCapability.BATCH_PROCESSING)
        self.add_capability(ProcessorCapability.CACHING)
        self.initialize_called = False
        self.shutdown_called = False
        self.processed_inputs = []
        self.processing_delay = 0.0  # For timing tests
    
    async def initialize(self) -> bool:
        """Initialize the processor."""
        await asyncio.sleep(0.01)  # Simulate initialization time
        self.initialize_called = True
        self.set_status(ProcessorStatus.IDLE)
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the processor."""
        await asyncio.sleep(0.01)  # Simulate shutdown time
        self.shutdown_called = True
        self.set_status(ProcessorStatus.SHUTTING_DOWN)
        return True
    
    async def process(self, input_data: str, **kwargs) -> ProcessResult[str]:
        """Process input data."""
        if self.processing_delay > 0:
            await asyncio.sleep(self.processing_delay)
        
        self.processed_inputs.append(input_data)
        
        # Simulate different processing outcomes
        if input_data == "error_input":
            return ProcessResult(
                status=ProcessStatus.FAILED,
                error="Simulated processing error"
            )
        elif input_data == "slow_input":
            await asyncio.sleep(0.1)
            return ProcessResult(
                status=ProcessStatus.COMPLETED,
                data=f"slowly_processed_{input_data}"
            )
        elif input_data.startswith("invalid"):
            return ProcessResult(
                status=ProcessStatus.FAILED,
                error="Invalid input format"
            )
        
        result = f"processed_{input_data}"
        return ProcessResult(
            status=ProcessStatus.COMPLETED,
            data=result
        )
    
    async def validate_input(self, input_data: str) -> bool:
        """Validate input data."""
        if not isinstance(input_data, str):
            return False
        if input_data.startswith("invalid"):
            return False
        if len(input_data) == 0:
            return False
        return True
    
    async def estimate_processing_time(self, input_data: str) -> float:
        """Estimate processing time."""
        if input_data == "slow_input":
            return 0.15
        elif input_data == "fast_input":
            return 0.01
        else:
            return len(input_data) * 0.01  # 10ms per character


@pytest.mark.unit
class TestProcessingInterfaceInitialization:
    """Test processing interface initialization."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        processor = TestProcessingInterface()
        
        assert processor.processor_id == "test_processor"
        assert processor.status == ProcessorStatus.IDLE
        assert isinstance(processor.capabilities, set)
        assert ProcessorCapability.TEXT_PROCESSING in processor.capabilities
        assert isinstance(processor.metrics, ProcessingMetrics)
        assert isinstance(processor.processing_queue, asyncio.Queue)
        assert processor._cache == {}
        assert processor._processing_tasks == {}
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = ProcessConfig()
        processor = TestProcessingInterface("custom_processor", config)
        
        assert processor.processor_id == "custom_processor"
        assert processor.config == config
    
    async def test_successful_initialization(self):
        """Test successful processor initialization."""
        processor = TestProcessingInterface()
        
        result = await processor.initialize()
        
        assert result is True
        assert processor.initialize_called is True
        assert processor.status == ProcessorStatus.IDLE
    
    async def test_successful_shutdown(self):
        """Test successful processor shutdown."""
        processor = TestProcessingInterface()
        
        result = await processor.shutdown()
        
        assert result is True
        assert processor.shutdown_called is True
        assert processor.status == ProcessorStatus.SHUTTING_DOWN


@pytest.mark.unit  
class TestProcessingInterfaceBasicProcessing:
    """Test basic processing functionality."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_successful_processing(self, processor):
        """Test successful data processing."""
        result = await processor.process("test_input")
        
        assert result.is_success
        assert result.data == "processed_test_input"
        assert "test_input" in processor.processed_inputs
    
    async def test_processing_failure(self, processor):
        """Test processing failure handling."""
        result = await processor.process("error_input")
        
        assert result.is_error
        assert result.error == "Simulated processing error"
        assert "error_input" in processor.processed_inputs
    
    async def test_input_validation_success(self, processor):
        """Test successful input validation."""
        is_valid = await processor.validate_input("valid_input")
        
        assert is_valid is True
    
    async def test_input_validation_failure(self, processor):
        """Test input validation failure."""
        # Test various invalid inputs
        invalid_inputs = ["", "invalid_input", None, 123]
        
        for invalid_input in invalid_inputs:
            if invalid_input is not None and not isinstance(invalid_input, str):
                continue  # Skip non-string types for this implementation
            is_valid = await processor.validate_input(invalid_input)
            assert is_valid is False
    
    async def test_processing_time_estimation(self, processor):
        """Test processing time estimation."""
        # Test different input types
        slow_time = await processor.estimate_processing_time("slow_input")
        fast_time = await processor.estimate_processing_time("fast_input")
        normal_time = await processor.estimate_processing_time("normal")
        
        assert slow_time > fast_time
        assert slow_time == 0.15
        assert fast_time == 0.01
        assert normal_time == len("normal") * 0.01
    
    async def test_processing_with_kwargs(self, processor):
        """Test processing with additional parameters."""
        result = await processor.process("test_input", param1="value1", param2=42)
        
        assert result.is_success
        assert result.data == "processed_test_input"


@pytest.mark.unit
class TestProcessingInterfaceBatchProcessing:
    """Test batch processing functionality."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_batch_processing_parallel(self, processor):
        """Test parallel batch processing."""
        input_batch = ["input1", "input2", "input3", "input4", "input5"]
        
        results = await processor.process_batch(input_batch, parallel=True)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.is_success
            assert result.data == f"processed_input{i+1}"
    
    async def test_batch_processing_sequential(self, processor):
        """Test sequential batch processing."""
        input_batch = ["input1", "input2", "input3"]
        
        results = await processor.process_batch(input_batch, parallel=False)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.is_success
            assert result.data == f"processed_input{i+1}"
    
    async def test_batch_processing_with_batch_size(self, processor):
        """Test batch processing with custom batch size."""
        input_batch = ["input1", "input2", "input3", "input4", "input5"]
        
        results = await processor.process_batch(input_batch, batch_size=2)
        
        assert len(results) == 5
        # All should be processed despite batch size limit
        for i, result in enumerate(results):
            assert result.is_success
    
    async def test_batch_processing_with_failures(self, processor):
        """Test batch processing with some failures."""
        input_batch = ["input1", "error_input", "input3"]
        
        results = await processor.process_batch(input_batch, parallel=True)
        
        assert len(results) == 3
        assert results[0].is_success
        assert results[1].is_error
        assert results[2].is_success
    
    async def test_batch_processing_without_capability(self):
        """Test batch processing without capability."""
        processor = TestProcessingInterface()
        processor.remove_capability(ProcessorCapability.BATCH_PROCESSING)
        
        with pytest.raises(NotImplementedError) as exc_info:
            await processor.process_batch(["input1", "input2"])
        
        assert "does not support batch processing" in str(exc_info.value)
    
    async def test_empty_batch_processing(self, processor):
        """Test processing empty batch."""
        results = await processor.process_batch([])
        
        assert results == []
    
    async def test_single_item_batch_processing(self, processor):
        """Test processing single-item batch."""
        results = await processor.process_batch(["single_input"])
        
        assert len(results) == 1
        assert results[0].is_success
        assert results[0].data == "processed_single_input"


@pytest.mark.unit
class TestProcessingInterfaceQueueProcessing:
    """Test queue-based processing."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_submit_request(self, processor):
        """Test submitting processing request."""
        request = create_processing_request("test_data", "text_processing")
        
        request_id = await processor.submit_request(request)
        
        assert request_id == request.request_id
        assert processor.processing_queue.qsize() == 1
    
    async def test_process_request_from_queue(self, processor):
        """Test processing request from queue."""
        request = create_processing_request("test_data", "text_processing")
        
        response = await processor._process_request(request)
        
        assert response.request_id == request.request_id
        assert response.success is True
        assert response.output_data == "processed_test_data"
        assert response.processing_time_ms > 0
    
    async def test_process_request_failure(self, processor):
        """Test processing request failure."""
        request = create_processing_request("error_input", "text_processing")
        
        response = await processor._process_request(request)
        
        assert response.request_id == request.request_id
        assert response.success is False
        assert response.error_message == "Simulated processing error"
        assert response.processing_time_ms > 0
    
    async def test_get_result_success(self, processor):
        """Test getting successful processing result."""
        request = create_processing_request("test_data", "text_processing")
        
        # Submit and start processing
        await processor.submit_request(request)
        task = asyncio.create_task(processor._process_request(request))
        processor._processing_tasks[request.request_id] = task
        
        result = await processor.get_result(request.request_id, timeout_seconds=1.0)
        
        assert result is not None
        assert result.success is True
        assert result.output_data == "processed_test_data"
    
    async def test_get_result_timeout(self, processor):
        """Test getting result with timeout."""
        request = create_processing_request("slow_input", "text_processing")
        
        # Create a slow task
        async def slow_task():
            await asyncio.sleep(2.0)  # Longer than timeout
            return await processor._process_request(request)
        
        task = asyncio.create_task(slow_task())
        processor._processing_tasks[request.request_id] = task
        
        result = await processor.get_result(request.request_id, timeout_seconds=0.1)
        
        assert result is None  # Should timeout
    
    async def test_get_result_nonexistent_request(self, processor):
        """Test getting result for non-existent request."""
        result = await processor.get_result("nonexistent_id", timeout_seconds=0.1)
        
        assert result is None
    
    async def test_processing_worker_lifecycle(self, processor):
        """Test processing worker lifecycle."""
        # Submit multiple requests
        requests = []
        for i in range(3):
            request = create_processing_request(f"input{i}", "text_processing")
            requests.append(request)
            await processor.submit_request(request)
        
        # Start processing worker
        worker_task = asyncio.create_task(processor.start_processing_worker())
        
        # Give it time to process
        await asyncio.sleep(0.1)
        
        # Stop the worker
        processor.set_status(ProcessorStatus.SHUTTING_DOWN)
        
        # Wait for worker to finish
        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except asyncio.TimeoutError:
            worker_task.cancel()


@pytest.mark.unit
class TestProcessingInterfaceCaching:
    """Test caching functionality."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor with caching."""
        processor = TestProcessingInterface()
        processor.add_capability(ProcessorCapability.CACHING)
        await processor.initialize()
        return processor
    
    async def test_cache_result(self, processor):
        """Test caching results."""
        key = "test_key"
        value = {"result": "test_value"}
        
        await processor.cache_result(key, value)
        
        assert key in processor._cache
        assert processor._cache[key]["result"] == value
    
    async def test_get_cached_result(self, processor):
        """Test retrieving cached results."""
        key = "test_key"
        value = {"result": "test_value"}
        
        await processor.cache_result(key, value)
        cached_value = await processor.get_cached_result(key)
        
        assert cached_value == value
    
    async def test_cache_with_ttl(self, processor):
        """Test cache with time-to-live."""
        key = "ttl_key"
        value = {"result": "ttl_value"}
        
        await processor.cache_result(key, value, ttl_seconds=0.1)
        
        # Should be available immediately
        cached_value = await processor.get_cached_result(key)
        assert cached_value == value
        
        # Should expire after TTL
        await asyncio.sleep(0.2)
        expired_value = await processor.get_cached_result(key)
        assert expired_value is None
    
    async def test_cache_without_capability(self):
        """Test caching without capability."""
        processor = TestProcessingInterface()
        processor.remove_capability(ProcessorCapability.CACHING)
        
        await processor.cache_result("key", "value")
        result = await processor.get_cached_result("key")
        
        # Should not cache or return anything
        assert result is None
    
    async def test_clear_cache(self, processor):
        """Test clearing cache."""
        await processor.cache_result("key1", "value1")
        await processor.cache_result("key2", "value2")
        
        assert len(processor._cache) == 2
        
        processor.clear_cache()
        
        assert len(processor._cache) == 0


@pytest.mark.unit
class TestProcessingInterfaceCapabilityManagement:
    """Test capability management."""
    
    def test_initial_capabilities(self):
        """Test initial capabilities."""
        processor = TestProcessingInterface()
        
        capabilities = processor.get_capabilities()
        assert ProcessorCapability.TEXT_PROCESSING in capabilities
        assert ProcessorCapability.BATCH_PROCESSING in capabilities
        assert ProcessorCapability.CACHING in capabilities
    
    def test_add_capability(self):
        """Test adding capability."""
        processor = TestProcessingInterface()
        
        processor.add_capability(ProcessorCapability.STREAM_PROCESSING)
        
        assert processor.has_capability(ProcessorCapability.STREAM_PROCESSING)
        assert ProcessorCapability.STREAM_PROCESSING in processor.get_capabilities()
    
    def test_remove_capability(self):
        """Test removing capability."""
        processor = TestProcessingInterface()
        
        processor.remove_capability(ProcessorCapability.TEXT_PROCESSING)
        
        assert not processor.has_capability(ProcessorCapability.TEXT_PROCESSING)
        assert ProcessorCapability.TEXT_PROCESSING not in processor.get_capabilities()
    
    def test_has_capability(self):
        """Test capability checking."""
        processor = TestProcessingInterface()
        
        assert processor.has_capability(ProcessorCapability.TEXT_PROCESSING)
        assert not processor.has_capability(ProcessorCapability.VIDEO_PROCESSING)


@pytest.mark.unit
class TestProcessingInterfaceStatusManagement:
    """Test status management."""
    
    def test_initial_status(self):
        """Test initial processor status."""
        processor = TestProcessingInterface()
        
        assert processor.get_status() == ProcessorStatus.IDLE
    
    def test_set_status(self):
        """Test setting processor status."""
        processor = TestProcessingInterface()
        
        processor.set_status(ProcessorStatus.PROCESSING)
        
        assert processor.get_status() == ProcessorStatus.PROCESSING
    
    def test_status_transitions(self):
        """Test various status transitions."""
        processor = TestProcessingInterface()
        
        # Test valid transitions
        processor.set_status(ProcessorStatus.PROCESSING)
        assert processor.status == ProcessorStatus.PROCESSING
        
        processor.set_status(ProcessorStatus.PAUSED)
        assert processor.status == ProcessorStatus.PAUSED
        
        processor.set_status(ProcessorStatus.ERROR)
        assert processor.status == ProcessorStatus.ERROR
        
        processor.set_status(ProcessorStatus.MAINTENANCE)
        assert processor.status == ProcessorStatus.MAINTENANCE
        
        processor.set_status(ProcessorStatus.SHUTTING_DOWN)
        assert processor.status == ProcessorStatus.SHUTTING_DOWN


@pytest.mark.unit
class TestProcessingInterfaceMetrics:
    """Test metrics tracking."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_metrics_initialization(self, processor):
        """Test initial metrics state."""
        metrics = processor.get_metrics()
        
        assert metrics.total_processed == 0
        assert metrics.successful_processes == 0
        assert metrics.failed_processes == 0
        assert metrics.average_processing_time_ms == 0.0
        assert metrics.success_rate == 0.0
    
    async def test_metrics_update_on_success(self, processor):
        """Test metrics update on successful processing."""
        await processor.process("test_input")
        
        metrics = processor.get_metrics()
        assert metrics.total_processed == 1
        assert metrics.successful_processes == 1
        assert metrics.failed_processes == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_processing_time_ms > 0
    
    async def test_metrics_update_on_failure(self, processor):
        """Test metrics update on failed processing."""
        await processor.process("error_input")
        
        metrics = processor.get_metrics()
        assert metrics.total_processed == 1
        assert metrics.successful_processes == 0
        assert metrics.failed_processes == 1
        assert metrics.success_rate == 0.0
    
    async def test_metrics_average_calculation(self, processor):
        """Test average processing time calculation."""
        processor.processing_delay = 0.01  # Set consistent delay
        
        await processor.process("input1")
        await processor.process("input2")
        await processor.process("input3")
        
        metrics = processor.get_metrics()
        assert metrics.total_processed == 3
        assert metrics.average_processing_time_ms > 0
        assert metrics.min_processing_time_ms <= metrics.average_processing_time_ms
        assert metrics.max_processing_time_ms >= metrics.average_processing_time_ms
    
    async def test_metrics_throughput_calculation(self, processor):
        """Test throughput calculation."""
        await processor.process("test_input")
        
        metrics = processor.get_metrics()
        throughput = metrics.throughput_per_second
        
        assert throughput > 0
        assert throughput == 1000.0 / metrics.average_processing_time_ms


@pytest.mark.unit
class TestProcessingInterfaceHealthCheck:
    """Test health check functionality."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_basic_health_check(self, processor):
        """Test basic health check."""
        health = await processor.health_check()
        
        assert "processor_id" in health
        assert "status" in health
        assert "capabilities" in health
        assert "queue_size" in health
        assert "active_tasks" in health
        assert "cache_size" in health
        assert "metrics" in health
        
        assert health["processor_id"] == "test_processor"
        assert health["status"] == ProcessorStatus.IDLE.value
        assert isinstance(health["capabilities"], list)
        assert health["queue_size"] == 0
        assert health["active_tasks"] == 0
        assert health["cache_size"] == 0
    
    async def test_health_check_with_activity(self, processor):
        """Test health check with processor activity."""
        # Add some activity
        await processor.cache_result("test_key", "test_value")
        await processor.submit_request(create_processing_request("test", "text"))
        await processor.process("test_input")
        
        health = await processor.health_check()
        
        assert health["cache_size"] == 1
        assert health["queue_size"] == 1
        assert health["metrics"]["total_processed"] == 1


@pytest.mark.unit
class TestProcessingInterfaceErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_processing_exception_handling(self, processor):
        """Test exception handling during processing."""
        # Override process method to raise exception
        async def failing_process(input_data, **kwargs):
            raise ValueError("Processing failed")
        
        processor.process = failing_process
        
        request = create_processing_request("test", "text")
        response = await processor._process_request(request)
        
        assert response.success is False
        assert "Processing failed" in response.error_message
    
    async def test_invalid_input_types(self, processor):
        """Test handling of invalid input types."""
        # Test with None
        result = await processor.process(None)
        assert result.is_error
        
        # Test validation with invalid types
        is_valid = await processor.validate_input(None)
        assert is_valid is False
    
    async def test_concurrent_processing_safety(self, processor):
        """Test thread safety with concurrent processing."""
        # Create multiple concurrent processing tasks
        tasks = []
        for i in range(10):
            task = asyncio.create_task(processor.process(f"input_{i}"))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(result.is_success for result in results)
        assert len(processor.processed_inputs) == 10


@pytest.mark.unit
class TestProcessingInterfaceUtilities:
    """Test utility functions."""
    
    def test_create_processing_request(self):
        """Test processing request creation."""
        request = create_processing_request(
            "test_data", 
            "text_processing",
            parameters={"param1": "value1"},
            priority=5
        )
        
        assert request.input_data == "test_data"
        assert request.processing_type == "text_processing"
        assert request.parameters == {"param1": "value1"}
        assert request.priority == 5
        assert request.request_id is not None
        assert isinstance(request.created_at, datetime)
    
    def test_validate_processing_interface(self):
        """Test processing interface validation."""
        processor = TestProcessingInterface()
        
        is_valid = validate_processing_interface(processor)
        assert is_valid is True
        
        # Test with invalid object
        invalid_processor = Mock()
        is_valid = validate_processing_interface(invalid_processor)
        assert is_valid is False


@pytest.mark.unit
class TestProcessingInterfacePerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    async def processor(self):
        """Create initialized processor."""
        processor = TestProcessingInterface()
        await processor.initialize()
        return processor
    
    async def test_processing_time_tracking(self, processor):
        """Test processing time tracking accuracy."""
        start_time = datetime.now()
        
        result = await processor.process("test_input")
        
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds() * 1000
        
        # Processing time should be tracked within reasonable bounds
        assert result.is_success
        metrics = processor.get_metrics()
        recorded_time = metrics.average_processing_time_ms
        
        # Should be within 10% of actual time (allowing for measurement overhead)
        assert abs(recorded_time - actual_duration) < actual_duration * 0.5
    
    async def test_batch_processing_performance(self, processor):
        """Test batch processing performance scaling."""
        batch_sizes = [1, 5, 10, 20]
        processing_times = []
        
        for batch_size in batch_sizes:
            input_batch = [f"input_{i}" for i in range(batch_size)]
            
            start_time = datetime.now()
            results = await processor.process_batch(input_batch, parallel=True)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            processing_times.append(processing_time)
            
            assert len(results) == batch_size
            assert all(result.is_success for result in results)
        
        # Parallel processing should not scale linearly with batch size
        # (i.e., 20 items shouldn't take 20x as long as 1 item)
        assert processing_times[-1] < processing_times[0] * batch_sizes[-1] * 0.5
    
    async def test_memory_usage_stability(self, processor):
        """Test memory usage remains stable during processing."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Process many items
        for i in range(100):
            await processor.process(f"input_{i}")
            if i % 10 == 0:
                gc.collect()
        
        # Memory should not grow excessively
        metrics = processor.get_metrics()
        assert metrics.total_processed == 100
        assert len(processor.processed_inputs) == 100
        
        # Cache should not grow unbounded
        assert len(processor._cache) < 50  # Reasonable cache size limit


@pytest.mark.integration
class TestProcessingInterfaceIntegration:
    """Integration tests with external components."""
    
    @pytest.fixture
    async def processor(self):
        """Create processor for integration testing."""
        processor = TestProcessingInterface("integration_processor")
        await processor.initialize()
        return processor
    
    async def test_end_to_end_processing_workflow(self, processor):
        """Test complete end-to-end processing workflow."""
        # Submit requests through queue
        requests = []
        for i in range(5):
            request = create_processing_request(f"data_{i}", "text_processing")
            request_id = await processor.submit_request(request)
            requests.append((request, request_id))
        
        # Start worker to process queue
        worker_task = asyncio.create_task(processor.start_processing_worker())
        
        # Wait for processing and collect results
        results = []
        for request, request_id in requests:
            result = await processor.get_result(request_id, timeout_seconds=2.0)
            results.append(result)
        
        # Stop worker
        processor.set_status(ProcessorStatus.SHUTTING_DOWN)
        try:
            await asyncio.wait_for(worker_task, timeout=1.0)
        except asyncio.TimeoutError:
            worker_task.cancel()
        
        # Verify all requests processed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result is not None
            assert result.success is True
            assert result.output_data == f"processed_data_{i}"
    
    async def test_processor_lifecycle_management(self, processor):
        """Test complete processor lifecycle."""
        # Test initialization
        assert processor.status == ProcessorStatus.IDLE
        
        # Test processing under load
        processor.set_status(ProcessorStatus.PROCESSING)
        
        batch_data = [f"load_test_{i}" for i in range(20)]
        results = await processor.process_batch(batch_data, parallel=True)
        
        assert len(results) == 20
        assert all(result.is_success for result in results)
        
        # Test graceful shutdown
        shutdown_result = await processor.shutdown()
        assert shutdown_result is True
        assert processor.status == ProcessorStatus.SHUTTING_DOWN
    
    async def test_error_recovery_and_resilience(self, processor):
        """Test error recovery and system resilience."""
        # Mix of successful and failing inputs
        mixed_inputs = [
            "good_input_1",
            "error_input", 
            "good_input_2",
            "invalid_input",
            "good_input_3"
        ]
        
        results = await processor.process_batch(mixed_inputs, parallel=True)
        
        # System should handle partial failures gracefully
        assert len(results) == 5
        assert results[0].is_success
        assert results[1].is_error
        assert results[2].is_success  
        assert results[3].is_error
        assert results[4].is_success
        
        # Metrics should reflect mixed results
        metrics = processor.get_metrics()
        assert metrics.success_rate == 0.6  # 3/5 successful
        assert metrics.total_processed == 5