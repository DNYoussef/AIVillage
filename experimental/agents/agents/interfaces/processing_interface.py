"""Standardized Processing Interface.

This module defines the standard interface for processing operations
across all AIVillage components, building on the BaseProcessHandler framework.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from agents.base import ProcessConfig, ProcessResult

from core import ErrorContext

T = TypeVar("T")
U = TypeVar("U")


class ProcessorStatus(Enum):
    """Standard processor status values."""

    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    MAINTENANCE = "maintenance"


class ProcessorCapability(Enum):
    """Standard processor capabilities."""

    # Input/Output capabilities
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_PROCESSING = "video_processing"
    MULTIMODAL_PROCESSING = "multimodal_processing"

    # Processing types
    BATCH_PROCESSING = "batch_processing"
    STREAM_PROCESSING = "stream_processing"
    REAL_TIME_PROCESSING = "real_time_processing"
    PARALLEL_PROCESSING = "parallel_processing"

    # Advanced capabilities
    CHAIN_PROCESSING = "chain_processing"
    CONDITIONAL_PROCESSING = "conditional_processing"
    ADAPTIVE_PROCESSING = "adaptive_processing"
    CACHING = "caching"

    # Quality features
    QUALITY_CONTROL = "quality_control"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"


class ProcessingMode(Enum):
    """Processing execution modes."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"
    PIPELINE = "pipeline"


@dataclass
class ProcessingMetrics:
    """Metrics for processing operations."""

    total_processed: int = 0
    successful_processes: int = 0
    failed_processes: int = 0
    average_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float("inf")
    max_processing_time_ms: float = 0.0
    total_input_size: int = 0
    total_output_size: int = 0
    cache_hit_rate: float = 0.0
    last_activity: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.total_processed == 0:
            return 0.0
        return self.successful_processes / self.total_processed

    @property
    def throughput_per_second(self) -> float:
        """Calculate average throughput per second."""
        if self.average_processing_time_ms == 0:
            return 0.0
        return 1000.0 / self.average_processing_time_ms


@dataclass
class ProcessingRequest:
    """Standard processing request."""

    request_id: str
    input_data: Any
    processing_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout_seconds: float | None = None
    callback: callable | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingResponse:
    """Standard processing response."""

    request_id: str
    output_data: Any
    processing_time_ms: float
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


class ProcessingInterface(ABC, Generic[T, U]):
    """Standard interface for processing operations.

    This interface extends the BaseProcessHandler framework with
    processor-specific functionality for managing processing pipelines.
    """

    def __init__(self, processor_id: str, config: ProcessConfig | None = None) -> None:
        self.processor_id = processor_id
        self.config = config or ProcessConfig()
        self.status = ProcessorStatus.IDLE
        self.capabilities: Set[ProcessorCapability] = set()
        self.metrics = ProcessingMetrics()
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self._cache: dict[str, Any] = {}
        self._processing_tasks: dict[str, asyncio.Task] = {}

    # Core Processing Interface

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the processor."""

    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the processor gracefully."""

    @abstractmethod
    async def process(self, input_data: T, **kwargs) -> ProcessResult[U]:
        """Process input data and return result.

        Args:
            input_data: Data to process
            **kwargs: Additional processing parameters

        Returns:
            ProcessResult containing output data
        """

    @abstractmethod
    async def validate_input(self, input_data: T) -> bool:
        """Validate input data before processing.

        Args:
            input_data: Data to validate

        Returns:
            bool: True if input is valid
        """

    @abstractmethod
    async def estimate_processing_time(self, input_data: T) -> float | None:
        """Estimate processing time for input data.

        Args:
            input_data: Data to estimate for

        Returns:
            Estimated processing time in seconds, or None if cannot estimate
        """

    # Batch Processing

    async def process_batch(
        self, input_batch: list[T], batch_size: int | None = None, parallel: bool = True
    ) -> list[ProcessResult[U]]:
        """Process batch of inputs.

        Args:
            input_batch: List of inputs to process
            batch_size: Size of processing batches (None = process all at once)
            parallel: Whether to process in parallel

        Returns:
            List of processing results
        """
        if not self.has_capability(ProcessorCapability.BATCH_PROCESSING):
            msg = "Processor does not support batch processing"
            raise NotImplementedError(msg)

        if batch_size is None:
            batch_size = len(input_batch)

        results = []

        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i : i + batch_size]

            if parallel and len(batch) > 1:
                # Process batch in parallel
                tasks = [self.process(item) for item in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Convert exceptions to failed ProcessResults
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        from agents.base import ProcessStatus

                        failed_result = ProcessResult(
                            status=ProcessStatus.FAILED, error=str(result)
                        )
                        batch_results[j] = failed_result

                results.extend(batch_results)
            else:
                # Process batch sequentially
                for item in batch:
                    result = await self.process(item)
                    results.append(result)

        return results

    # Queue-based Processing

    async def submit_request(self, request: ProcessingRequest) -> str:
        """Submit processing request to queue.

        Args:
            request: Processing request

        Returns:
            Request ID for tracking
        """
        await self.processing_queue.put(request)
        return request.request_id

    async def get_result(
        self, request_id: str, timeout_seconds: float | None = None
    ) -> ProcessingResponse | None:
        """Get processing result by request ID.

        Args:
            request_id: ID of processing request
            timeout_seconds: Timeout for waiting

        Returns:
            Processing response or None if not ready/timeout
        """
        # Check if processing task exists
        if request_id in self._processing_tasks:
            task = self._processing_tasks[request_id]

            try:
                if timeout_seconds:
                    result = await asyncio.wait_for(task, timeout=timeout_seconds)
                else:
                    result = await task

                # Clean up completed task
                del self._processing_tasks[request_id]
                return result

            except asyncio.TimeoutError:
                return None

        return None

    async def start_processing_worker(self) -> None:
        """Start background worker for processing queued requests."""
        while self.status != ProcessorStatus.SHUTTING_DOWN:
            try:
                # Get next request with timeout
                request = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )

                # Create processing task
                task = asyncio.create_task(self._process_request(request))
                self._processing_tasks[request.request_id] = task

            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except Exception as e:
                # Log error and continue
                print(f"Error in processing worker: {e}")
                await asyncio.sleep(1.0)

    async def _process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process individual request."""
        start_time = datetime.now()

        try:
            # Process the request
            result = await self.process(request.input_data, **request.parameters)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update metrics
            self._update_metrics(processing_time, True)

            # Call callback if provided
            if request.callback:
                if asyncio.iscoroutinefunction(request.callback):
                    await request.callback(result)
                else:
                    request.callback(result)

            return ProcessingResponse(
                request_id=request.request_id,
                output_data=result.data if result.is_success else None,
                processing_time_ms=processing_time,
                success=result.is_success,
                error_message=result.error if result.is_error else None,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)

            return ProcessingResponse(
                request_id=request.request_id,
                output_data=None,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e),
            )

    # Caching

    async def cache_result(
        self, key: str, result: Any, ttl_seconds: float | None = None
    ) -> None:
        """Cache processing result."""
        if self.has_capability(ProcessorCapability.CACHING):
            self._cache[key] = {
                "result": result,
                "timestamp": datetime.now(),
                "ttl_seconds": ttl_seconds,
            }

    async def get_cached_result(self, key: str) -> Any | None:
        """Get cached result if available and not expired."""
        if not self.has_capability(ProcessorCapability.CACHING):
            return None

        if key not in self._cache:
            return None

        cache_entry = self._cache[key]

        # Check TTL if specified
        if cache_entry["ttl_seconds"]:
            age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            if age > cache_entry["ttl_seconds"]:
                del self._cache[key]
                return None

        return cache_entry["result"]

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    # Capability Management

    def has_capability(self, capability: ProcessorCapability) -> bool:
        """Check if processor has specific capability."""
        return capability in self.capabilities

    def add_capability(self, capability: ProcessorCapability) -> None:
        """Add capability to processor."""
        self.capabilities.add(capability)

    def remove_capability(self, capability: ProcessorCapability) -> None:
        """Remove capability from processor."""
        self.capabilities.discard(capability)

    def get_capabilities(self) -> Set[ProcessorCapability]:
        """Get all processor capabilities."""
        return self.capabilities.copy()

    # Status Management

    def get_status(self) -> ProcessorStatus:
        """Get current processor status."""
        return self.status

    def set_status(self, status: ProcessorStatus) -> None:
        """Set processor status."""
        self.status = status

    # Metrics

    def _update_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Update processing metrics."""
        self.metrics.total_processed += 1

        if success:
            self.metrics.successful_processes += 1
        else:
            self.metrics.failed_processes += 1

        # Update timing statistics
        if self.metrics.total_processed == 1:
            self.metrics.average_processing_time_ms = processing_time_ms
        else:
            # Running average
            n = self.metrics.total_processed
            self.metrics.average_processing_time_ms = (
                self.metrics.average_processing_time_ms * (n - 1) + processing_time_ms
            ) / n

        self.metrics.min_processing_time_ms = min(
            self.metrics.min_processing_time_ms, processing_time_ms
        )
        self.metrics.max_processing_time_ms = max(
            self.metrics.max_processing_time_ms, processing_time_ms
        )

        self.metrics.last_activity = datetime.now()

    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        return self.metrics

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "processor_id": self.processor_id,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "queue_size": self.processing_queue.qsize(),
            "active_tasks": len(self._processing_tasks),
            "cache_size": len(self._cache),
            "metrics": {
                "total_processed": self.metrics.total_processed,
                "success_rate": self.metrics.success_rate,
                "average_processing_time_ms": self.metrics.average_processing_time_ms,
                "throughput_per_second": self.metrics.throughput_per_second,
            },
        }

    def create_error_context(self, operation: str, **metadata) -> ErrorContext:
        """Create error context for processing operations."""
        from core import ErrorContext

        return ErrorContext(
            component=f"Processor.{self.processor_id}",
            operation=operation,
            metadata=metadata,
        )


# Utility functions


def create_processing_request(
    input_data: Any, processing_type: str, **kwargs
) -> ProcessingRequest:
    """Create processing request with auto-generated ID.

    Args:
        input_data: Data to process
        processing_type: Type of processing
        **kwargs: Additional parameters

    Returns:
        ProcessingRequest instance
    """
    import uuid

    return ProcessingRequest(
        request_id=str(uuid.uuid4()),
        input_data=input_data,
        processing_type=processing_type,
        **kwargs,
    )


def validate_processing_interface(processor: Any) -> bool:
    """Validate that an object implements ProcessingInterface.

    Args:
        processor: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "initialize",
        "shutdown",
        "process",
        "validate_input",
        "estimate_processing_time",
        "health_check",
    ]

    for method in required_methods:
        if not hasattr(processor, method) or not callable(getattr(processor, method)):
            return False

    required_attributes = ["processor_id", "status", "capabilities", "metrics"]
    return all(hasattr(processor, attr) for attr in required_attributes)
