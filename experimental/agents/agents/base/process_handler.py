"""Base Process Handler for Standardized Processing Operations.

This module provides standardized base classes and interfaces for all process_*
methods across the AIVillage codebase to eliminate duplication and ensure
consistent error handling, logging, and processing patterns.
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time
import traceback
from typing import Any, Generic, TypeVar, Union

from core.error_handling import Message, MessageType

# Type definitions
T = TypeVar("T")
ProcessInput = Union[str, dict[str, Any], Message, Any]
ProcessOutput = Union[str, dict[str, Any], Any]


class ProcessStatus(Enum):
    """Standard processing status codes."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ProcessResult:
    """Standardized result wrapper for all processing operations."""

    status: ProcessStatus
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.status == ProcessStatus.COMPLETED

    @property
    def is_error(self) -> bool:
        return self.status in [ProcessStatus.FAILED, ProcessStatus.TIMEOUT]


@dataclass
class ProcessConfig:
    """Configuration for processing operations."""

    timeout_seconds: float | None = None
    retry_attempts: int = 0
    retry_delay_seconds: float = 1.0
    enable_logging: bool = True
    enable_metrics: bool = True
    validation_enabled: bool = True


class BaseProcessHandler(ABC, Generic[T]):
    """Abstract base class for all processing operations.

    Provides standardized:
    - Error handling and logging
    - Timeout management
    - Retry logic
    - Input validation
    - Performance metrics
    - Result standardization
    """

    def __init__(
        self,
        name: str,
        config: ProcessConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.name = name
        self.config = config or ProcessConfig()
        self.logger = logger or logging.getLogger(f"ProcessHandler.{name}")
        self._metrics: dict[str, Any] = {}

    async def process(self, input_data: ProcessInput, **kwargs) -> ProcessResult:
        """Main processing entry point with standardized error handling.

        Args:
            input_data: Input to be processed
            **kwargs: Additional processing parameters

        Returns:
            ProcessResult with standardized output format
        """
        start_time = time.time()

        try:
            # Input validation
            if self.config.validation_enabled:
                await self._validate_input(input_data)

            # Log processing start
            if self.config.enable_logging:
                self.logger.info(f"Starting {self.name} processing")

            # Execute with timeout and retries
            result_data = await self._execute_with_retry(input_data, **kwargs)

            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            if self.config.enable_metrics:
                self._update_metrics(processing_time, True)

            # Log success
            if self.config.enable_logging:
                self.logger.info(
                    f"Completed {self.name} processing in {processing_time:.2f}ms"
                )

            return ProcessResult(
                status=ProcessStatus.COMPLETED,
                data=result_data,
                processing_time_ms=processing_time,
                metadata={"processor": self.name},
            )

        except asyncio.TimeoutError:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Processing timeout after {self.config.timeout_seconds}s"
            self.logger.exception(error_msg)
            if self.config.enable_metrics:
                self._update_metrics(processing_time, False)

            return ProcessResult(
                status=ProcessStatus.TIMEOUT,
                error=error_msg,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Processing failed: {e!s}"
            self.logger.exception(f"{error_msg}\n{traceback.format_exc()}")
            if self.config.enable_metrics:
                self._update_metrics(processing_time, False)

            return ProcessResult(
                status=ProcessStatus.FAILED,
                error=error_msg,
                processing_time_ms=processing_time,
            )

    async def _execute_with_retry(self, input_data: ProcessInput, **kwargs) -> T:
        """Execute processing with retry logic and timeout."""
        last_exception = None

        for attempt in range(self.config.retry_attempts + 1):
            try:
                if self.config.timeout_seconds:
                    return await asyncio.wait_for(
                        self._process_impl(input_data, **kwargs),
                        timeout=self.config.timeout_seconds,
                    )
                return await self._process_impl(input_data, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {self.config.retry_delay_seconds}s"
                    )
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    raise

        # Should never reach here, but for type safety
        raise last_exception or Exception("Unknown error in retry logic")

    @abstractmethod
    async def _process_impl(self, input_data: ProcessInput, **kwargs) -> T:
        """Abstract method for actual processing implementation.

        Subclasses must implement this method with their specific logic.
        """

    async def _validate_input(self, input_data: ProcessInput) -> None:
        """Validate input data. Override in subclasses for specific validation.

        Args:
            input_data: Input to validate

        Raises:
            ValueError: If input is invalid
        """
        if input_data is None:
            msg = "Input data cannot be None"
            raise ValueError(msg)

    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update internal performance metrics."""
        if "total_processed" not in self._metrics:
            self._metrics["total_processed"] = 0
            self._metrics["success_count"] = 0
            self._metrics["error_count"] = 0
            self._metrics["avg_processing_time"] = 0.0
            self._metrics["min_processing_time"] = float("inf")
            self._metrics["max_processing_time"] = 0.0

        self._metrics["total_processed"] += 1

        if success:
            self._metrics["success_count"] += 1
        else:
            self._metrics["error_count"] += 1

        # Update timing statistics
        total = self._metrics["total_processed"]
        current_avg = self._metrics["avg_processing_time"]
        self._metrics["avg_processing_time"] = (
            current_avg * (total - 1) + processing_time
        ) / total
        self._metrics["min_processing_time"] = min(
            self._metrics["min_processing_time"], processing_time
        )
        self._metrics["max_processing_time"] = max(
            self._metrics["max_processing_time"], processing_time
        )

    @property
    def metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return self._metrics.copy()

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self._metrics.get("total_processed", 0)
        if total == 0:
            return 0.0
        return (self._metrics.get("success_count", 0) / total) * 100


class QueryProcessor(BaseProcessHandler[str]):
    """Standardized query processing handler."""

    async def _process_impl(self, input_data: ProcessInput, **kwargs) -> str:
        """Process query input and return string response."""
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = input_data.get("query", str(input_data))
        else:
            query = str(input_data)

        # Subclasses should override this for specific query processing
        return await self._process_query(query, **kwargs)

    @abstractmethod
    async def _process_query(self, query: str, **kwargs) -> str:
        """Abstract method for query-specific processing."""


class TaskProcessor(BaseProcessHandler[dict[str, Any]]):
    """Standardized task processing handler."""

    async def _process_impl(self, input_data: ProcessInput, **kwargs) -> dict[str, Any]:
        """Process task input and return structured response."""
        if isinstance(input_data, dict):
            task = input_data
        elif hasattr(input_data, "to_dict"):
            task = input_data.to_dict()
        else:
            task = {"content": str(input_data)}

        return await self._process_task(task, **kwargs)

    @abstractmethod
    async def _process_task(self, task: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Abstract method for task-specific processing."""


class MessageProcessor(BaseProcessHandler[dict[str, Any]]):
    """Standardized message processing handler."""

    async def _validate_input(self, input_data: ProcessInput) -> None:
        """Validate message input."""
        await super()._validate_input(input_data)

        if isinstance(input_data, Message):
            return  # Message objects are always valid

        if isinstance(input_data, dict):
            required_fields = ["type", "content"]
            missing = [field for field in required_fields if field not in input_data]
            if missing:
                msg = f"Message missing required fields: {missing}"
                raise ValueError(msg)
        else:
            msg = "Message input must be Message object or dict"
            raise ValueError(msg)

    async def _process_impl(self, input_data: ProcessInput, **kwargs) -> dict[str, Any]:
        """Process message input and return structured response."""
        if isinstance(input_data, Message):
            message = input_data
        elif isinstance(input_data, dict):
            message = Message(
                type=MessageType(input_data["type"]),
                content=input_data["content"],
                sender=input_data.get("sender", "unknown"),
                receiver=input_data.get("receiver", "unknown"),
            )
        else:
            msg = "Invalid message input type"
            raise ValueError(msg)

        return await self._process_message(message, **kwargs)

    @abstractmethod
    async def _process_message(self, message: Message, **kwargs) -> dict[str, Any]:
        """Abstract method for message-specific processing."""


class BatchProcessor(BaseProcessHandler[list[T]]):
    """Standardized batch processing handler."""

    def __init__(
        self,
        name: str,
        batch_size: int = 10,
        parallel_processing: bool = True,
        config: ProcessConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name, config, logger)
        self.batch_size = batch_size
        self.parallel_processing = parallel_processing

    async def _process_impl(self, input_data: ProcessInput, **kwargs) -> list[T]:
        """Process batch input with parallel or sequential processing."""
        if not isinstance(input_data, list):
            msg = "Batch processor requires list input"
            raise ValueError(msg)

        items = input_data
        results = []

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            if self.parallel_processing:
                batch_results = await asyncio.gather(
                    *[self._process_single_item(item, **kwargs) for item in batch],
                    return_exceptions=True,
                )
            else:
                batch_results = []
                for item in batch:
                    try:
                        result = await self._process_single_item(item, **kwargs)
                        batch_results.append(result)
                    except Exception as e:
                        batch_results.append(e)

            results.extend(batch_results)

        return results

    @abstractmethod
    async def _process_single_item(self, item: Any, **kwargs) -> T:
        """Abstract method for processing individual items."""


# Utility functions for migration
def create_query_processor(
    name: str, process_func: Callable[[str], str], config: ProcessConfig | None = None
) -> QueryProcessor:
    """Factory function to create QueryProcessor from existing function.

    Useful for migrating existing process_query methods.
    """

    class FunctionQueryProcessor(QueryProcessor):
        async def _process_query(self, query: str, **kwargs) -> str:
            if asyncio.iscoroutinefunction(process_func):
                return await process_func(query)
            return process_func(query)

    return FunctionQueryProcessor(name, config)


def create_task_processor(
    name: str,
    process_func: Callable[[dict[str, Any]], dict[str, Any]],
    config: ProcessConfig | None = None,
) -> TaskProcessor:
    """Factory function to create TaskProcessor from existing function.

    Useful for migrating existing process_task methods.
    """

    class FunctionTaskProcessor(TaskProcessor):
        async def _process_task(self, task: dict[str, Any], **kwargs) -> dict[str, Any]:
            if asyncio.iscoroutinefunction(process_func):
                return await process_func(task)
            return process_func(task)

    return FunctionTaskProcessor(name, config)


# Migration helper decorator
def standardized_process(
    processor_type: str = "query",
    config: ProcessConfig | None = None,
    name: str | None = None,
):
    """Decorator to wrap existing process methods with standardized handling.

    Args:
        processor_type: Type of processor ("query", "task", "message")
        config: Processing configuration
        name: Processor name (defaults to function name)

    Example:
        @standardized_process("query")
        async def process_query(self, query: str) -> str:
            return f"Processed: {query}"
    """

    def decorator(func):
        async def wrapper(self, input_data, **kwargs):
            processor_name = name or f"{self.__class__.__name__}.{func.__name__}"

            if processor_type == "query":
                processor = create_query_processor(processor_name, func, config)
            elif processor_type == "task":
                processor = create_task_processor(processor_name, func, config)
            else:
                msg = f"Unsupported processor type: {processor_type}"
                raise ValueError(msg)

            result = await processor.process(input_data, **kwargs)

            if result.is_error:
                raise Exception(result.error)

            return result.data

        return wrapper

    return decorator
