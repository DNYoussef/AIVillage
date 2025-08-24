"""Core error handling module for AIVillage.
Provides comprehensive error handling with categories, severity levels, and context.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import functools
import logging
import traceback
from typing import Any, TypeVar, cast


# Priority enum defined locally to avoid circular imports
# This could be moved to a shared types module in the future
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    NORMAL = 2  # Alias for MEDIUM
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


# Placeholder for protocol - will be injected at runtime to avoid circular import
BaseCommProtocol = None

# Configure logging
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categorization of error types for better error handling and reporting."""

    # System-level errors
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    COMMUNICATION = "communication"

    # Processing errors
    PROCESSING = "processing"
    VALIDATION = "validation"
    DECISION = "decision"
    ANALYSIS = "analysis"

    # Resource errors
    ACCESS = "access"
    RECORDING = "recording"
    CREATION = "creation"

    # Feature-specific errors
    NOT_IMPLEMENTED = "not_implemented"
    OPTIMIZATION = "optimization"
    EVOLUTION = "evolution"

    # General errors
    UNKNOWN = "unknown"
    NETWORK = "network"
    TIMEOUT = "timeout"


class ErrorSeverity(Enum):
    """Severity levels for error prioritization and handling."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MessageType(Enum):
    """Types of messages for communication protocol."""

    REQUEST = "request"
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    """Standard message format for inter-component communication."""

    type: MessageType
    content: Any
    sender: str
    receiver: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    id: str = field(default_factory=lambda: str(datetime.now(UTC).timestamp()))
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "type": self.type.value,
            "content": self.content,
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp,
            "id": self.id,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary format."""
        return cls(
            type=MessageType(data["type"]),
            content=data["content"],
            sender=data["sender"],
            receiver=data["receiver"],
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            id=data.get("id", str(datetime.now(UTC).timestamp())),
            metadata=data.get("metadata", {}),
            priority=Priority(data.get("priority", Priority.MEDIUM.value)),
            parent_id=data.get("parent_id"),
        )


class StandardCommunicationProtocol:
    """Standard protocol for communication between components.

    This class extends the full implementation from :mod:`src.communications` and
    adds convenience helpers for creating structured messages.
    """

    def __init__(self, agent_id: str = "anon", port: int = 8888) -> None:  # pragma: no cover - simple wrapper
        super().__init__(agent_id=agent_id, port=port)
        self.inboxes: dict[str, asyncio.PriorityQueue[tuple[int, Message]]] = {}
        self._running = True

    @staticmethod
    def create_request(
        content: Any,
        sender: str,
        recipient: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Create a request message."""
        return Message(
            type=MessageType.REQUEST,
            content=content,
            sender=sender,
            recipient=recipient,
            metadata=metadata or {},
        )

    @staticmethod
    def create_response(
        content: Any,
        sender: str,
        recipient: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Create a response message."""
        return Message(
            type=MessageType.RESPONSE,
            content=content,
            sender=sender,
            recipient=recipient,
            metadata=metadata or {},
        )

    @staticmethod
    def create_error(error: str, sender: str, recipient: str, metadata: dict[str, Any] | None = None) -> Message:
        """Create an error message."""
        return Message(
            type=MessageType.ERROR,
            content={"error": error},
            sender=sender,
            recipient=recipient,
            metadata=metadata or {},
        )

    # Compatibility wrappers -------------------------------------------------
    def subscribe(self, agent_id: str, handler: Callable[[Message], Any]) -> None:
        self.message_handlers[agent_id] = handler

    def unsubscribe(self, agent_id: str, handler: Callable[[Message], Any]) -> None:
        self.message_handlers.pop(agent_id, None)

    async def send_message(self, message: Message) -> None:
        self._store_message(message.sender, message.to_dict(), "sent")
        self._store_message(message.receiver, message.to_dict(), "received")
        queue = self.inboxes.setdefault(message.receiver, asyncio.PriorityQueue())
        await queue.put((-message.priority.value, message))
        handler = self.message_handlers.get(message.receiver)
        if handler is not None:
            await handler(message)

    async def receive_message(self, agent_id: str) -> Message:
        queue = self.inboxes.setdefault(agent_id, asyncio.PriorityQueue())
        _p, msg = await queue.get()
        return msg

    async def send_and_wait(self, message: Message, timeout: float = 1.0) -> Message:
        await self.send_message(message)
        return await asyncio.wait_for(self.receive_message(message.sender), timeout)

    async def broadcast(self, sender: str, message_type: MessageType, content: dict[str, Any]) -> None:
        for agent_id in list(self.message_handlers.keys()):
            msg = Message(
                type=message_type,
                sender=sender,
                receiver=agent_id,
                content=content,
            )
            await self.send_message(msg)

    async def process_messages(self, handler: Callable[[Message], Any]) -> None:
        while self._running:
            for queue in list(self.inboxes.values()):
                if not queue.empty():
                    _p, msg = await queue.get()
                    await handler(msg)
            await asyncio.sleep(0.01)

    def get_message_history(self, agent_id: str, message_type: MessageType | None = None) -> list[Message]:
        history = self.message_history.get(agent_id, [])
        msgs = [
            Message.from_dict(entry["message"]) for entry in history if isinstance(entry, dict) and "message" in entry
        ]
        if message_type is not None:
            msgs = [m for m in msgs if m.type == message_type]
        return msgs


@dataclass
class ErrorContext:
    """Context information for errors to aid debugging and resolution."""

    component: str
    operation: str
    details: dict[str, Any]
    stack_trace: str | None = None
    timestamp: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()


class AIVillageException(Exception):
    """Base exception class for AIVillage with structured error information."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: ErrorContext | None = None,
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.original_exception = original_exception

        # Log the error
        logger.error(
            f"AIVillageException: {message}",
            extra={
                "category": category.value,
                "severity": severity.value,
                "context": context.__dict__ if context else None,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": f"{self.category.value.upper()}_{self.severity.value.upper()}",
            "category": self.category.value.upper(),
            "severity": self.severity.value.upper(),
            "timestamp": (self.context.timestamp if self.context else datetime.now(UTC).isoformat()),
            "context": self.context.__dict__ if self.context else None,
            "original_exception": (str(self.original_exception) if self.original_exception else None),
        }


# Legacy alias for backward compatibility
AIVillageError = AIVillageException


class ServiceException(AIVillageException):
    """Exception for service-level errors."""


class ValidationException(AIVillageException):
    """Exception for validation errors."""

    def __init__(self, message: str, field: str, value: Any, **kwargs) -> None:
        context = kwargs.pop("context", None) or ErrorContext(
            component="validation",
            operation="validate_field",
            details={"field": field, "value": value},
        )
        super().__init__(message, category=ErrorCategory.VALIDATION, context=context, **kwargs)


class NetworkException(AIVillageException):
    """Exception for network-related errors."""

    def __init__(self, message: str, url: str, status_code: int | None = None, **kwargs) -> None:
        context = kwargs.pop("context", None) or ErrorContext(
            component="network",
            operation="request",
            details={"url": url, "status_code": status_code},
        )
        super().__init__(message, category=ErrorCategory.NETWORK, context=context, **kwargs)


class ConfigurationException(AIVillageException):
    """Exception for configuration-related errors."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs) -> None:
        context = kwargs.pop("context", None) or ErrorContext(
            component="configuration",
            operation="load_config",
            details={"config_key": config_key},
        )
        super().__init__(message, category=ErrorCategory.CONFIGURATION, context=context, **kwargs)


class ErrorContextManager:
    """Context manager for error handling with automatic context capture."""

    def __init__(self, component: str, operation: str, details: dict[str, Any] | None = None) -> None:
        self.component = component
        self.operation = operation
        self.details = details or {}
        self.context = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception occurred, create context
            self.context = ErrorContext(
                component=self.component,
                operation=self.operation,
                details=self.details,
                stack_trace="".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
            )
            # Re-raise the exception
            return False
        return None


def get_component_logger(component_name: str) -> logging.Logger:
    """Get a logger for a specific component."""
    return logging.getLogger(f"aivillage.{component_name}")


def migrate_from_legacy_exception(legacy_exception: Exception) -> AIVillageException:
    """Migrate from legacy exception to new AIVillageException."""
    if isinstance(legacy_exception, AIVillageException):
        return legacy_exception

    return AIVillageException(
        message=str(legacy_exception),
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.ERROR,
        original_exception=legacy_exception,
    )


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def safe_execute(
    func: Callable[..., T] | None = None,
    *,
    component: str = "unknown",
    operation: str = "unknown",
    fallback: T | None = None,
    raise_on_error: bool = True,
) -> T | None:
    """Safely execute a function with error handling."""
    if func is None:
        return None

    try:
        if asyncio.iscoroutinefunction(func):
            # Handle async functions
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        details={"args": str(args), "kwargs": str(kwargs)},
                    )
                    error = AIVillageException(
                        message=f"Error in {operation}: {e!s}",
                        category=ErrorCategory.PROCESSING,
                        context=context,
                        original_exception=e,
                    )
                    if raise_on_error:
                        raise error
                    return fallback

            return async_wrapper

        # Handle sync functions
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    details={"args": str(args), "kwargs": str(kwargs)},
                )
                error = AIVillageException(
                    message=f"Error in {operation}: {e!s}",
                    category=ErrorCategory.PROCESSING,
                    context=context,
                    original_exception=e,
                )
                if raise_on_error:
                    raise error
                return fallback

        return sync_wrapper
    except Exception as e:
        context = ErrorContext(component=component, operation=operation, details={})
        error = AIVillageException(
            message=f"Error setting up safe_execute: {e!s}",
            category=ErrorCategory.INITIALIZATION,
            context=context,
            original_exception=e,
        )
        if raise_on_error:
            raise error
        return fallback


def with_error_handling(*args, **kwargs) -> Callable[[F], F]:
    """Decorator for adding error handling to functions.

    Usage patterns:
    1. @with_error_handling(component="MyComponent", operation="my_operation")
    2. @with_error_handling(retries=2, context={"component": "MyComponent", "method": "my_method"})
    """
    # Handle both old and new usage patterns
    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
        # Old pattern: @with_error_handling("component", "operation")
        component, operation = args
        category = kwargs.pop("category", ErrorCategory.PROCESSING)
        severity = kwargs.pop("severity", ErrorSeverity.ERROR)
        fallback = kwargs.pop("fallback", None)
        raise_on_error = kwargs.pop("raise_on_error", True)
    elif kwargs:
        # New pattern: @with_error_handling(retries=2, context={...})
        context_data = kwargs.pop("context", {})
        component = context_data.get("component", "unknown")
        operation = context_data.get("method", "unknown")
        category = kwargs.pop("category", ErrorCategory.PROCESSING)
        severity = kwargs.pop("severity", ErrorSeverity.ERROR)
        fallback = kwargs.pop("fallback", None)
        raise_on_error = kwargs.pop("raise_on_error", True)
    else:
        # Default fallback
        component = "unknown"
        operation = "unknown"
        category = ErrorCategory.PROCESSING
        severity = ErrorSeverity.ERROR
        fallback = None
        raise_on_error = True

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorContext(
                    component=component,
                    operation=operation,
                    details={"args": str(args), "kwargs": str(kwargs)},
                )
                error = AIVillageException(
                    message=f"Error in {operation}: {e!s}",
                    category=category,
                    severity=severity,
                    context=error_context,
                    original_exception=e,
                )
                if raise_on_error:
                    raise error
                return fallback

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = ErrorContext(
                    component=component,
                    operation=operation,
                    details={"args": str(args), "kwargs": str(kwargs)},
                )
                error = AIVillageException(
                    message=f"Error in {operation}: {e!s}",
                    category=category,
                    severity=severity,
                    context=error_context,
                    original_exception=e,
                )
                if raise_on_error:
                    raise error
                return fallback

        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)

    return decorator


class ErrorHandler:
    """Centralized error handler for tracking and managing errors."""

    def __init__(self) -> None:
        self.errors: list[AIVillageException] = []
        self.error_counts: dict[str, int] = {}

    def handle_error(self, error: AIVillageException) -> None:
        """Handle an error by logging and tracking it."""
        self.errors.append(error)
        key = f"{error.category.value}:{error.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        logger.error(
            f"Handled error: {error.message}",
            extra={
                "category": error.category.value,
                "severity": error.severity.value,
                "count": self.error_counts[key],
            },
        )

    def get_errors_by_category(self, category: ErrorCategory) -> list[AIVillageException]:
        """Get all errors for a specific category."""
        return [e for e in self.errors if e.category == category]

    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[AIVillageException]:
        """Get all errors for a specific severity."""
        return [e for e in self.errors if e.severity == severity]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "categories": list({e.category.value for e in self.errors}),
            "severities": list({e.severity.value for e in self.errors}),
        }


# Global error handler instance
error_handler = ErrorHandler()
