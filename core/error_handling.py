"""Core error handling module for AIVillage.
Provides comprehensive error handling with categories, severity levels, and context.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import functools
import logging
import traceback
from typing import Any, TypeVar, cast

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
    recipient: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: str = field(default_factory=lambda: str(datetime.now(timezone.utc).timestamp()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "type": self.type.value,
            "content": self.content,
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary format."""
        return cls(
            type=MessageType(data["type"]),
            content=data["content"],
            sender=data["sender"],
            recipient=data["recipient"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            message_id=data.get("message_id", str(datetime.now(timezone.utc).timestamp())),
            metadata=data.get("metadata", {}),
        )


class StandardCommunicationProtocol:
    """Standard protocol for communication between components."""

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
    def create_error(
        error: str, sender: str, recipient: str, metadata: dict[str, Any] | None = None
    ) -> Message:
        """Create an error message."""
        return Message(
            type=MessageType.ERROR,
            content={"error": error},
            sender=sender,
            recipient=recipient,
            metadata=metadata or {},
        )


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
            self.timestamp = datetime.now(timezone.utc).isoformat()
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
    ):
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


# Legacy alias for backward compatibility
AIVillageError = AIVillageException


class ServiceException(AIVillageException):
    """Exception for service-level errors."""


class ValidationException(AIVillageException):
    """Exception for validation errors."""

    def __init__(self, message: str, field: str, value: Any, **kwargs):
        context = kwargs.pop("context", None) or ErrorContext(
            component="validation",
            operation="validate_field",
            details={"field": field, "value": value},
        )
        super().__init__(
            message, category=ErrorCategory.VALIDATION, context=context, **kwargs
        )


class NetworkException(AIVillageException):
    """Exception for network-related errors."""

    def __init__(
        self, message: str, url: str, status_code: int | None = None, **kwargs
    ):
        context = kwargs.pop("context", None) or ErrorContext(
            component="network",
            operation="request",
            details={"url": url, "status_code": status_code},
        )
        super().__init__(
            message, category=ErrorCategory.NETWORK, context=context, **kwargs
        )


class ConfigurationException(AIVillageException):
    """Exception for configuration-related errors."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        context = kwargs.pop("context", None) or ErrorContext(
            component="configuration",
            operation="load_config",
            details={"config_key": config_key},
        )
        super().__init__(
            message, category=ErrorCategory.CONFIGURATION, context=context, **kwargs
        )


class ErrorContextManager:
    """Context manager for error handling with automatic context capture."""

    def __init__(
        self, component: str, operation: str, details: dict[str, Any] | None = None
    ):
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
                stack_trace="".join(
                    traceback.format_exception(exc_type, exc_val, exc_tb)
                ),
            )
            # Re-raise the exception
            return False


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

    def __init__(self):
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

    def get_errors_by_category(
        self, category: ErrorCategory
    ) -> list[AIVillageException]:
        """Get all errors for a specific category."""
        return [e for e in self.errors if e.category == category]

    def get_errors_by_severity(
        self, severity: ErrorSeverity
    ) -> list[AIVillageException]:
        """Get all errors for a specific severity."""
        return [e for e in self.errors if e.severity == severity]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "categories": list(set(e.category.value for e in self.errors)),
            "severities": list(set(e.severity.value for e in self.errors)),
        }


# Global error handler instance
error_handler = ErrorHandler()
