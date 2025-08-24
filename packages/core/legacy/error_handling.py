"""
Legacy Error Handling Module - Agent 6 Compatibility Fix
Provides error handling functions required by the agent system
"""

from collections.abc import Callable
from functools import wraps
import logging
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)


class AIVillageError(Exception):
    """Base exception for AIVillage errors."""

    def __init__(self, message: str, error_code: str = "UNKNOWN", context: dict = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(message)


class AgentError(AIVillageError):
    """Agent-specific errors."""

    pass


class CommunicationError(AIVillageError):
    """Communication system errors."""

    pass


class ProcessingError(AIVillageError):
    """Message processing errors."""

    pass


class ErrorContext:
    """Context manager for error handling with additional context."""

    def __init__(self, operation_name: str, context: dict = None):
        self.operation_name = operation_name
        self.context = context or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            duration = time.time() - self.start_time if self.start_time else 0
            log_error(
                exc_val, context={**self.context, "operation": self.operation_name, "duration_ms": duration * 1000}
            )
        return False  # Don't suppress exceptions


def handle_errors(fallback_value: Any = None, log_errors: bool = True):
    """
    Decorator to handle errors in agent methods.

    Args:
        fallback_value: Value to return if error occurs
        log_errors: Whether to log errors
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.error(traceback.format_exc())

                # Return fallback value or raise based on error type
                if isinstance(e, AIVillageError):
                    raise
                else:
                    return fallback_value

        return wrapper

    return decorator


def safe_execute(func: Callable, *args, fallback_value: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        fallback_value: Value to return on error
        **kwargs: Function keyword arguments

    Returns:
        Function result or fallback_value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return fallback_value


def validate_input(value: Any, expected_type: type, field_name: str = "input") -> Any:
    """
    Validate input value type.

    Args:
        value: Value to validate
        expected_type: Expected type
        field_name: Field name for error messages

    Returns:
        Validated value

    Raises:
        ProcessingError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise ProcessingError(
            f"Invalid {field_name}: expected {expected_type.__name__}, got {type(value).__name__}",
            error_code="INVALID_INPUT",
        )
    return value


def create_error_response(error: Exception, request_id: str = None) -> dict:
    """
    Create standardized error response.

    Args:
        error: Exception that occurred
        request_id: Optional request identifier

    Returns:
        Error response dictionary
    """
    if isinstance(error, AIVillageError):
        error_code = error.error_code
        message = error.message
        context = error.context
    else:
        error_code = "INTERNAL_ERROR"
        message = str(error)
        context = {}

    response = {"success": False, "error": {"code": error_code, "message": message, "type": error.__class__.__name__}}

    if request_id:
        response["request_id"] = request_id

    if context:
        response["error"]["context"] = context

    return response


def log_error(error: Exception, context: dict = None, level: str = "error") -> None:
    """
    Log error with context information.

    Args:
        error: Exception to log
        context: Additional context information
        level: Logging level (error, warning, info)
    """
    context_str = ""
    if context:
        context_str = f" | Context: {context}"

    log_method = getattr(logger, level.lower(), logger.error)
    log_method(f"Error: {error.__class__.__name__}: {error}{context_str}")

    if level.lower() == "error":
        logger.error(traceback.format_exc())


# Backward compatibility aliases
ErrorHandler = handle_errors
SafeExecute = safe_execute
ValidateInput = validate_input
