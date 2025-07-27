"""Core module for AIVillage.
Provides essential components for error handling, communication, and utilities.
"""

from .error_handling import (
    AIVillageError,
    AIVillageException,
    ConfigurationException,
    ErrorCategory,
    ErrorContext,
    ErrorContextManager,
    ErrorHandler,
    ErrorSeverity,
    Message,
    MessageType,
    NetworkException,
    ServiceException,
    StandardCommunicationProtocol,
    ValidationException,
    error_handler,
    get_component_logger,
    migrate_from_legacy_exception,
    safe_execute,
    with_error_handling,
)

from .communication import (
    AgentMessage,
    AgentMessageType,
    AgentCommunicationProtocol,
    Priority,
)

__all__ = [
    "AIVillageError",
    "AIVillageException",
    "ConfigurationException",
    "ErrorCategory",
    "ErrorContext",
    "ErrorContextManager",
    "ErrorHandler",
    "ErrorSeverity",
    "Message",
    "MessageType",
    "NetworkException",
    "ServiceException",
    "StandardCommunicationProtocol",
    "ValidationException",
    "error_handler",
    "get_component_logger",
    "migrate_from_legacy_exception",
    "safe_execute",
    "with_error_handling",
    # Communication classes (renamed to avoid conflicts)
    "AgentMessage",
    "AgentMessageType",
    "AgentCommunicationProtocol",
    "Priority",
]
