"""
Custom Exception Hierarchy for AIVillage

This module provides a structured exception hierarchy to improve error handling
consistency across the entire codebase.
"""

from typing import Optional, Dict, Any
import traceback


class AIVillageError(Exception):
    """Base exception for all AIVillage-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.stack_trace = traceback.format_exc() if cause else None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "stack_trace": self.stack_trace
        }


class ConfigurationError(AIVillageError):
    """Raised when there are configuration-related errors."""
    pass


class ModelError(AIVillageError):
    """Base class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training encounters errors."""
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass


class AgentForgeError(AIVillageError):
    """Base class for Agent Forge specific errors."""
    pass


class PhaseExecutionError(AgentForgeError):
    """Raised when a pipeline phase fails to execute."""
    
    def __init__(
        self, 
        message: str, 
        phase_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.phase_name = phase_name


class PipelineError(AgentForgeError):
    """Raised when pipeline execution fails."""
    pass


class ResourceError(AIVillageError):
    """Base class for resource-related errors."""
    pass


class MemoryError(ResourceError):
    """Raised when memory allocation/management fails."""
    pass


class ComputeResourceError(ResourceError):
    """Raised when compute resources are insufficient or unavailable."""
    pass


class NetworkError(AIVillageError):
    """Base class for network-related errors."""
    pass


class P2PNetworkError(NetworkError):
    """Raised when P2P network operations fail."""
    pass


class FederatedLearningError(NetworkError):
    """Raised when federated learning operations fail."""
    pass


class DataError(AIVillageError):
    """Base class for data-related errors."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""
    pass


class SecurityError(AIVillageError):
    """Base class for security-related errors."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


class IntegrationError(AIVillageError):
    """Base class for integration-related errors."""
    pass


class MCPServerError(IntegrationError):
    """Raised when MCP server operations fail."""
    pass


class ExternalServiceError(IntegrationError):
    """Raised when external service integration fails."""
    pass


def handle_exception(
    exception: Exception, 
    context: Optional[str] = None,
    reraise_as: Optional[type] = None
) -> None:
    """
    Standardized exception handling utility.
    
    Args:
        exception: The caught exception
        context: Additional context for the error
        reraise_as: Exception class to reraise as (default: AIVillageError)
    
    Raises:
        AIVillageError: Standardized exception with context
    """
    if isinstance(exception, AIVillageError):
        raise exception
        
    error_class = reraise_as or AIVillageError
    
    details = {"original_exception": str(exception)}
    if context:
        details["context"] = context
        
    raise error_class(
        message=f"{context}: {str(exception)}" if context else str(exception),
        cause=exception,
        details=details
    )


def create_error_handler(error_class: type, context: str):
    """Factory function to create context-specific error handlers."""
    def handler(exception: Exception):
        handle_exception(exception, context, error_class)
    return handler


# Pre-configured error handlers for common scenarios
phase_error_handler = create_error_handler(PhaseExecutionError, "Phase execution failed")
model_error_handler = create_error_handler(ModelError, "Model operation failed")
config_error_handler = create_error_handler(ConfigurationError, "Configuration error")
data_error_handler = create_error_handler(DataError, "Data operation failed")