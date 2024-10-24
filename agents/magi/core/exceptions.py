"""Custom exceptions for MAGI agent system."""

from typing import Optional, Any, Dict
from .constants import ErrorLevel

class MAGIException(Exception):
    """Base exception for MAGI agent system."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.level = level
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "level": self.level.name,
            "details": self.details
        }

class ConfigurationError(MAGIException):
    """Raised when there is a configuration error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class ValidationError(MAGIException):
    """Raised when validation fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class ExecutionError(MAGIException):
    """Raised when task execution fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EXECUTION_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class ResourceError(MAGIException):
    """Raised when there is a resource-related error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class ToolError(MAGIException):
    """Raised when there is a tool-related error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TOOL_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class SecurityError(MAGIException):
    """Raised when there is a security-related error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            level=ErrorLevel.CRITICAL,
            details=details
        )

class CommunicationError(MAGIException):
    """Raised when there is a communication error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="COMMUNICATION_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class KnowledgeError(MAGIException):
    """Raised when there is a knowledge base error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="KNOWLEDGE_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class EvolutionError(MAGIException):
    """Raised when there is an evolution-related error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EVOLUTION_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class PlanningError(MAGIException):
    """Raised when there is a planning error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PLANNING_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class OptimizationError(MAGIException):
    """Raised when there is an optimization error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class TimeoutError(MAGIException):
    """Raised when an operation times out."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class ConcurrencyError(MAGIException):
    """Raised when there is a concurrency-related error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONCURRENCY_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class StateError(MAGIException):
    """Raised when there is a state-related error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="STATE_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

class IntegrationError(MAGIException):
    """Raised when there is an integration error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INTEGRATION_ERROR",
            level=ErrorLevel.ERROR,
            details=details
        )

# Error handling utilities
def handle_error(error: Exception) -> MAGIException:
    """Convert any exception to a MAGIException."""
    if isinstance(error, MAGIException):
        return error
    
    return MAGIException(
        message=str(error),
        error_code="UNKNOWN_ERROR",
        level=ErrorLevel.ERROR,
        details={"original_error": error.__class__.__name__}
    )

def raise_from_response(response: Dict[str, Any]) -> None:
    """Raise appropriate exception from API response."""
    if "error" in response:
        error_data = response["error"]
        error_code = error_data.get("error_code", "UNKNOWN_ERROR")
        message = error_data.get("message", "Unknown error occurred")
        level = ErrorLevel[error_data.get("level", "ERROR")]
        details = error_data.get("details", {})
        
        exception_map = {
            "CONFIG_ERROR": ConfigurationError,
            "VALIDATION_ERROR": ValidationError,
            "EXECUTION_ERROR": ExecutionError,
            "RESOURCE_ERROR": ResourceError,
            "TOOL_ERROR": ToolError,
            "SECURITY_ERROR": SecurityError,
            "COMMUNICATION_ERROR": CommunicationError,
            "KNOWLEDGE_ERROR": KnowledgeError,
            "EVOLUTION_ERROR": EvolutionError,
            "PLANNING_ERROR": PlanningError,
            "OPTIMIZATION_ERROR": OptimizationError,
            "TIMEOUT_ERROR": TimeoutError,
            "CONCURRENCY_ERROR": ConcurrencyError,
            "STATE_ERROR": StateError,
            "INTEGRATION_ERROR": IntegrationError
        }
        
        exception_class = exception_map.get(error_code, MAGIException)
        raise exception_class(message=message, details=details)
