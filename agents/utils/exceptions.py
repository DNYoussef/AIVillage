"""Custom exceptions for AI Village system."""

class AIVillageException(Exception):
    """Base exception class for AI Village."""
    pass

class ConfigurationError(AIVillageException):
    """Raised when there is a configuration error."""
    pass

class InitializationError(AIVillageException):
    """Raised when component initialization fails."""
    pass

class TaskExecutionError(AIVillageException):
    """Raised when task execution fails."""
    pass

class CommunicationError(AIVillageException):
    """Raised when there is a communication error."""
    pass

class ResourceError(AIVillageException):
    """Raised when there is a resource allocation error."""
    pass

class ValidationError(AIVillageException):
    """Raised when validation fails."""
    pass

class QualityAssuranceError(AIVillageException):
    """Raised when quality assurance checks fail."""
    pass

class KnowledgeError(AIVillageException):
    """Raised when there is an error with knowledge management."""
    pass

class CompressionError(AIVillageException):
    """Raised when there is an error during model compression."""
    pass
