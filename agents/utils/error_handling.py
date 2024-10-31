"""Error handling utilities for AI Village."""

import logging
import functools
from typing import Any, Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

def error_handler(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator for handling errors in agent functions.
    
    Args:
        func: The function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class AIVillageError(Exception):
    """Base exception class for AI Village errors."""
    pass

class AgentError(AIVillageError):
    """Exception raised for errors in agent operations."""
    pass

class CommunicationError(AIVillageError):
    """Exception raised for errors in communication between components."""
    pass

class ConfigurationError(AIVillageError):
    """Exception raised for configuration-related errors."""
    pass

class RAGError(AIVillageError):
    """Exception raised for errors in RAG system operations."""
    pass

class UIError(AIVillageError):
    """Exception raised for errors in UI operations."""
    pass

def handle_agent_error(error: Exception) -> AgentError:
    """Convert generic exceptions to AgentError with appropriate context."""
    return AgentError(f"Agent operation failed: {str(error)}")

def handle_communication_error(error: Exception) -> CommunicationError:
    """Convert generic exceptions to CommunicationError with appropriate context."""
    return CommunicationError(f"Communication failed: {str(error)}")

def handle_configuration_error(error: Exception) -> ConfigurationError:
    """Convert generic exceptions to ConfigurationError with appropriate context."""
    return ConfigurationError(f"Configuration error: {str(error)}")

def handle_rag_error(error: Exception) -> RAGError:
    """Convert generic exceptions to RAGError with appropriate context."""
    return RAGError(f"RAG system error: {str(error)}")

def handle_ui_error(error: Exception) -> UIError:
    """Convert generic exceptions to UIError with appropriate context."""
    return UIError(f"UI operation failed: {str(error)}")
