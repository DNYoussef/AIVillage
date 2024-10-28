"""Error handling utilities for RAG system."""

import logging
import functools
from typing import Any, Callable, TypeVar, ParamSpec, Dict, Optional, Type
import traceback
import asyncio
from contextlib import AbstractAsyncContextManager
import types

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""
    pass

class ComponentInitializationError(RAGSystemError):
    """Raised when a component fails to initialize."""
    pass

class ComponentShutdownError(RAGSystemError):
    """Raised when a component fails to shutdown."""
    pass

class ProcessingError(RAGSystemError):
    """Raised when processing fails."""
    pass

class ConfigurationError(RAGSystemError):
    """Raised when configuration is invalid."""
    pass

def log_and_handle_errors(error_type: Optional[Type[Exception]] = None):
    """
    Decorator to log and handle errors in RAG system components.
    
    Args:
        error_type: Optional specific error type to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                if error_type and not isinstance(e, error_type):
                    raise
                logger.error(f"{func.__name__} failed: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if isinstance(e, RAGSystemError):
                    raise
                raise ProcessingError(f"Error in {func.__name__}: {str(e)}") from e
        return async_wrapper
    return decorator

def handle_component_error(
    component_name: str,
    error: Exception,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Handle component errors and return standardized error response.
    
    Args:
        component_name: Name of the component where error occurred
        error: The error that occurred
        include_traceback: Whether to include traceback in response
        
    Returns:
        Dictionary containing error information
    """
    error_info = {
        "status": "error",
        "component": component_name,
        "error_type": error.__class__.__name__,
        "error_message": str(error)
    }
    
    if include_traceback:
        error_info["traceback"] = traceback.format_exc()
    
    logger.error(f"Error in {component_name}: {str(error)}")
    if include_traceback:
        logger.debug(f"Error details: {error_info}")
    
    return error_info

def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is recoverable.
    
    Args:
        error: The error to check
        
    Returns:
        Boolean indicating if error is recoverable
    """
    # List of error types that are considered recoverable
    recoverable_errors = (
        TimeoutError,
        ConnectionError,
        ProcessingError,
        asyncio.TimeoutError
    )
    
    return isinstance(error, recoverable_errors)

def should_retry(
    error: Exception,
    attempt: int,
    max_retries: int,
    force_retry: bool = False
) -> bool:
    """
    Determine if operation should be retried.
    
    Args:
        error: The error that occurred
        attempt: Current attempt number
        max_retries: Maximum number of retries allowed
        force_retry: Whether to force retry regardless of error type
        
    Returns:
        Boolean indicating if operation should be retried
    """
    return (attempt < max_retries and 
            (force_retry or is_recoverable_error(error)))

class ErrorContext(AbstractAsyncContextManager):
    """Async context manager for handling errors in RAG system components."""
    
    def __init__(
        self,
        component_name: str,
        error_handler: Optional[Callable[[str, Exception], Dict[str, Any]]] = None,
        suppress_errors: bool = False
    ):
        """
        Initialize error context.
        
        Args:
            component_name: Name of the component
            error_handler: Optional custom error handler function
            suppress_errors: Whether to suppress errors
        """
        self.component_name = component_name
        self.error_handler = error_handler or handle_component_error
        self.suppress_errors = suppress_errors
    
    async def __aenter__(self) -> 'ErrorContext':
        """Enter the async context."""
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType]
    ) -> bool:
        """
        Handle errors on async context exit.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            Boolean indicating if exception should be suppressed
        """
        if exc_val is not None:
            error_info = self.error_handler(self.component_name, exc_val)
            logger.error(f"Error in {self.component_name}: {error_info}")
            return self.suppress_errors
        return False

async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    **kwargs: Any
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result
        
    Raises:
        ProcessingError: If all retries fail
    """
    attempt = 0
    while True:
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if not should_retry(e, attempt, max_retries):
                raise ProcessingError(f"Failed after {attempt} attempts: {str(e)}") from e
            
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(f"Attempt {attempt} failed, retrying in {delay:.1f}s: {str(e)}")
            await asyncio.sleep(delay)
