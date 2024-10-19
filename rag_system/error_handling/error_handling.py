import logging
import traceback
from typing import Dict, Any, Callable
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class InputError(RAGSystemError):
    """Exception raised for errors in the input."""
    pass

class ProcessingError(RAGSystemError):
    """Exception raised for errors during processing."""
    pass

class OutputError(RAGSystemError):
    """Exception raised for errors in the output."""
    pass

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log the error with additional context."""
    error_message = f"{type(error).__name__}: {str(error)}"
    if isinstance(error, RAGSystemError):
        error_message += f" (Error Code: {error.error_code})"
    
    logger.error(error_message)
    if context:
        logger.error(f"Error Context: {context}")
    logger.error(f"Traceback: {''.join(traceback.format_tb(error.__traceback__))}")

def error_handler(func: Callable):
    """Decorator for handling errors in functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except RAGSystemError as e:
            log_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            # Re-raise the error to be handled by the caller
            raise
        except Exception as e:
            # For unexpected errors, wrap them in a ProcessingError
            error = ProcessingError(f"Unexpected error in {func.__name__}: {str(e)}", "UNEXPECTED_ERROR", 
                                    {"original_error": str(e), "error_type": type(e).__name__})
            log_error(error, {"function": func.__name__, "args": args, "kwargs": kwargs})
            raise error
    return wrapper

class ErrorHandler:
    @staticmethod
    def handle_error(error: Exception, context: Dict[str, Any] = None):
        """Handle errors globally."""
        log_error(error, context)
        
        if isinstance(error, InputError):
            # Handle input errors (e.g., invalid user queries)
            return {"error": "Invalid input", "details": error.details}
        elif isinstance(error, ProcessingError):
            # Handle processing errors (e.g., errors in the RAG pipeline)
            return {"error": "Processing error", "details": error.details}
        elif isinstance(error, OutputError):
            # Handle output errors (e.g., errors in formatting the response)
            return {"error": "Output error", "details": error.details}
        else:
            # Handle unexpected errors
            return {"error": "An unexpected error occurred", "details": {"message": str(error)}}

    @staticmethod
    def raise_input_error(message: str, details: Dict[str, Any] = None):
        """Raise an InputError with the given message and details."""
        raise InputError(message, "INPUT_ERROR", details)

    @staticmethod
    def raise_processing_error(message: str, details: Dict[str, Any] = None):
        """Raise a ProcessingError with the given message and details."""
        raise ProcessingError(message, "PROCESSING_ERROR", details)

    @staticmethod
    def raise_output_error(message: str, details: Dict[str, Any] = None):
        """Raise an OutputError with the given message and details."""
        raise OutputError(message, "OUTPUT_ERROR", details)
