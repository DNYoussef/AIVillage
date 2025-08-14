import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

# Re-export shared logging setup for consumers
from common.logging import setup_logging

logger = logging.getLogger(__name__)


class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""


class ConfigurationError(RAGSystemError):
    """Raised when there's an error in the configuration."""


class ProcessingError(RAGSystemError):
    """Raised when there's an error during data processing."""


class RetrievalError(RAGSystemError):
    """Raised when there's an error during information retrieval."""


def log_and_handle_errors(func: Callable) -> Callable:
    """A decorator to log errors and handle them gracefully."""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except RAGSystemError as e:
            logger.exception(f"RAG System Error in {func.__name__}: {e!s}")
            # Here you can add custom error handling logic
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e!s}")
            # Here you can add custom error handling logic
            msg = f"An unexpected error occurred: {e!s}"
            raise RAGSystemError(msg)

    return wrapper
