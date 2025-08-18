import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from common.logging import setup_logging

logger = logging.getLogger(__name__)
setup_logging(log_file="rag_system.log")


class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""


class ConfigurationError(RAGSystemError):
    """Raised when there's an error in the configuration."""


class ProcessingError(RAGSystemError):
    """Raised when there's an error during data processing."""


class RetrievalError(RAGSystemError):
    ""Raised when there's an error during information retrieval."""


def log_and_handle_errors(func: Callable) -> Callable:
    """A decorator to log errors and handle them gracefully."""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except RAGSystemError as e:
            logger.exception(f"RAG System Error in {func.__name__}: {e\!s}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e\!s}")
            msg = f"An unexpected error occurred: {e\!s}"
            raise RAGSystemError(msg)

    return wrapper


__all__ = [
    "ConfigurationError",
    "ProcessingError",
    "RAGSystemError",
    "RetrievalError",
    "log_and_handle_errors",
]
