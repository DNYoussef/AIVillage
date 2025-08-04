from collections.abc import Callable
from functools import wraps
import logging
from typing import Any

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


def setup_logging(
    log_file: str = "rag_system.log", log_level: int = logging.INFO
) -> None:
    """Set up logging for the RAG system."""
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
