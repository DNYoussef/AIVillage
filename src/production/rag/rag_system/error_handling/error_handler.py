import logging
import traceback
from functools import wraps
import logging
import traceback
from functools import wraps
from typing import Any

from common.logging import setup_logging


class AIVillageException(Exception):
    """Custom exception class for AI Village errors."""


class ErrorHandler:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        setup_logging(log_file="ai_village.log")

    def log_error(self, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Log an error with optional context."""
        error_message = f"Error: {error!s}"
        if context:
            error_message += f" Context: {context}"
        self.logger.error(error_message, exc_info=True)

    def handle_error(self, func):
        """Decorator to handle errors in functions."""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.log_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
                msg = f"Error in {func.__name__}: {e!s}"
                raise AIVillageException(msg)

        return wrapper


error_handler = ErrorHandler()


def safe_execute(func):
    """Decorator to safely execute a function and handle any errors."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_handler.log_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            return {"error": str(e), "traceback": traceback.format_exc()}

    return wrapper
