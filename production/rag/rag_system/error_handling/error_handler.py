import logging
from typing import Any, Dict, Optional
from functools import wraps
import traceback

class AIVillageException(Exception):
    """Custom exception class for AI Village errors."""
    pass

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='ai_village.log'
        )

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with optional context."""
        error_message = f"Error: {str(error)}"
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
                self.log_error(e, {'function': func.__name__, 'args': args, 'kwargs': kwargs})
                raise AIVillageException(f"Error in {func.__name__}: {str(e)}")
        return wrapper

error_handler = ErrorHandler()

def safe_execute(func):
    """Decorator to safely execute a function and handle any errors."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_handler.log_error(e, {'function': func.__name__, 'args': args, 'kwargs': kwargs})
            return {'error': str(e), 'traceback': traceback.format_exc()}
    return wrapper
