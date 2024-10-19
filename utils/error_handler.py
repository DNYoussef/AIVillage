import logging
import traceback
from functools import wraps
from typing import Callable, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIVillageException(Exception):
    """Base exception class for AI Village errors."""
    pass

def error_handler(func: Callable) -> Callable:
    """
    A decorator for handling exceptions in async functions.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AIVillageException as e:
            logger.error(f"AIVillageException in {func.__name__}: {str(e)}")
            # You might want to handle this differently, e.g., return a specific error response
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            # You might want to handle this differently, e.g., return a generic error response
            raise AIVillageException(f"An unexpected error occurred in {func.__name__}")
    return wrapper

def safe_execute(func: Callable) -> Any:
    """
    A decorator for handling exceptions in sync functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AIVillageException as e:
            logger.error(f"AIVillageException in {func.__name__}: {str(e)}")
            # You might want to handle this differently, e.g., return a specific error response
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            # You might want to handle this differently, e.g., return a generic error response
            raise AIVillageException(f"An unexpected error occurred in {func.__name__}")
    return wrapper

# Example usage:
# @error_handler
# async def some_async_function():
#     # Your code here
#     pass

# @safe_execute
# def some_sync_function():
#     # Your code here
#     pass
