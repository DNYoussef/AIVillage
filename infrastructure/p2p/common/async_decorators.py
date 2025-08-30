"""
Async decorator utilities for P2P infrastructure.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


def timeout_after(seconds: float = DEFAULT_TIMEOUT_SECONDS):
    """Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {seconds}s")
                raise
        return wrapper
    return decorator


def retry_on_failure(*, max_attempts: int = DEFAULT_MAX_ATTEMPTS, 
                     delay: float = DEFAULT_RETRY_DELAY, 
                     backoff: float = DEFAULT_BACKOFF_MULTIPLIER):
    """Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.debug(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")
            
            raise last_exception
        return wrapper
    return decorator