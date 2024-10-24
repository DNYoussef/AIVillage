"""Helper utilities for MAGI agent system."""

import os
import json
import yaml
import time
import hashlib
import inspect
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
from ..core.exceptions import MAGIException, ExecutionError, TimeoutError
from ..core.constants import SYSTEM_CONSTANTS

logger = logging.getLogger(__name__)

T = TypeVar('T')

def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Retry async function with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator

def memoize(ttl: int = 3600) -> Callable:
    """Memoize function results with time-to-live."""
    cache: Dict[str, Dict[str, Any]] = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = hashlib.md5(
                f"{func.__name__}{str(args)}{str(kwargs)}".encode()
            ).hexdigest()
            
            now = time.time()
            if key in cache:
                result, timestamp = cache[key]['result'], cache[key]['timestamp']
                if now - timestamp < ttl:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = {'result': result, 'timestamp': now}
            return result
        return wrapper
    return decorator

async def run_in_executor(func: Callable, *args, **kwargs) -> Any:
    """Run blocking function in thread pool."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, lambda: func(*args, **kwargs))

def safe_json_loads(data: str) -> Dict[str, Any]:
    """Safely load JSON data."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {}

def safe_yaml_load(path: Union[str, Path]) -> Dict[str, Any]:
    """Safely load YAML file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.error(f"Failed to load YAML file {path}: {e}")
        return {}

def validate_type(value: Any, expected_type: type) -> bool:
    """Validate value type."""
    return isinstance(value, expected_type)

def validate_schema(data: Dict[str, Any], schema: Dict[str, type]) -> bool:
    """Validate dictionary against schema."""
    try:
        for key, expected_type in schema.items():
            if key not in data or not validate_type(data[key], expected_type):
                return False
        return True
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        return False

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def format_timestamp(timestamp: Optional[float] = None) -> str:
    """Format timestamp as ISO 8601 string."""
    dt = datetime.fromtimestamp(timestamp or time.time(), tz=timezone.utc)
    return dt.isoformat()

def parse_timestamp(timestamp_str: str) -> float:
    """Parse ISO 8601 timestamp string to Unix timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.timestamp()
    except ValueError as e:
        logger.error(f"Failed to parse timestamp {timestamp_str}: {e}")
        return 0.0

def get_function_args(func: Callable) -> List[str]:
    """Get function argument names."""
    return list(inspect.signature(func).parameters.keys())

def create_directory(path: Union[str, Path]) -> bool:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def clean_directory(path: Union[str, Path], max_age: int = 86400) -> None:
    """Clean old files from directory."""
    try:
        now = time.time()
        for entry in os.scandir(path):
            if entry.is_file() and now - entry.stat().st_mtime > max_age:
                os.remove(entry.path)
    except OSError as e:
        logger.error(f"Failed to clean directory {path}: {e}")

def calculate_hash(data: Union[str, bytes]) -> str:
    """Calculate SHA-256 hash of data."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()

def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to maximum length."""
    return text[:max_length] + "..." if len(text) > max_length else text

def format_exception(e: Exception) -> str:
    """Format exception with traceback."""
    import traceback
    return f"{type(e).__name__}: {str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"

def measure_time(func: Callable) -> Callable:
    """Measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def timeout(seconds: int) -> Callable:
    """Timeout decorator for async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
        return wrapper
    return decorator

def rate_limit(calls: int, period: float) -> Callable:
    """Rate limit decorator."""
    timestamps: List[float] = []
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            now = time.time()
            timestamps.append(now)
            
            # Remove timestamps outside the window
            while timestamps and timestamps[0] < now - period:
                timestamps.pop(0)
            
            if len(timestamps) > calls:
                wait_time = timestamps[0] + period - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def chunk_list(lst: List[T], size: int) -> List[List[T]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary with dot notation keys."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result
