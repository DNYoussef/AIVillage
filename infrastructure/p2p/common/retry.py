"""
Retry strategies and backoff utilities for P2P operations.

Provides standardized retry mechanisms with different backoff strategies
for handling transient failures in P2P communications.
"""

import asyncio
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union, Type, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class RetryResult(Enum):
    """Result of a retry attempt."""
    SUCCESS = "success"
    RETRY = "retry"
    FAILED = "failed"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_type: str = "exponential"  # exponential, linear, constant
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.retryable_exceptions:
            self.retryable_exceptions = [ConnectionError, TimeoutError, OSError]


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempt = 0
        self.total_delay = 0.0
        self.start_time = time.time()
    
    @abstractmethod
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        pass
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if exception type is retryable
        if self.config.retryable_exceptions:
            return any(isinstance(exception, exc_type) 
                      for exc_type in self.config.retryable_exceptions)
        
        # Default: retry on common network exceptions
        return isinstance(exception, (ConnectionError, TimeoutError, OSError))
    
    def add_jitter(self, delay: float) -> float:
        """Add jitter to delay if enabled."""
        if self.config.jitter:
            # Add up to 25% jitter
            jitter_amount = delay * 0.25 * random.random()
            return delay + jitter_amount
        return delay
    
    def reset(self):
        """Reset retry state."""
        self.attempt = 0
        self.total_delay = 0.0
        self.start_time = time.time()


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff retry strategy."""
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        return self.add_jitter(delay)


class LinearBackoff(RetryStrategy):
    """Linear backoff retry strategy."""
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        delay = self.config.base_delay * (attempt + 1)
        delay = min(delay, self.config.max_delay)
        return self.add_jitter(delay)


class ConstantBackoff(RetryStrategy):
    """Constant delay retry strategy."""
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate constant delay."""
        return self.add_jitter(self.config.base_delay)


class FibonacciBackoff(RetryStrategy):
    """Fibonacci sequence backoff retry strategy."""
    
    def __init__(self, config: RetryConfig):
        super().__init__(config)
        self._fib_cache = {0: 0, 1: 1}
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number with memoization."""
        if n not in self._fib_cache:
            self._fib_cache[n] = self._fibonacci(n-1) + self._fibonacci(n-2)
        return self._fib_cache[n]
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate Fibonacci backoff delay."""
        fib_multiplier = self._fibonacci(attempt + 1)
        delay = self.config.base_delay * fib_multiplier
        delay = min(delay, self.config.max_delay)
        return self.add_jitter(delay)


def get_retry_strategy(config: RetryConfig) -> RetryStrategy:
    """Get retry strategy based on configuration."""
    if config.backoff_type == "exponential":
        return ExponentialBackoff(config)
    elif config.backoff_type == "linear":
        return LinearBackoff(config)
    elif config.backoff_type == "constant":
        return ConstantBackoff(config)
    elif config.backoff_type == "fibonacci":
        return FibonacciBackoff(config)
    else:
        raise ValueError(f"Unknown backoff type: {config.backoff_type}")


async def with_retry(func: Callable, config: Optional[RetryConfig] = None, *args, **kwargs) -> Any:
    """
    Execute function with retry logic.
    
    Args:
        func: Function to execute with retries
        config: Retry configuration
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Function result on success
        
    Raises:
        Last exception if all retries failed
    """
    if config is None:
        config = RetryConfig()
    
    strategy = get_retry_strategy(config)
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success!
            logger.debug(f"Function succeeded on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            last_exception = e
            logger.debug(f"Attempt {attempt + 1} failed: {e}")
            
            # Check if we should retry
            if not strategy.should_retry(e, attempt + 1):
                logger.info(f"Not retrying after attempt {attempt + 1}: {e}")
                break
            
            # Calculate delay and wait
            if attempt < config.max_attempts - 1:  # Don't delay after last attempt
                delay = strategy.calculate_delay(attempt)
                logger.debug(f"Waiting {delay:.2f}s before retry {attempt + 2}")
                await asyncio.sleep(delay)
    
    # All attempts failed
    logger.error(f"All {config.max_attempts} attempts failed")
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("All retry attempts failed")


def retry_decorator(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        config: Retry configuration
        
    Example:
        @retry_decorator(RetryConfig(max_attempts=5, backoff_type="exponential"))
        async def unreliable_network_call():
            # Your code here
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await with_retry(func, config, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Convert sync function to async for retry logic
                async def async_func(*a, **kw):
                    return func(*a, **kw)
                
                # Run in event loop
                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(with_retry(async_func, config, *args, **kwargs))
                except RuntimeError:
                    # No event loop running, create new one
                    return asyncio.run(with_retry(async_func, config, *args, **kwargs))
            return sync_wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    Tracks failure rate and opens circuit when threshold is exceeded.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "half_open"
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has elapsed
            if (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout):
                self.state = "half_open"
                return True
            return False
        
        # half_open state
        return True
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if not self.can_attempt():
            raise RuntimeError("Circuit breaker is open")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.record_success()
            return result
            
        except self.expected_exception as e:
            self.record_failure()
            raise e


# Predefined retry configurations for common scenarios
QUICK_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
    backoff_type="exponential"
)

STANDARD_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    backoff_type="exponential"
)

PERSISTENT_RETRY = RetryConfig(
    max_attempts=10,
    base_delay=2.0,
    max_delay=120.0,
    backoff_type="exponential"
)

NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    backoff_type="exponential",
    retryable_exceptions=[ConnectionError, TimeoutError, OSError]
)
