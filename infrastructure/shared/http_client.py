"""
Resilient HTTP Client with Retries, Circuit Breakers, and Idempotency

This module provides a production-ready HTTP client with:
- Exponential backoff retry logic with jitter
- Circuit breaker pattern for fail-fast behavior
- Idempotency key management
- Timeout handling and connection pooling
- Request/response logging and metrics
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import random
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_status_codes: set[int] = field(default_factory=lambda: {500, 502, 503, 504, 520, 521, 522, 523, 524})


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # failures before opening
    success_threshold: int = 2  # successes needed to close
    timeout: float = 60.0  # seconds before trying again
    monitor_window: float = 300.0  # seconds for failure counting


@dataclass
class IdempotencyConfig:
    """Configuration for idempotency handling."""

    enabled: bool = True
    header_name: str = "Idempotency-Key"
    ttl_seconds: int = 3600  # 1 hour
    auto_generate: bool = True  # Generate keys for non-idempotent methods


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class IdempotencyViolationError(Exception):
    """Raised when idempotency key conflicts are detected."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""

    def __init__(self, config: CircuitBreakerConfig, service_name: str = "default"):
        self.config = config
        self.service_name = service_name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.failure_times: list[float] = []

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.config.timeout:
                raise CircuitBreakerError(f"Circuit breaker open for service: {self.service_name}")
            else:
                # Transition to half-open
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e

    async def acall(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.config.timeout:
                raise CircuitBreakerError(f"Circuit breaker open for service: {self.service_name}")
            else:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise e

    def _record_success(self):
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.failure_times.clear()
                logger.info(f"Circuit breaker closed for service: {self.service_name}")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            self.failure_count = max(0, self.failure_count - 1)

    def _record_failure(self):
        """Record failed call."""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failure_times.append(current_time)

        # Clean old failures outside monitor window
        cutoff_time = current_time - self.config.monitor_window
        self.failure_times = [t for t in self.failure_times if t > cutoff_time]

        self.failure_count = len(self.failure_times)

        if self.failure_count >= self.config.failure_threshold and self.state == CircuitState.CLOSED:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened for service: {self.service_name} after {self.failure_count} failures"
            )
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "time_until_retry": max(0, self.config.timeout - (time.time() - self.last_failure_time)),
        }


class IdempotencyStore:
    """In-memory store for idempotency keys with TTL."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, float] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        """Get cached response for idempotency key."""
        self._cleanup_expired()
        return self._store.get(key)

    def set(self, key: str, response_data: dict[str, Any], ttl_seconds: int = 3600):
        """Store response for idempotency key."""
        self._cleanup_expired()
        current_time = time.time()

        self._store[key] = {
            "response": response_data,
            "created_at": current_time,
            "expires_at": current_time + ttl_seconds,
        }
        self._access_times[key] = current_time

    def has(self, key: str) -> bool:
        """Check if idempotency key exists."""
        self._cleanup_expired()
        return key in self._store

    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [key for key, data in self._store.items() if data.get("expires_at", 0) < current_time]

        for key in expired_keys:
            self._store.pop(key, None)
            self._access_times.pop(key, None)


class ResilientHttpClient:
    """Production HTTP client with retries, circuit breakers, and idempotency."""

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        idempotency_config: IdempotencyConfig | None = None,
        timeout: float = 30.0,
        **httpx_kwargs,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.idempotency_config = idempotency_config or IdempotencyConfig()

        # Circuit breakers per service (based on hostname)
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Idempotency store
        self.idempotency_store = IdempotencyStore()

        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            **httpx_kwargs,
        )

    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        try:
            parsed_url = httpx.URL(url)
            service_name = f"{parsed_url.scheme}://{parsed_url.host}:{parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)}"
        except Exception:
            service_name = "unknown"

        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(self.circuit_config, service_name)

        return self.circuit_breakers[service_name]

    def _generate_idempotency_key(self, method: str, url: str, **kwargs) -> str:
        """Generate idempotency key for request."""
        # Include method, URL, and request body/params in key generation
        key_data = {
            "method": method.upper(),
            "url": str(url),
            "json": kwargs.get("json"),
            "data": kwargs.get("data"),
            "params": kwargs.get("params"),
        }

        # Create deterministic hash
        content = str(sorted(key_data.items())).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:32]

    def _should_retry(self, response: httpx.Response, attempt: int) -> bool:
        """Determine if request should be retried."""
        if attempt >= self.retry_config.max_attempts:
            return False

        return response.status_code in self.retry_config.retry_on_status_codes

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry with exponential backoff and jitter."""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base**attempt), self.retry_config.max_delay
        )

        if self.retry_config.jitter:
            # Add jitter to prevent thundering herd
            delay *= 0.5 + random.random() * 0.5

        return delay

    async def _make_request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic."""
        last_exception = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                response = await self.client.request(method, url, **kwargs)

                if not self._should_retry(response, attempt):
                    return response

                # Log retry attempt
                logger.warning(
                    f"Request failed with status {response.status_code}, "
                    f"retrying attempt {attempt + 1}/{self.retry_config.max_attempts}"
                )

                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(
                    f"Request failed with error: {e}, "
                    f"retrying attempt {attempt + 1}/{self.retry_config.max_attempts}"
                )

                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        if last_exception:
            raise last_exception

        return response  # Return last response even if it failed

    async def request(self, method: str, url: str, idempotency_key: str | None = None, **kwargs) -> httpx.Response:
        """Make resilient HTTP request with circuit breaker and retry logic."""

        # Handle idempotency
        if self.idempotency_config.enabled:
            # Generate idempotency key if not provided and auto-generation enabled
            if idempotency_key is None and self.idempotency_config.auto_generate:
                if method.upper() in ["POST", "PUT", "PATCH"]:
                    idempotency_key = self._generate_idempotency_key(method, url, **kwargs)

            # Check for cached response
            if idempotency_key:
                cached = self.idempotency_store.get(idempotency_key)
                if cached:
                    logger.info(f"Returning cached response for idempotency key: {idempotency_key[:16]}...")
                    return httpx.Response(
                        status_code=cached["response"]["status_code"],
                        json=cached["response"]["json"],
                        headers=cached["response"]["headers"],
                    )

                # Add idempotency header
                headers = kwargs.get("headers", {})
                headers[self.idempotency_config.header_name] = idempotency_key
                kwargs["headers"] = headers

        # Get circuit breaker for this service
        circuit_breaker = self._get_circuit_breaker(url)

        # Make request through circuit breaker
        try:
            response = await circuit_breaker.acall(self._make_request_with_retry, method, url, **kwargs)

            # Cache successful idempotent responses
            if idempotency_key and self.idempotency_config.enabled and 200 <= response.status_code < 300:
                try:
                    response_data = {
                        "status_code": response.status_code,
                        "json": response.json(),
                        "headers": dict(response.headers),
                    }
                    self.idempotency_store.set(idempotency_key, response_data, self.idempotency_config.ttl_seconds)
                except Exception as e:
                    logger.warning(f"Failed to cache idempotent response: {e}")

            return response

        except CircuitBreakerError:
            # Circuit breaker is open
            logger.error(f"Circuit breaker open for {url}")
            raise
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    # Convenience methods
    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, idempotency_key: str | None = None, **kwargs) -> httpx.Response:
        return await self.request("POST", url, idempotency_key=idempotency_key, **kwargs)

    async def put(self, url: str, idempotency_key: str | None = None, **kwargs) -> httpx.Response:
        return await self.request("PUT", url, idempotency_key=idempotency_key, **kwargs)

    async def patch(self, url: str, idempotency_key: str | None = None, **kwargs) -> httpx.Response:
        return await self.request("PATCH", url, idempotency_key=idempotency_key, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", url, **kwargs)

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {service: breaker.get_state() for service, breaker in self.circuit_breakers.items()}

    def get_idempotency_stats(self) -> dict[str, Any]:
        """Get idempotency cache statistics."""
        return {
            "total_keys": len(self.idempotency_store._store),
            "cache_enabled": self.idempotency_config.enabled,
            "ttl_seconds": self.idempotency_config.ttl_seconds,
            "auto_generate": self.idempotency_config.auto_generate,
        }

    async def close(self):
        """Close HTTP client and cleanup resources."""
        await self.client.aclose()


# Global client instance
_global_client: ResilientHttpClient | None = None


def get_http_client() -> ResilientHttpClient:
    """Get global HTTP client instance."""
    global _global_client
    if _global_client is None:
        _global_client = ResilientHttpClient()
    return _global_client


# Context manager for dependency outage simulation (testing)
class DependencyOutageSimulator:
    """Simulate dependency outages for testing circuit breaker behavior."""

    def __init__(self, client: ResilientHttpClient, service_pattern: str):
        self.client = client
        self.service_pattern = service_pattern
        self.original_request = None

    async def __aenter__(self):
        # Replace request method with failing version
        self.original_request = self.client.client.request

        async def failing_request(*args, **kwargs):
            url = args[1] if len(args) > 1 else kwargs.get("url", "")
            if self.service_pattern in str(url):
                raise httpx.RequestError("Simulated service outage")
            return await self.original_request(*args, **kwargs)

        self.client.client.request = failing_request
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Restore original request method
        if self.original_request:
            self.client.client.request = self.original_request


# Usage examples
if __name__ == "__main__":
    import asyncio

    async def example_usage():
        # Create configured client
        client = ResilientHttpClient(
            retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
            circuit_config=CircuitBreakerConfig(failure_threshold=3),
            idempotency_config=IdempotencyConfig(enabled=True),
        )

        try:
            # Make idempotent POST request
            response = await client.post(
                "https://api.example.com/users",
                json={"name": "John", "email": "john@example.com"},
                idempotency_key="create-user-123",
            )
            print(f"Response: {response.status_code}")

            # Check circuit breaker status
            status = client.get_circuit_breaker_status()
            print(f"Circuit breaker status: {status}")

        except CircuitBreakerError as e:
            print(f"Circuit breaker blocked request: {e}")
        except Exception as e:
            print(f"Request failed: {e}")
        finally:
            await client.close()

    # Run example
    asyncio.run(example_usage())
