"""Error Handling & Resilience Framework - Prompt F

Comprehensive error handling and system resilience including:
- Circuit breaker patterns for external dependencies
- Retry mechanisms with exponential backoff
- Graceful degradation strategies
- Error categorization and recovery
- System health monitoring
- Failover and fallback mechanisms

Integration Point: Resilience layer for Phase 4 testing
"""

import functools
import logging
import random
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categorization."""

    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    RESOURCE = "resource"
    EXTERNAL = "external"
    INTERNAL = "internal"
    TIMEOUT = "timeout"
    PERMISSION = "permission"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RetryStrategy(Enum):
    """Retry strategy types."""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


@dataclass
class ErrorContext:
    """Error context information."""

    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    error_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovered: bool = False
    recovery_time_ms: float | None = None


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_attempts: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: list[type] = field(default_factory=list)
    non_retryable_exceptions: list[type] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_ms: float = 60000.0
    half_open_max_calls: int = 5
    failure_rate_threshold: float = 0.5
    minimum_calls: int = 10


@dataclass
class DegradationConfig:
    """Graceful degradation configuration."""

    enable_fallback: bool = True
    fallback_timeout_ms: float = 5000.0
    cache_stale_data: bool = True
    reduce_functionality: bool = True
    offline_mode: bool = True
    user_notification: bool = True


class HealthStatus(Enum):
    """System health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class HealthCheck:
    """Health check definition."""

    name: str
    check_func: Callable
    timeout_ms: float = 5000.0
    interval_ms: float = 30000.0
    critical: bool = False
    dependencies: list[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """System health state."""

    overall_status: HealthStatus
    component_health: dict[str, HealthStatus] = field(default_factory=dict)
    last_check: float = 0.0
    error_count: int = 0
    degraded_services: list[str] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)


class ErrorTracker:
    """Tracks and categorizes errors across the system."""

    def __init__(self):
        self.errors: list[ErrorContext] = []
        self.error_counts: dict[str, int] = {}
        self.component_health: dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

    def record_error(
        self,
        component: str,
        operation: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an error occurrence."""
        error_id = f"{component}_{operation}_{int(time.time() * 1000)}"

        # Auto-categorize if not provided
        if category is None:
            category = self._categorize_error(error)

        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            message=str(error),
            metadata=metadata or {},
        )

        with self._lock:
            self.errors.append(error_context)

            # Update error counts
            error_key = f"{component}:{category.value}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            # Update component health
            self._update_component_health(component, severity)

        return error_id

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Automatically categorize error based on type."""
        error_type = type(error).__name__.lower()

        if any(
            net_error in error_type for net_error in ["connection", "network", "socket"]
        ):
            return ErrorCategory.NETWORK
        elif any(
            auth_error in error_type
            for auth_error in ["auth", "permission", "unauthorized"]
        ):
            return ErrorCategory.AUTHENTICATION
        elif any(
            val_error in error_type for val_error in ["validation", "value", "type"]
        ):
            return ErrorCategory.VALIDATION
        elif any(
            res_error in error_type for res_error in ["memory", "resource", "limit"]
        ):
            return ErrorCategory.RESOURCE
        elif any(
            timeout_error in error_type for timeout_error in ["timeout", "deadline"]
        ):
            return ErrorCategory.TIMEOUT
        else:
            return ErrorCategory.INTERNAL

    def _update_component_health(self, component: str, severity: ErrorSeverity):
        """Update component health based on error severity."""
        if severity == ErrorSeverity.CRITICAL:
            self.component_health[component] = HealthStatus.CRITICAL
        elif severity == ErrorSeverity.HIGH:
            current = self.component_health.get(component, HealthStatus.HEALTHY)
            if current == HealthStatus.HEALTHY:
                self.component_health[component] = HealthStatus.DEGRADED
        elif severity == ErrorSeverity.MEDIUM:
            if component not in self.component_health:
                self.component_health[component] = HealthStatus.DEGRADED

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            recent_errors = [
                e for e in self.errors if time.time() - e.timestamp < 3600
            ]  # Last hour

            return {
                "total_errors": len(self.errors),
                "recent_errors": len(recent_errors),
                "error_by_severity": {
                    severity.value: len(
                        [e for e in recent_errors if e.severity == severity]
                    )
                    for severity in ErrorSeverity
                },
                "error_by_category": {
                    category.value: len(
                        [e for e in recent_errors if e.category == category]
                    )
                    for category in ErrorCategory
                },
                "component_health": dict(self.component_health),
                "top_error_sources": sorted(
                    self.error_counts.items(), key=lambda x: x[1], reverse=True
                )[:10],
            }


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self._lock = threading.Lock()

        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = []

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1

            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.timeout_ms / 1000:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN"
                    )
                else:
                    # Transition to half-open
                    self._change_state(CircuitBreakerState.HALF_OPEN)

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} half-open call limit exceeded"
                    )
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception:
            self._record_failure()
            raise

    def _record_success(self):
        """Record successful call."""
        with self._lock:
            self.total_successes += 1
            self.success_count += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._change_state(CircuitBreakerState.CLOSED)
                    self.failure_count = 0
                    self.half_open_calls = 0

    def _record_failure(self):
        """Record failed call."""
        with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._change_state(CircuitBreakerState.OPEN)

            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._change_state(CircuitBreakerState.OPEN)
                self.half_open_calls = 0

    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.state_changes.append(
            {
                "timestamp": time.time(),
                "from_state": old_state.value,
                "to_state": new_state.value,
            }
        )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": self.total_failures / max(self.total_calls, 1),
            "state_changes": len(self.state_changes),
            "last_state_change": self.state_changes[-1] if self.state_changes else None,
        }


class RetryHandler:
    """Retry mechanism with various backoff strategies."""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)

        return wrapper

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logging.info(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                    )
                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable(e):
                    logging.warning(f"Non-retryable exception in {func.__name__}: {e}")
                    raise

                # Don't delay on last attempt
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

        # All attempts failed
        logging.error(
            f"All {self.config.max_attempts} attempts failed for {func.__name__}"
        )
        raise last_exception

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        if self.config.non_retryable_exceptions:
            if any(
                isinstance(exception, exc_type)
                for exc_type in self.config.non_retryable_exceptions
            ):
                return False

        if self.config.retryable_exceptions:
            return any(
                isinstance(exception, exc_type)
                for exc_type in self.config.retryable_exceptions
            )

        # Default retryable exceptions
        retryable_types = (
            ConnectionError,
            TimeoutError,
            OSError,
        )

        return isinstance(exception, retryable_types)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.initial_delay_ms / 1000

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = (self.config.initial_delay_ms * (attempt + 1)) / 1000

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = (
                self.config.initial_delay_ms * (self.config.backoff_multiplier**attempt)
            ) / 1000

        elif self.config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = (
                self.config.initial_delay_ms * (self.config.backoff_multiplier**attempt)
            ) / 1000
            jitter = random.uniform(0.1, 0.3) * base_delay
            delay = base_delay + jitter

        else:
            delay = self.config.initial_delay_ms / 1000

        # Apply maximum delay limit
        max_delay = self.config.max_delay_ms / 1000
        return min(delay, max_delay)


class GracefulDegradation:
    """Handles graceful degradation strategies."""

    def __init__(self, config: DegradationConfig | None = None):
        self.config = config or DegradationConfig()
        self.degraded_services: list[str] = []
        self.fallback_cache: dict[str, Any] = {}
        self.offline_mode = False
        self._lock = threading.Lock()

    def degrade_service(self, service_name: str, reason: str = ""):
        """Mark service as degraded."""
        with self._lock:
            if service_name not in self.degraded_services:
                self.degraded_services.append(service_name)
                logging.warning(f"Service {service_name} degraded: {reason}")

    def restore_service(self, service_name: str):
        """Restore degraded service."""
        with self._lock:
            if service_name in self.degraded_services:
                self.degraded_services.remove(service_name)
                logging.info(f"Service {service_name} restored")

    def is_degraded(self, service_name: str) -> bool:
        """Check if service is degraded."""
        return service_name in self.degraded_services

    def enable_offline_mode(self):
        """Enable offline mode."""
        self.offline_mode = True
        logging.warning("System entering offline mode")

    def disable_offline_mode(self):
        """Disable offline mode."""
        self.offline_mode = False
        logging.info("System exiting offline mode")

    def cache_fallback_data(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Cache data for fallback use."""
        expiry = time.time() + ttl_seconds
        self.fallback_cache[key] = {
            "data": data,
            "expiry": expiry,
            "cached_at": time.time(),
        }

    def get_fallback_data(self, key: str) -> Any | None:
        """Get cached fallback data."""
        if key not in self.fallback_cache:
            return None

        cached_item = self.fallback_cache[key]
        if time.time() > cached_item["expiry"]:
            del self.fallback_cache[key]
            return None

        return cached_item["data"]

    def fallback_response(
        self, service_name: str, operation: str, default_response: Any = None
    ) -> Any:
        """Generate fallback response for degraded service."""
        if not self.config.enable_fallback:
            raise ServiceDegradedError(
                f"Service {service_name} is degraded and fallback disabled"
            )

        # Try cached data first
        cache_key = f"{service_name}:{operation}"
        cached_data = self.get_fallback_data(cache_key)
        if cached_data is not None:
            logging.info(f"Using cached fallback data for {service_name}:{operation}")
            return cached_data

        # Return default response
        if default_response is not None:
            return default_response

        # Offline mode response
        if self.offline_mode and self.config.offline_mode:
            return {
                "status": "offline",
                "message": f"Service {service_name} unavailable in offline mode",
                "operation": operation,
                "timestamp": time.time(),
            }

        raise ServiceDegradedError(
            f"No fallback available for {service_name}:{operation}"
        )


class HealthMonitor:
    """System health monitoring."""

    def __init__(self):
        self.health_checks: dict[str, HealthCheck] = {}
        self.health_status = SystemHealth(overall_status=HealthStatus.HEALTHY)
        self.monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()

    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check

    def start_monitoring(self, interval_ms: float = 30000.0):
        """Start health monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_ms,), daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _monitoring_loop(self, interval_ms: float):
        """Health monitoring loop."""
        interval_seconds = interval_ms / 1000

        while self.monitoring_active:
            try:
                self.check_system_health()
                time.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                time.sleep(interval_seconds)

    def check_system_health(self) -> SystemHealth:
        """Check overall system health."""
        with self._lock:
            self.health_status.last_check = time.time()
            self.health_status.component_health.clear()
            self.health_status.degraded_services.clear()
            self.health_status.critical_failures.clear()

            for name, check in self.health_checks.items():
                try:
                    # Execute health check with timeout
                    start_time = time.time()
                    result = self._execute_health_check(check)
                    (time.time() - start_time) * 1000

                    if result:
                        self.health_status.component_health[name] = HealthStatus.HEALTHY
                    else:
                        if check.critical:
                            self.health_status.component_health[name] = (
                                HealthStatus.CRITICAL
                            )
                            self.health_status.critical_failures.append(name)
                        else:
                            self.health_status.component_health[name] = (
                                HealthStatus.DEGRADED
                            )
                            self.health_status.degraded_services.append(name)

                except Exception as e:
                    logging.error(f"Health check {name} failed: {e}")
                    self.health_status.component_health[name] = HealthStatus.CRITICAL
                    if check.critical:
                        self.health_status.critical_failures.append(name)
                    else:
                        self.health_status.degraded_services.append(name)

            # Determine overall status
            self._update_overall_status()

        return self.health_status

    def _execute_health_check(self, check: HealthCheck) -> bool:
        """Execute individual health check."""
        try:
            return check.check_func()
        except Exception:
            return False

    def _update_overall_status(self):
        """Update overall system status."""
        if self.health_status.critical_failures:
            self.health_status.overall_status = HealthStatus.CRITICAL
        elif self.health_status.degraded_services:
            self.health_status.overall_status = HealthStatus.DEGRADED
        else:
            self.health_status.overall_status = HealthStatus.HEALTHY

    def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "overall_status": self.health_status.overall_status.value,
            "last_check": self.health_status.last_check,
            "component_health": {
                name: status.value
                for name, status in self.health_status.component_health.items()
            },
            "degraded_services": self.health_status.degraded_services,
            "critical_failures": self.health_status.critical_failures,
            "total_checks": len(self.health_checks),
            "monitoring_active": self.monitoring_active,
        }


class ResilienceManager:
    """Main resilience management system."""

    def __init__(self):
        self.error_tracker = ErrorTracker()
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_handlers: dict[str, RetryHandler] = {}
        self.degradation = GracefulDegradation()
        self.health_monitor = HealthMonitor()

        # Register default health checks
        self._register_default_health_checks()

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def get_retry_handler(
        self, name: str, config: RetryConfig | None = None
    ) -> RetryHandler:
        """Get or create retry handler."""
        if name not in self.retry_handlers:
            self.retry_handlers[name] = RetryHandler(config)
        return self.retry_handlers[name]

    def resilient_call(
        self,
        component: str,
        operation: str,
        func: Callable,
        *args,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        retry_config: RetryConfig | None = None,
        fallback_func: Callable | None = None,
        **kwargs,
    ) -> Any:
        """Execute function with full resilience protection."""

        # Get resilience components
        cb_name = f"{component}_{operation}"
        circuit_breaker = self.get_circuit_breaker(cb_name, circuit_breaker_config)
        retry_handler = self.get_retry_handler(cb_name, retry_config)

        try:
            # Wrap function with circuit breaker and retry
            resilient_func = circuit_breaker(retry_handler(func))
            return resilient_func(*args, **kwargs)

        except Exception as e:
            # Record error
            error_id = self.error_tracker.record_error(
                component=component,
                operation=operation,
                error=e,
                severity=self._determine_error_severity(e),
                metadata={
                    "circuit_breaker_state": circuit_breaker.state.value,
                    "retry_attempts": retry_config.max_attempts if retry_config else 3,
                },
            )

            # Try fallback if available
            if fallback_func:
                try:
                    logging.warning(
                        f"Using fallback for {component}:{operation} (error: {error_id})"
                    )
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    self.error_tracker.record_error(
                        component=component,
                        operation=f"{operation}_fallback",
                        error=fallback_error,
                        severity=ErrorSeverity.HIGH,
                    )

            # Check if service should be degraded
            if circuit_breaker.state == CircuitBreakerState.OPEN:
                self.degradation.degrade_service(
                    component, f"Circuit breaker open: {error_id}"
                )

            raise

    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(error, SystemExit | KeyboardInterrupt):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, ConnectionError | TimeoutError):
            return ErrorSeverity.HIGH
        elif isinstance(error, ValueError | TypeError):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _register_default_health_checks(self):
        """Register default system health checks."""

        def check_memory_usage():
            """Check system memory usage."""
            try:
                import psutil

                memory = psutil.virtual_memory()
                return memory.percent < 85  # Healthy if under 85%
            except ImportError:
                return True  # Assume healthy if psutil not available

        def check_disk_space():
            """Check disk space."""
            try:
                import psutil

                disk = psutil.disk_usage("/")
                return disk.percent < 90  # Healthy if under 90%
            except ImportError:
                return True

        def check_python_import():
            """Check that core Python modules can be imported."""
            try:
                import json
                import os
                import sys

                return True
            except ImportError:
                return False

        # Register health checks
        self.health_monitor.register_health_check(
            HealthCheck("memory_usage", check_memory_usage, critical=False)
        )
        self.health_monitor.register_health_check(
            HealthCheck("disk_space", check_disk_space, critical=False)
        )
        self.health_monitor.register_health_check(
            HealthCheck("python_imports", check_python_import, critical=True)
        )

    def start_monitoring(self):
        """Start health monitoring."""
        self.health_monitor.start_monitoring()

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.health_monitor.stop_monitoring()

    def get_resilience_report(self) -> dict[str, Any]:
        """Get comprehensive resilience report."""
        return {
            "error_stats": self.error_tracker.get_error_stats(),
            "circuit_breakers": {
                name: cb.get_stats() for name, cb in self.circuit_breakers.items()
            },
            "degraded_services": self.degradation.degraded_services,
            "offline_mode": self.degradation.offline_mode,
            "health_status": self.health_monitor.get_health_report(),
            "fallback_cache_size": len(self.degradation.fallback_cache),
            "monitoring_active": self.health_monitor.monitoring_active,
        }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class ServiceDegradedError(Exception):
    """Raised when service is degraded and no fallback available."""

    pass


class ResilienceConfigurationError(Exception):
    """Raised when resilience configuration is invalid."""

    pass


# Global resilience manager instance
_resilience_manager = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


def resilient(
    component: str,
    operation: str = "default",
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    retry_config: RetryConfig | None = None,
    fallback_func: Callable | None = None,
):
    """Decorator for adding resilience to functions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            return manager.resilient_call(
                component=component,
                operation=operation or func.__name__,
                func=func,
                *args,
                circuit_breaker_config=circuit_breaker_config,
                retry_config=retry_config,
                fallback_func=fallback_func,
                **kwargs,
            )

        return wrapper

    return decorator


@contextmanager
def resilience_context(component: str, operation: str = "context"):
    """Context manager for resilience protection."""
    manager = get_resilience_manager()
    start_time = time.time()

    try:
        yield manager
    except Exception as e:
        manager.error_tracker.record_error(
            component=component,
            operation=operation,
            error=e,
            metadata={"execution_time_ms": (time.time() - start_time) * 1000},
        )
        raise


def setup_resilience_logging():
    """Setup logging for resilience components."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create resilience-specific logger
    resilience_logger = logging.getLogger("resilience")
    resilience_logger.setLevel(logging.INFO)

    return resilience_logger


# Initialize resilience system
resilience_logger = setup_resilience_logging()
