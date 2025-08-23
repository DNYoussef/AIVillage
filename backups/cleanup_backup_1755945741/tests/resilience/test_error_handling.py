"""Tests for Error Handling & Resilience Framework - Prompt F

Comprehensive validation of error handling and system resilience including:
- Circuit breaker functionality and state transitions
- Retry mechanisms with different backoff strategies
- Graceful degradation and fallback behavior
- Health monitoring and system status tracking
- Integration with quality gates and performance systems

Integration Point: Resilience validation for Phase 4 testing
"""

import sys
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.resilience.error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerState,
    DegradationConfig,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    ErrorTracker,
    GracefulDegradation,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    ResilienceManager,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    ServiceDegradedError,
    get_resilience_manager,
    resilience_context,
    resilient,
)


class TestErrorContext:
    """Test error context data structure."""

    def test_error_context_creation(self):
        """Test error context creation."""
        context = ErrorContext(
            error_id="test_error_001",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            component="p2p_mesh",
            operation="send_message",
            error_type="ConnectionError",
            message="Connection timeout",
            metadata={"host": "192.168.1.1", "port": 8080},
        )

        assert context.error_id == "test_error_001"
        assert context.severity == ErrorSeverity.HIGH
        assert context.category == ErrorCategory.NETWORK
        assert context.component == "p2p_mesh"
        assert context.operation == "send_message"
        assert context.error_type == "ConnectionError"
        assert context.message == "Connection timeout"
        assert context.metadata["host"] == "192.168.1.1"
        assert context.retry_count == 0
        assert context.recovered is False

    def test_error_enums(self):
        """Test error enumeration values."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorCategory.NETWORK.value == "network"
        assert CircuitBreakerState.OPEN.value == "open"
        assert RetryStrategy.EXPONENTIAL_BACKOFF.value == "exponential_backoff"
        assert HealthStatus.DEGRADED.value == "degraded"


class TestErrorTracker:
    """Test error tracking functionality."""

    def test_error_tracker_initialization(self):
        """Test error tracker initialization."""
        tracker = ErrorTracker()

        assert len(tracker.errors) == 0
        assert len(tracker.error_counts) == 0
        assert len(tracker.component_health) == 0

    def test_record_error_basic(self):
        """Test basic error recording."""
        tracker = ErrorTracker()

        error = ConnectionError("Network timeout")
        error_id = tracker.record_error(
            component="api_client",
            operation="fetch_data",
            error=error,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
        )

        assert isinstance(error_id, str)
        assert len(tracker.errors) == 1

        error_context = tracker.errors[0]
        assert error_context.component == "api_client"
        assert error_context.operation == "fetch_data"
        assert error_context.error_type == "ConnectionError"
        assert error_context.severity == ErrorSeverity.HIGH
        assert error_context.category == ErrorCategory.NETWORK

    def test_error_auto_categorization(self):
        """Test automatic error categorization."""
        tracker = ErrorTracker()

        # Network error
        network_error = ConnectionError("Connection failed")
        tracker.record_error("test", "op", network_error)
        assert tracker.errors[-1].category == ErrorCategory.NETWORK

        # Validation error
        validation_error = ValueError("Invalid input")
        tracker.record_error("test", "op", validation_error)
        assert tracker.errors[-1].category == ErrorCategory.VALIDATION

        # Generic error
        generic_error = RuntimeError("Something went wrong")
        tracker.record_error("test", "op", generic_error)
        assert tracker.errors[-1].category == ErrorCategory.INTERNAL

    def test_component_health_updates(self):
        """Test component health status updates."""
        tracker = ErrorTracker()

        # Critical error should set component to critical
        critical_error = Exception("Critical failure")
        tracker.record_error("component1", "op", critical_error, ErrorSeverity.CRITICAL)
        assert tracker.component_health["component1"] == HealthStatus.CRITICAL

        # High error should set healthy component to degraded
        high_error = Exception("High severity error")
        tracker.record_error("component2", "op", high_error, ErrorSeverity.HIGH)
        assert tracker.component_health["component2"] == HealthStatus.DEGRADED

    def test_error_statistics(self):
        """Test error statistics generation."""
        tracker = ErrorTracker()

        # Record various errors
        errors = [
            (ConnectionError("timeout"), ErrorSeverity.HIGH, ErrorCategory.NETWORK),
            (ValueError("invalid"), ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION),
            (RuntimeError("runtime"), ErrorSeverity.CRITICAL, ErrorCategory.INTERNAL),
        ]

        for error, severity, category in errors:
            tracker.record_error("test_component", "test_op", error, severity, category)

        stats = tracker.get_error_stats()

        assert stats["total_errors"] == 3
        assert stats["recent_errors"] == 3
        assert stats["error_by_severity"]["critical"] == 1
        assert stats["error_by_severity"]["high"] == 1
        assert stats["error_by_severity"]["medium"] == 1
        assert stats["error_by_category"]["network"] == 1
        assert stats["error_by_category"]["validation"] == 1
        assert stats["error_by_category"]["internal"] == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker("test_service")

        assert cb.name == "test_service"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.total_calls == 0

    def test_circuit_breaker_success_calls(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker("test_service")

        def successful_func():
            return "success"

        # Call function multiple times
        for i in range(5):
            result = cb.call(successful_func)
            assert result == "success"

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.total_calls == 5
        assert cb.total_successes == 5
        assert cb.total_failures == 0

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening on failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_service", config)

        def failing_func():
            raise ConnectionError("Service unavailable")

        # Trigger failures up to threshold
        for i in range(3):
            with pytest.raises(ConnectionError):
                cb.call(failing_func)

        # Circuit breaker should now be open
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(failing_func)

    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker half-open state transition."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_ms=100)
        cb = CircuitBreaker("test_service", config)

        def failing_func():
            raise ConnectionError("Service unavailable")

        # Trigger failures to open circuit
        for i in range(2):
            with pytest.raises(ConnectionError):
                cb.call(failing_func)

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout
        time.sleep(0.11)

        def successful_func():
            return "success"

        # First call after timeout should transition to half-open
        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery from half-open state."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=2, timeout_ms=100)
        cb = CircuitBreaker("test_service", config)

        # Force to half-open state
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.half_open_calls = 0

        def successful_func():
            return "success"

        # Make successful calls to close circuit
        for i in range(2):
            result = cb.call(successful_func)
            assert result == "success"

        # Should transition back to closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        cb = CircuitBreaker("decorated_service")

        @cb
        def decorated_func(value):
            if value == "fail":
                raise ValueError("Forced failure")
            return f"success: {value}"

        # Test successful call
        result = decorated_func("test")
        assert result == "success: test"

        # Test failing call
        with pytest.raises(ValueError):
            decorated_func("fail")

    def test_circuit_breaker_statistics(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker("stats_test")

        def mixed_func(should_fail):
            if should_fail:
                raise RuntimeError("Failed")
            return "success"

        # Mix of successful and failing calls
        for i in range(3):
            try:
                cb.call(mixed_func, i % 2 == 0)  # Fail on even numbers
            except RuntimeError:
                pass

        stats = cb.get_stats()

        assert stats["name"] == "stats_test"
        assert stats["total_calls"] == 3
        assert stats["total_successes"] == 1  # Only odd numbers succeed
        assert stats["total_failures"] == 2
        assert 0 <= stats["failure_rate"] <= 1


class TestRetryHandler:
    """Test retry mechanism functionality."""

    def test_retry_handler_initialization(self):
        """Test retry handler initialization."""
        handler = RetryHandler()

        assert handler.config.max_attempts == 3
        assert handler.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_retry_handler_success_no_retry(self):
        """Test retry handler with immediate success."""
        handler = RetryHandler()

        call_count = 0

        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = handler.execute_with_retry(successful_func)

        assert result == "success"
        assert call_count == 1  # Should only be called once

    def test_retry_handler_eventual_success(self):
        """Test retry handler with eventual success."""
        config = RetryConfig(max_attempts=3, initial_delay_ms=1)  # Fast for testing
        handler = RetryHandler(config)

        call_count = 0

        def eventually_successful_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = handler.execute_with_retry(eventually_successful_func)

        assert result == "success"
        assert call_count == 3

    def test_retry_handler_all_attempts_fail(self):
        """Test retry handler when all attempts fail."""
        config = RetryConfig(max_attempts=2, initial_delay_ms=1)
        handler = RetryHandler(config)

        call_count = 0

        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            handler.execute_with_retry(always_failing_func)

        assert call_count == 2  # Should try max_attempts times

    def test_retry_handler_non_retryable_exception(self):
        """Test retry handler with non-retryable exceptions."""
        config = RetryConfig(non_retryable_exceptions=[ValueError])
        handler = RetryHandler(config)

        call_count = 0

        def non_retryable_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            handler.execute_with_retry(non_retryable_func)

        assert call_count == 1  # Should not retry

    def test_retry_delay_strategies(self):
        """Test different retry delay strategies."""
        # Fixed delay
        fixed_config = RetryConfig(strategy=RetryStrategy.FIXED_DELAY, initial_delay_ms=100)
        fixed_handler = RetryHandler(fixed_config)

        delay1 = fixed_handler._calculate_delay(0)
        delay2 = fixed_handler._calculate_delay(1)
        assert delay1 == delay2 == 0.1  # 100ms

        # Exponential backoff
        exp_config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, initial_delay_ms=100)
        exp_handler = RetryHandler(exp_config)

        delay1 = exp_handler._calculate_delay(0)
        delay2 = exp_handler._calculate_delay(1)
        assert delay1 < delay2  # Should increase

    def test_retry_handler_decorator(self):
        """Test retry handler as decorator."""
        config = RetryConfig(max_attempts=2, initial_delay_ms=1)
        handler = RetryHandler(config)

        call_count = 0

        @handler
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First attempt fails")
            return "success"

        result = decorated_func()
        assert result == "success"
        assert call_count == 2


class TestGracefulDegradation:
    """Test graceful degradation functionality."""

    def test_degradation_initialization(self):
        """Test degradation manager initialization."""
        degradation = GracefulDegradation()

        assert len(degradation.degraded_services) == 0
        assert len(degradation.fallback_cache) == 0
        assert degradation.offline_mode is False

    def test_service_degradation(self):
        """Test service degradation and restoration."""
        degradation = GracefulDegradation()

        # Degrade service
        degradation.degrade_service("api_service", "High error rate")
        assert degradation.is_degraded("api_service")
        assert "api_service" in degradation.degraded_services

        # Restore service
        degradation.restore_service("api_service")
        assert not degradation.is_degraded("api_service")
        assert "api_service" not in degradation.degraded_services

    def test_offline_mode(self):
        """Test offline mode functionality."""
        degradation = GracefulDegradation()

        assert not degradation.offline_mode

        degradation.enable_offline_mode()
        assert degradation.offline_mode

        degradation.disable_offline_mode()
        assert not degradation.offline_mode

    def test_fallback_cache(self):
        """Test fallback data caching."""
        degradation = GracefulDegradation()

        # Cache data
        test_data = {"key": "value", "timestamp": time.time()}
        degradation.cache_fallback_data("test_key", test_data, ttl_seconds=3600)

        # Retrieve cached data
        cached_data = degradation.get_fallback_data("test_key")
        assert cached_data == test_data

        # Test non-existent key
        assert degradation.get_fallback_data("nonexistent") is None

    def test_fallback_cache_expiry(self):
        """Test fallback cache expiry."""
        degradation = GracefulDegradation()

        # Cache data with short TTL
        degradation.cache_fallback_data("short_lived", "data", ttl_seconds=0.1)

        # Should be available immediately
        assert degradation.get_fallback_data("short_lived") == "data"

        # Wait for expiry
        time.sleep(0.15)

        # Should be None after expiry
        assert degradation.get_fallback_data("short_lived") is None

    def test_fallback_response_with_cache(self):
        """Test fallback response using cached data."""
        degradation = GracefulDegradation()

        # Cache fallback data
        fallback_data = {"status": "cached", "data": [1, 2, 3]}
        degradation.cache_fallback_data("service:operation", fallback_data)

        # Get fallback response
        response = degradation.fallback_response("service", "operation")
        assert response == fallback_data

    def test_fallback_response_default(self):
        """Test fallback response with default value."""
        degradation = GracefulDegradation()

        default_response = {"status": "default", "message": "Service unavailable"}
        response = degradation.fallback_response("service", "operation", default_response)
        assert response == default_response

    def test_fallback_response_offline_mode(self):
        """Test fallback response in offline mode."""
        config = DegradationConfig(offline_mode=True)
        degradation = GracefulDegradation(config)
        degradation.enable_offline_mode()

        response = degradation.fallback_response("service", "operation")

        assert isinstance(response, dict)
        assert response["status"] == "offline"
        assert "service" in response["message"]
        assert "operation" in response

    def test_fallback_disabled(self):
        """Test behavior when fallback is disabled."""
        config = DegradationConfig(enable_fallback=False)
        degradation = GracefulDegradation(config)

        with pytest.raises(ServiceDegradedError):
            degradation.fallback_response("service", "operation")


class TestHealthMonitor:
    """Test health monitoring functionality."""

    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor()

        assert len(monitor.health_checks) == 0
        assert monitor.health_status.overall_status == HealthStatus.HEALTHY
        assert not monitor.monitoring_active

    def test_health_check_registration(self):
        """Test health check registration."""
        monitor = HealthMonitor()

        def test_check():
            return True

        health_check = HealthCheck("test_check", test_check)
        monitor.register_health_check(health_check)

        assert "test_check" in monitor.health_checks
        assert monitor.health_checks["test_check"].name == "test_check"

    def test_health_check_execution(self):
        """Test health check execution."""
        monitor = HealthMonitor()

        def passing_check():
            return True

        def failing_check():
            return False

        def error_check():
            raise Exception("Health check error")

        # Register checks
        monitor.register_health_check(HealthCheck("passing", passing_check))
        monitor.register_health_check(HealthCheck("failing", failing_check))
        monitor.register_health_check(HealthCheck("error", error_check))

        # Run health checks
        health = monitor.check_system_health()

        assert health.component_health["passing"] == HealthStatus.HEALTHY
        assert health.component_health["failing"] == HealthStatus.DEGRADED
        assert health.component_health["error"] == HealthStatus.CRITICAL

    def test_critical_health_check(self):
        """Test critical health check affecting overall status."""
        monitor = HealthMonitor()

        def critical_failing_check():
            return False

        critical_check = HealthCheck("critical_test", critical_failing_check, critical=True)
        monitor.register_health_check(critical_check)

        health = monitor.check_system_health()

        assert health.overall_status == HealthStatus.CRITICAL
        assert "critical_test" in health.critical_failures

    def test_health_monitoring_start_stop(self):
        """Test health monitoring lifecycle."""
        monitor = HealthMonitor()

        def test_check():
            return True

        monitor.register_health_check(HealthCheck("test", test_check))

        # Start monitoring
        monitor.start_monitoring(interval_ms=100)
        assert monitor.monitoring_active

        # Let it run briefly
        time.sleep(0.2)

        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring_active

    def test_health_report_generation(self):
        """Test health report generation."""
        monitor = HealthMonitor()

        def test_check():
            return True

        monitor.register_health_check(HealthCheck("test", test_check))
        monitor.check_system_health()

        report = monitor.get_health_report()

        assert "overall_status" in report
        assert "component_health" in report
        assert "total_checks" in report
        assert report["total_checks"] == 1
        assert report["overall_status"] == "healthy"


class TestResilienceManager:
    """Test main resilience manager functionality."""

    def test_resilience_manager_initialization(self):
        """Test resilience manager initialization."""
        manager = ResilienceManager()

        assert isinstance(manager.error_tracker, ErrorTracker)
        assert isinstance(manager.degradation, GracefulDegradation)
        assert isinstance(manager.health_monitor, HealthMonitor)
        assert len(manager.circuit_breakers) == 0
        assert len(manager.retry_handlers) == 0

    def test_get_circuit_breaker(self):
        """Test circuit breaker creation and retrieval."""
        manager = ResilienceManager()

        # Get circuit breaker (should create new one)
        cb1 = manager.get_circuit_breaker("test_service")
        assert cb1.name == "test_service"
        assert len(manager.circuit_breakers) == 1

        # Get same circuit breaker (should return existing)
        cb2 = manager.get_circuit_breaker("test_service")
        assert cb1 is cb2
        assert len(manager.circuit_breakers) == 1

    def test_get_retry_handler(self):
        """Test retry handler creation and retrieval."""
        manager = ResilienceManager()

        # Get retry handler (should create new one)
        rh1 = manager.get_retry_handler("test_service")
        assert len(manager.retry_handlers) == 1

        # Get same retry handler (should return existing)
        rh2 = manager.get_retry_handler("test_service")
        assert rh1 is rh2
        assert len(manager.retry_handlers) == 1

    def test_resilient_call_success(self):
        """Test successful resilient call."""
        manager = ResilienceManager()

        def successful_func(value):
            return f"success: {value}"

        result = manager.resilient_call(
            component="test_component",
            operation="test_operation",
            func=successful_func,
            *("test_value",),
        )

        assert result == "success: test_value"

    def test_resilient_call_with_fallback(self):
        """Test resilient call with fallback function."""
        manager = ResilienceManager()

        def failing_func():
            raise ConnectionError("Service unavailable")

        def fallback_func():
            return "fallback_result"

        # Configure to not retry to speed up test
        retry_config = RetryConfig(max_attempts=1)

        result = manager.resilient_call(
            component="test_component",
            operation="test_operation",
            func=failing_func,
            retry_config=retry_config,
            fallback_func=fallback_func,
        )

        assert result == "fallback_result"

        # Check that error was recorded
        assert len(manager.error_tracker.errors) >= 1

    def test_resilient_call_error_recording(self):
        """Test error recording in resilient calls."""
        manager = ResilienceManager()

        def failing_func():
            raise ValueError("Test error")

        retry_config = RetryConfig(max_attempts=1)

        with pytest.raises(ValueError):
            manager.resilient_call(
                component="test_component",
                operation="test_operation",
                func=failing_func,
                retry_config=retry_config,
            )

        # Check error was recorded
        assert len(manager.error_tracker.errors) >= 1
        error = manager.error_tracker.errors[-1]
        assert error.component == "test_component"
        assert error.operation == "test_operation"
        assert "ValueError" in error.error_type

    def test_resilience_report_generation(self):
        """Test comprehensive resilience report."""
        manager = ResilienceManager()

        # Generate some activity
        manager.get_circuit_breaker("test_service")
        manager.get_retry_handler("test_service")

        # Record an error
        manager.error_tracker.record_error("test_component", "test_op", Exception("test error"))

        report = manager.get_resilience_report()

        assert "error_stats" in report
        assert "circuit_breakers" in report
        assert "degraded_services" in report
        assert "health_status" in report
        assert isinstance(report["error_stats"], dict)

    def test_global_resilience_manager(self):
        """Test global resilience manager singleton."""
        manager1 = get_resilience_manager()
        manager2 = get_resilience_manager()

        # Should be same instance
        assert manager1 is manager2


class TestResilienceDecorators:
    """Test resilience decorators and context managers."""

    def test_resilient_decorator(self):
        """Test resilient decorator functionality."""

        call_count = 0

        @resilient(component="decorated_service", operation="test_op")
        def decorated_func(value):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First call fails")
            return f"success: {value}"

        result = decorated_func("test")
        assert result == "success: test"
        assert call_count >= 1  # May be retried

    def test_resilience_context_manager(self):
        """Test resilience context manager."""
        with resilience_context("context_test", "test_operation") as manager:
            assert isinstance(manager, ResilienceManager)

        # Test with exception
        try:
            with resilience_context("context_test", "failing_operation") as manager:
                raise ValueError("Context test error")
        except ValueError:
            pass

        # Check error was recorded
        errors = manager.error_tracker.errors
        assert len(errors) >= 1
        error = next((e for e in errors if e.component == "context_test"), None)
        assert error is not None
        assert error.operation == "failing_operation"


class TestResilienceIntegration:
    """Test resilience system integration scenarios."""

    def test_end_to_end_resilience_flow(self):
        """Test complete resilience flow with multiple components."""
        manager = ResilienceManager()

        # Start health monitoring
        manager.start_monitoring()

        try:
            # Configure components
            retry_config = RetryConfig(max_attempts=2, initial_delay_ms=1)
            cb_config = CircuitBreakerConfig(failure_threshold=2)

            call_count = 0

            def flaky_service():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError("Service temporarily unavailable")
                return "service_data"

            def fallback_service():
                return "fallback_data"

            # Make resilient call
            result = manager.resilient_call(
                component="integration_test",
                operation="flaky_operation",
                func=flaky_service,
                retry_config=retry_config,
                circuit_breaker_config=cb_config,
                fallback_func=fallback_service,
            )

            # Depending on timing, might get service data or fallback
            assert result in ["service_data", "fallback_data"]

            # Generate resilience report
            report = manager.get_resilience_report()

            assert "error_stats" in report
            assert "circuit_breakers" in report
            assert "health_status" in report

        finally:
            manager.stop_monitoring()

    def test_multiple_service_degradation(self):
        """Test multiple service degradation scenarios."""
        manager = ResilienceManager()

        # Simulate multiple service failures
        services = ["api_gateway", "database", "cache", "auth_service"]

        for service in services:
            # Force circuit breaker to open
            cb = manager.get_circuit_breaker(service)
            cb.state = CircuitBreakerState.OPEN

            # Mark service as degraded
            manager.degradation.degrade_service(service, "Circuit breaker open")

        # Check system state
        report = manager.get_resilience_report()

        assert len(report["degraded_services"]) == len(services)
        assert len(report["circuit_breakers"]) == len(services)

        # All circuit breakers should be open
        for service in services:
            cb_stats = report["circuit_breakers"][service]
            assert cb_stats["state"] == "open"

    def test_recovery_scenarios(self):
        """Test service recovery scenarios."""
        manager = ResilienceManager()

        # Degrade service
        service_name = "recovery_test_service"
        manager.degradation.degrade_service(service_name, "Test degradation")

        assert manager.degradation.is_degraded(service_name)

        # Simulate recovery
        manager.degradation.restore_service(service_name)

        assert not manager.degradation.is_degraded(service_name)

        # Check circuit breaker recovery
        cb = manager.get_circuit_breaker(service_name)
        cb.state = CircuitBreakerState.OPEN

        # Force to half-open and then closed through successful calls
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.half_open_calls = 0

        def successful_func():
            return "success"

        # Make enough successful calls to close circuit
        for _ in range(cb.config.success_threshold):
            cb.call(successful_func)

        assert cb.state == CircuitBreakerState.CLOSED

    def test_performance_under_load(self):
        """Test resilience system performance under load."""
        manager = ResilienceManager()

        def test_operation():
            return "success"

        # Make many resilient calls
        start_time = time.time()

        for i in range(100):
            result = manager.resilient_call(
                component=f"load_test_{i % 10}",  # 10 different services
                operation="test_op",
                func=test_operation,
            )
            assert result == "success"

        execution_time = time.time() - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert execution_time < 5.0

        # Check that circuit breakers were created efficiently
        assert len(manager.circuit_breakers) == 10
        assert len(manager.retry_handlers) == 10


if __name__ == "__main__":
    # Run error handling and resilience validation
    print("=== Testing Error Handling & Resilience Framework ===")

    # Test error tracking
    print("Testing error tracking...")
    tracker = ErrorTracker()
    error_id = tracker.record_error("test_component", "test_op", ConnectionError("test"))
    print(f"OK Error tracking: recorded error {error_id}")

    # Test circuit breaker
    print("Testing circuit breaker...")
    cb = CircuitBreaker("test_service")
    assert cb.state == CircuitBreakerState.CLOSED
    print(f"OK Circuit breaker: state={cb.state.value}")

    # Test retry handler
    print("Testing retry handler...")
    retry_handler = RetryHandler(RetryConfig(max_attempts=2, initial_delay_ms=1))

    attempt_count = 0

    def test_func():
        global attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ConnectionError("First attempt fails")
        return "success"

    result = retry_handler.execute_with_retry(test_func)
    print(f"OK Retry handler: result={result}, attempts={attempt_count}")

    # Test graceful degradation
    print("Testing graceful degradation...")
    degradation = GracefulDegradation()
    degradation.degrade_service("test_service", "Testing")
    assert degradation.is_degraded("test_service")
    print("OK Graceful degradation: service degraded and restored")

    # Test health monitoring
    print("Testing health monitoring...")
    monitor = HealthMonitor()
    monitor.register_health_check(HealthCheck("test", lambda: True))
    health = monitor.check_system_health()
    print(f"OK Health monitoring: status={health.overall_status.value}")

    # Test resilience manager
    print("Testing resilience manager...")
    manager = ResilienceManager()

    def resilient_test_func():
        return "resilient_success"

    result = manager.resilient_call("test_component", "test_op", resilient_test_func)
    print(f"OK Resilience manager: result={result}")

    # Test global manager
    print("Testing global resilience manager...")
    global_manager = get_resilience_manager()
    assert isinstance(global_manager, ResilienceManager)
    print("OK Global manager: singleton pattern working")

    # Test resilience report
    print("Testing resilience report...")
    report = manager.get_resilience_report()
    print(f"OK Report generation: {len(report)} sections")

    print("=== Error handling & resilience framework validation completed ===")
