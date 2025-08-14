"""Tests for Monitoring & Observability System - Prompt I

Comprehensive validation of monitoring and observability including:
- Metrics collection and aggregation
- Distributed tracing functionality
- Structured logging and log management
- Health monitoring and alerting

Integration Point: Observability validation for Phase 4 testing
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from monitoring.observability_system import (
    AlertManager,
    AlertSeverity,
    DistributedTracer,
    HealthMonitor,
    LogHandler,
    LogLevel,
    MetricsCollector,
    ObservabilitySystem,
    monitored_operation,
    timed_operation,
    traced_operation,
)


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.collector = MetricsCollector(self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        assert len(self.collector.metrics_buffer) == 0
        assert len(self.collector.counters) == 0
        assert len(self.collector.gauges) == 0

    def test_counter_metrics(self):
        """Test counter metric recording."""
        self.collector.record_counter("test_counter", 1.0)
        self.collector.record_counter("test_counter", 2.0)

        assert len(self.collector.metrics_buffer) == 2
        assert self.collector.counters["test_counter"] == 3.0

    def test_gauge_metrics(self):
        """Test gauge metric recording."""
        self.collector.record_gauge("test_gauge", 42.0)
        self.collector.record_gauge("test_gauge", 43.0)

        assert len(self.collector.metrics_buffer) == 2
        assert self.collector.gauges["test_gauge"] == 43.0

    def test_histogram_metrics(self):
        """Test histogram metric recording."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.collector.record_histogram("test_histogram", value)

        assert len(self.collector.metrics_buffer) == 5
        assert len(self.collector.histograms["test_histogram"]) == 5

    def test_timer_metrics(self):
        """Test timer metric recording."""
        self.collector.record_timer("test_timer", 100.0)

        assert len(self.collector.metrics_buffer) == 1
        assert len(self.collector.histograms["test_timer_duration_ms"]) == 1

    def test_metrics_with_labels(self):
        """Test metrics with labels."""
        labels = {"service": "test", "endpoint": "/api"}
        self.collector.record_counter("requests", 1.0, labels)

        key = self.collector._metric_key("requests", labels)
        assert "service=test" in key
        assert "endpoint=/api" in key

    def test_flush_to_storage(self):
        """Test flushing metrics to storage."""
        self.collector.record_counter("test", 1.0)
        self.collector.record_gauge("test2", 2.0)

        flushed = self.collector.flush_to_storage()
        assert flushed == 2
        assert len(self.collector.metrics_buffer) == 0

    def test_metric_summary(self):
        """Test metric summary generation."""
        # Record some values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.collector.record_gauge("summary_test", value)

        self.collector.flush_to_storage()

        summary = self.collector.get_metric_summary("summary_test")

        assert summary["count"] == 5
        assert summary["min"] == 1.0
        assert summary["max"] == 5.0
        assert summary["mean"] == 3.0


class TestDistributedTracer:
    """Test distributed tracing functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.tracer = DistributedTracer("test_service", self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_tracer_initialization(self):
        """Test tracer initialization."""
        assert self.tracer.service_name == "test_service"
        assert len(self.tracer.active_spans) == 0
        assert len(self.tracer.completed_spans) == 0

    def test_span_creation_and_finishing(self):
        """Test span lifecycle."""
        span = self.tracer.start_span("test_operation")

        assert span.service_name == "test_service"
        assert span.operation_name == "test_operation"
        assert span.start_time > 0
        assert span.end_time is None
        assert span.span_id in self.tracer.active_spans

        self.tracer.finish_span(span)

        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.span_id not in self.tracer.active_spans
        assert len(self.tracer.completed_spans) == 1

    def test_span_hierarchy(self):
        """Test parent-child span relationships."""
        parent_span = self.tracer.start_span("parent_operation")
        child_span = self.tracer.start_span(
            "child_operation",
            parent_span_id=parent_span.span_id,
            trace_id=parent_span.trace_id,
        )

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id

    def test_span_attributes_and_events(self):
        """Test span attributes and events."""
        span = self.tracer.start_span("test_operation")

        self.tracer.set_span_attribute(span, "user_id", "12345")
        self.tracer.add_span_event(span, "cache_miss", {"key": "test_key"})

        assert span.attributes["user_id"] == "12345"
        assert len(span.events) == 1
        assert span.events[0]["name"] == "cache_miss"

    def test_traced_operation_context_manager(self):
        """Test traced operation context manager."""
        with traced_operation(self.tracer, "context_test") as span:
            assert span.operation_name == "context_test"
            assert span.span_id in self.tracer.active_spans

        assert span.span_id not in self.tracer.active_spans
        assert span.status == "ok"

    def test_traced_operation_exception_handling(self):
        """Test traced operation with exception."""
        with pytest.raises(ValueError):
            with traced_operation(self.tracer, "failing_operation") as span:
                raise ValueError("Test error")

        assert span.status == "error"
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"

    def test_flush_to_storage_and_retrieval(self):
        """Test flushing spans and retrieving traces."""
        # Create a trace with multiple spans
        parent_span = self.tracer.start_span("parent")
        child_span = self.tracer.start_span(
            "child", parent_span_id=parent_span.span_id, trace_id=parent_span.trace_id
        )

        self.tracer.finish_span(child_span)
        self.tracer.finish_span(parent_span)

        # Flush to storage
        flushed = self.tracer.flush_to_storage()
        assert flushed == 2

        # Retrieve trace
        trace_spans = self.tracer.get_trace(parent_span.trace_id)
        assert len(trace_spans) == 2

        # Verify span order and relationships
        span_by_id = {span.span_id: span for span in trace_spans}
        retrieved_parent = span_by_id[parent_span.span_id]
        retrieved_child = span_by_id[child_span.span_id]

        assert retrieved_child.parent_span_id == retrieved_parent.span_id


class TestLogHandler:
    """Test logging functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.logger = LogHandler("test_service", self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_logger_initialization(self):
        """Test logger initialization."""
        assert self.logger.service_name == "test_service"
        assert len(self.logger.log_buffer) == 0

    def test_logging_levels(self):
        """Test different logging levels."""
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")

        assert len(self.logger.log_buffer) == 5

        levels = [entry.level for entry in self.logger.log_buffer]
        assert LogLevel.DEBUG in levels
        assert LogLevel.INFO in levels
        assert LogLevel.WARNING in levels
        assert LogLevel.ERROR in levels
        assert LogLevel.CRITICAL in levels

    def test_logging_with_trace_context(self):
        """Test logging with trace context."""
        trace_id = "test_trace_123"
        span_id = "test_span_456"

        self.logger.info(
            "Test message", trace_id=trace_id, span_id=span_id, user_id="12345"
        )

        log_entry = self.logger.log_buffer[0]
        assert log_entry.trace_id == trace_id
        assert log_entry.span_id == span_id
        assert log_entry.attributes["user_id"] == "12345"

    def test_flush_to_storage(self):
        """Test flushing logs to storage."""
        self.logger.info("Test log message")
        self.logger.error("Test error message")

        flushed = self.logger.flush_to_storage()
        assert flushed == 2
        assert len(self.logger.log_buffer) == 0


class TestAlertManager:
    """Test alert management functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.alert_manager = AlertManager(self.temp_db.name)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        assert len(self.alert_manager.active_alerts) == 0
        assert len(self.alert_manager.alert_rules) == 0

    def test_alert_rule_creation(self):
        """Test creating alert rules."""
        self.alert_manager.add_alert_rule(
            name="High CPU",
            condition="cpu_usage",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            description="CPU usage is high",
        )

        assert len(self.alert_manager.alert_rules) == 1
        rule = self.alert_manager.alert_rules[0]
        assert rule["name"] == "High CPU"
        assert rule["threshold"] == 80.0

    def test_alert_triggering(self):
        """Test triggering alerts."""
        alert = self.alert_manager.trigger_alert(
            name="Test Alert",
            description="Test alert description",
            severity=AlertSeverity.ERROR,
            value=95.0,
            threshold=80.0,
        )

        assert alert.name == "Test Alert"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.value == 95.0
        assert alert.triggered_at > 0
        assert alert.resolved_at is None

        assert alert.alert_id in self.alert_manager.active_alerts

    def test_alert_resolution(self):
        """Test resolving alerts."""
        alert = self.alert_manager.trigger_alert(
            name="Test Alert",
            description="Test description",
            severity=AlertSeverity.WARNING,
        )

        alert_id = alert.alert_id
        assert alert_id in self.alert_manager.active_alerts

        self.alert_manager.resolve_alert(alert_id)

        assert alert_id not in self.alert_manager.active_alerts
        assert alert.resolved_at is not None

    def test_alert_condition_checking(self):
        """Test automatic alert condition checking."""
        self.alert_manager.add_alert_rule(
            name="High Memory",
            condition="memory_usage",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
        )

        # Trigger condition
        metrics = {"memory_usage": 95.0}
        self.alert_manager.check_alert_conditions(metrics)

        active_alerts = self.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].name == "High Memory"


class TestHealthMonitor:
    """Test health monitoring functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.health_monitor = HealthMonitor()

    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        assert len(self.health_monitor.health_checks) == 0
        assert len(self.health_monitor.health_status) == 0
        assert self.health_monitor.monitoring_active is False

    def test_health_check_registration(self):
        """Test registering health checks."""

        def dummy_check():
            return True

        self.health_monitor.register_health_check("test_check", dummy_check)

        assert "test_check" in self.health_monitor.health_checks
        assert self.health_monitor.health_checks["test_check"] == dummy_check

    def test_health_check_execution(self):
        """Test executing health checks."""

        def passing_check():
            return True

        def failing_check():
            return False

        def detailed_check():
            return {"status": "healthy", "details": {"uptime": 3600}}

        self.health_monitor.register_health_check("passing", passing_check)
        self.health_monitor.register_health_check("failing", failing_check)
        self.health_monitor.register_health_check("detailed", detailed_check)

        self.health_monitor.run_health_checks()

        status = self.health_monitor.get_health_status()

        assert len(status) == 3
        assert status["passing"].status == "healthy"
        assert status["failing"].status == "unhealthy"
        assert status["detailed"].status == "healthy"
        assert status["detailed"].details["uptime"] == 3600

    def test_health_check_error_handling(self):
        """Test health check error handling."""

        def error_check():
            raise Exception("Health check failed")

        self.health_monitor.register_health_check("error_check", error_check)
        self.health_monitor.run_health_checks()

        status = self.health_monitor.get_health_status()
        assert status["error_check"].status == "error"
        assert "Health check failed" in status["error_check"].details["error"]


class TestObservabilitySystem:
    """Test complete observability system."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.observability = ObservabilitySystem(
            service_name="test_service",
            storage_backend=self.temp_db.name,
            flush_interval=0.1,  # Fast flush for testing
        )

    def teardown_method(self):
        """Cleanup after each test."""
        self.observability.stop()
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_observability_system_initialization(self):
        """Test observability system initialization."""
        assert self.observability.service_name == "test_service"
        assert hasattr(self.observability, "metrics")
        assert hasattr(self.observability, "tracer")
        assert hasattr(self.observability, "logger")
        assert hasattr(self.observability, "alerts")
        assert hasattr(self.observability, "health")

    def test_system_startup_and_shutdown(self):
        """Test system startup and shutdown."""
        self.observability.start()

        # System should be running
        assert self.observability.health.monitoring_active is True
        assert self.observability._flush_active is True

        self.observability.stop()

        # System should be stopped
        assert self.observability.health.monitoring_active is False
        assert self.observability._flush_active is False

    def test_integrated_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        self.observability.start()

        # Record some metrics
        self.observability.metrics.record_counter("requests", 1.0)
        self.observability.metrics.record_gauge("cpu_usage", 75.0)

        # Create a trace
        with traced_operation(self.observability.tracer, "test_operation") as span:
            self.observability.logger.info(
                "Processing request",
                trace_id=span.trace_id,
                span_id=span.span_id,
                user_id="12345",
            )

            # Simulate some work
            time.sleep(0.01)

        # Trigger an alert
        self.observability.alerts.trigger_alert(
            name="Test Alert",
            description="Test alert for integration",
            severity=AlertSeverity.INFO,
        )

        # Wait for auto-flush
        time.sleep(0.2)

        # Verify data was collected
        dashboard_data = self.observability.get_dashboard_data()

        assert dashboard_data["service_name"] == "test_service"
        assert len(dashboard_data["active_alerts"]) == 1
        assert dashboard_data["active_alerts"][0]["name"] == "Test Alert"

    def test_timed_operation_decorator(self):
        """Test timed operation decorator."""

        @timed_operation(self.observability, "decorated_operation")
        def test_function():
            time.sleep(0.01)
            return "success"

        result = test_function()

        assert result == "success"
        # Verify metrics were recorded
        assert len(self.observability.metrics.metrics_buffer) > 0

    def test_monitored_operation_context_manager(self):
        """Test monitored operation context manager."""
        with monitored_operation(self.observability, "context_operation") as context:
            assert "span" in context
            assert "start_time" in context
            assert context["operation_name"] == "context_operation"

        # Verify metrics were recorded
        assert len(self.observability.metrics.metrics_buffer) > 0

    def test_dashboard_data_generation(self):
        """Test dashboard data generation."""
        # Add some test data
        self.observability.metrics.record_counter("test_metric", 1.0)
        self.observability.alerts.trigger_alert(
            "Test Alert", "Description", AlertSeverity.WARNING
        )

        dashboard_data = self.observability.get_dashboard_data()

        required_keys = [
            "service_name",
            "timestamp",
            "health_status",
            "active_alerts",
            "system_stats",
        ]

        for key in required_keys:
            assert key in dashboard_data

        assert dashboard_data["service_name"] == "test_service"
        assert len(dashboard_data["active_alerts"]) == 1


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_microservice_observability_simulation(self):
        """Test microservice observability simulation."""
        # Create observability for multiple services
        services = {}

        for service_name in ["api_gateway", "user_service", "order_service"]:
            temp_db = tempfile.NamedTemporaryFile(delete=False)
            temp_db.close()

            obs = ObservabilitySystem(
                service_name=service_name, storage_backend=temp_db.name
            )
            obs.start()
            services[service_name] = (obs, temp_db.name)

        try:
            # Simulate request flow across services
            gateway_obs = services["api_gateway"][0]
            user_obs = services["user_service"][0]
            order_obs = services["order_service"][0]

            # Start trace in API gateway
            with traced_operation(gateway_obs.tracer, "handle_request") as gateway_span:
                gateway_obs.metrics.record_counter("requests_received", 1.0)

                # Continue trace in user service
                with traced_operation(
                    user_obs.tracer,
                    "get_user",
                    parent_span_id=gateway_span.span_id,
                    attributes={"trace_id": gateway_span.trace_id},
                ) as user_span:
                    user_obs.metrics.record_timer("user_lookup_time", 50.0)

                    # Continue trace in order service
                    with traced_operation(
                        order_obs.tracer,
                        "create_order",
                        parent_span_id=user_span.span_id,
                        attributes={"trace_id": gateway_span.trace_id},
                    ) as order_span:
                        order_obs.metrics.record_counter("orders_created", 1.0)

            # Verify metrics were collected in all services
            for service_name, (obs, _) in services.items():
                assert len(obs.metrics.metrics_buffer) > 0

        finally:
            # Cleanup
            for obs, db_path in services.values():
                obs.stop()
                try:
                    os.unlink(db_path)
                except PermissionError:
                    pass


if __name__ == "__main__":
    # Run observability system validation
    print("=== Testing Monitoring & Observability System ===")

    # Test metrics collection
    print("Testing metrics collection...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        collector = MetricsCollector(tmp.name)
        collector.record_counter("test_counter", 5.0)
        collector.record_gauge("test_gauge", 42.0)
        flushed = collector.flush_to_storage()
        print(f"OK Metrics: {flushed} metrics flushed")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test distributed tracing
    print("Testing distributed tracing...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tracer = DistributedTracer("test_service", tmp.name)

        with traced_operation(tracer, "test_operation") as span:
            tracer.set_span_attribute(span, "test_attr", "test_value")
            tracer.add_span_event(span, "test_event")

        flushed = tracer.flush_to_storage()
        print(f"OK Tracing: {flushed} spans flushed")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test logging
    print("Testing structured logging...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        logger = LogHandler("test_service", tmp.name)
        logger.info("Test log message", user_id="12345")
        logger.error("Test error message", error_code="E001")
        flushed = logger.flush_to_storage()
        print(f"OK Logging: {flushed} logs flushed")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test alerting
    print("Testing alert management...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        alert_mgr = AlertManager(tmp.name)
        alert = alert_mgr.trigger_alert(
            "Test Alert", "Test description", AlertSeverity.WARNING
        )
        active_alerts = alert_mgr.get_active_alerts()
        print(f"OK Alerting: {len(active_alerts)} active alerts")
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    # Test health monitoring
    print("Testing health monitoring...")
    health_monitor = HealthMonitor()
    health_monitor.register_health_check("test", lambda: True)
    health_monitor.run_health_checks()
    status = health_monitor.get_health_status()
    print(f"OK Health monitoring: {len(status)} checks completed")

    # Test complete observability system
    print("Testing complete observability system...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        obs = ObservabilitySystem("test_service", tmp.name, flush_interval=0.1)
        obs.start()

        # Generate some observability data
        obs.metrics.record_counter("test_requests", 10.0)
        with traced_operation(obs.tracer, "test_op") as span:
            obs.logger.info("Test message", trace_id=span.trace_id)

        time.sleep(0.2)  # Wait for flush

        dashboard_data = obs.get_dashboard_data()
        print(f"OK Complete system: service={dashboard_data['service_name']}")

        obs.stop()
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass

    print("=== Monitoring & observability system validation completed ===")
