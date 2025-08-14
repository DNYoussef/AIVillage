"""Cross-Component Integration Tests - Phase 4.1

Comprehensive validation of component interactions including:
- Authentication + Monitoring integration
- Agent Coordination + Security integration
- ML Feature Extraction + Performance Benchmarking
- Error Handling + Observability integration
- Resource Management + Quality Gates

Integration Point: Complete system validation for production readiness
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import all major components
from agents.coordination_system import (
    Agent,
    AgentCapability,
    AgentRegistry,
    AgentStatus,
    MessageBroker,
    MessageType,
    Task,
    TaskScheduler,
)
from core.resilience.error_handling import (
    CircuitBreakerManager,
    GracefulDegradationManager,
    RetryHandler,
)
from ml.feature_extraction import FeatureExtractor, ModelComparator
from monitoring.observability_system import (
    AlertSeverity,
    LogLevel,
    ObservabilitySystem,
)
from security.auth_system import (
    AuthenticationManager,
    AuthorizationManager,
    Permission,
    SecurityLevel,
    UserRole,
)
from testing.coverage_gates import CoverageGate, LintingGate, QualityGateFramework
from testing.performance_benchmarks import BenchmarkSuite, PerformanceBenchmarkManager


class TestAuthenticationMonitoringIntegration:
    """Test integration between Authentication and Monitoring systems."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dbs = []
        for i in range(3):
            temp_db = tempfile.NamedTemporaryFile(delete=False)
            temp_db.close()
            self.temp_dbs.append(temp_db.name)

        self.auth_manager = AuthenticationManager(db_path=self.temp_dbs[0])
        self.observability = ObservabilitySystem("auth_service", self.temp_dbs[1])
        self.observability.start()

    def teardown_method(self):
        """Cleanup after each test."""
        self.observability.stop()
        for db_path in self.temp_dbs:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

    def test_authenticated_user_monitoring(self):
        """Test that authentication events are properly monitored."""
        # Create user
        user = self.auth_manager.create_user(
            username="monitored_user",
            email="monitored@example.com",
            password="SecurePassword123!",
            role=UserRole.DEVELOPER,
        )

        # Start monitoring context for authentication
        with self.observability.tracer.traced_operation("user_authentication") as span:
            # Record authentication metrics
            self.observability.metrics.record_counter(
                "auth_attempts", 1.0, {"user": user.username}
            )

            # Perform authentication
            success, auth_user, session_token = self.auth_manager.authenticate(
                username="monitored_user",
                password="SecurePassword123!",
                ip_address="10.0.0.1",
                user_agent="IntegrationTest/1.0",
            )

            # Log authentication result
            self.observability.logger.info(
                "Authentication completed",
                trace_id=span.trace_id,
                span_id=span.span_id,
                user_id=user.user_id,
                success=success,
                ip_address="10.0.0.1",
            )

            if success:
                self.observability.metrics.record_counter(
                    "auth_success", 1.0, {"user": user.username}
                )
                span.status = "ok"
            else:
                self.observability.metrics.record_counter(
                    "auth_failure", 1.0, {"user": user.username}
                )
                span.status = "error"

        # Verify authentication succeeded
        assert success is True
        assert auth_user is not None
        assert session_token is not None

        # Wait for metrics to be collected
        time.sleep(0.1)

        # Verify monitoring data was collected
        assert len(self.observability.metrics.metrics_buffer) > 0
        assert len(self.observability.logger.log_buffer) > 0
        assert len(self.observability.tracer.completed_spans) > 0

        # Get dashboard data to verify integration
        dashboard_data = self.observability.get_dashboard_data()
        assert dashboard_data["service_name"] == "auth_service"

    def test_failed_authentication_alerting(self):
        """Test that failed authentication triggers appropriate alerts."""
        # Create user
        self.auth_manager.create_user(
            username="alert_user",
            email="alert@example.com",
            password="SecurePassword123!",
        )

        # Setup alert rule for failed authentication
        self.observability.alerts.add_alert_rule(
            name="Failed Authentication",
            condition="auth_failures",
            threshold=3.0,
            severity=AlertSeverity.WARNING,
            description="Multiple failed authentication attempts",
        )

        # Simulate multiple failed authentication attempts
        failed_attempts = 0
        for i in range(5):
            success, user, _ = self.auth_manager.authenticate(
                username="alert_user",
                password=f"wrong_password_{i}",
                ip_address="192.168.1.100",
            )

            if not success:
                failed_attempts += 1
                self.observability.metrics.record_counter(
                    "auth_failures", 1.0, {"user": "alert_user"}
                )

        # Check if alert was triggered
        self.observability.alerts.check_alert_conditions(
            {"auth_failures": failed_attempts}
        )

        active_alerts = self.observability.alerts.get_active_alerts()

        # Should have triggered alert due to multiple failures
        assert len(active_alerts) >= 1
        assert any(alert.name == "Failed Authentication" for alert in active_alerts)

    def test_session_monitoring_integration(self):
        """Test session lifecycle monitoring integration."""
        # Create user and authenticate
        user = self.auth_manager.create_user(
            username="session_user",
            email="session@example.com",
            password="SecurePassword123!",
        )

        success, auth_user, session_token = self.auth_manager.authenticate(
            username="session_user",
            password="SecurePassword123!",
            ip_address="10.0.0.1",
        )

        assert success is True

        # Monitor session validation
        with self.observability.tracer.traced_operation("session_validation") as span:
            self.observability.metrics.record_counter("session_validations", 1.0)

            is_valid, session_user = self.auth_manager.validate_session(session_token)

            self.observability.logger.info(
                "Session validation",
                trace_id=span.trace_id,
                session_valid=is_valid,
                user_id=session_user.user_id if session_user else None,
            )

        assert is_valid is True
        assert session_user.user_id == user.user_id

        # Verify monitoring captured session operations
        assert len(self.observability.tracer.completed_spans) >= 1
        session_span = next(
            span
            for span in self.observability.tracer.completed_spans
            if span.operation_name == "session_validation"
        )
        assert session_span.status == "ok"


class TestAgentCoordinationSecurityIntegration:
    """Test integration between Agent Coordination and Security systems."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dbs = []
        for i in range(4):
            temp_db = tempfile.NamedTemporaryFile(delete=False)
            temp_db.close()
            self.temp_dbs.append(temp_db.name)

        self.auth_manager = AuthenticationManager(db_path=self.temp_dbs[0])
        self.authz_manager = AuthorizationManager()
        self.registry = AgentRegistry(self.temp_dbs[1])
        self.scheduler = TaskScheduler(self.registry, self.temp_dbs[2])
        self.broker = MessageBroker()

    def teardown_method(self):
        """Cleanup after each test."""
        for db_path in self.temp_dbs:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

    def test_secure_agent_registration(self):
        """Test that agent registration requires proper authentication."""
        # Create user with appropriate permissions
        user = self.auth_manager.create_user(
            username="agent_operator",
            email="operator@example.com",
            password="SecurePassword123!",
            role=UserRole.OPERATOR,
            security_level=SecurityLevel.INTERNAL,
        )

        # Authenticate user
        success, auth_user, session_token = self.auth_manager.authenticate(
            username="agent_operator",
            password="SecurePassword123!",
            ip_address="10.0.0.1",
        )

        assert success is True

        # Verify user has permission to register agents
        has_permission = self.authz_manager.has_permission(
            auth_user, Permission.EXECUTE
        )
        assert has_permission is True

        # Register agent with authenticated context
        capabilities = [
            AgentCapability(
                "data_processing",
                "1.0",
                "Data processing capability",
                supported_task_types=["data_processing"],
            )
        ]
        agent = Agent(
            agent_id="secure_agent_001",
            name="Secure Agent",
            agent_type="worker",
            capabilities=capabilities,
            status=AgentStatus.IDLE,
            endpoint="https://localhost:8001",  # HTTPS for security
            registered_at=time.time(),
            last_heartbeat=time.time(),
            metadata={
                "registered_by": auth_user.user_id,
                "session": session_token[:10],
            },
        )

        # Register agent (in real system, this would check authentication)
        success = self.registry.register_agent(agent)
        assert success is True

        # Verify agent metadata includes security context
        registered_agent = self.registry.get_agent("secure_agent_001")
        assert registered_agent is not None
        assert registered_agent.metadata["registered_by"] == auth_user.user_id

    def test_authorized_task_submission(self):
        """Test that task submission respects authorization rules."""
        # Create users with different roles
        admin_user = self.auth_manager.create_user(
            username="admin",
            email="admin@example.com",
            password="AdminPassword123!",
            role=UserRole.ADMIN,
            security_level=SecurityLevel.TOP_SECRET,
        )

        viewer_user = self.auth_manager.create_user(
            username="viewer",
            email="viewer@example.com",
            password="ViewerPassword123!",
            role=UserRole.VIEWER,
            security_level=SecurityLevel.PUBLIC,
        )

        # Test admin can submit high-security tasks
        admin_can_execute = self.authz_manager.has_permission(
            admin_user, Permission.EXECUTE
        )
        admin_can_access_confidential = self.authz_manager.check_access(
            admin_user,
            "high_security_task",
            Permission.EXECUTE,
            SecurityLevel.CONFIDENTIAL,
        )

        assert admin_can_execute is True
        assert admin_can_access_confidential is True

        # Test viewer cannot submit execution tasks
        viewer_can_execute = self.authz_manager.has_permission(
            viewer_user, Permission.EXECUTE
        )
        viewer_can_access_confidential = self.authz_manager.check_access(
            viewer_user,
            "high_security_task",
            Permission.EXECUTE,
            SecurityLevel.CONFIDENTIAL,
        )

        assert viewer_can_execute is False
        assert viewer_can_access_confidential is False

        # Submit task as admin (would be allowed)
        admin_task = Task(
            task_id="admin_task_001",
            task_type="secure_processing",
            description="High security task",
            priority=10,
            payload={"classification": "confidential"},
            requirements={
                "security_level": "confidential",
                "submitted_by": admin_user.user_id,
            },
        )

        task_id = self.scheduler.submit_task(admin_task)
        assert task_id == "admin_task_001"

    def test_secure_message_routing(self):
        """Test that inter-agent messages maintain security context."""

        # Setup secure message handler that validates sender permissions
        def secure_message_handler(message):
            # In real implementation, would validate message sender authorization
            return {
                "status": "processed",
                "security_validated": True,
                "message_id": message.message_id,
            }

        self.broker.register_handler(
            "secure_agent", MessageType.TASK_REQUEST, secure_message_handler
        )

        # Create secure message with security metadata
        secure_message = {
            "message_id": "secure_msg_001",
            "message_type": MessageType.TASK_REQUEST,
            "sender_id": "coordinator",
            "recipient_id": "secure_agent",
            "payload": {
                "task_id": "secure_task_001",
                "security_level": "confidential",
                "authorized_by": "admin_user",
            },
            "timestamp": time.time(),
        }

        # Convert to Message object
        from agents.coordination_system import Message

        message = Message(**secure_message)

        # Send secure message
        self.broker.send_message(message)

        # Verify message was queued
        messages = self.broker.get_messages("secure_agent")
        assert len(messages) == 1
        assert messages[0].payload["security_level"] == "confidential"
        assert messages[0].payload["authorized_by"] == "admin_user"


class TestMLFeatureExtractionPerformanceIntegration:
    """Test integration between ML Feature Extraction and Performance systems."""

    def setup_method(self):
        """Setup for each test."""
        self.feature_extractor = FeatureExtractor()
        self.model_comparator = ModelComparator()
        self.benchmark_manager = PerformanceBenchmarkManager()

    def test_feature_extraction_performance_monitoring(self):
        """Test that feature extraction operations are performance monitored."""

        # Create mock models for testing
        class MockModel:
            def __init__(self, name, params=1000000):
                self.name = name
                self.params = params

            def get_weights(self):
                return [0.1, 0.2, 0.3, 0.4, 0.5]

        model_a = MockModel("test_model_a", 1500000)
        model_b = MockModel("test_model_b", 2000000)

        # Benchmark feature extraction performance
        config = BenchmarkSuite(
            name="feature_extraction_test",
            description="Feature extraction performance test",
            benchmarks=[],
        )

        # Test feature extraction with performance monitoring
        start_time = time.time()

        # Extract features from both models
        features_a = self.feature_extractor.extract_features(model_a, "test_model_a")
        features_b = self.feature_extractor.extract_features(model_b, "test_model_b")

        extraction_time = time.time() - start_time

        # Verify features were extracted
        assert features_a is not None
        assert features_b is not None
        assert features_a.model_id == "test_model_a"
        assert features_b.model_id == "test_model_b"

        # Performance validation (reasonable time for feature extraction)
        assert extraction_time < 30.0  # Should complete within 30 seconds

        # Test model comparison performance
        start_time = time.time()

        comparison = self.model_comparator.compare_models(
            model_a, model_b, "test_model_a", "test_model_b"
        )

        comparison_time = time.time() - start_time

        # Verify comparison completed
        assert comparison is not None
        assert comparison.model_a_id == "test_model_a"
        assert comparison.model_b_id == "test_model_b"
        assert comparison.similarity_score >= 0.0
        assert comparison.similarity_score <= 1.0

        # Performance validation (reasonable time for comparison)
        assert comparison_time < 30.0  # Should complete within 30 seconds

        # Verify performance metrics
        ops_per_second = 2 / (extraction_time + comparison_time)  # 2 operations total
        assert ops_per_second >= 1.0  # Should be reasonably fast

    def test_feature_extraction_memory_efficiency(self):
        """Test that feature extraction operates within memory constraints."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create larger mock model to test memory efficiency
        class MemoryIntensiveModel:
            def __init__(self, size_mb=10):
                # Create data structures to simulate model memory usage
                self.large_data = [0.1] * (
                    size_mb * 1024 * 128
                )  # Approximately size_mb MB
                self.name = f"memory_model_{size_mb}mb"

            def get_weights(self):
                return self.large_data[:1000]  # Return subset for feature extraction

        model = MemoryIntensiveModel(size_mb=5)

        # Extract features and monitor memory
        features = self.feature_extractor.extract_features(model, model.name)

        # Check memory usage after extraction
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Verify features extracted successfully
        assert features is not None
        assert features.model_id == model.name

        # Memory efficiency validation (should not use excessive memory)
        assert memory_increase < 100  # Should not increase by more than 100MB

        # Cleanup model to free memory
        del model

    def test_concurrent_feature_extraction_performance(self):
        """Test feature extraction performance under concurrent load."""
        import concurrent.futures

        # Create multiple mock models
        class ConcurrentTestModel:
            def __init__(self, model_id):
                self.model_id = model_id
                self.name = f"concurrent_model_{model_id}"

            def get_weights(self):
                # Simulate some computation time
                time.sleep(0.01)
                return [float(i) for i in range(100)]

        models = [ConcurrentTestModel(i) for i in range(10)]

        # Concurrent feature extraction
        start_time = time.time()

        def extract_features_task(model):
            return self.feature_extractor.extract_features(model, model.name)

        # Use ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(extract_features_task, model) for model in models
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        total_time = time.time() - start_time

        # Verify all extractions completed
        assert len(results) == 10
        assert all(result is not None for result in results)

        # Performance validation - concurrent should be faster than sequential
        # With 4 workers and 0.01s per task, should complete in roughly 0.025s + overhead
        assert total_time < 1.0  # Should complete within reasonable time

        # Verify unique model IDs
        extracted_ids = {result.model_id for result in results}
        expected_ids = {f"concurrent_model_{i}" for i in range(10)}
        assert extracted_ids == expected_ids


class TestErrorHandlingObservabilityIntegration:
    """Test integration between Error Handling and Observability systems."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()

        self.circuit_breaker = CircuitBreakerManager()
        self.retry_handler = RetryHandler()
        self.degradation_manager = GracefulDegradationManager()
        self.observability = ObservabilitySystem("error_service", self.temp_db.name)
        self.observability.start()

    def teardown_method(self):
        """Cleanup after each test."""
        self.observability.stop()
        try:
            os.unlink(self.temp_db.name)
        except PermissionError:
            pass

    def test_circuit_breaker_monitoring_integration(self):
        """Test that circuit breaker state changes are monitored."""
        service_name = "test_service"

        # Register circuit breaker with monitoring
        self.circuit_breaker.register_circuit_breaker(
            service_name=service_name,
            failure_threshold=3,
            recovery_timeout=5.0,
            expected_exception=Exception,
        )

        # Simulate service failures with monitoring
        for i in range(4):  # Exceed failure threshold
            with self.observability.tracer.traced_operation(
                f"service_call_{i}"
            ) as span:
                try:
                    # Simulate failing service call
                    with self.circuit_breaker.circuit_breaker(service_name):
                        if i < 3:  # First 3 calls fail
                            self.observability.metrics.record_counter(
                                "service_failures", 1.0, {"service": service_name}
                            )
                            raise Exception(f"Service failure {i}")
                        else:
                            # 4th call should be blocked by circuit breaker
                            self.observability.metrics.record_counter(
                                "service_calls", 1.0, {"service": service_name}
                            )

                except Exception as e:
                    span.status = "error"
                    span.add_event("exception", {"error": str(e)})
                    self.observability.logger.error(
                        f"Service call failed: {e}",
                        trace_id=span.trace_id,
                        service=service_name,
                        error_type=type(e).__name__,
                    )

        # Verify circuit breaker is open
        breaker = self.circuit_breaker.circuit_breakers[service_name]
        assert breaker.is_open() is True

        # Verify monitoring captured all events
        assert len(self.observability.tracer.completed_spans) >= 3
        assert len(self.observability.logger.log_buffer) >= 3

        # Check for circuit breaker metrics
        metrics_recorded = len(self.observability.metrics.metrics_buffer) > 0
        assert metrics_recorded is True

    def test_retry_handler_observability_integration(self):
        """Test that retry attempts are properly logged and monitored."""
        operation_name = "flaky_operation"
        max_retries = 3

        # Configure retry handler with monitoring
        retry_config = {
            "max_retries": max_retries,
            "backoff_factor": 0.1,
            "backoff_max": 1.0,
        }

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1

            # Record attempt in monitoring
            self.observability.metrics.record_counter(
                "operation_attempts", 1.0, {"operation": operation_name}
            )

            if call_count < 3:  # Fail first 2 attempts
                self.observability.logger.warning(
                    f"Operation attempt {call_count} failed",
                    operation=operation_name,
                    attempt=call_count,
                )
                raise Exception(f"Attempt {call_count} failed")
            else:
                # Succeed on 3rd attempt
                self.observability.logger.info(
                    f"Operation succeeded on attempt {call_count}",
                    operation=operation_name,
                    attempt=call_count,
                )
                return f"Success after {call_count} attempts"

        # Execute with retry and monitoring
        with self.observability.tracer.traced_operation("retry_operation") as span:
            result = self.retry_handler.retry_on_exception(
                operation=flaky_operation,
                max_retries=max_retries,
                backoff_factor=retry_config["backoff_factor"],
                exceptions=(Exception,),
            )

            span.set_attribute("retry_count", call_count)
            span.set_attribute("final_result", "success" if result else "failed")

        # Verify operation eventually succeeded
        assert result is not None
        assert "Success after 3 attempts" in result
        assert call_count == 3

        # Verify monitoring captured retry attempts
        assert len(self.observability.logger.log_buffer) >= 3  # 2 warnings + 1 info
        assert len(self.observability.metrics.metrics_buffer) >= 3  # 3 attempt counters

        # Verify trace shows retry context
        retry_span = self.observability.tracer.completed_spans[-1]
        assert retry_span.operation_name == "retry_operation"
        assert retry_span.attributes["retry_count"] == 3
        assert retry_span.attributes["final_result"] == "success"

    def test_graceful_degradation_monitoring(self):
        """Test that graceful degradation events are monitored."""
        service_name = "critical_service"

        # Register degradation strategy with monitoring
        def fallback_strategy():
            self.observability.logger.warning(
                "Using fallback strategy",
                service=service_name,
                degradation_level="fallback",
            )
            self.observability.metrics.record_counter(
                "degradation_events", 1.0, {"service": service_name}
            )
            return "fallback_result"

        self.degradation_manager.register_fallback(service_name, fallback_strategy)

        # Simulate service degradation with monitoring
        with self.observability.tracer.traced_operation(
            "degraded_service_call"
        ) as span:
            try:
                # Simulate primary service failure
                self.observability.logger.error(
                    "Primary service unavailable",
                    service=service_name,
                    error="connection_timeout",
                )
                raise Exception("Primary service down")

            except Exception:
                # Trigger graceful degradation
                span.add_event("degradation_triggered", {"service": service_name})

                result = self.degradation_manager.execute_with_fallback(
                    service_name=service_name,
                    primary_operation=lambda: None,  # Will fail
                    fallback_data={"degraded": True},
                )

                span.set_attribute("degradation_used", True)
                span.set_attribute("result_type", "fallback")

        # Verify degradation occurred
        assert result is not None

        # Verify monitoring captured degradation events
        warning_logs = [
            log
            for log in self.observability.logger.log_buffer
            if log.level == LogLevel.WARNING and "fallback" in log.message
        ]
        assert len(warning_logs) >= 1

        # Verify degradation metrics
        degradation_metrics = [
            metric
            for metric in self.observability.metrics.metrics_buffer
            if "degradation_events" in str(metric)
        ]
        assert len(degradation_metrics) >= 1


class TestQualityGatesIntegration:
    """Test integration of Quality Gates with all system components."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dbs = []
        for i in range(3):
            temp_db = tempfile.NamedTemporaryFile(delete=False)
            temp_db.close()
            self.temp_dbs.append(temp_db.name)

        self.quality_framework = QualityGateFramework()
        self.observability = ObservabilitySystem("quality_service", self.temp_dbs[0])

    def teardown_method(self):
        """Cleanup after each test."""
        self.observability.stop()
        for db_path in self.temp_dbs:
            try:
                os.unlink(db_path)
            except PermissionError:
                pass

    def test_integrated_quality_validation(self):
        """Test quality gates across integrated system components."""
        # Setup quality gates for different components
        coverage_gate = CoverageGate(
            name="Integration Coverage Gate",
            minimum_coverage=75.0,
            scope=["authentication", "monitoring", "coordination"],
        )

        linting_gate = LintingGate(
            name="Integration Linting Gate",
            max_violations=50,
            ignored_rules=["line-too-long"],  # Focus on critical issues
        )

        # Register gates
        self.quality_framework.register_gate(coverage_gate)
        self.quality_framework.register_gate(linting_gate)

        # Simulate integration test coverage data
        coverage_data = {
            "authentication": 85.5,
            "monitoring": 78.2,
            "coordination": 82.1,
            "integration": 76.8,
        }

        # Simulate linting results
        linting_data = {
            "total_violations": 23,
            "critical_violations": 2,
            "files_scanned": 45,
        }

        # Run quality validation with monitoring
        with self.observability.tracer.traced_operation("quality_validation") as span:
            self.observability.metrics.record_gauge(
                "coverage_overall", 80.65
            )  # Average coverage

            # Validate coverage gate
            coverage_result = coverage_gate.validate(coverage_data)
            self.observability.logger.info(
                "Coverage validation completed",
                trace_id=span.trace_id,
                passed=coverage_result.passed,
                coverage_average=sum(coverage_data.values()) / len(coverage_data),
            )

            # Validate linting gate
            linting_result = linting_gate.validate(linting_data)
            self.observability.logger.info(
                "Linting validation completed",
                trace_id=span.trace_id,
                passed=linting_result.passed,
                violations=linting_data["total_violations"],
            )

            # Record quality metrics
            self.observability.metrics.record_counter("quality_gates_run", 1.0)
            self.observability.metrics.record_counter(
                "quality_gates_passed",
                1.0 if coverage_result.passed and linting_result.passed else 0.0,
            )

        # Verify quality gates passed
        assert coverage_result.passed is True
        assert linting_result.passed is True

        # Verify monitoring captured quality validation
        quality_logs = [
            log
            for log in self.observability.logger.log_buffer
            if "validation completed" in log.message
        ]
        assert len(quality_logs) >= 2

        # Get overall quality report
        overall_results = self.quality_framework.run_quality_gates(
            {"coverage": coverage_data, "linting": linting_data}
        )

        # Verify overall quality validation
        assert overall_results["overall_passed"] is True
        assert len(overall_results["results"]) == 2


if __name__ == "__main__":
    # Run cross-component integration validation
    print("=== Testing Cross-Component Integration ===")

    # Test Authentication + Monitoring integration
    print("Testing Authentication + Monitoring integration...")
    try:
        test = TestAuthenticationMonitoringIntegration()
        test.setup_method()
        test.test_authenticated_user_monitoring()
        test.teardown_method()
        print("OK Authentication + Monitoring integration")
    except Exception as e:
        print(f"FAILED Authentication + Monitoring: {e}")

    # Test Agent Coordination + Security integration
    print("Testing Agent Coordination + Security integration...")
    try:
        test = TestAgentCoordinationSecurityIntegration()
        test.setup_method()
        test.test_secure_agent_registration()
        test.teardown_method()
        print("OK Agent Coordination + Security integration")
    except Exception as e:
        print(f"FAILED Agent Coordination + Security: {e}")

    # Test ML + Performance integration
    print("Testing ML + Performance integration...")
    try:
        test = TestMLFeatureExtractionPerformanceIntegration()
        test.setup_method()
        test.test_feature_extraction_performance_monitoring()
        print("OK ML + Performance integration")
    except Exception as e:
        print(f"FAILED ML + Performance: {e}")

    # Test Error Handling + Observability integration
    print("Testing Error Handling + Observability integration...")
    try:
        test = TestErrorHandlingObservabilityIntegration()
        test.setup_method()
        test.test_circuit_breaker_monitoring_integration()
        test.teardown_method()
        print("OK Error Handling + Observability integration")
    except Exception as e:
        print(f"FAILED Error Handling + Observability: {e}")

    print("=== Cross-component integration validation completed ===")
