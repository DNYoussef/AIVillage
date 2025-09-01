"""
Consolidated Test Helper Utilities
==================================

Shared test utilities to eliminate duplicate test logic across 741+ test files.
These helpers provide common test patterns, assertions, and utilities.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import Mock, patch

import pytest
import torch


# ============================================================================
# Test Assertion Helpers
# ============================================================================


class TestAssertions:
    """Enhanced assertions for common test patterns."""

    @staticmethod
    def assert_agent_response_valid(response: Dict[str, Any]):
        """Assert agent response has valid structure."""
        assert isinstance(response, dict), "Response must be a dictionary"
        assert "result" in response or "data" in response, "Response must contain result or data"
        if "error" in response:
            assert response["error"] is None, f"Unexpected error: {response['error']}"

    @staticmethod
    def assert_security_validation_passed(validator_result: Dict[str, Any]):
        """Assert security validation passed without threats."""
        assert validator_result is not None, "Validator result cannot be None"
        assert "threat_detected" not in validator_result or not validator_result["threat_detected"]
        assert "security_score" not in validator_result or validator_result["security_score"] > 0.8

    @staticmethod
    def assert_p2p_message_delivered(delivery_status: Dict[str, Any]):
        """Assert P2P message was delivered successfully."""
        assert delivery_status["status"] in ["delivered", "acknowledged"]
        assert "message_id" in delivery_status
        assert delivery_status.get("error") is None

    @staticmethod
    def assert_model_structure_valid(model: torch.nn.Module):
        """Assert model has valid structure for Agent Forge."""
        assert isinstance(model, torch.nn.Module)

        # Check model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model must have parameters"

        # Check model can forward pass
        with torch.no_grad():
            # Try with a sample input (adjust size as needed)
            try:
                sample_input = torch.randn(1, 768)  # Common embedding size
                output = model(sample_input)
                assert output is not None, "Model forward pass failed"
            except RuntimeError:
                # Try with different input size
                sample_input = torch.randn(1, 512)
                output = model(sample_input)
                assert output is not None, "Model forward pass failed"

    @staticmethod
    def assert_performance_metrics_within_bounds(metrics: Dict[str, float], bounds: Dict[str, tuple]):
        """Assert performance metrics are within acceptable bounds."""
        for metric_name, (min_val, max_val) in bounds.items():
            assert metric_name in metrics, f"Missing metric: {metric_name}"
            actual_val = metrics[metric_name]
            assert (
                min_val <= actual_val <= max_val
            ), f"Metric {metric_name}={actual_val} outside bounds [{min_val}, {max_val}]"

    @staticmethod
    def assert_integration_test_successful(test_results: Dict[str, Any]):
        """Assert integration test completed successfully."""
        assert test_results.get("success", False), "Integration test failed"
        assert test_results.get("duration", 0) > 0, "Test must have measurable duration"

        if "error_count" in test_results:
            assert test_results["error_count"] == 0, f"Test had {test_results['error_count']} errors"


# ============================================================================
# Test Data Generation
# ============================================================================


class TestDataGenerator:
    """Generate realistic test data for various scenarios."""

    @staticmethod
    def generate_agent_conversation(turns: int = 5, agent_count: int = 2):
        """Generate multi-turn agent conversation."""
        conversation = []
        agents = [f"Agent_{i}" for i in range(agent_count)]

        for turn in range(turns):
            speaker = agents[turn % agent_count]
            message = {
                "id": f"msg_{turn:03d}",
                "speaker": speaker,
                "content": f"Message {turn + 1} from {speaker}",
                "timestamp": time.time() + turn,
                "turn": turn,
                "metadata": {
                    "confidence": 0.85 + (turn * 0.02),
                    "processing_time_ms": 50 + (turn * 5),
                },
            }
            conversation.append(message)

        return conversation

    @staticmethod
    def generate_security_test_payloads(threat_types: List[str]):
        """Generate security test payloads for specified threat types."""
        payloads = {}

        threat_patterns = {
            "code_injection": [
                'eval("malicious_code")',
                "exec(\"import os; os.system('ls')\")",
                '__import__("subprocess").call(["whoami"])',
            ],
            "command_injection": [
                'subprocess.call(["rm", "-rf", "/"])',
                'os.system("cat /etc/passwd")',
                'os.popen("whoami").read()',
            ],
            "script_injection": [
                '<script>alert("XSS")</script>',
                'javascript:alert("Injected")',
                'onerror=alert("XSS")',
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "/etc/shadow",
            ],
            "sql_injection": [
                "1' OR '1'='1",
                "test'; DROP TABLE users; --",
                "name UNION SELECT password FROM admin",
            ],
        }

        for threat_type in threat_types:
            if threat_type in threat_patterns:
                payloads[threat_type] = [
                    json.dumps({"type": "test", "payload": pattern}) for pattern in threat_patterns[threat_type]
                ]

        return payloads

    @staticmethod
    def generate_p2p_network_topology(node_count: int = 10, connectivity: float = 0.3):
        """Generate P2P network topology for testing."""
        import random

        nodes = [f"node_{i:03d}" for i in range(node_count)]
        connections = {}

        for node in nodes:
            connections[node] = {
                "peers": [],
                "transports": ["bitchat", "betanet"],
                "status": "active",
                "latency_ms": random.uniform(5, 50),
                "reliability": random.uniform(0.85, 0.99),
            }

        # Create connections based on connectivity factor
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes[i + 1 :], i + 1):
                if random.random() < connectivity:
                    connections[node_a]["peers"].append(node_b)
                    connections[node_b]["peers"].append(node_a)

        return {
            "nodes": nodes,
            "connections": connections,
            "total_nodes": node_count,
            "avg_connections_per_node": sum(len(c["peers"]) for c in connections.values()) / node_count,
        }

    @staticmethod
    def generate_model_performance_data(model_sizes: List[int], batch_sizes: List[int]):
        """Generate model performance test data."""
        import random

        performance_data = []

        for model_size in model_sizes:
            for batch_size in batch_sizes:
                # Simulate realistic performance metrics
                params = model_size * 1000000  # Convert to millions
                flops = params * batch_size * 2  # Rough FLOPS estimate

                data_point = {
                    "model_size_params": params,
                    "batch_size": batch_size,
                    "inference_time_ms": (params / 1000000) * batch_size * random.uniform(0.8, 1.2),
                    "memory_usage_mb": (params * 4 / 1024 / 1024) * random.uniform(0.9, 1.1),
                    "throughput_samples_per_sec": batch_size
                    / ((params / 1000000) * batch_size * random.uniform(0.8, 1.2) / 1000),
                    "flops": flops,
                    "accuracy": random.uniform(0.85, 0.95),
                }

                performance_data.append(data_point)

        return performance_data


# ============================================================================
# Mock Builders
# ============================================================================


class MockBuilder:
    """Builder for creating complex mock objects with realistic behavior."""

    @classmethod
    def create_agent_forge_pipeline_mock(cls, phases: List[str], success_rate: float = 0.9):
        """Create realistic Agent Forge pipeline mock."""
        pipeline = Mock()
        pipeline.phases = [(phase, Mock()) for phase in phases]
        pipeline.config = Mock()
        pipeline.metrics = Mock()

        # Mock pipeline execution
        async def mock_run_pipeline():
            import random

            result = Mock()
            result.success = random.random() < success_rate
            result.error = None if result.success else "Mock pipeline failure"
            result.metrics = {
                "phases_completed": len(phases) if result.success else random.randint(0, len(phases) - 1),
                "total_time_seconds": random.uniform(10, 60),
                "memory_peak_mb": random.uniform(100, 500),
            }
            return result

        pipeline.run_pipeline = mock_run_pipeline
        return pipeline

    @classmethod
    def create_security_validator_mock(cls, threat_detection_rate: float = 0.95):
        """Create security validator mock with configurable detection."""
        validator = Mock()
        validator.threat_patterns = {
            "CODE_INJECTION": [r"eval\s*\(", r"exec\s*\("],
            "COMMAND_INJECTION": [r"subprocess\.", r"os\.system"],
            "SCRIPT_INJECTION": [r"<script", r"javascript:"],
            "PATH_TRAVERSAL": [r"\.\./", r"\.\.\\"],
            "SQL_INJECTION": [r"'\s+OR\s+'", r"UNION\s+SELECT"],
        }

        async def mock_validate_message(message, client_info):
            import random
            import re

            # Check for threats
            threat_detected = False
            threat_type = None

            for pattern_type, patterns in validator.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        if random.random() < threat_detection_rate:
                            threat_detected = True
                            threat_type = pattern_type
                            break
                if threat_detected:
                    break

            if threat_detected:
                from unittest.mock import Mock as MockClass

                threat = MockClass()
                threat.threat_type = threat_type
                threat.severity = "CRITICAL" if "INJECTION" in threat_type else "HIGH"

                error = MockClass()
                error.threat = threat
                raise error

            return json.loads(message)

        validator.validate_message = mock_validate_message
        validator.security_events = []
        validator.get_security_report = Mock(
            return_value={"total_events": 0, "threat_type_counts": {}, "critical_events_24h": 0}
        )

        return validator

    @classmethod
    def create_p2p_network_mock(cls, node_count: int = 5, reliability: float = 0.95):
        """Create P2P network mock with realistic behavior."""
        network = Mock()
        network.nodes = {f"node_{i}": Mock() for i in range(node_count)}
        network.reliability = reliability

        async def mock_send_message(sender_id, recipient_id, message):
            import random

            if random.random() < reliability:
                return {
                    "status": "delivered",
                    "message_id": f"msg_{int(time.time())}",
                    "delivery_time_ms": random.uniform(10, 100),
                }
            else:
                raise ConnectionError(f"Failed to deliver message from {sender_id} to {recipient_id}")

        network.send_message = mock_send_message
        network.get_topology = Mock(
            return_value={
                "nodes": list(network.nodes.keys()),
                "connections": node_count * 2,  # Rough estimate
            }
        )

        return network


# ============================================================================
# Test Environment Utilities
# ============================================================================


class TestEnvironment:
    """Utilities for managing test environments."""

    @staticmethod
    @contextmanager
    def temporary_env_vars(**env_vars):
        """Temporarily set environment variables for testing."""
        import os

        original_values = {}

        # Set new values and store originals
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = str(value)

        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    @staticmethod
    @contextmanager
    def mock_imports(*module_names):
        """Mock imports for testing without dependencies."""
        mocks = {}

        for module_name in module_names:
            mock_module = Mock()
            mocks[module_name] = mock_module

        with patch.dict("sys.modules", mocks):
            yield mocks

    @staticmethod
    @contextmanager
    def capture_logs(logger_name: str = None, level: int = logging.INFO):
        """Capture log messages for testing."""
        import logging
        from io import StringIO

        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(level)

        logger = logging.getLogger(logger_name)
        original_level = logger.level
        logger.setLevel(level)
        logger.addHandler(handler)

        try:
            yield log_capture
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    @staticmethod
    @asynccontextmanager
    async def async_timeout(seconds: float):
        """Async context manager with timeout."""
        try:
            async with asyncio.timeout(seconds):
                yield
        except asyncio.TimeoutError:
            pytest.fail(f"Operation timed out after {seconds} seconds")


# ============================================================================
# Performance Testing Utilities
# ============================================================================


class PerformanceTester:
    """Utilities for performance testing."""

    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
        """Measure function execution time."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        return result, end_time - start_time

    @staticmethod
    async def measure_async_execution_time(coro) -> tuple:
        """Measure async function execution time."""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()

        return result, end_time - start_time

    @staticmethod
    def memory_usage_context():
        """Context manager to measure memory usage."""
        import psutil
        import os

        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.initial_memory = None
                self.peak_memory = None

            def __enter__(self):
                self.initial_memory = self.process.memory_info().rss
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                final_memory = self.process.memory_info().rss
                self.peak_memory = final_memory

            @property
            def memory_increase_mb(self):
                if self.initial_memory and self.peak_memory:
                    return (self.peak_memory - self.initial_memory) / 1024 / 1024
                return 0

        return MemoryMonitor()

    @staticmethod
    def benchmark_model_inference(
        model: torch.nn.Module, input_shape: tuple, iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark model inference performance."""
        model.eval()

        # Warmup
        with torch.no_grad():
            dummy_input = torch.randn(*input_shape)
            for _ in range(10):
                _ = model(dummy_input)

        # Measure
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(dummy_input)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / iterations

        return {
            "total_time_seconds": total_time,
            "average_time_ms": avg_time * 1000,
            "throughput_inferences_per_second": 1.0 / avg_time,
            "iterations": iterations,
        }


# ============================================================================
# Integration Test Utilities
# ============================================================================


class IntegrationTestRunner:
    """Runner for integration test suites."""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None

    async def run_test_suite(self, test_functions: List[Callable]) -> Dict[str, Any]:
        """Run a suite of integration tests."""
        self.start_time = time.time()

        for test_func in test_functions:
            test_name = test_func.__name__

            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()

                self.results[test_name] = {
                    "success": True,
                    "result": result,
                    "error": None,
                    "duration": time.time() - self.start_time,
                }

            except Exception as e:
                self.results[test_name] = {
                    "success": False,
                    "result": None,
                    "error": str(e),
                    "duration": time.time() - self.start_time,
                }

        self.end_time = time.time()
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["success"])

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "results": self.results,
        }


# ============================================================================
# Test Report Generation
# ============================================================================


class TestReporter:
    """Generate comprehensive test reports."""

    @staticmethod
    def generate_consolidation_report(
        test_results: Dict[str, Any], original_test_count: int, consolidated_test_count: int
    ) -> str:
        """Generate test consolidation report."""
        reduction_percentage = ((original_test_count - consolidated_test_count) / original_test_count) * 100

        report = f"""
Test Consolidation Report
========================

## Summary
- Original test files: {original_test_count}
- Consolidated test files: {consolidated_test_count}  
- Reduction: {reduction_percentage:.1f}%

## Test Results
- Total tests run: {test_results.get('total_tests', 0)}
- Successful tests: {test_results.get('successful_tests', 0)}
- Failed tests: {test_results.get('failed_tests', 0)}
- Success rate: {test_results.get('success_rate', 0):.1%}

## Performance Impact
- Total execution time: {test_results.get('total_duration', 0):.2f}s
- Average test time: {test_results.get('total_duration', 0) / max(test_results.get('total_tests', 1), 1):.3f}s

## Benefits Achieved
- Reduced duplicate test code
- Standardized test patterns
- Improved maintainability
- Better test isolation
- Enhanced error reporting
"""
        return report.strip()

    @staticmethod
    def save_test_metrics(metrics: Dict[str, Any], output_path: Path):
        """Save test metrics to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)


# ============================================================================
# Export All Utilities
# ============================================================================

__all__ = [
    "TestAssertions",
    "TestDataGenerator",
    "MockBuilder",
    "TestEnvironment",
    "PerformanceTester",
    "IntegrationTestRunner",
    "TestReporter",
]
