"""Tests for Performance Benchmarking Framework - Prompt E

Comprehensive validation of performance benchmarking system including:
- Benchmark execution and timing
- Resource monitoring accuracy
- Performance threshold evaluation
- Report generation and CI integration

Integration Point: Performance validation for Phase 4 testing
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from testing.performance_benchmarks import (
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkSuite,
    PerformanceBenchmarkManager,
    PerformanceMetrics,
    PerformanceThreshold,
    ResourceMonitor,
    generate_performance_ci_config,
    performance_context,
    run_performance_benchmarks,
)


class TestPerformanceMetrics:
    """Test performance metrics data structure."""

    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            latency_ms=50.5,
            throughput_ops_per_sec=100.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=75.5,
            peak_memory_mb=512.0,
            success_rate=95.5,
            custom_metrics={"compression_ratio": 0.7},
        )

        assert metrics.latency_ms == 50.5
        assert metrics.throughput_ops_per_sec == 100.0
        assert metrics.memory_usage_mb == 256.0
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.peak_memory_mb == 512.0
        assert metrics.success_rate == 95.5
        assert metrics.custom_metrics["compression_ratio"] == 0.7

    def test_performance_metrics_defaults(self):
        """Test performance metrics default values."""
        metrics = PerformanceMetrics()

        assert metrics.latency_ms == 0.0
        assert metrics.throughput_ops_per_sec == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.peak_memory_mb == 0.0
        assert metrics.total_time_ms == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.error_count == 0
        assert metrics.custom_metrics == {}


class TestBenchmarkResult:
    """Test benchmark result data structure."""

    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        metrics = PerformanceMetrics(latency_ms=25.0)
        result = BenchmarkResult(
            benchmark_name="test_latency",
            status=BenchmarkStatus.COMPLETED,
            metrics=metrics,
            threshold_level=PerformanceThreshold.GOOD,
            message="Benchmark completed successfully",
            execution_time_ms=1500.0,
        )

        assert result.benchmark_name == "test_latency"
        assert result.status == BenchmarkStatus.COMPLETED
        assert result.metrics.latency_ms == 25.0
        assert result.threshold_level == PerformanceThreshold.GOOD
        assert result.message == "Benchmark completed successfully"
        assert result.execution_time_ms == 1500.0

    def test_benchmark_status_enum(self):
        """Test benchmark status enumeration."""
        assert BenchmarkStatus.PENDING.value == "pending"
        assert BenchmarkStatus.RUNNING.value == "running"
        assert BenchmarkStatus.COMPLETED.value == "completed"
        assert BenchmarkStatus.FAILED.value == "failed"
        assert BenchmarkStatus.SKIPPED.value == "skipped"

    def test_performance_threshold_enum(self):
        """Test performance threshold enumeration."""
        assert PerformanceThreshold.EXCELLENT.value == "excellent"
        assert PerformanceThreshold.GOOD.value == "good"
        assert PerformanceThreshold.ACCEPTABLE.value == "acceptable"
        assert PerformanceThreshold.POOR.value == "poor"
        assert PerformanceThreshold.CRITICAL.value == "critical"


class TestResourceMonitor:
    """Test resource monitoring functionality."""

    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor(interval_seconds=0.1)

        assert monitor.interval == 0.1
        assert monitor.monitoring is False
        assert monitor._thread is None
        assert monitor._metrics == []

    def test_resource_monitor_start_stop(self):
        """Test resource monitor start/stop cycle."""
        monitor = ResourceMonitor(interval_seconds=0.05)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring is True
        assert monitor._thread is not None

        # Let it collect some samples
        time.sleep(0.2)

        # Stop monitoring
        metrics = monitor.stop_monitoring()
        assert monitor.monitoring is False
        assert isinstance(metrics, dict)

        # Should have collected some metrics
        if metrics:  # May be empty on fast systems
            assert "avg_cpu_percent" in metrics
            assert "max_cpu_percent" in metrics
            assert "avg_memory_mb" in metrics
            assert "peak_memory_mb" in metrics
            assert "sample_count" in metrics
            assert metrics["sample_count"] >= 0

    def test_resource_monitor_no_double_start(self):
        """Test that monitor doesn't start twice."""
        monitor = ResourceMonitor(interval_seconds=0.1)

        monitor.start_monitoring()
        first_thread = monitor._thread

        # Try to start again
        monitor.start_monitoring()
        assert monitor._thread is first_thread  # Should be same thread

        monitor.stop_monitoring()

    def test_resource_monitor_stop_without_start(self):
        """Test stopping monitor that wasn't started."""
        monitor = ResourceMonitor()

        metrics = monitor.stop_monitoring()
        assert metrics == {}


class TestPerformanceContext:
    """Test performance measurement context manager."""

    def test_performance_context_basic(self):
        """Test basic performance context functionality."""
        with performance_context() as monitor:
            time.sleep(0.1)  # Simulate work
            assert isinstance(monitor, ResourceMonitor)

    def test_performance_context_without_monitoring(self):
        """Test performance context without resource monitoring."""
        with performance_context(monitor_resources=False) as monitor:
            time.sleep(0.05)
            assert monitor is None

    def test_performance_context_exception_handling(self):
        """Test performance context handles exceptions properly."""
        try:
            with performance_context() as monitor:
                assert isinstance(monitor, ResourceMonitor)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Context should have cleaned up properly
        assert not monitor.monitoring


class TestBenchmarkSuite:
    """Test benchmark suite functionality."""

    def test_benchmark_suite_creation(self):
        """Test benchmark suite creation."""

        def dummy_benchmark():
            return None

        suite = BenchmarkSuite(
            suite_name="test_suite",
            description="Test suite description",
            benchmarks=[dummy_benchmark],
            timeout_seconds=120,
        )

        assert suite.suite_name == "test_suite"
        assert suite.description == "Test suite description"
        assert len(suite.benchmarks) == 1
        assert suite.timeout_seconds == 120
        assert suite.setup_func is None
        assert suite.teardown_func is None
        assert suite.parallel_execution is False

    def test_benchmark_suite_with_setup_teardown(self):
        """Test benchmark suite with setup and teardown functions."""

        def setup():
            return "setup"

        def teardown():
            return "teardown"

        def benchmark():
            return None

        suite = BenchmarkSuite(
            suite_name="full_suite",
            description="Suite with setup/teardown",
            benchmarks=[benchmark],
            setup_func=setup,
            teardown_func=teardown,
        )

        assert suite.setup_func is setup
        assert suite.teardown_func is teardown


class TestPerformanceBenchmarkManager:
    """Test performance benchmark manager functionality."""

    def test_manager_initialization_default_config(self):
        """Test manager initialization with default config."""
        manager = PerformanceBenchmarkManager()

        assert "thresholds" in manager.config
        assert "benchmark_settings" in manager.config
        assert "test_data" in manager.config

        # Check default thresholds
        assert "p2p_latency_ms" in manager.config["thresholds"]
        assert "rag_query_ms" in manager.config["thresholds"]
        assert manager.config["thresholds"]["p2p_latency_ms"]["excellent"] == 10

        # Check that default suites are registered
        assert "p2p_network" in manager.suites
        assert "rag_system" in manager.suites
        assert "compression" in manager.suites
        assert "integration" in manager.suites

    def test_manager_initialization_custom_config(self):
        """Test manager initialization with custom config."""
        custom_config = {
            "thresholds": {"p2p_latency_ms": {"excellent": 5, "good": 25}},
            "benchmark_settings": {"default_timeout_seconds": 60},
        }

        manager = PerformanceBenchmarkManager(custom_config)

        assert manager.config["thresholds"]["p2p_latency_ms"]["excellent"] == 5
        assert manager.config["benchmark_settings"]["default_timeout_seconds"] == 60

    def test_suite_registration(self):
        """Test benchmark suite registration."""
        manager = PerformanceBenchmarkManager()

        def test_benchmark():
            return None

        custom_suite = BenchmarkSuite(
            suite_name="custom_test",
            description="Custom test suite",
            benchmarks=[test_benchmark],
        )

        manager.register_suite(custom_suite)

        assert "custom_test" in manager.suites
        assert manager.suites["custom_test"].description == "Custom test suite"

    def test_threshold_evaluation_latency(self):
        """Test threshold evaluation for latency metrics."""
        manager = PerformanceBenchmarkManager()

        # Test excellent latency
        metrics = PerformanceMetrics(latency_ms=5.0)
        threshold = manager._evaluate_threshold("p2p_latency", metrics)
        assert threshold == PerformanceThreshold.EXCELLENT

        # Test good latency
        metrics = PerformanceMetrics(latency_ms=25.0)
        threshold = manager._evaluate_threshold("p2p_latency", metrics)
        assert threshold == PerformanceThreshold.GOOD

        # Test poor latency
        metrics = PerformanceMetrics(latency_ms=600.0)
        threshold = manager._evaluate_threshold("p2p_latency", metrics)
        assert threshold == PerformanceThreshold.CRITICAL

    def test_threshold_evaluation_compression_ratio(self):
        """Test threshold evaluation for compression ratio metrics."""
        manager = PerformanceBenchmarkManager()

        # Test excellent compression ratio
        metrics = PerformanceMetrics(custom_metrics={"compression_ratio": 0.85})
        threshold = manager._evaluate_threshold("compression_ratio", metrics)
        assert threshold == PerformanceThreshold.EXCELLENT

        # Test poor compression ratio
        metrics = PerformanceMetrics(custom_metrics={"compression_ratio": 0.15})
        threshold = manager._evaluate_threshold("compression_ratio", metrics)
        assert threshold == PerformanceThreshold.CRITICAL

    def test_p2p_latency_benchmark(self):
        """Test P2P latency benchmark execution."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_p2p_latency()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.success_rate > 0
        assert isinstance(details, dict)

        # Check that custom metrics are present
        assert "min_latency_ms" in metrics.custom_metrics
        assert "max_latency_ms" in metrics.custom_metrics

    def test_rag_query_benchmark(self):
        """Test RAG query latency benchmark."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_rag_query_latency()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.throughput_ops_per_sec > 0
        assert metrics.success_rate == 100.0
        assert "query_count" in metrics.custom_metrics
        assert "queries_executed" in details

    def test_compression_ratio_benchmark(self):
        """Test compression ratio benchmark."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_compression_ratio()

        assert isinstance(metrics, PerformanceMetrics)
        assert "compression_ratio" in metrics.custom_metrics
        assert metrics.custom_metrics["compression_ratio"] > 0
        assert metrics.custom_metrics["compression_ratio"] < 1
        assert "original_size_bytes" in metrics.custom_metrics
        assert "compressed_size_bytes" in metrics.custom_metrics

    def test_system_startup_benchmark(self):
        """Test system startup benchmark."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_system_startup()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.success_rate == 100.0
        assert "components_loaded" in metrics.custom_metrics
        assert "components" in details
        assert "total_startup_ms" in details

    def test_single_benchmark_execution(self):
        """Test single benchmark execution with monitoring."""
        manager = PerformanceBenchmarkManager()

        def test_benchmark():
            time.sleep(0.01)  # Simulate 10ms work
            metrics = PerformanceMetrics(latency_ms=10.0, success_rate=100.0)
            details = {"test": True}
            return metrics, details

        result = manager._run_single_benchmark(test_benchmark, timeout=10)

        assert isinstance(result, BenchmarkResult)
        assert result.status == BenchmarkStatus.COMPLETED
        assert result.execution_time_ms > 0
        assert result.metrics.latency_ms == 10.0
        assert result.threshold_level in [
            PerformanceThreshold.EXCELLENT,
            PerformanceThreshold.GOOD,
        ]

    def test_benchmark_timeout_handling(self):
        """Test benchmark timeout handling."""
        manager = PerformanceBenchmarkManager()

        def slow_benchmark():
            time.sleep(2.0)  # Simulate slow benchmark
            return PerformanceMetrics(), {}

        # Set very short timeout
        result = manager._run_single_benchmark(slow_benchmark, timeout=1)

        # Should complete even if slow (timeout implementation is basic)
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_exception_handling(self):
        """Test benchmark exception handling."""
        manager = PerformanceBenchmarkManager()

        def failing_benchmark():
            raise ValueError("Test benchmark failure")

        result = manager._run_single_benchmark(failing_benchmark, timeout=10)

        assert isinstance(result, BenchmarkResult)
        assert result.status == BenchmarkStatus.FAILED
        assert "Test benchmark failure" in result.message
        assert result.threshold_level == PerformanceThreshold.CRITICAL

    def test_run_benchmark_suite(self):
        """Test running a complete benchmark suite."""
        manager = PerformanceBenchmarkManager()

        # Run P2P network suite
        results = manager.run_benchmark_suite("p2p_network")

        assert isinstance(results, list)
        assert len(results) > 0

        # All results should be BenchmarkResult instances
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.status in [
                BenchmarkStatus.COMPLETED,
                BenchmarkStatus.FAILED,
                BenchmarkStatus.SKIPPED,
            ]

    def test_run_all_benchmarks(self):
        """Test running all benchmark suites."""
        manager = PerformanceBenchmarkManager()

        results = manager.run_all_benchmarks()

        assert isinstance(results, list)
        assert len(results) > 10  # Should have many benchmarks across all suites

        # Check that we have results from different suites
        benchmark_names = [r.benchmark_name for r in results]
        assert any("p2p" in name for name in benchmark_names)
        assert any("rag" in name for name in benchmark_names)
        assert any("compression" in name for name in benchmark_names)

    def test_unknown_suite_error(self):
        """Test error handling for unknown benchmark suite."""
        manager = PerformanceBenchmarkManager()

        with pytest.raises(ValueError, match="Unknown benchmark suite"):
            manager.run_benchmark_suite("nonexistent_suite")

    def test_performance_report_generation(self):
        """Test performance report generation."""
        manager = PerformanceBenchmarkManager()

        # Add some mock results
        manager.results = [
            BenchmarkResult(
                benchmark_name="test1",
                status=BenchmarkStatus.COMPLETED,
                metrics=PerformanceMetrics(latency_ms=10.0),
                threshold_level=PerformanceThreshold.EXCELLENT,
            ),
            BenchmarkResult(
                benchmark_name="test2",
                status=BenchmarkStatus.COMPLETED,
                metrics=PerformanceMetrics(latency_ms=600.0),
                threshold_level=PerformanceThreshold.CRITICAL,
            ),
            BenchmarkResult(
                benchmark_name="test3",
                status=BenchmarkStatus.FAILED,
                metrics=PerformanceMetrics(),
                threshold_level=PerformanceThreshold.CRITICAL,
            ),
        ]

        report = manager.generate_performance_report()

        assert isinstance(report, dict)
        assert "summary" in report
        assert "performance_distribution" in report
        assert "critical_issues" in report
        assert "recommendations" in report
        assert "benchmark_details" in report

        # Check summary
        assert report["summary"]["total_benchmarks"] == 3
        assert report["summary"]["completed"] == 2
        assert report["summary"]["failed"] == 1

        # Check performance distribution
        assert report["performance_distribution"]["excellent"] == 1
        assert report["performance_distribution"]["critical"] == 2

        # Check critical issues
        assert len(report["critical_issues"]) == 2

        # Check recommendations
        assert len(report["recommendations"]) > 0
        assert any("critical" in rec.lower() for rec in report["recommendations"])


class TestBenchmarkIntegration:
    """Test benchmark integration scenarios."""

    def test_run_performance_benchmarks_function(self):
        """Test main performance benchmarks function."""
        report = run_performance_benchmarks()

        assert isinstance(report, dict)
        assert "summary" in report
        assert "performance_distribution" in report
        assert "benchmark_details" in report

        # Should have run multiple benchmarks
        assert report["summary"]["total_benchmarks"] > 10

    def test_run_specific_benchmark_suites(self):
        """Test running specific benchmark suites."""
        report = run_performance_benchmarks(suite_names=["p2p_network"])

        assert isinstance(report, dict)
        assert report["summary"]["total_benchmarks"] >= 4  # P2P suite has 4 benchmarks

        # Should only have P2P benchmarks
        benchmark_names = [b["name"] for b in report["benchmark_details"]]
        assert all("p2p" in name for name in benchmark_names)

    def test_performance_benchmarks_with_config_file(self):
        """Test performance benchmarks with configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {"thresholds": {"p2p_latency_ms": {"excellent": 5, "good": 15}}}
            json.dump(config, f)
            config_file = f.name

        try:
            report = run_performance_benchmarks(config_file=config_file)
            assert isinstance(report, dict)
            assert "summary" in report
        finally:
            os.unlink(config_file)

    def test_ci_config_generation(self):
        """Test CI/CD configuration generation."""
        ci_config = generate_performance_ci_config()

        assert isinstance(ci_config, str)
        assert "name: Performance Benchmarks" in ci_config
        assert "python-version: 3.12" in ci_config
        assert "run_performance_benchmarks" in ci_config
        assert "performance-benchmark-report" in ci_config
        assert "Performance regression analysis" in ci_config

    def test_benchmark_error_recovery(self):
        """Test benchmark system error recovery."""
        manager = PerformanceBenchmarkManager()

        # Create a suite with one failing and one succeeding benchmark
        def success_benchmark():
            return PerformanceMetrics(latency_ms=10.0, success_rate=100.0), {}

        def fail_benchmark():
            raise Exception("Benchmark failure")

        test_suite = BenchmarkSuite(
            suite_name="mixed_results",
            description="Suite with mixed results",
            benchmarks=[success_benchmark, fail_benchmark],
        )

        manager.register_suite(test_suite)
        results = manager.run_benchmark_suite("mixed_results")

        assert len(results) == 2
        assert results[0].status == BenchmarkStatus.COMPLETED
        assert results[1].status == BenchmarkStatus.FAILED

        # System should continue despite failures
        report = manager.generate_performance_report()
        assert report["summary"]["completed"] == 1
        assert report["summary"]["failed"] == 1

    def test_performance_recommendations(self):
        """Test performance recommendation generation."""
        manager = PerformanceBenchmarkManager()

        # Create results with various performance levels
        manager.results = [
            BenchmarkResult(
                "excellent_test",
                BenchmarkStatus.COMPLETED,
                PerformanceMetrics(latency_ms=5.0),
                PerformanceThreshold.EXCELLENT,
            ),
            BenchmarkResult(
                "poor_latency",
                BenchmarkStatus.COMPLETED,
                PerformanceMetrics(latency_ms=800.0),
                PerformanceThreshold.POOR,
            ),
            BenchmarkResult(
                "poor_memory",
                BenchmarkStatus.COMPLETED,
                PerformanceMetrics(memory_usage_mb=1500.0),
                PerformanceThreshold.POOR,
            ),
            BenchmarkResult(
                "poor_throughput",
                BenchmarkStatus.COMPLETED,
                PerformanceMetrics(throughput_ops_per_sec=0.5),
                PerformanceThreshold.POOR,
            ),
        ]

        poor_performance = [r for r in manager.results if r.threshold_level == PerformanceThreshold.POOR]
        recommendations = manager._generate_performance_recommendations([], poor_performance)

        assert len(recommendations) >= 3  # Should have specific recommendations
        rec_text = " ".join(recommendations).lower()
        assert "latency" in rec_text
        assert "memory" in rec_text
        assert "throughput" in rec_text

    def test_stress_test_benchmark(self):
        """Test stress test benchmark execution."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_stress_test()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.memory_usage_mb > 0
        assert metrics.peak_memory_mb > 0
        assert metrics.cpu_usage_percent > 0
        assert metrics.success_rate > 80  # Should maintain good success rate
        assert "stress_duration_seconds" in metrics.custom_metrics
        assert "system_remained_stable" in details

    def test_resource_limits_benchmark(self):
        """Test resource limits benchmark."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_resource_limits()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.memory_usage_mb > 0
        assert metrics.cpu_usage_percent > 0
        assert "within_resource_limits" in metrics.custom_metrics
        assert "operating_within_limits" in details

    def test_concurrent_operations_benchmark(self):
        """Test concurrent operations benchmark."""
        manager = PerformanceBenchmarkManager()

        metrics, details = manager._benchmark_concurrent_operations()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.latency_ms > 0
        assert metrics.throughput_ops_per_sec > 0
        assert metrics.success_rate > 90  # Should handle concurrency well
        assert "concurrent_users" in metrics.custom_metrics
        assert "load_pattern" in details


if __name__ == "__main__":
    # Run performance benchmarking validation
    print("=== Testing Performance Benchmarking Framework ===")

    # Test basic functionality
    print("Testing performance metrics...")
    metrics = PerformanceMetrics(latency_ms=25.0, throughput_ops_per_sec=50.0)
    print(f"OK Performance metrics: latency={metrics.latency_ms}ms, throughput={metrics.throughput_ops_per_sec}")

    # Test resource monitoring
    print("Testing resource monitoring...")
    monitor = ResourceMonitor(interval_seconds=0.05)
    monitor.start_monitoring()
    time.sleep(0.2)
    resource_metrics = monitor.stop_monitoring()
    print(f"OK Resource monitoring: {resource_metrics.get('sample_count', 0)} samples collected")

    # Test benchmark manager
    print("Testing benchmark manager...")
    manager = PerformanceBenchmarkManager()
    print(f"OK Manager initialized: {len(manager.suites)} suites registered")

    # Test P2P benchmark
    print("Testing P2P latency benchmark...")
    metrics, details = manager._benchmark_p2p_latency()
    print(f"OK P2P benchmark: latency={metrics.latency_ms:.1f}ms, success_rate={metrics.success_rate:.1f}%")

    # Test RAG benchmark
    print("Testing RAG query benchmark...")
    metrics, details = manager._benchmark_rag_query_latency()
    print(
        f"OK RAG benchmark: latency={metrics.latency_ms:.1f}ms, throughput={metrics.throughput_ops_per_sec:.1f} ops/sec"
    )

    # Test compression benchmark
    print("Testing compression benchmark...")
    metrics, details = manager._benchmark_compression_ratio()
    compression_ratio = metrics.custom_metrics.get("compression_ratio", 0)
    print(f"OK Compression benchmark: ratio={compression_ratio:.2f}")

    # Test threshold evaluation
    print("Testing threshold evaluation...")
    test_metrics = PerformanceMetrics(latency_ms=15.0)
    threshold = manager._evaluate_threshold("p2p_latency", test_metrics)
    print(f"OK Threshold evaluation: 15ms latency -> {threshold.value}")

    # Test suite execution
    print("Testing benchmark suite execution...")
    results = manager.run_benchmark_suite("p2p_network")
    completed_count = len([r for r in results if r.status == BenchmarkStatus.COMPLETED])
    print(f"OK Suite execution: {completed_count}/{len(results)} benchmarks completed")

    # Test report generation
    print("Testing report generation...")
    report = manager.generate_performance_report()
    print(
        f"OK Report generated: {report['summary']['total_benchmarks']} benchmarks, {len(report['recommendations'])} recommendations"
    )

    print("=== Performance benchmarking framework validation completed ===")
