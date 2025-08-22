"""Performance Validation Tests

Tests that validate agent performance characteristics and ensure they meet
specified performance bounds. These tests focus on behavioral performance
rather than implementation details.
"""

import asyncio
import gc
import os
import time
from unittest.mock import patch

import psutil
import pytest

from packages.agents.core.base_agent_template import ReflectionType
from tests.agents.core.fixtures.conftest import MockTestAgent
from tests.agents.core.fixtures.test_builders import MessageInterfaceBuilder, TaskInterfaceBuilder


class TestAgentPerformance:
    """Performance validation tests for agent operations."""

    @pytest.mark.slow
    async def test_task_processing_latency_bounds(self, mock_agent: MockTestAgent):
        """Task processing should complete within acceptable latency bounds."""
        # Given: Various task sizes
        test_cases = [
            ("small", "x" * 10),
            ("medium", "x" * 100),
            ("large", "x" * 1000),
        ]

        latencies = []

        for size_name, content in test_cases:
            task = TaskInterfaceBuilder().with_content(content).with_type("performance").build()

            # When: Task is processed
            start_time = time.perf_counter()
            result = await mock_agent.process_task(task)
            end_time = time.perf_counter()

            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append((size_name, latency))

            # Then: Task should complete successfully
            assert result["status"] == "success"

            # And: Latency should be within bounds
            if size_name == "small":
                assert latency < 100, f"Small task took {latency}ms, expected < 100ms"
            elif size_name == "medium":
                assert latency < 200, f"Medium task took {latency}ms, expected < 200ms"
            elif size_name == "large":
                assert latency < 500, f"Large task took {latency}ms, expected < 500ms"

        # Verify latency scaling is reasonable
        small_latency = next(lat for name, lat in latencies if name == "small")
        large_latency = next(lat for name, lat in latencies if name == "large")

        # Large tasks shouldn't be more than 10x slower than small tasks
        assert large_latency / small_latency < 10, "Latency scaling is too steep"

    @pytest.mark.slow
    async def test_concurrent_processing_performance(self, mock_agent: MockTestAgent):
        """Agent should handle concurrent operations efficiently."""
        # Given: Multiple concurrent tasks
        num_concurrent = 10
        tasks = [
            TaskInterfaceBuilder()
            .with_id(f"concurrent-{i}")
            .with_content(f"Concurrent task {i}")
            .with_type("concurrent")
            .build()
            for i in range(num_concurrent)
        ]

        # When: Tasks are processed concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*[mock_agent.process_task(task) for task in tasks])
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # milliseconds

        # Then: All tasks should complete successfully
        assert len(results) == num_concurrent
        assert all(result["status"] == "success" for result in results)

        # And: Concurrent processing should be efficient
        # Should complete faster than sequential processing
        estimated_sequential_time = num_concurrent * 50  # 50ms per task
        assert (
            total_time < estimated_sequential_time * 0.8
        ), f"Concurrent processing took {total_time}ms, expected < {estimated_sequential_time * 0.8}ms"

    async def test_memory_usage_bounds(self, mock_agent: MockTestAgent):
        """Agent memory usage should stay within reasonable bounds."""
        # Given: Initial memory measurement
        initial_memory = self._get_memory_usage_mb()

        # When: Agent processes many operations
        num_operations = 50
        for i in range(num_operations):
            # Mix of operations that could cause memory growth
            task = TaskInterfaceBuilder().with_content(f"Memory test {i}").build()
            await mock_agent.process_task(task)

            await mock_agent.record_quiet_star_reflection(
                ReflectionType.TASK_COMPLETION,
                f"Memory test operation {i}",
                f"Processing memory test {i}",
                f"Completed memory test {i}",
                tags=["memory_test"],
            )

            await mock_agent.update_geometric_self_awareness()

        # Force garbage collection
        gc.collect()
        final_memory = self._get_memory_usage_mb()

        # Then: Memory growth should be reasonable
        memory_growth = final_memory - initial_memory
        memory_per_operation = memory_growth / num_operations

        # Should not use more than 1MB per operation on average
        assert memory_per_operation < 1.0, f"Memory usage per operation: {memory_per_operation:.2f}MB, expected < 1.0MB"

        # Total memory growth should be bounded
        assert memory_growth < 50, f"Total memory growth: {memory_growth:.2f}MB, expected < 50MB"

    async def test_response_time_consistency(self, mock_agent: MockTestAgent):
        """Agent response times should be consistent for similar operations."""
        # Given: Multiple identical operations
        num_samples = 20
        response_times = []

        for i in range(num_samples):
            task = TaskInterfaceBuilder().with_content("Consistency test").build()

            start_time = time.perf_counter()
            result = await mock_agent.process_task(task)
            end_time = time.perf_counter()

            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)

            assert result["status"] == "success"

        # Then: Response times should be consistent
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min(response_times)

        # Coefficient of variation should be reasonable (< 50%)
        std_dev = (sum((t - avg_time) ** 2 for t in response_times) / len(response_times)) ** 0.5
        coefficient_of_variation = std_dev / avg_time

        assert (
            coefficient_of_variation < 0.5
        ), f"Response time variability too high: {coefficient_of_variation:.2f}, expected < 0.5"

        # No outliers should be more than 3x the average
        assert max_time < avg_time * 3, f"Maximum response time {max_time:.2f}ms is {max_time/avg_time:.1f}x average"

    @pytest.mark.slow
    async def test_throughput_performance(self, mock_agent: MockTestAgent):
        """Agent should maintain minimum throughput for sustained operations."""
        # Given: Sustained workload
        duration_seconds = 5
        end_time = time.time() + duration_seconds
        completed_tasks = 0

        # When: Processing tasks for sustained period
        while time.time() < end_time:
            task = TaskInterfaceBuilder().with_content(f"Throughput test {completed_tasks}").build()
            result = await mock_agent.process_task(task)

            if result["status"] == "success":
                completed_tasks += 1

        # Then: Should maintain minimum throughput
        throughput = completed_tasks / duration_seconds
        min_expected_throughput = 10  # 10 tasks per second minimum

        assert (
            throughput >= min_expected_throughput
        ), f"Throughput {throughput:.1f} tasks/sec, expected >= {min_expected_throughput}"

    async def test_resource_cleanup_efficiency(self, mock_agent: MockTestAgent):
        """Agent should efficiently clean up resources after operations."""
        # Given: Initial resource state
        initial_handles = self._get_file_handle_count()
        initial_memory = self._get_memory_usage_mb()

        # When: Performing operations that create/destroy resources
        for i in range(20):
            # Operations that might create temporary resources
            task = TaskInterfaceBuilder().with_content(f"Resource test {i}").build()
            await mock_agent.process_task(task)

            # Simulate resource-intensive operations
            await mock_agent.query_group_memory(f"test query {i}")

            message = MessageInterfaceBuilder().with_content(f"Test message {i}").build()
            await mock_agent.send_message(message)

        # Force cleanup
        gc.collect()
        await asyncio.sleep(0.1)  # Allow async cleanup

        final_handles = self._get_file_handle_count()
        final_memory = self._get_memory_usage_mb()

        # Then: Resources should be cleaned up
        handle_growth = final_handles - initial_handles
        memory_growth = final_memory - initial_memory

        # Should not leak file handles
        assert handle_growth < 10, f"File handle leak: {handle_growth} handles"

        # Memory growth should be minimal after cleanup
        assert memory_growth < 20, f"Memory not cleaned up: {memory_growth:.2f}MB growth"

    async def test_performance_under_load(self, mock_agent: MockTestAgent):
        """Agent performance should degrade gracefully under high load."""
        # Given: Increasing load levels
        load_levels = [1, 5, 10, 20]
        performance_metrics = []

        for load in load_levels:
            # Create concurrent tasks for this load level
            tasks = [TaskInterfaceBuilder().with_content(f"Load test {i}").build() for i in range(load)]

            # Measure performance under this load
            start_time = time.perf_counter()
            results = await asyncio.gather(*[mock_agent.process_task(task) for task in tasks])
            end_time = time.perf_counter()

            duration = end_time - start_time
            throughput = load / duration
            success_rate = sum(1 for r in results if r["status"] == "success") / len(results)

            performance_metrics.append(
                {"load": load, "duration": duration, "throughput": throughput, "success_rate": success_rate}
            )

        # Then: Performance should degrade gracefully
        for i in range(1, len(performance_metrics)):
            current = performance_metrics[i]
            previous = performance_metrics[i - 1]

            # Success rate should not drop dramatically
            assert (
                current["success_rate"] >= 0.9
            ), f"Success rate dropped to {current['success_rate']:.2f} under load {current['load']}"

            # Throughput efficiency should not degrade too much
            current_efficiency = current["throughput"] / current["load"]
            previous_efficiency = previous["throughput"] / previous["load"]
            efficiency_ratio = current_efficiency / previous_efficiency

            assert efficiency_ratio > 0.5, f"Efficiency dropped to {efficiency_ratio:.2f} under load {current['load']}"

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_file_handle_count(self) -> int:
        """Get current file handle count."""
        try:
            process = psutil.Process(os.getpid())
            return process.num_fds() if hasattr(process, "num_fds") else len(process.open_files())
        except Exception:
            return 0


class TestPerformanceRegression:
    """Performance regression tests to catch performance degradations."""

    @pytest.fixture
    def performance_baseline(self):
        """Provide performance baseline expectations."""
        return {
            "task_processing_max_ms": 100,
            "memory_growth_max_mb": 10,
            "throughput_min_ops_sec": 20,
            "concurrent_efficiency_min": 0.7,
        }

    async def test_task_processing_regression(self, mock_agent: MockTestAgent, performance_baseline):
        """Verify task processing performance hasn't regressed."""
        # Given: Standard test task
        task = TaskInterfaceBuilder().with_content("Regression test task").build()

        # When: Processing multiple times to get average
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = await mock_agent.process_task(task)
            end_time = time.perf_counter()

            assert result["status"] == "success"
            times.append((end_time - start_time) * 1000)

        # Then: Average time should be within baseline
        avg_time = sum(times) / len(times)
        assert (
            avg_time <= performance_baseline["task_processing_max_ms"]
        ), f"Task processing regressed: {avg_time:.2f}ms > {performance_baseline['task_processing_max_ms']}ms"

    async def test_memory_usage_regression(self, mock_agent: MockTestAgent, performance_baseline):
        """Verify memory usage hasn't regressed."""
        # Given: Initial memory state
        initial_memory = self._get_memory_usage_mb()

        # When: Processing many operations
        for i in range(100):
            task = TaskInterfaceBuilder().with_content(f"Memory regression test {i}").build()
            await mock_agent.process_task(task)

        gc.collect()
        final_memory = self._get_memory_usage_mb()

        # Then: Memory growth should be within baseline
        memory_growth = final_memory - initial_memory
        assert (
            memory_growth <= performance_baseline["memory_growth_max_mb"]
        ), f"Memory usage regressed: {memory_growth:.2f}MB > {performance_baseline['memory_growth_max_mb']}MB"

    async def test_throughput_regression(self, mock_agent: MockTestAgent, performance_baseline):
        """Verify throughput hasn't regressed."""
        # Given: Throughput test setup
        test_duration = 2  # seconds
        end_time = time.time() + test_duration
        completed_operations = 0

        # When: Processing operations for duration
        while time.time() < end_time:
            task = TaskInterfaceBuilder().with_content(f"Throughput test {completed_operations}").build()
            result = await mock_agent.process_task(task)

            if result["status"] == "success":
                completed_operations += 1

        # Then: Throughput should meet baseline
        throughput = completed_operations / test_duration
        assert (
            throughput >= performance_baseline["throughput_min_ops_sec"]
        ), f"Throughput regressed: {throughput:.1f} ops/sec < {performance_baseline['throughput_min_ops_sec']}"

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class TestPerformanceIsolation:
    """Tests that ensure performance tests are isolated and don't interfere."""

    async def test_performance_test_isolation(self, agent_factory):
        """Performance tests should not interfere with each other."""
        # Given: Multiple agents for isolation testing
        agents = []
        for i in range(3):
            agent = await agent_factory(agent_type=f"IsolationAgent{i}")
            agents.append(agent)

        # When: Running performance tests on different agents concurrently
        async def performance_test(agent, test_id):
            results = []
            for i in range(10):
                task = TaskInterfaceBuilder().with_content(f"Isolation test {test_id}-{i}").build()
                start_time = time.perf_counter()
                result = await agent.process_task(task)
                end_time = time.perf_counter()

                results.append({"success": result["status"] == "success", "duration": (end_time - start_time) * 1000})
            return results

        # Run tests concurrently
        all_results = await asyncio.gather(*[performance_test(agent, i) for i, agent in enumerate(agents)])

        # Then: All tests should succeed without interference
        for i, results in enumerate(all_results):
            success_rate = sum(1 for r in results if r["success"]) / len(results)
            assert success_rate == 1.0, f"Agent {i} had failures: {success_rate:.2f}"

            avg_duration = sum(r["duration"] for r in results) / len(results)
            assert avg_duration < 200, f"Agent {i} performance degraded: {avg_duration:.2f}ms"

    async def test_resource_isolation(self, mock_agent: MockTestAgent):
        """Resource usage should be isolated between test runs."""
        # Given: Clean initial state
        initial_memory = self._get_memory_usage_mb()

        # When: Running resource-intensive operations
        for batch in range(3):
            # Process batch of operations
            for i in range(20):
                task = TaskInterfaceBuilder().with_content(f"Resource batch {batch}-{i}").build()
                await mock_agent.process_task(task)

            # Force cleanup between batches
            gc.collect()
            await asyncio.sleep(0.1)

            # Check memory after each batch
            current_memory = self._get_memory_usage_mb()
            memory_growth = current_memory - initial_memory

            # Memory growth should be bounded for each batch
            assert memory_growth < 30, f"Memory not isolated between batches: {memory_growth:.2f}MB after batch {batch}"

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class TestPerformanceMonitoring:
    """Tests for performance monitoring and metrics collection."""

    async def test_performance_metrics_collection(self, mock_agent: MockTestAgent):
        """Agent should collect accurate performance metrics."""
        # Given: Clean metrics state
        initial_metrics = mock_agent.get_performance_metrics()

        # When: Processing tasks with known characteristics
        fast_tasks = 5
        slow_tasks = 3

        # Process fast tasks
        for i in range(fast_tasks):
            task = TaskInterfaceBuilder().with_content("x").build()  # Small content
            await mock_agent.process_task(task)

        # Process slow tasks (larger content)
        for i in range(slow_tasks):
            task = TaskInterfaceBuilder().with_content("x" * 1000).build()  # Large content
            await mock_agent.process_task(task)

        final_metrics = mock_agent.get_performance_metrics()

        # Then: Metrics should accurately reflect operations
        total_tasks = fast_tasks + slow_tasks
        tasks_processed = final_metrics.total_tasks_processed - initial_metrics.total_tasks_processed
        assert tasks_processed == total_tasks

        # Success rate should be 100%
        assert final_metrics.successful_tasks >= initial_metrics.successful_tasks + total_tasks

        # Average processing time should be reasonable
        assert final_metrics.average_processing_time_ms > 0
        assert final_metrics.average_processing_time_ms < 1000

    async def test_performance_trend_detection(self, mock_agent: MockTestAgent):
        """Agent should be able to detect performance trends."""
        # Given: Baseline performance measurement
        baseline_times = []
        for i in range(10):
            task = TaskInterfaceBuilder().with_content("Baseline test").build()
            start_time = time.perf_counter()
            await mock_agent.process_task(task)
            end_time = time.perf_counter()
            baseline_times.append((end_time - start_time) * 1000)

        baseline_avg = sum(baseline_times) / len(baseline_times)

        # When: Simulating performance degradation
        degraded_times = []
        with patch("asyncio.sleep", return_value=None):  # Remove natural delays
            # Add artificial delay to simulate degradation
            original_process = mock_agent.process_specialized_task

            async def slower_process(task_data):
                await asyncio.sleep(0.01)  # 10ms delay
                return await original_process(task_data)

            mock_agent.process_specialized_task = slower_process

            for i in range(10):
                task = TaskInterfaceBuilder().with_content("Degraded test").build()
                start_time = time.perf_counter()
                await mock_agent.process_task(task)
                end_time = time.perf_counter()
                degraded_times.append((end_time - start_time) * 1000)

        degraded_avg = sum(degraded_times) / len(degraded_times)

        # Then: Should detect performance degradation
        performance_ratio = degraded_avg / baseline_avg
        assert performance_ratio > 1.5, f"Performance degradation not significant enough: {performance_ratio:.2f}x"

        # And: Degradation should be measurable but bounded
        assert performance_ratio < 10, f"Performance degradation too severe: {performance_ratio:.2f}x"
