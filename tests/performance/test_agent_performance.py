"""Performance and stress tests for agent implementations.

Tests performance benchmarks, memory usage, throughput, and chaos engineering
scenarios to validate system resilience and performance characteristics.
"""

import asyncio
import gc
import psutil
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Dict
import random

from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.king.analytics.base_analytics import BaseAnalytics
from agents.interfaces.processing_interface import ProcessingInterface, ProcessorCapability
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore


class PerformanceAnalytics(BaseAnalytics):
    """Analytics optimized for performance testing."""

    def __init__(self):
        super().__init__()
        self.performance_baseline = {}

    def generate_analytics_report(self) -> Dict[str, Any]:
        report = {
            "total_metrics": len(self.metrics),
            "performance_summary": {},
            "baseline_comparison": {},
            "timestamp": datetime.now().isoformat(),
        }

        for metric, values in self.metrics.items():
            if values:
                report["performance_summary"][metric] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1],
                }

                # Compare to baseline if available
                if metric in self.performance_baseline:
                    baseline = self.performance_baseline[metric]
                    current_avg = report["performance_summary"][metric]["avg"]
                    report["baseline_comparison"][metric] = {
                        "baseline": baseline,
                        "current": current_avg,
                        "improvement": (baseline - current_avg) / baseline * 100 if baseline > 0 else 0,
                    }

        return report

    def set_baseline(self, metric: str, value: float):
        """Set performance baseline for comparison."""
        self.performance_baseline[metric] = value


class PerformanceProcessor(ProcessingInterface[Dict[str, Any], Dict[str, Any]]):
    """High-performance processor for stress testing."""

    def __init__(self, processor_id: str = "perf_processor"):
        super().__init__(processor_id)
        self.add_capability(ProcessorCapability.TEXT_PROCESSING)
        self.add_capability(ProcessorCapability.PARALLEL_PROCESSING)
        self.add_capability(ProcessorCapability.BATCH_PROCESSING)
        self.add_capability(ProcessorCapability.REAL_TIME_PROCESSING)
        self.processing_counter = 0
        self.error_simulation_rate = 0.0  # For chaos testing

    async def initialize(self) -> bool:
        await asyncio.sleep(0.001)  # Minimal initialization
        return True

    async def shutdown(self) -> bool:
        await asyncio.sleep(0.001)  # Minimal shutdown
        return True

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        from agents.base import ProcessResult, ProcessStatus

        self.processing_counter += 1

        # Simulate processing error for chaos testing
        if random.random() < self.error_simulation_rate:
            return ProcessResult(status=ProcessStatus.FAILED, error=f"Simulated error #{self.processing_counter}")

        # Minimal processing for maximum throughput
        processing_type = input_data.get("processing_type", "default")

        if processing_type == "cpu_intensive":
            # CPU-intensive task simulation
            result = sum(i * i for i in range(100))  # Compute squares
        elif processing_type == "memory_intensive":
            # Memory-intensive task simulation
            temp_data = [i for i in range(1000)]  # Create temporary list
            result = len(temp_data)
        elif processing_type == "io_intensive":
            # IO-intensive task simulation
            await asyncio.sleep(0.001)  # Simulate IO wait
            result = "io_completed"
        else:
            result = f"processed_{self.processing_counter}"

        return ProcessResult(
            status=ProcessStatus.COMPLETED,
            data={"result": result, "processing_id": self.processing_counter, "processing_type": processing_type},
        )

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict)  # Minimal validation for speed

    async def estimate_processing_time(self, input_data: Dict[str, Any]) -> float:
        processing_type = input_data.get("processing_type", "default")
        estimates = {"cpu_intensive": 0.002, "memory_intensive": 0.001, "io_intensive": 0.001, "default": 0.0005}
        return estimates.get(processing_type, 0.0005)


@pytest.mark.performance
class TestAgentPerformanceBenchmarks:
    """Performance benchmark tests for agents."""

    @pytest.fixture
    async def performance_agent(self, mock_communication_protocol, performance_thresholds):
        """Create high-performance agent for benchmarking."""
        vector_store = Mock(spec=VectorStore)
        vector_store.add_texts = AsyncMock()
        vector_store.similarity_search = AsyncMock(return_value=[])

        config = UnifiedAgentConfig(
            name="PerformanceAgent",
            description="High-performance agent for benchmarking",
            capabilities=["high_throughput", "low_latency", "scalable_processing"],
            rag_config=UnifiedConfig(),
            vector_store=vector_store,
            model="gpt-4",
            instructions="Optimize for maximum performance and throughput",
        )

        with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
            agent = UnifiedBaseAgent(config, mock_communication_protocol)
            agent.analytics = PerformanceAnalytics()
            agent.processor = PerformanceProcessor("perf_processor")
            await agent.processor.initialize()

            # High-performance _process_task implementation
            async def high_performance_process_task(task):
                start_time = time.perf_counter()

                # Minimal processing for maximum speed
                task_data = {"content": task.content, "processing_type": getattr(task, "processing_type", "default")}

                result = await agent.processor.process(task_data)

                # Record performance metrics
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                agent.analytics.record_metric("processing_time_ms", processing_time_ms)
                agent.analytics.record_metric("throughput", 1.0)

                return result.data if result.is_success else {"error": result.error}

            agent._process_task = high_performance_process_task

            # Optimize layers for performance
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(return_value="fast_decision")
            agent.continuous_learning_layer.update = AsyncMock()

            return agent

    async def test_single_task_latency_benchmark(self, performance_agent, performance_thresholds):
        """Benchmark single task processing latency."""
        task = Mock()
        task.content = "latency_benchmark_task"
        task.type = "benchmark"
        task.processing_type = "default"

        # Warm up
        for _ in range(5):
            await performance_agent.execute_task(task)

        # Clear metrics for clean measurement
        performance_agent.analytics.metrics.clear()

        # Benchmark
        num_iterations = 100
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            result = await performance_agent.execute_task(task)
            assert result["result"] == "fast_decision"

        end_time = time.perf_counter()

        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / num_iterations

        # Verify performance threshold
        latency_threshold = performance_thresholds["task_execution_ms"]
        assert (
            avg_latency_ms <= latency_threshold
        ), f"Average latency {avg_latency_ms:.2f}ms exceeds threshold {latency_threshold}ms"

        # Verify analytics recorded measurements
        analytics_report = performance_agent.analytics.generate_analytics_report()
        performance_summary = analytics_report["performance_summary"]

        assert "processing_time_ms" in performance_summary
        assert performance_summary["processing_time_ms"]["count"] == num_iterations

        # Set baseline for future comparisons
        performance_agent.analytics.set_baseline("processing_time_ms", avg_latency_ms)

    async def test_throughput_benchmark(self, performance_agent, performance_thresholds):
        """Benchmark maximum throughput with concurrent processing."""
        num_concurrent_tasks = 50
        tasks = [
            Mock(content=f"throughput_task_{i}", type="throughput", processing_type="default")
            for i in range(num_concurrent_tasks)
        ]

        # Warm up
        await asyncio.gather(*[performance_agent.execute_task(task) for task in tasks[:5]])

        # Clear metrics
        performance_agent.analytics.metrics.clear()

        # Benchmark throughput
        start_time = time.perf_counter()

        results = await asyncio.gather(*[performance_agent.execute_task(task) for task in tasks])

        end_time = time.perf_counter()

        # Calculate throughput
        total_time_seconds = end_time - start_time
        throughput_per_second = num_concurrent_tasks / total_time_seconds

        # Verify results
        assert len(results) == num_concurrent_tasks
        assert all(result["result"] == "fast_decision" for result in results)

        # Verify throughput threshold
        min_throughput = performance_thresholds["batch_processing_throughput"]
        assert (
            throughput_per_second >= min_throughput
        ), f"Throughput {throughput_per_second:.2f} tasks/sec below threshold {min_throughput}"

        # Verify analytics
        analytics_report = performance_agent.analytics.generate_analytics_report()
        performance_summary = analytics_report["performance_summary"]

        assert "throughput" in performance_summary
        assert performance_summary["throughput"]["count"] == num_concurrent_tasks

    async def test_memory_usage_benchmark(self, performance_agent, performance_thresholds):
        """Benchmark memory usage under load."""
        process = psutil.Process()

        # Baseline memory usage
        gc.collect()  # Force garbage collection
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024

        # Create memory-intensive tasks
        num_tasks = 100
        tasks = [
            Mock(content=f"memory_task_{i}", type="memory", processing_type="memory_intensive")
            for i in range(num_tasks)
        ]

        # Process tasks in batches to monitor memory growth
        batch_size = 20
        memory_measurements = []

        for i in range(0, num_tasks, batch_size):
            batch = tasks[i : i + batch_size]

            # Process batch
            await asyncio.gather(*[performance_agent.execute_task(task) for task in batch])

            # Measure memory
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory_mb)

            # Record memory usage
            performance_agent.analytics.record_metric("memory_usage_mb", current_memory_mb)

        # Force cleanup
        gc.collect()
        final_memory_mb = process.memory_info().rss / 1024 / 1024

        # Calculate memory metrics
        peak_memory_mb = max(memory_measurements)
        memory_growth_mb = peak_memory_mb - baseline_memory_mb

        # Verify memory threshold
        memory_threshold = performance_thresholds["memory_usage_mb"]
        assert (
            memory_growth_mb <= memory_threshold
        ), f"Memory growth {memory_growth_mb:.2f}MB exceeds threshold {memory_threshold}MB"

        # Memory should return close to baseline after cleanup
        memory_retained_mb = final_memory_mb - baseline_memory_mb
        assert (
            memory_retained_mb <= memory_threshold * 0.2
        ), f"Memory retained {memory_retained_mb:.2f}MB suggests memory leak"

    async def test_cpu_intensive_benchmark(self, performance_agent):
        """Benchmark CPU-intensive task processing."""
        num_cpu_tasks = 20
        tasks = [
            Mock(content=f"cpu_task_{i}", type="cpu", processing_type="cpu_intensive") for i in range(num_cpu_tasks)
        ]

        # Clear metrics
        performance_agent.analytics.metrics.clear()

        # Benchmark CPU performance
        start_time = time.perf_counter()
        cpu_start = time.process_time()

        results = await asyncio.gather(*[performance_agent.execute_task(task) for task in tasks])

        end_time = time.perf_counter()
        cpu_end = time.process_time()

        # Calculate CPU metrics
        wall_time = end_time - start_time
        cpu_time = cpu_end - cpu_start
        cpu_utilization = cpu_time / wall_time if wall_time > 0 else 0

        # Verify results
        assert len(results) == num_cpu_tasks
        assert all(result["result"] == "fast_decision" for result in results)

        # CPU utilization should be reasonable for concurrent processing
        assert 0.1 <= cpu_utilization <= 2.0, f"CPU utilization {cpu_utilization:.2f} seems abnormal"

        # Record CPU metrics
        performance_agent.analytics.record_metric("cpu_utilization", cpu_utilization)
        performance_agent.analytics.record_metric("wall_time_seconds", wall_time)


@pytest.mark.performance
class TestSystemScalabilityBenchmarks:
    """Test system scalability with multiple agents."""

    @pytest.fixture
    async def scalable_system(self, mock_communication_protocol):
        """Create scalable multi-agent system."""
        agents = []

        for i in range(5):  # 5 agents for scalability testing
            vector_store = Mock(spec=VectorStore)
            vector_store.add_texts = AsyncMock()
            vector_store.similarity_search = AsyncMock(return_value=[])

            config = UnifiedAgentConfig(
                name=f"ScalableAgent{i}",
                description=f"Scalable agent {i}",
                capabilities=[f"capability_{i}", "shared_processing"],
                rag_config=UnifiedConfig(),
                vector_store=vector_store,
                model="gpt-4",
                instructions=f"Agent {i} optimized for scalable processing",
            )

            with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
                agent = UnifiedBaseAgent(config, mock_communication_protocol)
                agent.analytics = PerformanceAnalytics()
                agent.processor = PerformanceProcessor(f"processor_{i}")
                await agent.processor.initialize()

                # Scalable _process_task
                async def scalable_process_task(task):
                    start_time = time.perf_counter()

                    task_data = {"content": task.content, "processing_type": "default"}
                    result = await agent.processor.process(task_data)

                    processing_time = (time.perf_counter() - start_time) * 1000
                    agent.analytics.record_metric("processing_time_ms", processing_time)
                    agent.analytics.record_metric("tasks_processed", 1.0)

                    return result.data if result.is_success else {"error": result.error}

                agent._process_task = scalable_process_task

                # Optimize layers
                agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
                agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
                agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
                agent.decision_making_layer.make_decision = AsyncMock(return_value=f"decision_from_agent_{i}")
                agent.continuous_learning_layer.update = AsyncMock()

                agents.append(agent)

        return SelfEvolvingSystem(agents)

    async def test_horizontal_scaling_performance(self, scalable_system):
        """Test performance scaling with multiple agents."""
        # Test with different numbers of agents
        agent_counts = [1, 3, 5]
        performance_results = {}

        for num_agents in agent_counts:
            # Use subset of agents
            active_agents = scalable_system.agents[:num_agents]

            # Create tasks for each agent
            tasks_per_agent = 20
            total_tasks = num_agents * tasks_per_agent

            tasks = []
            for i in range(total_tasks):
                agent_index = i % num_agents
                task = Mock()
                task.content = f"scaling_task_{i}"
                task.type = f"capability_{agent_index}"  # Route to specific agent
                tasks.append(task)

            # Clear metrics
            for agent in active_agents:
                agent.analytics.metrics.clear()

            # Benchmark scaling
            start_time = time.perf_counter()

            # Process tasks through system
            results = []
            for task in tasks:
                result = await scalable_system.process_task(task)
                results.append(result)

            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            throughput = total_tasks / total_time

            performance_results[num_agents] = {
                "total_tasks": total_tasks,
                "total_time": total_time,
                "throughput": throughput,
                "results_count": len(results),
            }

            # Verify all tasks completed
            assert len(results) == total_tasks

        # Verify scaling efficiency
        single_agent_throughput = performance_results[1]["throughput"]
        multi_agent_throughput = performance_results[5]["throughput"]

        # Multi-agent should provide better throughput (not necessarily linear)
        scaling_efficiency = multi_agent_throughput / (single_agent_throughput * 5)

        # Should achieve at least 50% scaling efficiency
        assert scaling_efficiency >= 0.5, f"Scaling efficiency {scaling_efficiency:.2f} is too low"

    async def test_load_balancing_performance(self, scalable_system):
        """Test load balancing across agents."""
        # Create many tasks that can be handled by multiple agents
        num_tasks = 100
        tasks = []

        for i in range(num_tasks):
            task = Mock()
            task.content = f"load_balance_task_{i}"
            task.type = "shared_processing"  # Can be handled by any agent
            tasks.append(task)

        # Clear metrics
        for agent in scalable_system.agents:
            agent.analytics.metrics.clear()

        # Process tasks
        time.perf_counter()

        results = []
        for task in tasks:
            result = await scalable_system.process_task(task)
            results.append(result)

        time.perf_counter()

        # Verify all tasks completed
        assert len(results) == num_tasks

        # Analyze load distribution
        agent_task_counts = []
        for agent in scalable_system.agents:
            if "tasks_processed" in agent.analytics.metrics:
                task_count = len(agent.analytics.metrics["tasks_processed"])
                agent_task_counts.append(task_count)
            else:
                agent_task_counts.append(0)

        total_distributed_tasks = sum(agent_task_counts)
        assert total_distributed_tasks == num_tasks

        # Check load balancing (no agent should be completely idle if tasks > agents)
        if num_tasks >= len(scalable_system.agents):
            active_agents = sum(1 for count in agent_task_counts if count > 0)
            assert (
                active_agents >= len(scalable_system.agents) * 0.6
            ), f"Only {active_agents}/{len(scalable_system.agents)} agents were used"


@pytest.mark.performance
@pytest.mark.chaos
class TestChaosEngineeringResilience:
    """Chaos engineering tests for system resilience."""

    @pytest.fixture
    async def chaos_agent(self, mock_communication_protocol, chaos_scenarios):
        """Create agent for chaos testing."""
        vector_store = Mock(spec=VectorStore)
        vector_store.add_texts = AsyncMock()
        vector_store.similarity_search = AsyncMock(return_value=[])

        config = UnifiedAgentConfig(
            name="ChaosAgent",
            description="Agent for chaos engineering tests",
            capabilities=["resilience", "error_recovery", "fault_tolerance"],
            rag_config=UnifiedConfig(),
            vector_store=vector_store,
            model="gpt-4",
            instructions="Maintain functionality under adverse conditions",
        )

        with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
            agent = UnifiedBaseAgent(config, mock_communication_protocol)
            agent.analytics = PerformanceAnalytics()
            agent.processor = PerformanceProcessor("chaos_processor")
            agent.processor.error_simulation_rate = 0.1  # 10% error rate
            await agent.processor.initialize()

            # Resilient _process_task with retry logic
            async def resilient_process_task(task):
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        start_time = time.perf_counter()

                        task_data = {"content": task.content, "processing_type": "default"}
                        result = await agent.processor.process(task_data)

                        if result.is_success:
                            processing_time = (time.perf_counter() - start_time) * 1000
                            agent.analytics.record_metric("successful_processing_time_ms", processing_time)
                            agent.analytics.record_metric("successful_tasks", 1.0)
                            agent.analytics.record_metric("retry_attempts", retry_count)

                            return result.data
                        else:
                            retry_count += 1
                            agent.analytics.record_metric("processing_errors", 1.0)

                            if retry_count < max_retries:
                                # Exponential backoff
                                await asyncio.sleep(0.001 * (2**retry_count))
                            else:
                                agent.analytics.record_metric("failed_tasks", 1.0)
                                return {"error": result.error, "retries_exhausted": True}

                    except Exception as e:
                        retry_count += 1
                        agent.analytics.record_metric("exception_errors", 1.0)

                        if retry_count >= max_retries:
                            agent.analytics.record_metric("failed_tasks", 1.0)
                            return {"error": str(e), "retries_exhausted": True}

                        await asyncio.sleep(0.001 * (2**retry_count))

            agent._process_task = resilient_process_task

            # Setup layers with potential failures
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(return_value="resilient_decision")
            agent.continuous_learning_layer.update = AsyncMock()

            return agent

    async def test_error_injection_resilience(self, chaos_agent):
        """Test resilience to random error injection."""
        num_tasks = 50
        tasks = [Mock(content=f"chaos_task_{i}", type="chaos") for i in range(num_tasks)]

        # Clear metrics
        chaos_agent.analytics.metrics.clear()

        # Process tasks with error injection
        results = []
        for task in tasks:
            result = await chaos_agent.execute_task(task)
            results.append(result)

        # Verify resilience
        assert len(results) == num_tasks

        # Analyze results
        successful_results = [r for r in results if r["result"] == "resilient_decision"]
        [r for r in results if "error" in str(r)]

        # Should have mostly successful results despite errors
        success_rate = len(successful_results) / num_tasks
        assert success_rate >= 0.7, f"Success rate {success_rate:.2f} too low for resilient system"

        # Verify analytics recorded resilience metrics
        analytics_report = chaos_agent.analytics.generate_analytics_report()
        performance_summary = analytics_report["performance_summary"]

        assert "successful_tasks" in performance_summary
        assert "processing_errors" in performance_summary

        # Should have some retries due to errors
        if "retry_attempts" in performance_summary:
            assert performance_summary["retry_attempts"]["count"] > 0

    async def test_network_partition_simulation(self, chaos_agent):
        """Test behavior during simulated network partitions."""
        # Create tasks that require communication
        comm_tasks = [Mock(content=f"network_task_{i}", type="communication") for i in range(10)]

        # Simulate network partition by making communication fail
        chaos_agent.communication_protocol.query.side_effect = ConnectionError("Network partition")
        chaos_agent.communication_protocol.send_message.side_effect = ConnectionError("Network partition")

        # Override communicate method to handle network failures gracefully
        original_communicate = chaos_agent.communicate

        async def resilient_communicate(message, recipient):
            try:
                return await original_communicate(message, recipient)
            except ConnectionError:
                chaos_agent.analytics.record_metric("network_failures", 1.0)
                return "fallback_response_due_to_network_partition"

        chaos_agent.communicate = resilient_communicate

        # Clear metrics
        chaos_agent.analytics.metrics.clear()

        # Process tasks during network partition
        results = []
        for task in comm_tasks:
            try:
                result = await chaos_agent.execute_task(task)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "network_partition": True})

        # Verify system continued operating
        assert len(results) == len(comm_tasks)

        # Should have recorded network failures
        if "network_failures" in chaos_agent.analytics.metrics:
            network_failures = len(chaos_agent.analytics.metrics["network_failures"])
            assert network_failures > 0, "Network failures should have been recorded"

    async def test_memory_pressure_simulation(self, chaos_agent):
        """Test behavior under memory pressure."""
        # Create memory-intensive tasks
        memory_tasks = [
            Mock(content=f"memory_pressure_task_{i}", type="memory", processing_type="memory_intensive")
            for i in range(20)
        ]

        # Monitor memory during processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Clear metrics
        chaos_agent.analytics.metrics.clear()

        # Process tasks while monitoring memory
        results = []
        memory_readings = []

        for i, task in enumerate(memory_tasks):
            result = await chaos_agent.execute_task(task)
            results.append(result)

            # Record memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            chaos_agent.analytics.record_metric("memory_usage_mb", current_memory)

            # Force garbage collection every few tasks
            if i % 5 == 0:
                gc.collect()

        # Verify system handled memory pressure
        assert len(results) == len(memory_tasks)

        # Memory should not grow excessively
        peak_memory = max(memory_readings)
        memory_growth = peak_memory - initial_memory

        # Should not exceed reasonable memory growth
        assert memory_growth < 200, f"Memory growth {memory_growth:.2f}MB seems excessive"

        # Verify analytics tracked memory usage
        analytics_report = chaos_agent.analytics.generate_analytics_report()
        assert "memory_usage_mb" in analytics_report["performance_summary"]

    async def test_concurrent_chaos_scenarios(self, chaos_agent):
        """Test multiple chaos scenarios simultaneously."""
        # Increase error rates for maximum chaos
        chaos_agent.processor.error_simulation_rate = 0.2  # 20% error rate

        # Simulate multiple failure types
        chaos_agent.communication_protocol.query.side_effect = lambda *args: (
            ConnectionError("Network chaos") if random.random() < 0.3 else {"chaos_response": "ok"}
        )

        # Create mixed workload
        num_tasks = 30
        tasks = []

        for i in range(num_tasks):
            task = Mock()
            task.content = f"chaos_concurrent_task_{i}"
            task.type = random.choice(["processing", "communication", "memory"])
            task.processing_type = random.choice(["cpu_intensive", "memory_intensive", "io_intensive"])
            tasks.append(task)

        # Clear metrics
        chaos_agent.analytics.metrics.clear()

        # Process tasks concurrently under chaos
        start_time = time.perf_counter()

        # Process in batches to simulate realistic load
        batch_size = 10
        all_results = []

        for i in range(0, num_tasks, batch_size):
            batch = tasks[i : i + batch_size]

            batch_results = await asyncio.gather(
                *[chaos_agent.execute_task(task) for task in batch], return_exceptions=True
            )

            # Convert exceptions to error results
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    processed_results.append({"error": str(result), "exception": True})
                else:
                    processed_results.append(result)

            all_results.extend(processed_results)

            # Brief pause between batches
            await asyncio.sleep(0.01)

        end_time = time.perf_counter()

        # Verify system survived concurrent chaos
        assert len(all_results) == num_tasks

        # Calculate survival metrics
        successful_results = [r for r in all_results if r.get("result") == "resilient_decision"]
        error_results = [r for r in all_results if "error" in r]

        survival_rate = len(successful_results) / num_tasks
        error_rate = len(error_results) / num_tasks

        # System should maintain reasonable survival rate under chaos
        assert survival_rate >= 0.5, f"Survival rate {survival_rate:.2f} too low under chaos"

        # Should have some errors (proving chaos was active)
        assert error_rate > 0, "Error rate should be > 0 during chaos testing"

        # Performance should degrade but not collapse
        total_time = end_time - start_time
        throughput = num_tasks / total_time
        assert throughput > 5, f"Throughput {throughput:.2f} tasks/sec collapsed under chaos"

        # Verify comprehensive analytics
        analytics_report = chaos_agent.analytics.generate_analytics_report()
        performance_summary = analytics_report["performance_summary"]

        # Should have recorded various types of events
        expected_metrics = ["successful_tasks", "processing_errors", "retry_attempts"]
        recorded_metrics = [m for m in expected_metrics if m in performance_summary]

        assert (
            len(recorded_metrics) >= 2
        ), f"Only recorded {recorded_metrics}, expected more comprehensive chaos metrics"
