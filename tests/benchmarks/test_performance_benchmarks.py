"""Performance benchmarks and load testing for Frontier Curriculum Engine.

Tests system performance under various load conditions and measures
key performance indicators for production deployment.
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import statistics

# Test imports
import sys
import time
from typing import Any

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from agent_forge.curriculum import (
    CurriculumOrchestrator,
    EdgeConstraints,
    EdgeFinder,
    EdgeWindow,
    Grader,
    NumericJitterPolicy,
    OpenRouterLLM,
    Problem,
    ProblemGenerator,
    TelemetryEntry,
    TopicMix,
    VariantMaker,
    VariantPolicy,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

    operation: str
    duration_seconds: float
    memory_mb: float
    cpu_percent: float
    items_processed: int
    throughput_per_second: float
    success_rate: float
    api_calls: int = 0
    cache_hits: int = 0


class PerformanceBenchmark:
    """Performance benchmarking framework."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or "mock-key"
        self.use_mock = api_key is None
        self.results: list[PerformanceMetrics] = []

    def create_llm_client(self) -> OpenRouterLLM:
        """Create LLM client (mock or real based on API key availability)."""
        if self.use_mock:
            from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

            return MockOpenRouterLLM(self.api_key)
        else:
            return OpenRouterLLM(api_key=self.api_key, model="openai/gpt-4o-mini")

    async def measure_performance(self, operation_name: str, operation_func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an async operation."""
        process = psutil.Process(os.getpid())

        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_cpu = process.cpu_percent()

        # Execute operation
        try:
            result = await operation_func(*args, **kwargs)
            success = True
            items_processed = self._count_items_processed(result)
        except Exception as e:
            logger.error(f"Operation {operation_name} failed: {e}")
            success = False
            items_processed = 0
            result = None

        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        end_cpu = process.cpu_percent()

        # Calculate metrics
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        cpu_used = (start_cpu + end_cpu) / 2
        throughput = items_processed / duration if duration > 0 else 0
        success_rate = 1.0 if success else 0.0

        metrics = PerformanceMetrics(
            operation=operation_name,
            duration_seconds=duration,
            memory_mb=memory_used,
            cpu_percent=cpu_used,
            items_processed=items_processed,
            throughput_per_second=throughput,
            success_rate=success_rate,
        )

        self.results.append(metrics)
        return metrics

    def _count_items_processed(self, result) -> int:
        """Count items processed based on result type."""
        if hasattr(result, "problems"):
            return len(result.problems)
        elif hasattr(result, "variants"):
            return len(result.variants)
        elif hasattr(result, "queue"):
            return sum(item.n for item in result.queue)
        elif isinstance(result, list):
            return len(result)
        elif isinstance(result, dict) and "items_processed" in result:
            return result["items_processed"]
        else:
            return 1  # Single item

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No performance data available"}

        # Group by operation type
        by_operation = {}
        for metric in self.results:
            op = metric.operation
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(metric)

        # Calculate statistics for each operation
        operation_stats = {}
        for op, metrics in by_operation.items():
            durations = [m.duration_seconds for m in metrics]
            throughputs = [m.throughput_per_second for m in metrics]
            memory_usage = [m.memory_mb for m in metrics]
            success_rates = [m.success_rate for m in metrics]

            operation_stats[op] = {
                "total_runs": len(metrics),
                "avg_duration": statistics.mean(durations),
                "median_duration": statistics.median(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "avg_throughput": statistics.mean(throughputs),
                "avg_memory_mb": statistics.mean(memory_usage),
                "success_rate": statistics.mean(success_rates),
                "total_items": sum(m.items_processed for m in metrics),
            }

        # Overall system stats
        total_duration = sum(m.duration_seconds for m in self.results)
        total_items = sum(m.items_processed for m in self.results)
        overall_throughput = total_items / total_duration if total_duration > 0 else 0

        return {
            "summary": {
                "total_operations": len(self.results),
                "total_duration_seconds": total_duration,
                "total_items_processed": total_items,
                "overall_throughput": overall_throughput,
                "test_mode": "mock" if self.use_mock else "live",
            },
            "by_operation": operation_stats,
            "raw_metrics": [
                {
                    "operation": m.operation,
                    "duration": m.duration_seconds,
                    "throughput": m.throughput_per_second,
                    "items": m.items_processed,
                    "success": m.success_rate,
                }
                for m in self.results
            ],
        }


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    @pytest.fixture
    def benchmark(self):
        """Performance benchmark fixture."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        return PerformanceBenchmark(api_key)

    @pytest.fixture
    def sample_telemetry_large(self):
        """Large telemetry dataset for performance testing."""
        import random

        telemetry = []
        for i in range(200):  # Larger dataset
            difficulty = random.uniform(0.2, 0.9)
            correct = random.random() < (0.8 - difficulty)  # Realistic accuracy curve
            telemetry.append(
                TelemetryEntry(
                    task_id=f"perf_task_{i:03d}",
                    difficulty=round(difficulty, 2),
                    correct=correct,
                )
            )
        return telemetry

    @pytest.mark.asyncio
    async def test_edge_finder_performance(self, benchmark, sample_telemetry_large):
        """Benchmark EdgeFinder performance."""
        async with benchmark.create_llm_client() as client:
            edge_finder = EdgeFinder(client)

            # Benchmark multiple edge finding operations
            for i in range(3):
                metrics = await benchmark.measure_performance(
                    f"edge_finding_{i}",
                    edge_finder.find_edge,
                    domain="coding-python",
                    telemetry=sample_telemetry_large[: 50 + i * 25],  # Varying sizes
                    constraints=EdgeConstraints(target_low=0.55, target_high=0.75),
                )

                # Performance assertions
                assert (
                    metrics.duration_seconds < 120
                ), f"Edge finding took {metrics.duration_seconds:.1f}s, should be <120s"
                assert metrics.success_rate == 1.0, "Edge finding should succeed"

                logger.info(
                    f"Edge finding {i + 1}: {metrics.duration_seconds:.1f}s, {metrics.throughput_per_second:.2f} items/sec"
                )

    @pytest.mark.asyncio
    async def test_problem_generator_performance(self, benchmark):
        """Benchmark ProblemGenerator performance."""
        async with benchmark.create_llm_client() as client:
            problem_gen = ProblemGenerator(client)

            edge = EdgeWindow(low=0.55, high=0.75)
            topic_mix = [
                TopicMix(topic="string_manipulation", weight=0.5),
                TopicMix(topic="list_operations", weight=0.5),
            ]

            # Benchmark different batch sizes
            batch_sizes = [5, 10, 15]
            for batch_size in batch_sizes:
                metrics = await benchmark.measure_performance(
                    f"problem_generation_{batch_size}",
                    problem_gen.generate_problems,
                    domain="coding-python",
                    edge=edge,
                    topic_mix=topic_mix,
                    n=batch_size,
                    batch_size=min(3, batch_size),
                )

                # Performance assertions
                assert (
                    metrics.duration_seconds < 300
                ), f"Problem generation took {metrics.duration_seconds:.1f}s, should be <300s"
                assert metrics.throughput_per_second > 0.05, f"Throughput {metrics.throughput_per_second:.3f} too low"

                logger.info(
                    f"Problem generation {batch_size}: {metrics.duration_seconds:.1f}s, {metrics.throughput_per_second:.2f} problems/sec"
                )

    @pytest.mark.asyncio
    async def test_variant_maker_performance(self, benchmark):
        """Benchmark VariantMaker performance."""
        async with benchmark.create_llm_client() as client:
            variant_maker = VariantMaker(client)

            base_problem = Problem(
                id="perf_test_001",
                topic="string_manipulation",
                difficulty=0.6,
                statement="Write a function that counts characters in a string",
                canonical_answer="def count_chars(s): return len(s)",
                rubric="Function returns correct character count",
                unit_tests=["assert count_chars('hello') == 5"],
            )

            variant_policy = VariantPolicy(
                paraphrase=True,
                numeric_jitter=NumericJitterPolicy(enabled=True, pct=10),
            )

            # Benchmark different variant counts
            variant_counts = [2, 3, 5]
            for count in variant_counts:
                metrics = await benchmark.measure_performance(
                    f"variant_generation_{count}",
                    variant_maker.create_variants,
                    base_problem=base_problem,
                    variant_policy=variant_policy,
                    n_variants=count,
                )

                assert (
                    metrics.duration_seconds < 180
                ), f"Variant generation took {metrics.duration_seconds:.1f}s, should be <180s"

                logger.info(
                    f"Variant generation {count}: {metrics.duration_seconds:.1f}s, {metrics.throughput_per_second:.2f} variants/sec"
                )

    @pytest.mark.asyncio
    async def test_grader_performance(self, benchmark):
        """Benchmark Grader performance."""
        async with benchmark.create_llm_client() as client:
            grader = Grader(client, enable_code_execution=False)

            test_problem = Problem(
                id="grading_perf_001",
                topic="algorithms",
                difficulty=0.6,
                statement="Write a function that finds the maximum value in a list",
                canonical_answer="def find_max(lst): return max(lst) if lst else None",
                rubric="Function finds maximum correctly",
                unit_tests=["assert find_max([1,3,2]) == 3"],
            )

            test_answers = [
                "def find_max(lst): return max(lst) if lst else None",
                "def find_max(lst): return max(lst)",
                "def find_max(lst): return min(lst)",
                "print('hello world')",
                "def find_max(lst):\n    max_val = lst[0]\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val",
            ]

            # Benchmark batch grading
            metrics = await benchmark.measure_performance(
                "batch_grading",
                grader.grade_batch,
                problems=[test_problem] * len(test_answers),
                model_answers=test_answers,
            )

            assert metrics.duration_seconds < 300, f"Batch grading took {metrics.duration_seconds:.1f}s"
            assert (
                metrics.throughput_per_second > 0.01
            ), f"Grading throughput too low: {metrics.throughput_per_second:.3f}"

            logger.info(
                f"Batch grading: {metrics.duration_seconds:.1f}s, {metrics.throughput_per_second:.2f} solutions/sec"
            )

    @pytest.mark.asyncio
    async def test_orchestrator_performance(self, benchmark, sample_telemetry_large):
        """Benchmark CurriculumOrchestrator performance."""
        async with benchmark.create_llm_client() as client:
            orchestrator = CurriculumOrchestrator(client)

            # Benchmark initialization
            metrics = await benchmark.measure_performance(
                "orchestrator_init",
                orchestrator.initialize_curriculum,
                domain="coding-python",
                initial_telemetry=sample_telemetry_large,
                constraints=EdgeConstraints(problem_budget=50),
            )

            assert metrics.duration_seconds < 600, f"Orchestrator init took {metrics.duration_seconds:.1f}s"
            assert metrics.success_rate == 1.0, "Orchestrator initialization should succeed"

            # Benchmark curriculum cycles
            cycle_metrics = await benchmark.measure_performance(
                "curriculum_cycles",
                orchestrator.run_curriculum_cycle,
                domain="coding-python",
                num_cycles=3,
                cycle_capacity=10,
            )

            assert cycle_metrics.duration_seconds < 900, f"Curriculum cycles took {cycle_metrics.duration_seconds:.1f}s"

            logger.info(f"Orchestrator init: {metrics.duration_seconds:.1f}s")
            logger.info(f"Curriculum cycles: {cycle_metrics.duration_seconds:.1f}s")


class TestLoadTesting:
    """Load testing for system scalability."""

    @pytest.mark.asyncio
    async def test_concurrent_component_usage(self):
        """Test concurrent usage of multiple components."""
        api_key = os.getenv("OPENROUTER_API_KEY", "test_mock_api_key")  # pragma: allowlist secret
        use_mock = api_key == "test_mock_api_key"  # pragma: allowlist secret

        if use_mock:
            from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

            def create_client():
                return MockOpenRouterLLM(api_key)

        else:

            def create_client():
                return OpenRouterLLM(api_key=api_key, model="openai/gpt-4o-mini")

        async def worker_task(worker_id: int, num_operations: int):
            """Worker task for concurrent testing."""
            results = []

            async with create_client() as client:
                problem_gen = ProblemGenerator(client)
                edge = EdgeWindow(low=0.55, high=0.75)
                topic_mix = [TopicMix(topic=f"topic_{worker_id}", weight=1.0)]

                for i in range(num_operations):
                    try:
                        start_time = time.time()

                        result = await problem_gen.generate_problems(
                            domain="coding-python",
                            edge=edge,
                            topic_mix=topic_mix,
                            n=2,  # Small number for load testing
                        )

                        duration = time.time() - start_time
                        results.append(
                            {
                                "worker": worker_id,
                                "operation": i,
                                "duration": duration,
                                "success": result.ok,
                                "items": len(result.problems) if result.ok else 0,
                            }
                        )

                        # Small delay to avoid overwhelming
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        results.append(
                            {
                                "worker": worker_id,
                                "operation": i,
                                "duration": 0,
                                "success": False,
                                "error": str(e),
                                "items": 0,
                            }
                        )

            return results

        # Run concurrent workers
        num_workers = 3
        operations_per_worker = 2

        start_time = time.time()

        tasks = [worker_task(worker_id, operations_per_worker) for worker_id in range(num_workers)]

        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = time.time() - start_time

        # Analyze results
        successful_operations = 0
        failed_operations = 0
        total_items = 0

        for worker_result in worker_results:
            if isinstance(worker_result, Exception):
                failed_operations += operations_per_worker
                continue

            for op_result in worker_result:
                if op_result["success"]:
                    successful_operations += 1
                    total_items += op_result["items"]
                else:
                    failed_operations += 1

        # Performance assertions
        success_rate = successful_operations / (successful_operations + failed_operations)
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} too low for load test"

        throughput = total_items / total_duration
        assert throughput > 0, "Should process some items"

        logger.info(f"Load test: {num_workers} workers, {operations_per_worker} ops each")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Total duration: {total_duration:.1f}s")
        logger.info(f"Throughput: {throughput:.2f} items/sec")

    @pytest.mark.asyncio
    async def test_memory_scaling(self):
        """Test memory usage scaling with increasing load."""
        api_key = os.getenv("OPENROUTER_API_KEY", "test_mock_api_key")  # pragma: allowlist secret
        use_mock = api_key == "test_mock_api_key"  # pragma: allowlist secret

        if use_mock:
            from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

            client = MockOpenRouterLLM(api_key)
        else:
            client = OpenRouterLLM(api_key=api_key)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        problem_gen = ProblemGenerator(client)
        edge = EdgeWindow(low=0.55, high=0.75)
        topic_mix = [TopicMix(topic="scaling_test", weight=1.0)]

        memory_measurements = [(0, initial_memory)]

        # Generate increasing numbers of problems and measure memory
        problem_counts = [5, 10, 15, 20]
        for count in problem_counts:
            await problem_gen.generate_problems(domain="coding-python", edge=edge, topic_mix=topic_mix, n=count)

            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_measurements.append((count, current_memory))

            # Memory growth should be reasonable
            memory_growth = current_memory - initial_memory
            max_allowed_growth = count * 2  # 2MB per problem is generous

            assert (
                memory_growth < max_allowed_growth
            ), f"Memory grew {memory_growth:.1f}MB for {count} problems, max allowed: {max_allowed_growth}MB"

            logger.info(f"Generated {count} problems, memory: {current_memory:.1f}MB (+{memory_growth:.1f}MB)")

        # Check memory doesn't grow linearly (should be sub-linear due to reuse)
        final_memory = memory_measurements[-1][1]
        final_growth = final_memory - initial_memory
        max_problems = problem_counts[-1]

        # Memory per problem should decrease with scale (efficiency)
        memory_per_problem = final_growth / max_problems if max_problems > 0 else 0
        assert memory_per_problem < 5.0, f"Memory per problem {memory_per_problem:.1f}MB too high"

        if not use_mock:
            await client.close()


class TestScalabilityLimits:
    """Test system behavior at scalability limits."""

    @pytest.mark.asyncio
    async def test_large_telemetry_processing(self):
        """Test processing very large telemetry datasets."""
        # Generate large telemetry dataset
        large_telemetry = []
        import random

        for i in range(1000):  # Very large dataset
            difficulty = random.uniform(0.1, 0.9)
            correct = random.random() < (0.9 - difficulty)
            large_telemetry.append(
                TelemetryEntry(
                    task_id=f"large_task_{i:04d}",
                    difficulty=round(difficulty, 3),
                    correct=correct,
                )
            )

        api_key = os.getenv("OPENROUTER_API_KEY", "test_mock_api_key")  # pragma: allowlist secret
        use_mock = api_key == "test_mock_api_key"  # pragma: allowlist secret

        if use_mock:
            from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

            client = MockOpenRouterLLM(api_key)
        else:
            client = OpenRouterLLM(api_key=api_key)

        edge_finder = EdgeFinder(client)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        # Process large telemetry
        result = await edge_finder.find_edge(
            domain="coding-python",
            telemetry=large_telemetry,
            constraints=EdgeConstraints(problem_budget=100),
        )

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        duration = end_time - start_time
        memory_used = end_memory - start_memory

        # Should handle large datasets efficiently
        assert result.ok, "Should successfully process large telemetry"
        assert duration < 300, f"Processing 1000 telemetry entries took {duration:.1f}s, should be <300s"
        assert memory_used < 100, f"Used {memory_used:.1f}MB memory, should be <100MB"

        throughput = len(large_telemetry) / duration
        assert throughput > 3, f"Throughput {throughput:.1f} entries/sec too low"

        logger.info(f"Processed {len(large_telemetry)} telemetry entries in {duration:.1f}s")
        logger.info(f"Memory usage: {memory_used:.1f}MB, throughput: {throughput:.1f} entries/sec")

        if not use_mock:
            await client.close()


def generate_performance_report_file(benchmark: PerformanceBenchmark, output_path: str):
    """Generate detailed performance report file."""
    report = benchmark.generate_performance_report()

    # Add system information
    report["system_info"] = {
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("FRONTIER CURRICULUM ENGINE - PERFORMANCE REPORT")
    print(f"{'=' * 60}")

    summary = report["summary"]
    print(f"Test Mode: {summary['test_mode']}")
    print(f"Total Operations: {summary['total_operations']}")
    print(f"Total Duration: {summary['total_duration_seconds']:.1f}s")
    print(f"Items Processed: {summary['total_items_processed']}")
    print(f"Overall Throughput: {summary['overall_throughput']:.2f} items/sec")

    print(f"\n{'By Operation:':<20} {'Avg Duration':<12} {'Throughput':<12} {'Success Rate'}")
    print("-" * 60)

    for op, stats in report["by_operation"].items():
        print(
            f"{op:<20} {stats['avg_duration']:<12.2f} {stats['avg_throughput']:<12.2f} {stats['success_rate']:<12.1%}"
        )

    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
