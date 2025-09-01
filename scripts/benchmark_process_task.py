#!/usr/bin/env python3
"""
Performance benchmarking script for UnifiedBaseAgent._process_task method.

This script provides comprehensive performance analysis including:
- Latency measurements
- Throughput analysis
- Memory usage profiling
- Scalability testing
- Performance regression detection
"""

import asyncio
import time
import statistics
import json
import sys
import os
from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
import argparse

# Add the agents module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../experiments/agents/agents"))

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory profiling disabled.")

from unittest.mock import Mock, AsyncMock, patch


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""

    test_name: str
    num_tasks: int
    total_time_seconds: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tasks_per_second: float
    success_rate: float
    avg_memory_mb: float
    peak_memory_mb: float
    meets_100ms_target_percentage: float
    timestamp: str


class ProcessTaskBenchmarker:
    """Performance benchmarker for _process_task method."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    async def create_benchmark_agent(self):
        """Create an agent optimized for benchmarking."""
        # Mock configuration
        config = Mock()
        config.name = "BenchmarkAgent"
        config.description = "High-performance benchmark agent"
        config.capabilities = ["general", "text_generation", "data_analysis", "question_answering", "classification"]
        config.model = "gpt-4"
        config.instructions = "You are a high-performance AI agent for benchmarking."
        config.rag_config = Mock()
        config.vector_store = Mock()

        communication_protocol = Mock()
        communication_protocol.subscribe = Mock()
        communication_protocol.send_message = AsyncMock()
        communication_protocol.query = AsyncMock(return_value="Benchmark response")

        # Create agent with mocked dependencies
        with patch("experiments.agents.agents.unified_base_agent.EnhancedRAGPipeline"):
            with patch("experiments.agents.agents.unified_base_agent.OpenAIGPTConfig"):
                from experiments.agents.agents.unified_base_agent import UnifiedBaseAgent

                agent = UnifiedBaseAgent(config, communication_protocol)

        # Mock external services for consistent benchmarking
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=Mock(text="Benchmark response"))
        agent.rag_pipeline = Mock()
        agent.rag_pipeline.process_query = AsyncMock(
            return_value={"answer": "Benchmark RAG response", "confidence": 0.95}
        )

        # Disable logging for cleaner benchmark output
        agent.logger = Mock()

        return agent

    def create_benchmark_task(self, task_id: str, task_type: str = "general", content: str = None, size: str = "small"):
        """Create a benchmark task with specified characteristics."""
        from dataclasses import dataclass

        @dataclass
        class BenchmarkTask:
            content: str
            type: str = "general"
            id: str = "benchmark_task"
            timeout: float = 30.0
            recipient: str = None
            target_agent: str = None

        if content is None:
            content_sizes = {
                "small": "Analyze this data point.",
                "medium": "Analyze this data: " + "sample data, " * 100,
                "large": "Analyze this large dataset: " + "data point, " * 1000,
            }
            content = content_sizes.get(size, content_sizes["small"])

        return BenchmarkTask(content=content, type=task_type, id=task_id, timeout=30.0)

    async def run_latency_benchmark(self, agent, num_tasks: int = 100) -> BenchmarkResult:
        """Run latency benchmark with sequential task processing."""
        print(f"Running latency benchmark with {num_tasks} tasks...")

        latencies = []
        memory_readings = []
        successes = 0

        start_time = time.perf_counter()

        for i in range(num_tasks):
            task = self.create_benchmark_task(f"latency_{i}", "general", size="small")

            task_start = time.perf_counter()
            try:
                result = await agent._process_task(task)
                task_end = time.perf_counter()

                latency_ms = (task_end - task_start) * 1000
                latencies.append(latency_ms)

                # Track memory usage
                if "performance_metrics" in result.get("metadata", {}):
                    memory_mb = result["metadata"]["performance_metrics"]["memory_usage_mb"]
                    if memory_mb > 0:
                        memory_readings.append(memory_mb)

                successes += 1

            except Exception as e:
                print(f"Task {i} failed: {e}")
                continue

            # Progress indicator
            if (i + 1) % (num_tasks // 10) == 0:
                print(f"  Completed {i + 1}/{num_tasks} tasks")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate statistics
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max_latency
            meets_100ms = len([l for l in latencies if l < 100]) / len(latencies) * 100
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = meets_100ms = 0

        avg_memory = statistics.mean(memory_readings) if memory_readings else 0
        peak_memory = max(memory_readings) if memory_readings else 0

        return BenchmarkResult(
            test_name="Latency Benchmark",
            num_tasks=num_tasks,
            total_time_seconds=total_time,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_tasks_per_second=successes / total_time,
            success_rate=successes / num_tasks * 100,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            meets_100ms_target_percentage=meets_100ms,
            timestamp=datetime.now().isoformat(),
        )

    async def run_throughput_benchmark(self, agent, num_tasks: int = 50, concurrency: int = 10) -> BenchmarkResult:
        """Run throughput benchmark with concurrent task processing."""
        print(f"Running throughput benchmark with {num_tasks} tasks, concurrency={concurrency}...")

        tasks = [self.create_benchmark_task(f"throughput_{i}", "general", size="small") for i in range(num_tasks)]

        latencies = []
        successes = 0

        start_time = time.perf_counter()

        # Process tasks in batches for controlled concurrency
        batch_size = concurrency
        for i in range(0, num_tasks, batch_size):
            batch_tasks = tasks[i : i + batch_size]
            time.perf_counter()

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self._process_task_with_timing(agent, task) for task in batch_tasks], return_exceptions=True
            )

            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    continue
                else:
                    latencies.append(result["latency_ms"])
                    successes += 1

            print(f"  Completed batch {i//batch_size + 1}/{(num_tasks + batch_size - 1)//batch_size}")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate statistics
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max_latency
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max_latency
            meets_100ms = len([l for l in latencies if l < 100]) / len(latencies) * 100
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = meets_100ms = 0

        return BenchmarkResult(
            test_name=f"Throughput Benchmark (Concurrency={concurrency})",
            num_tasks=num_tasks,
            total_time_seconds=total_time,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_tasks_per_second=successes / total_time,
            success_rate=successes / num_tasks * 100,
            avg_memory_mb=0,  # Not tracked in throughput test
            peak_memory_mb=0,
            meets_100ms_target_percentage=meets_100ms,
            timestamp=datetime.now().isoformat(),
        )

    async def _process_task_with_timing(self, agent, task):
        """Process a single task and return timing information."""
        try:
            start = time.perf_counter()
            result = await agent._process_task(task)
            end = time.perf_counter()

            return {"latency_ms": (end - start) * 1000, "success": True, "result": result}
        except Exception as e:
            return Exception(f"Task failed: {e}")

    async def run_scalability_benchmark(self, agent) -> List[BenchmarkResult]:
        """Run scalability benchmark with increasing load."""
        print("Running scalability benchmark...")

        results = []
        task_counts = [10, 25, 50, 100, 200]

        for num_tasks in task_counts:
            print(f"  Testing with {num_tasks} tasks...")
            result = await self.run_throughput_benchmark(agent, num_tasks, concurrency=10)
            result.test_name = f"Scalability Test ({num_tasks} tasks)"
            results.append(result)

        return results

    async def run_task_type_benchmark(self, agent) -> List[BenchmarkResult]:
        """Benchmark different task types."""
        print("Running task type benchmark...")

        task_types = [
            "general",
            "text_generation",
            "question_answering",
            "data_analysis",
            "classification",
            "summarization",
        ]

        results = []

        for task_type in task_types:
            print(f"  Testing {task_type} tasks...")

            latencies = []
            successes = 0
            num_tasks = 20

            start_time = time.perf_counter()

            for i in range(num_tasks):
                task = self.create_benchmark_task(f"{task_type}_{i}", task_type, size="small")

                try:
                    task_start = time.perf_counter()
                    result = await agent._process_task(task)
                    task_end = time.perf_counter()

                    latencies.append((task_end - task_start) * 1000)
                    successes += 1

                except Exception:
                    continue

            end_time = time.perf_counter()
            total_time = end_time - start_time

            if latencies:
                avg_latency = statistics.mean(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                meets_100ms = len([l for l in latencies if l < 100]) / len(latencies) * 100
            else:
                avg_latency = min_latency = max_latency = meets_100ms = 0

            result = BenchmarkResult(
                test_name=f"Task Type: {task_type}",
                num_tasks=num_tasks,
                total_time_seconds=total_time,
                avg_latency_ms=avg_latency,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                p95_latency_ms=max_latency,  # Simplified for small sample
                p99_latency_ms=max_latency,
                throughput_tasks_per_second=successes / total_time if total_time > 0 else 0,
                success_rate=successes / num_tasks * 100,
                avg_memory_mb=0,
                peak_memory_mb=0,
                meets_100ms_target_percentage=meets_100ms,
                timestamp=datetime.now().isoformat(),
            )

            results.append(result)

        return results

    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 100)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 100)

        for result in results:
            print(f"\nTest: {result.test_name}")
            print(f"  Tasks: {result.num_tasks}")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Total Time: {result.total_time_seconds:.2f}s")
            print(f"  Throughput: {result.throughput_tasks_per_second:.1f} tasks/sec")
            print("  Latency:")
            print(f"    Average: {result.avg_latency_ms:.2f}ms")
            print(f"    Min: {result.min_latency_ms:.2f}ms")
            print(f"    Max: {result.max_latency_ms:.2f}ms")
            print(f"    P95: {result.p95_latency_ms:.2f}ms")
            print(f"    P99: {result.p99_latency_ms:.2f}ms")
            print(f"  Performance Target (<100ms): {result.meets_100ms_target_percentage:.1f}%")
            if result.avg_memory_mb > 0:
                print("  Memory:")
                print(f"    Average: {result.avg_memory_mb:.1f}MB")
                print(f"    Peak: {result.peak_memory_mb:.1f}MB")

    def save_results(self, results: List[BenchmarkResult], filename: str):
        """Save benchmark results to JSON file."""
        results_data = [asdict(result) for result in results]

        with open(filename, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to: {filename}")

    async def run_comprehensive_benchmark(self):
        """Run all benchmark suites."""
        print("Starting comprehensive performance benchmark...")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        if PSUTIL_AVAILABLE:
            print(f"Memory available: {psutil.virtual_memory().available // (1024**3)}GB")
        print()

        # Create benchmark agent
        agent = await self.create_benchmark_agent()

        all_results = []

        # 1. Latency benchmark
        latency_result = await self.run_latency_benchmark(agent, num_tasks=100)
        all_results.append(latency_result)

        # 2. Throughput benchmark
        throughput_result = await self.run_throughput_benchmark(agent, num_tasks=50, concurrency=10)
        all_results.append(throughput_result)

        # 3. Scalability benchmark
        scalability_results = await self.run_scalability_benchmark(agent)
        all_results.extend(scalability_results)

        # 4. Task type benchmark
        task_type_results = await self.run_task_type_benchmark(agent)
        all_results.extend(task_type_results)

        # Print and save results
        self.print_results(all_results)

        # Save to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        self.save_results(all_results, filename)

        # Performance analysis
        self.analyze_performance(all_results)

        return all_results

    def analyze_performance(self, results: List[BenchmarkResult]):
        """Analyze performance results and provide insights."""
        print("\n" + "=" * 100)
        print("PERFORMANCE ANALYSIS")
        print("=" * 100)

        # Find best and worst performing tests
        latency_results = [r for r in results if "Latency" in r.test_name or "Throughput" in r.test_name]

        if latency_results:
            best_latency = min(latency_results, key=lambda r: r.avg_latency_ms)
            worst_latency = max(latency_results, key=lambda r: r.avg_latency_ms)

            print("\nLatency Analysis:")
            print(f"  Best: {best_latency.test_name} - {best_latency.avg_latency_ms:.2f}ms avg")
            print(f"  Worst: {worst_latency.test_name} - {worst_latency.avg_latency_ms:.2f}ms avg")

        # Throughput analysis
        throughput_results = [r for r in results if r.throughput_tasks_per_second > 0]
        if throughput_results:
            best_throughput = max(throughput_results, key=lambda r: r.throughput_tasks_per_second)
            print("\nThroughput Analysis:")
            print(f"  Peak: {best_throughput.test_name} - {best_throughput.throughput_tasks_per_second:.1f} tasks/sec")

        # Performance target analysis
        target_meeting_results = [r for r in results if r.meets_100ms_target_percentage > 0]
        if target_meeting_results:
            avg_target_meeting = sum(r.meets_100ms_target_percentage for r in target_meeting_results) / len(
                target_meeting_results
            )
            print("\n100ms Performance Target:")
            print(f"  Average across tests: {avg_target_meeting:.1f}%")

            if avg_target_meeting >= 80:
                print("  ✓ EXCELLENT: Consistently meets performance target")
            elif avg_target_meeting >= 60:
                print("  ⚠ GOOD: Usually meets performance target")
            elif avg_target_meeting >= 40:
                print("  ⚠ FAIR: Sometimes meets performance target")
            else:
                print("  ✗ POOR: Rarely meets performance target")

        print()


async def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark UnifiedBaseAgent._process_task performance")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer tasks")
    parser.add_argument("--output", type=str, help="Output filename for results")

    args = parser.parse_args()

    benchmarker = ProcessTaskBenchmarker()

    if args.quick:
        print("Running quick benchmark...")
        agent = await benchmarker.create_benchmark_agent()
        results = [
            await benchmarker.run_latency_benchmark(agent, num_tasks=20),
            await benchmarker.run_throughput_benchmark(agent, num_tasks=20, concurrency=5),
        ]
    else:
        results = await benchmarker.run_comprehensive_benchmark()

    if args.output:
        benchmarker.save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
