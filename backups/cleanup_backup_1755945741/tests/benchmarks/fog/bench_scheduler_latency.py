"""
Fog Scheduler Latency Benchmarks

Comprehensive benchmark suite to validate scheduler performance requirements:
- p95 placement latency <500ms in local environment
- Load testing under various conditions
- SLA class performance validation
- NSGA-II algorithm performance benchmarking

Tests scheduler performance under different scenarios:
- Single job placement
- Batch job submission
- High load stress testing
- Node failure scenarios
- SLA class distribution testing
"""

import asyncio
from dataclasses import dataclass
import statistics
import time

import pytest

from packages.fog.gateway.monitoring.metrics import FogMetricsCollector

# Mock imports for testing (would be real in production)
from packages.fog.gateway.scheduler.placement import FogNode, FogScheduler, JobRequest, NSGA2PlacementEngine, SLAClass
from packages.fog.gateway.scheduler.sla_classes import SLAManager


@dataclass
class BenchmarkResult:
    """Benchmark execution result"""

    test_name: str
    total_jobs: int
    duration_seconds: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_jobs_per_sec: float
    success_rate: float
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    def meets_sla_requirement(self) -> bool:
        """Check if p95 latency meets <500ms requirement"""
        return self.p95_latency_ms < 500.0

    def __str__(self) -> str:
        sla_status = "✅ PASS" if self.meets_sla_requirement() else "❌ FAIL"
        return (
            f"Test: {self.test_name}\n"
            f"  Jobs: {self.total_jobs}, Duration: {self.duration_seconds:.2f}s\n"
            f"  Latency - Avg: {self.avg_latency_ms:.1f}ms, P95: {self.p95_latency_ms:.1f}ms {sla_status}\n"
            f"  Throughput: {self.throughput_jobs_per_sec:.1f} jobs/sec\n"
            f"  Success Rate: {self.success_rate:.1%}"
        )


class SchedulerBenchmark:
    """
    Comprehensive scheduler benchmark suite

    Tests fog scheduler performance under various load conditions
    and validates SLA requirements for placement latency.
    """

    def __init__(self, node_count: int = 10):
        """Initialize benchmark with mock fog nodes"""
        self.metrics_collector = FogMetricsCollector()
        self.sla_manager = SLAManager(self.metrics_collector)
        self.scheduler = self._create_test_scheduler(node_count)
        self.results: list[BenchmarkResult] = []

    def _create_test_scheduler(self, node_count: int) -> FogScheduler:
        """Create test scheduler with mock fog nodes"""

        # Create mock fog nodes with varying capabilities
        nodes = []
        for i in range(node_count):
            node = FogNode(
                node_id=f"test-node-{i:02d}",
                cpu_cores=4.0 + (i % 4),  # 4-7 cores
                memory_gb=8.0 + (i % 8) * 2,  # 8-22 GB
                available_cpu=0.8,  # 80% available
                available_memory=0.7,  # 70% available
                trust_score=0.8 + (i % 3) * 0.1,  # 0.8-1.0 trust
                region=f"region-{i % 3}",  # 3 regions
                latency_ms=50 + (i % 5) * 20,  # 50-130ms latency
            )
            nodes.append(node)

        # Create NSGA-II placement engine
        placement_engine = NSGA2PlacementEngine(
            population_size=30,  # Smaller for faster benchmarks
            max_generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
        )

        return FogScheduler(
            placement_engine=placement_engine,
            metrics_collector=self.metrics_collector,
            sla_manager=self.sla_manager,
            available_nodes=nodes,
        )

    async def benchmark_single_job_placement(self, job_count: int = 100) -> BenchmarkResult:
        """Benchmark single job placement latency"""

        latencies = []
        successful_placements = 0

        start_time = time.time()

        for i in range(job_count):
            job = JobRequest(
                job_id=f"single-job-{i}",
                sla_class=SLAClass.B,  # Best-effort for baseline
                cpu_request=1.0,
                memory_request_gb=2.0,
                namespace="benchmark",
            )

            placement_start = time.time()

            try:
                placement = await self.scheduler.schedule_job(job)
                placement_end = time.time()

                if placement and placement.selected_nodes:
                    latency_ms = (placement_end - placement_start) * 1000
                    latencies.append(latency_ms)
                    successful_placements += 1

            except Exception as e:
                print(f"Failed to place job {job.job_id}: {e}")

        end_time = time.time()
        duration = end_time - start_time

        return self._calculate_benchmark_result(
            "Single Job Placement", job_count, duration, latencies, successful_placements
        )

    async def benchmark_batch_placement(self, batch_size: int = 50, batch_count: int = 10) -> BenchmarkResult:
        """Benchmark batch job placement performance"""

        latencies = []
        successful_placements = 0
        total_jobs = batch_size * batch_count

        start_time = time.time()

        for batch_id in range(batch_count):
            # Create batch of jobs
            jobs = []
            for i in range(batch_size):
                job = JobRequest(
                    job_id=f"batch-{batch_id}-job-{i}",
                    sla_class=SLAClass.A,  # A-class for higher requirements
                    cpu_request=1.5,
                    memory_request_gb=3.0,
                    namespace="benchmark",
                )
                jobs.append(job)

            # Schedule batch
            batch_start = time.time()

            try:
                placements = await self.scheduler.schedule_batch(jobs)
                batch_end = time.time()

                # Calculate per-job latency from batch
                batch_latency_ms = (batch_end - batch_start) * 1000
                per_job_latency = batch_latency_ms / batch_size

                for placement in placements:
                    if placement and placement.selected_nodes:
                        latencies.append(per_job_latency)
                        successful_placements += 1

            except Exception as e:
                print(f"Failed to place batch {batch_id}: {e}")

        end_time = time.time()
        duration = end_time - start_time

        return self._calculate_benchmark_result(
            "Batch Job Placement", total_jobs, duration, latencies, successful_placements
        )

    async def benchmark_high_load_stress(
        self, jobs_per_second: int = 20, duration_seconds: int = 30
    ) -> BenchmarkResult:
        """Benchmark scheduler under high load stress"""

        latencies = []
        successful_placements = 0
        job_count = 0

        start_time = time.time()
        end_time = start_time + duration_seconds

        while time.time() < end_time:
            # Submit jobs at target rate
            batch_start = time.time()

            # Create burst of jobs
            jobs_in_burst = min(jobs_per_second, 10)  # Max 10 jobs per burst

            for i in range(jobs_in_burst):
                job = JobRequest(
                    job_id=f"stress-{job_count}",
                    sla_class=SLAClass.S,  # S-class for maximum stress
                    cpu_request=2.0,
                    memory_request_gb=4.0,
                    namespace="stress-test",
                )

                placement_start = time.time()

                try:
                    placement = await self.scheduler.schedule_job(job)
                    placement_end = time.time()

                    if placement and placement.selected_nodes:
                        latency_ms = (placement_end - placement_start) * 1000
                        latencies.append(latency_ms)
                        successful_placements += 1

                except Exception as e:
                    print(f"Failed to place stress job {job_count}: {e}")

                job_count += 1

            # Control rate
            batch_duration = time.time() - batch_start
            target_duration = 1.0  # 1 second per batch
            if batch_duration < target_duration:
                await asyncio.sleep(target_duration - batch_duration)

        actual_duration = time.time() - start_time

        return self._calculate_benchmark_result(
            f"High Load Stress ({jobs_per_second} jobs/sec)",
            job_count,
            actual_duration,
            latencies,
            successful_placements,
        )

    async def benchmark_sla_class_distribution(self, job_count: int = 300) -> BenchmarkResult:
        """Benchmark scheduler with realistic SLA class distribution"""

        latencies = []
        successful_placements = 0

        start_time = time.time()

        for i in range(job_count):
            # Realistic SLA distribution: 10% S, 30% A, 60% B
            if i % 10 == 0:
                sla_class = SLAClass.S
                cpu_req = 3.0
                memory_req = 6.0
            elif i % 10 < 4:
                sla_class = SLAClass.A
                cpu_req = 2.0
                memory_req = 4.0
            else:
                sla_class = SLAClass.B
                cpu_req = 1.0
                memory_req = 2.0

            job = JobRequest(
                job_id=f"sla-dist-{i}",
                sla_class=sla_class,
                cpu_request=cpu_req,
                memory_request_gb=memory_req,
                namespace="sla-test",
            )

            placement_start = time.time()

            try:
                placement = await self.scheduler.schedule_job(job)
                placement_end = time.time()

                if placement and placement.selected_nodes:
                    latency_ms = (placement_end - placement_start) * 1000
                    latencies.append(latency_ms)
                    successful_placements += 1

            except Exception as e:
                print(f"Failed to place SLA job {i}: {e}")

        end_time = time.time()
        duration = end_time - start_time

        return self._calculate_benchmark_result(
            "SLA Class Distribution", job_count, duration, latencies, successful_placements
        )

    async def benchmark_node_failure_resilience(self, job_count: int = 100) -> BenchmarkResult:
        """Benchmark scheduler resilience to node failures"""

        latencies = []
        successful_placements = 0

        start_time = time.time()

        for i in range(job_count):
            # Simulate node failures at 25% and 75% progress
            if i == job_count // 4:
                # Fail 2 nodes
                self.scheduler.mark_node_failed("test-node-01")
                self.scheduler.mark_node_failed("test-node-02")
                print("Simulated failure of 2 nodes at 25% progress")
            elif i == 3 * job_count // 4:
                # Recover 1 node
                self.scheduler.mark_node_recovered("test-node-01")
                print("Simulated recovery of 1 node at 75% progress")

            job = JobRequest(
                job_id=f"resilience-{i}",
                sla_class=SLAClass.A,
                cpu_request=1.5,
                memory_request_gb=3.0,
                namespace="resilience-test",
            )

            placement_start = time.time()

            try:
                placement = await self.scheduler.schedule_job(job)
                placement_end = time.time()

                if placement and placement.selected_nodes:
                    latency_ms = (placement_end - placement_start) * 1000
                    latencies.append(latency_ms)
                    successful_placements += 1

            except Exception as e:
                print(f"Failed to place resilience job {i}: {e}")

        end_time = time.time()
        duration = end_time - start_time

        return self._calculate_benchmark_result(
            "Node Failure Resilience", job_count, duration, latencies, successful_placements
        )

    def _calculate_benchmark_result(
        self, test_name: str, job_count: int, duration: float, latencies: list[float], successful_placements: int
    ) -> BenchmarkResult:
        """Calculate benchmark statistics"""

        if not latencies:
            return BenchmarkResult(
                test_name=test_name,
                total_jobs=job_count,
                duration_seconds=duration,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=float("inf"),  # Fail SLA requirement
                p99_latency_ms=0.0,
                max_latency_ms=0.0,
                throughput_jobs_per_sec=0.0,
                success_rate=0.0,
            )

        # Calculate latency percentiles
        sorted_latencies = sorted(latencies)

        return BenchmarkResult(
            test_name=test_name,
            total_jobs=job_count,
            duration_seconds=duration,
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=self._percentile(sorted_latencies, 95),
            p99_latency_ms=self._percentile(sorted_latencies, 99),
            max_latency_ms=max(latencies),
            throughput_jobs_per_sec=successful_placements / duration if duration > 0 else 0,
            success_rate=successful_placements / job_count if job_count > 0 else 0,
        )

    def _percentile(self, sorted_data: list[float], percentile: int) -> float:
        """Calculate percentile from sorted data"""
        if not sorted_data:
            return 0.0

        index = (percentile / 100.0) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    async def run_all_benchmarks(self) -> list[BenchmarkResult]:
        """Run complete benchmark suite"""

        print("🚀 Starting Fog Scheduler Benchmark Suite")
        print("=" * 60)

        # Run all benchmark tests
        benchmarks = [
            ("Single Job Placement", self.benchmark_single_job_placement()),
            ("Batch Placement", self.benchmark_batch_placement()),
            ("High Load Stress", self.benchmark_high_load_stress()),
            ("SLA Distribution", self.benchmark_sla_class_distribution()),
            ("Node Failure Resilience", self.benchmark_node_failure_resilience()),
        ]

        results = []

        for test_name, benchmark_coro in benchmarks:
            print(f"\n⏱️  Running {test_name}...")
            try:
                result = await benchmark_coro
                results.append(result)
                print(result)

                if not result.meets_sla_requirement():
                    print(f"⚠️  WARNING: {test_name} failed p95 < 500ms requirement!")

            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")

        self.results = results
        return results

    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""

        if not self.results:
            return "No benchmark results available"

        report = []
        report.append("FOG SCHEDULER BENCHMARK REPORT")
        report.append("=" * 50)
        report.append("")

        # Summary statistics
        total_jobs = sum(r.total_jobs for r in self.results)
        avg_p95_latency = statistics.mean([r.p95_latency_ms for r in self.results])
        sla_compliance_rate = sum(1 for r in self.results if r.meets_sla_requirement()) / len(self.results)

        report.append("📊 SUMMARY")
        report.append(f"  Total Jobs Tested: {total_jobs:,}")
        report.append(f"  Average P95 Latency: {avg_p95_latency:.1f}ms")
        report.append(f"  SLA Compliance Rate: {sla_compliance_rate:.1%}")
        report.append(f"  Overall Status: {'✅ PASS' if avg_p95_latency < 500 else '❌ FAIL'}")
        report.append("")

        # Detailed results
        report.append("📋 DETAILED RESULTS")
        for result in self.results:
            report.append("")
            report.append(str(result))

        # Performance recommendations
        report.append("")
        report.append("🔧 RECOMMENDATIONS")

        if avg_p95_latency > 500:
            report.append("  • P95 latency exceeds 500ms SLA requirement")
            report.append("  • Consider optimizing NSGA-II parameters")
            report.append("  • Review node selection algorithms")

        if any(r.success_rate < 0.95 for r in self.results):
            report.append("  • Success rate below 95% in some tests")
            report.append("  • Check resource availability and constraints")

        return "\n".join(report)


# Pytest integration
class TestSchedulerLatencyBenchmarks:
    """Pytest-compatible benchmark tests"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_single_job_placement_latency(self):
        """Test single job placement meets p95 < 500ms requirement"""
        benchmark = SchedulerBenchmark(node_count=5)
        result = await benchmark.benchmark_single_job_placement(job_count=50)

        print(f"\n{result}")
        assert result.meets_sla_requirement(), f"P95 latency {result.p95_latency_ms:.1f}ms exceeds 500ms requirement"
        assert result.success_rate > 0.95, f"Success rate {result.success_rate:.1%} below 95%"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_batch_placement_performance(self):
        """Test batch placement performance"""
        benchmark = SchedulerBenchmark(node_count=8)
        result = await benchmark.benchmark_batch_placement(batch_size=20, batch_count=5)

        print(f"\n{result}")
        assert result.meets_sla_requirement(), f"P95 latency {result.p95_latency_ms:.1f}ms exceeds 500ms requirement"
        assert result.throughput_jobs_per_sec > 10, f"Throughput {result.throughput_jobs_per_sec:.1f} jobs/sec too low"

    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_high_load_stress_performance(self):
        """Test scheduler under high load stress"""
        benchmark = SchedulerBenchmark(node_count=10)
        result = await benchmark.benchmark_high_load_stress(jobs_per_second=15, duration_seconds=20)

        print(f"\n{result}")
        # Allow higher latency under stress but still reasonable
        assert result.p95_latency_ms < 1000, f"P95 latency {result.p95_latency_ms:.1f}ms too high under stress"
        assert result.success_rate > 0.90, f"Success rate {result.success_rate:.1%} too low under stress"


# CLI execution
async def main():
    """Run benchmark suite from command line"""

    benchmark = SchedulerBenchmark(node_count=10)
    results = await benchmark.run_all_benchmarks()

    print("\n" + "=" * 60)
    print(benchmark.generate_benchmark_report())

    # Check overall pass/fail
    overall_pass = all(r.meets_sla_requirement() for r in results)
    print(f"\n🏁 FINAL RESULT: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")

    return overall_pass


if __name__ == "__main__":
    # Run benchmarks
    success = asyncio.run(main())
    exit(0 if success else 1)
