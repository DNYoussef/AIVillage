#!/usr/bin/env python3
"""
Soak Test Orchestrator for AIVillage Production Validation
=========================================================

Long-running stability testing system that validates:
- 24-hour continuous operation
- Memory leak detection
- Performance degradation over time
- Error rate stability
- Resource utilization trends
- Automatic recovery capabilities

Usage:
    python soak_test_orchestrator.py --duration 86400  # 24 hours
    python soak_test_orchestrator.py --profile production --report-interval 3600
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import tracemalloc
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import psutil
except ImportError:
    psutil = None

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("soak_test.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class SoakTestConfig:
    """Configuration for soak testing"""

    test_name: str = "aivillage_soak_test"
    duration_hours: float = 24.0
    report_interval_seconds: int = 3600  # 1 hour
    checkpoint_interval_seconds: int = 300  # 5 minutes

    # System under test
    base_url: str = "http://localhost:8000"
    health_endpoints: list[str] = field(
        default_factory=lambda: ["/health", "/v1/agents/health", "/v1/rag/health", "/v1/p2p/health", "/v1/forge/health"]
    )

    # Load parameters
    concurrent_users: int = 50
    request_rate_per_second: float = 10.0

    # Thresholds for failure detection
    max_error_rate: float = 0.02  # 2%
    max_response_time_p99: float = 3000.0  # 3 seconds
    max_memory_growth_mb_per_hour: float = 50.0
    max_cpu_sustained_percent: float = 85.0

    # Recovery testing
    enable_chaos_testing: bool = False
    chaos_interval_minutes: int = 120  # Every 2 hours

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("soak_test_results"))
    generate_plots: bool = True


@dataclass
class SoakTestMetrics:
    """Metrics collected during soak testing"""

    test_start: datetime
    current_time: datetime
    elapsed_hours: float = 0.0

    # Request metrics over time
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # Response time trends
    avg_response_time: float = 0.0
    p99_response_time: float = 0.0
    response_time_trend: list[float] = field(default_factory=list)

    # Resource trends
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_trend: list[float] = field(default_factory=list)
    cpu_trend: list[float] = field(default_factory=list)

    # Stability indicators
    memory_growth_rate_mb_per_hour: float = 0.0
    performance_degradation_percent: float = 0.0
    error_rate_stability: float = 0.0

    # Health indicators
    endpoints_healthy: int = 0
    endpoints_total: int = 0
    system_health_score: float = 1.0

    # Recovery metrics
    recovery_events: list[dict[str, Any]] = field(default_factory=list)
    downtime_seconds: float = 0.0


class MemoryLeakDetector:
    """Detect memory leaks during soak testing"""

    def __init__(self, window_size: int = 12):  # 12 hours by default
        self.memory_samples = deque(maxlen=window_size)
        self.leak_threshold_mb_per_hour = 10.0

    def add_sample(self, memory_mb: float, timestamp: datetime):
        """Add memory sample"""
        self.memory_samples.append((timestamp, memory_mb))

    def detect_leak(self) -> tuple[bool, float]:
        """Detect if there's a memory leak"""
        if len(self.memory_samples) < 6:  # Need at least 6 hours of data
            return False, 0.0

        # Calculate linear regression to detect trend
        times = [(sample[0] - self.memory_samples[0][0]).total_seconds() / 3600 for sample in self.memory_samples]
        memories = [sample[1] for sample in self.memory_samples]

        # Simple linear regression
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(memories)
        sum_xy = sum(x * y for x, y in zip(times, memories))
        sum_x2 = sum(x * x for x in times)

        if n * sum_x2 - sum_x * sum_x == 0:
            return False, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # slope is MB per hour
        leak_detected = slope > self.leak_threshold_mb_per_hour
        return leak_detected, slope


class PerformanceDegradationDetector:
    """Detect performance degradation over time"""

    def __init__(self, baseline_window: int = 6):
        self.response_times = deque(maxlen=100)  # Last 100 samples
        self.baseline_window = baseline_window
        self.baseline_p99 = None

    def add_sample(self, response_time_p99: float):
        """Add response time sample"""
        self.response_times.append(response_time_p99)

        # Set baseline after initial period
        if len(self.response_times) == self.baseline_window and self.baseline_p99 is None:
            self.baseline_p99 = sum(list(self.response_times)[: self.baseline_window]) / self.baseline_window
            logger.info(f"Performance baseline set: {self.baseline_p99:.1f}ms P99")

    def get_degradation_percent(self) -> float:
        """Calculate performance degradation percentage"""
        if self.baseline_p99 is None or len(self.response_times) < 10:
            return 0.0

        recent_avg = sum(list(self.response_times)[-10:]) / 10
        degradation = ((recent_avg - self.baseline_p99) / self.baseline_p99) * 100
        return max(0.0, degradation)  # Only positive degradation


class ChaosTestingEngine:
    """Inject chaos to test system resilience"""

    def __init__(self, config: SoakTestConfig):
        self.config = config
        self.chaos_events = []

    async def inject_chaos(self, event_type: str = "random"):
        """Inject chaos event"""
        chaos_types = ["network_delay", "high_cpu_load", "memory_pressure", "disk_io_stress"]

        if event_type == "random":
            event_type = __import__("random").choice(chaos_types)

        logger.warning(f"Injecting chaos event: {event_type}")

        start_time = datetime.now()
        recovery_time = None

        try:
            if event_type == "network_delay":
                await self._simulate_network_delay()
            elif event_type == "high_cpu_load":
                await self._simulate_cpu_load()
            elif event_type == "memory_pressure":
                await self._simulate_memory_pressure()
            elif event_type == "disk_io_stress":
                await self._simulate_disk_stress()

            recovery_time = datetime.now()

        except Exception as e:
            logger.error(f"Chaos injection failed: {e}")

        chaos_event = {
            "type": event_type,
            "start_time": start_time.isoformat(),
            "recovery_time": recovery_time.isoformat() if recovery_time else None,
            "duration_seconds": (recovery_time - start_time).total_seconds() if recovery_time else None,
        }

        self.chaos_events.append(chaos_event)
        return chaos_event

    async def _simulate_network_delay(self):
        """Simulate network latency"""
        # Add artificial delay to requests
        await asyncio.sleep(2.0)  # Simulate 2-second network issue

    async def _simulate_cpu_load(self):
        """Simulate high CPU load"""

        def cpu_stress():
            end_time = time.time() + 30  # 30 seconds of stress
            while time.time() < end_time:
                # CPU-intensive calculation
                sum(i * i for i in range(10000))

        # Run CPU stress in background
        import threading

        thread = threading.Thread(target=cpu_stress)
        thread.start()
        await asyncio.sleep(35)  # Wait for stress to complete

    async def _simulate_memory_pressure(self):
        """Simulate memory pressure"""
        # Allocate and hold memory for a short period
        memory_hog = []
        try:
            for _ in range(100):
                # Allocate 1MB chunks
                chunk = bytearray(1024 * 1024)
                memory_hog.append(chunk)
                await asyncio.sleep(0.1)

            await asyncio.sleep(10)  # Hold memory for 10 seconds

        finally:
            # Release memory
            memory_hog.clear()

    async def _simulate_disk_stress(self):
        """Simulate disk I/O stress"""
        temp_file = Path("/tmp/chaos_test_file.tmp")
        try:
            # Write and read large file
            data = b"x" * (10 * 1024 * 1024)  # 10MB
            with open(temp_file, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Read it back multiple times
            for _ in range(5):
                with open(temp_file, "rb") as f:
                    f.read()

        finally:
            # Cleanup
            temp_file.unlink(missing_ok=True)


class SoakTestOrchestrator:
    """Main soak test orchestrator"""

    def __init__(self, config: SoakTestConfig):
        self.config = config
        self.metrics = SoakTestMetrics(test_start=datetime.now(), current_time=datetime.now())

        # Detectors
        self.memory_leak_detector = MemoryLeakDetector()
        self.performance_detector = PerformanceDegradationDetector()
        self.chaos_engine = ChaosTestingEngine(config) if config.enable_chaos_testing else None

        # State management
        self.running = False
        self.checkpoints = []
        self.last_chaos_time = datetime.now()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Enable memory tracking
        tracemalloc.start()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    async def run_soak_test(self) -> SoakTestMetrics:
        """Run the complete soak test"""
        logger.info(f"Starting soak test: {self.config.test_name}")
        logger.info(f"Duration: {self.config.duration_hours} hours")
        logger.info(f"Target load: {self.config.concurrent_users} users, {self.config.request_rate_per_second} RPS")

        self.running = True
        end_time = datetime.now() + timedelta(hours=self.config.duration_hours)

        # Start background tasks
        monitor_task = asyncio.create_task(self._monitor_system())
        load_task = asyncio.create_task(self._generate_load())
        checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        report_task = asyncio.create_task(self._report_loop())

        chaos_task = None
        if self.chaos_engine:
            chaos_task = asyncio.create_task(self._chaos_loop())

        try:
            # Main test loop
            while self.running and datetime.now() < end_time:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Update current metrics
                self._update_current_metrics()

                # Check for critical failures
                if self._check_critical_failures():
                    logger.error("Critical failure detected, stopping test")
                    break

        except Exception as e:
            logger.error(f"Soak test failed: {e}")

        finally:
            # Stop all tasks
            self.running = False

            tasks = [monitor_task, load_task, checkpoint_task, report_task]
            if chaos_task:
                tasks.append(chaos_task)

            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

            # Generate final report
            await self._generate_final_report()

        return self.metrics

    async def _monitor_system(self):
        """Monitor system resources"""
        while self.running:
            try:
                if psutil:
                    # System metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    memory_mb = (memory.total - memory.available) / 1024 / 1024

                    # Update metrics
                    self.metrics.cpu_usage_percent = cpu_percent
                    self.metrics.memory_usage_mb = memory_mb

                    # Add to trends
                    self.metrics.cpu_trend.append(cpu_percent)
                    self.metrics.memory_trend.append(memory_mb)

                    # Keep only recent data (last 24 hours at 1-minute intervals)
                    max_samples = 24 * 60
                    if len(self.metrics.cpu_trend) > max_samples:
                        self.metrics.cpu_trend = self.metrics.cpu_trend[-max_samples:]
                    if len(self.metrics.memory_trend) > max_samples:
                        self.metrics.memory_trend = self.metrics.memory_trend[-max_samples:]

                    # Memory leak detection
                    self.memory_leak_detector.add_sample(memory_mb, datetime.now())

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)

    async def _generate_load(self):
        """Generate sustained load"""
        try:
            from .production_load_test_suite import LoadTestClient, LoadTestConfig, WorkloadGenerator
        except ImportError:
            from production_load_test_suite import LoadTestClient, LoadTestConfig, WorkloadGenerator

        # Create load test config for sustained load
        load_config = LoadTestConfig(
            max_concurrent_users=self.config.concurrent_users,
            target_rps=self.config.request_rate_per_second,
            base_url=self.config.base_url,
        )

        workload_gen = WorkloadGenerator(load_config)

        async with LoadTestClient(self.config.base_url) as client:
            while self.running:
                try:
                    # Generate requests
                    tasks = []
                    for _ in range(self.config.concurrent_users):
                        workload = workload_gen.select_workload()
                        task = asyncio.create_task(client.make_request(**workload))
                        tasks.append(task)

                    # Wait for all requests to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Update metrics
                    for result in results:
                        if isinstance(result, dict):
                            self.metrics.total_requests += 1
                            if result.get("success", False):
                                self.metrics.successful_requests += 1
                            else:
                                self.metrics.failed_requests += 1

                            # Track response times
                            response_time = result.get("response_time", 0)
                            self.metrics.response_time_trend.append(response_time)

                    # Rate limiting
                    request_interval = self.config.concurrent_users / self.config.request_rate_per_second
                    await asyncio.sleep(max(0.1, request_interval))

                except Exception as e:
                    logger.error(f"Load generation error: {e}")
                    await asyncio.sleep(5)

    async def _chaos_loop(self):
        """Periodic chaos injection"""
        while self.running:
            try:
                # Wait for chaos interval
                await asyncio.sleep(self.config.chaos_interval_minutes * 60)

                if self.running:
                    event = await self.chaos_engine.inject_chaos()
                    self.metrics.recovery_events.append(event)
                    logger.info(f"Chaos event completed: {event['type']}")

            except Exception as e:
                logger.error(f"Chaos testing error: {e}")

    async def _checkpoint_loop(self):
        """Periodic checkpointing"""
        while self.running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_seconds)

                if self.running:
                    checkpoint = self._create_checkpoint()
                    self.checkpoints.append(checkpoint)

                    # Save checkpoint to file
                    checkpoint_file = self.config.output_dir / f"checkpoint_{len(self.checkpoints)}.json"
                    with open(checkpoint_file, "w") as f:
                        json.dump(checkpoint, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Checkpointing error: {e}")

    async def _report_loop(self):
        """Periodic reporting"""
        while self.running:
            try:
                await asyncio.sleep(self.config.report_interval_seconds)

                if self.running:
                    await self._generate_interim_report()

            except Exception as e:
                logger.error(f"Reporting error: {e}")

    def _update_current_metrics(self):
        """Update current metrics calculations"""
        self.metrics.current_time = datetime.now()
        self.metrics.elapsed_hours = (self.metrics.current_time - self.metrics.test_start).total_seconds() / 3600

        # Calculate error rate
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests

        # Calculate response time metrics
        if self.metrics.response_time_trend:
            recent_times = self.metrics.response_time_trend[-100:]  # Last 100 requests
            self.metrics.avg_response_time = sum(recent_times) / len(recent_times)
            recent_times.sort()
            p99_index = int(len(recent_times) * 0.99)
            self.metrics.p99_response_time = recent_times[p99_index] if recent_times else 0

        # Performance degradation detection
        self.performance_detector.add_sample(self.metrics.p99_response_time)
        self.metrics.performance_degradation_percent = self.performance_detector.get_degradation_percent()

        # Memory growth rate
        if len(self.metrics.memory_trend) > 1:
            leak_detected, growth_rate = self.memory_leak_detector.detect_leak()
            self.metrics.memory_growth_rate_mb_per_hour = growth_rate

    def _check_critical_failures(self) -> bool:
        """Check for critical failures that should stop the test"""
        failures = []

        # High error rate
        if self.metrics.error_rate > self.config.max_error_rate:
            failures.append(f"Error rate {self.metrics.error_rate:.3f} exceeds threshold {self.config.max_error_rate}")

        # Performance degradation
        if self.metrics.performance_degradation_percent > 50:  # 50% degradation
            failures.append(f"Performance degraded by {self.metrics.performance_degradation_percent:.1f}%")

        # Memory leak
        if self.metrics.memory_growth_rate_mb_per_hour > self.config.max_memory_growth_mb_per_hour:
            failures.append(f"Memory leak detected: {self.metrics.memory_growth_rate_mb_per_hour:.1f} MB/hour")

        if failures:
            for failure in failures:
                logger.error(f"Critical failure: {failure}")
            return True

        return False

    def _create_checkpoint(self) -> dict[str, Any]:
        """Create test checkpoint"""
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_hours": self.metrics.elapsed_hours,
            "total_requests": self.metrics.total_requests,
            "error_rate": self.metrics.error_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "cpu_usage_percent": self.metrics.cpu_usage_percent,
            "memory_growth_rate": self.metrics.memory_growth_rate_mb_per_hour,
            "performance_degradation": self.metrics.performance_degradation_percent,
        }

    async def _generate_interim_report(self):
        """Generate interim progress report"""
        logger.info("=== SOAK TEST INTERIM REPORT ===")
        logger.info(f"Elapsed: {self.metrics.elapsed_hours:.1f} hours")
        logger.info(f"Requests: {self.metrics.total_requests} total, {self.metrics.error_rate:.3f} error rate")
        logger.info(
            f"Performance: {self.metrics.avg_response_time:.1f}ms avg, {self.metrics.p99_response_time:.1f}ms P99"
        )
        logger.info(
            f"Resources: {self.metrics.memory_usage_mb:.1f}MB memory, {self.metrics.cpu_usage_percent:.1f}% CPU"
        )
        logger.info(f"Trends: {self.metrics.memory_growth_rate_mb_per_hour:.1f} MB/hour memory growth")
        logger.info(f"Degradation: {self.metrics.performance_degradation_percent:.1f}%")

    async def _generate_final_report(self):
        """Generate final comprehensive report"""
        logger.info("Generating final soak test report...")

        # Final metrics calculation
        self._update_current_metrics()

        # Create comprehensive report
        report = {
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "checkpoints": self.checkpoints,
            "test_summary": {
                "duration_hours": self.metrics.elapsed_hours,
                "total_requests": self.metrics.total_requests,
                "final_error_rate": self.metrics.error_rate,
                "memory_leak_detected": self.metrics.memory_growth_rate_mb_per_hour
                > self.config.max_memory_growth_mb_per_hour,
                "performance_degraded": self.metrics.performance_degradation_percent > 10,
                "test_passed": self._evaluate_test_success(),
            },
        }

        # Save main report
        report_file = self.config.output_dir / "soak_test_final_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate plots if enabled
        if self.config.generate_plots and PLOTTING_AVAILABLE:
            self._generate_plots()

        logger.info(f"Final report saved to {report_file}")

    def _evaluate_test_success(self) -> bool:
        """Evaluate overall test success"""
        if self.metrics.error_rate > self.config.max_error_rate:
            return False
        if self.metrics.memory_growth_rate_mb_per_hour > self.config.max_memory_growth_mb_per_hour:
            return False
        if self.metrics.performance_degradation_percent > 25:  # 25% degradation threshold
            return False
        return True

    def _generate_plots(self):
        """Generate visualization plots"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Memory usage over time
            if self.metrics.memory_trend:
                hours = [i / 60 for i in range(len(self.metrics.memory_trend))]  # Convert minutes to hours
                ax1.plot(hours, self.metrics.memory_trend)
                ax1.set_title("Memory Usage Over Time")
                ax1.set_xlabel("Hours")
                ax1.set_ylabel("Memory (MB)")
                ax1.grid(True)

            # CPU usage over time
            if self.metrics.cpu_trend:
                hours = [i / 60 for i in range(len(self.metrics.cpu_trend))]
                ax2.plot(hours, self.metrics.cpu_trend)
                ax2.set_title("CPU Usage Over Time")
                ax2.set_xlabel("Hours")
                ax2.set_ylabel("CPU (%)")
                ax2.grid(True)

            # Response time trend
            if self.metrics.response_time_trend:
                # Sample every 100th point to avoid overcrowding
                sampled_times = self.metrics.response_time_trend[::100]
                hours = [i * 100 / 3600 for i in range(len(sampled_times))]  # Approximate hours
                ax3.plot(hours, sampled_times)
                ax3.set_title("Response Time Trend")
                ax3.set_xlabel("Hours")
                ax3.set_ylabel("Response Time (ms)")
                ax3.grid(True)

            # Error rate over time (calculate moving average)
            if len(self.checkpoints) > 1:
                hours = [cp["elapsed_hours"] for cp in self.checkpoints]
                error_rates = [cp["error_rate"] for cp in self.checkpoints]
                ax4.plot(hours, error_rates)
                ax4.set_title("Error Rate Over Time")
                ax4.set_xlabel("Hours")
                ax4.set_ylabel("Error Rate")
                ax4.grid(True)

            plt.tight_layout()
            plot_file = self.config.output_dir / "soak_test_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Plots saved to {plot_file}")

        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIVillage Soak Test Orchestrator")
    parser.add_argument("--duration", type=float, default=24.0, help="Test duration in hours")
    parser.add_argument("--concurrent-users", type=int, default=50, help="Concurrent users")
    parser.add_argument("--request-rate", type=float, default=10.0, help="Requests per second")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--output-dir", type=Path, default="soak_test_results", help="Output directory")
    parser.add_argument("--enable-chaos", action="store_true", help="Enable chaos testing")
    parser.add_argument("--report-interval", type=int, default=3600, help="Report interval in seconds")
    parser.add_argument("--profile", choices=["development", "staging", "production"], default="development")

    args = parser.parse_args()

    # Create configuration
    config = SoakTestConfig(
        duration_hours=args.duration,
        concurrent_users=args.concurrent_users,
        request_rate_per_second=args.request_rate,
        base_url=args.base_url,
        output_dir=args.output_dir,
        enable_chaos_testing=args.enable_chaos,
        report_interval_seconds=args.report_interval,
    )

    # Adjust thresholds based on profile
    if args.profile == "production":
        config.max_error_rate = 0.001  # 0.1% for production
        config.max_response_time_p99 = 1000.0  # 1 second for production
    elif args.profile == "staging":
        config.max_error_rate = 0.01  # 1% for staging
        config.max_response_time_p99 = 2000.0  # 2 seconds for staging

    logger.info(f"Starting soak test with {args.profile} profile")
    logger.info(f"Duration: {config.duration_hours} hours")
    logger.info(f"Load: {config.concurrent_users} users, {config.request_rate_per_second} RPS")

    # Run soak test
    orchestrator = SoakTestOrchestrator(config)
    metrics = await orchestrator.run_soak_test()

    # Print final summary
    print("\n" + "=" * 70)
    print("SOAK TEST FINAL SUMMARY")
    print("=" * 70)
    print(f"Duration: {metrics.elapsed_hours:.1f} hours")
    print(f"Total Requests: {metrics.total_requests}")
    print(f"Error Rate: {metrics.error_rate:.4f} ({metrics.error_rate * 100:.2f}%)")
    print(f"Memory Growth: {metrics.memory_growth_rate_mb_per_hour:.1f} MB/hour")
    print(f"Performance Degradation: {metrics.performance_degradation_percent:.1f}%")
    print(f"Recovery Events: {len(metrics.recovery_events)}")

    test_passed = orchestrator._evaluate_test_success()
    if test_passed:
        print("\n✅ SOAK TEST PASSED")
        return 0
    else:
        print("\n❌ SOAK TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
