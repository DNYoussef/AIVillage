#!/usr/bin/env python3
"""
Production Load Testing Suite for AIVillage
============================================

Comprehensive load testing infrastructure for production validation including:
- System-wide load testing across all components
- Soak testing for long-term stability
- Performance regression detection
- Resource utilization monitoring
- Failure rate validation
- Scalability testing

Usage:
    python production_load_test_suite.py --profile basic --duration 300
    python production_load_test_suite.py --profile soak --duration 3600
    python production_load_test_suite.py --profile scale --max-users 1000
"""

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import sys
import threading
import time
import tracemalloc
from typing import Any

# Core imports with fallbacks
try:
    import psutil
except ImportError:
    psutil = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("load_test.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing"""

    test_profile: str = "basic"
    duration_seconds: int = 300
    max_concurrent_users: int = 100
    ramp_up_seconds: int = 60
    ramp_down_seconds: int = 30
    target_rps: float = 10.0
    error_threshold: float = 0.01  # 1% error rate
    response_time_p99_threshold: float = 2000.0  # 2 seconds
    memory_threshold_mb: float = 1024.0  # 1GB
    cpu_threshold_percent: float = 80.0

    # Test endpoints
    base_url: str = "http://localhost:8000"
    endpoints: list[str] = field(
        default_factory=lambda: [
            "/health",
            "/v1/agents/health",
            "/v1/rag/query",
            "/v1/p2p/status",
            "/v1/compression/status",
            "/v1/forge/status",
        ]
    )

    # Workload distribution
    workload_simple: float = 0.6
    workload_complex: float = 0.3
    workload_stress: float = 0.1


@dataclass
class TestMetrics:
    """Metrics collected during load testing"""

    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # Response time metrics
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0

    # Throughput metrics
    requests_per_second: float = 0.0
    peak_rps: float = 0.0

    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0

    # System metrics
    connection_errors: int = 0
    timeout_errors: int = 0
    server_errors: int = 0

    # Test status
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)


class SystemResourceMonitor:
    """Monitor system resources during load testing"""

    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self._monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring"""
        if not psutil:
            logger.warning("psutil not available, skipping resource monitoring")
            return

        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Started system resource monitoring")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped system resource monitoring")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU and memory metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                network = psutil.net_io_counters()

                metric = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / 1024 / 1024,
                    "disk_percent": disk.percent,
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv,
                }

                self.metrics.append(metric)

                # Keep only last 1000 metrics to prevent memory issues
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            time.sleep(1)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of resource usage"""
        if not self.metrics:
            return {}

        cpu_values = [m["cpu_percent"] for m in self.metrics]
        memory_values = [m["memory_percent"] for m in self.metrics]

        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_avg": sum(memory_values) / len(memory_values),
            "memory_max": max(memory_values),
            "samples": len(self.metrics),
        }


class LoadTestClient:
    """HTTP client for load testing"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = None

    async def __aenter__(self):
        if aiohttp:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, endpoint: str, method: str = "GET", **kwargs) -> dict[str, Any]:
        """Make HTTP request and return metrics"""
        start_time = time.time()
        url = f"{self.base_url}{endpoint}"

        result = {
            "endpoint": endpoint,
            "method": method,
            "start_time": start_time,
            "status_code": 0,
            "response_time": 0.0,
            "success": False,
            "error": None,
        }

        try:
            if self.session:
                # Use aiohttp if available
                async with self.session.request(method, url, **kwargs) as response:
                    result["status_code"] = response.status
                    result["response_time"] = (time.time() - start_time) * 1000
                    result["success"] = 200 <= response.status < 400

                    if not result["success"]:
                        result["error"] = f"HTTP {response.status}"
            else:
                # Fallback to requests-like behavior
                import urllib.request

                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as response:
                    result["status_code"] = response.getcode()
                    result["response_time"] = (time.time() - start_time) * 1000
                    result["success"] = 200 <= response.getcode() < 400

        except Exception as e:
            result["response_time"] = (time.time() - start_time) * 1000
            result["error"] = str(e)
            result["success"] = False

        return result


class WorkloadGenerator:
    """Generate different types of workloads"""

    def __init__(self, config: LoadTestConfig):
        self.config = config

    def get_simple_workload(self) -> dict[str, Any]:
        """Simple health check workload"""
        endpoint = random.choice(["/health", "/v1/agents/health"])
        return {"endpoint": endpoint, "method": "GET", "weight": self.config.workload_simple}

    def get_complex_workload(self) -> dict[str, Any]:
        """Complex query workload"""
        queries = [
            "What is machine learning?",
            "Explain quantum computing",
            "How does blockchain work?",
            "What are the benefits of AI?",
            "Describe neural networks",
        ]

        return {
            "endpoint": "/v1/rag/query",
            "method": "POST",
            "json": {"query": random.choice(queries), "mode": "balanced", "max_results": 5},
            "weight": self.config.workload_complex,
        }

    def get_stress_workload(self) -> dict[str, Any]:
        """Stress test workload"""
        return {
            "endpoint": "/v1/forge/status",
            "method": "GET",
            "params": {"detailed": "true", "include_metrics": "true"},
            "weight": self.config.workload_stress,
        }

    def select_workload(self) -> dict[str, Any]:
        """Select workload based on distribution"""
        rand = random.random()

        if rand < self.config.workload_simple:
            return self.get_simple_workload()
        elif rand < self.config.workload_simple + self.config.workload_complex:
            return self.get_complex_workload()
        else:
            return self.get_stress_workload()


class ProductionLoadTestSuite:
    """Main load testing suite"""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = TestMetrics(start_time=datetime.now())
        self.resource_monitor = SystemResourceMonitor()
        self.workload_generator = WorkloadGenerator(config)
        self.response_times = []
        self.request_results = []

        # Enable memory tracking
        tracemalloc.start()

    async def run_load_test(self) -> TestMetrics:
        """Run complete load test"""
        logger.info(f"Starting {self.config.test_profile} load test")
        logger.info(f"Duration: {self.config.duration_seconds}s, Max users: {self.config.max_concurrent_users}")

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        try:
            # Run the actual load test
            await self._execute_load_test()

            # Calculate final metrics
            self._calculate_final_metrics()

            # Validate results
            self._validate_results()

        except Exception as e:
            logger.error(f"Load test failed: {e}")
            self.metrics.failure_reasons.append(f"Test execution failed: {str(e)}")

        finally:
            # Stop monitoring
            self.resource_monitor.stop_monitoring()
            self.metrics.end_time = datetime.now()

        return self.metrics

    async def _execute_load_test(self):
        """Execute the main load test"""
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        # Calculate user ramp schedule
        ramp_up_end = start_time + self.config.ramp_up_seconds
        ramp_down_start = end_time - self.config.ramp_down_seconds

        # Track active tasks
        active_tasks = set()

        async with LoadTestClient(self.config.base_url) as client:
            current_time = start_time

            while current_time < end_time:
                # Calculate current user count based on ramp schedule
                if current_time < ramp_up_end:
                    # Ramp up phase
                    progress = (current_time - start_time) / self.config.ramp_up_seconds
                    current_users = int(self.config.max_concurrent_users * progress)
                elif current_time > ramp_down_start:
                    # Ramp down phase
                    progress = (end_time - current_time) / self.config.ramp_down_seconds
                    current_users = int(self.config.max_concurrent_users * progress)
                else:
                    # Steady state phase
                    current_users = self.config.max_concurrent_users

                # Adjust active tasks to match target user count
                while len(active_tasks) < current_users:
                    workload = self.workload_generator.select_workload()
                    task = asyncio.create_task(self._execute_request(client, workload))
                    active_tasks.add(task)

                # Remove completed tasks
                completed_tasks = [task for task in active_tasks if task.done()]
                for task in completed_tasks:
                    active_tasks.remove(task)
                    try:
                        result = await task
                        self.request_results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")

                # Sleep briefly to control request rate
                await asyncio.sleep(0.1)
                current_time = time.time()

            # Wait for remaining tasks to complete
            if active_tasks:
                logger.info(f"Waiting for {len(active_tasks)} remaining tasks...")
                await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _execute_request(self, client: LoadTestClient, workload: dict[str, Any]):
        """Execute a single request"""
        try:
            # Add some jitter to prevent thundering herd
            await asyncio.sleep(random.uniform(0, 0.5))

            result = await client.make_request(**workload)

            # Track metrics
            self.metrics.total_requests += 1
            if result["success"]:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1

                # Categorize errors
                if "timeout" in str(result.get("error", "")).lower():
                    self.metrics.timeout_errors += 1
                elif "connection" in str(result.get("error", "")).lower():
                    self.metrics.connection_errors += 1
                elif result.get("status_code", 0) >= 500:
                    self.metrics.server_errors += 1

            # Track response times
            response_time = result["response_time"]
            self.response_times.append(response_time)

            # Update min/max response times
            self.metrics.min_response_time = min(self.metrics.min_response_time, response_time)
            self.metrics.max_response_time = max(self.metrics.max_response_time, response_time)

            return result

        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            self.metrics.failed_requests += 1
            return {"success": False, "error": str(e)}

    def _calculate_final_metrics(self):
        """Calculate final test metrics"""
        if not self.response_times:
            logger.warning("No response times recorded")
            return

        # Duration
        if self.metrics.end_time:
            self.metrics.duration_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        # Error rate
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests

        # Response time percentiles
        self.response_times.sort()
        count = len(self.response_times)

        if count > 0:
            self.metrics.avg_response_time = sum(self.response_times) / count
            self.metrics.p50_response_time = self.response_times[int(count * 0.5)]
            self.metrics.p95_response_time = self.response_times[int(count * 0.95)]
            self.metrics.p99_response_time = self.response_times[int(count * 0.99)]

        # Throughput
        if self.metrics.duration_seconds > 0:
            self.metrics.requests_per_second = self.metrics.total_requests / self.metrics.duration_seconds

        # Resource metrics
        resource_summary = self.resource_monitor.get_summary()
        if resource_summary:
            self.metrics.avg_cpu_percent = resource_summary.get("cpu_avg", 0)
            self.metrics.peak_cpu_percent = resource_summary.get("cpu_max", 0)

        # Memory metrics
        current, peak = tracemalloc.get_traced_memory()
        self.metrics.peak_memory_mb = peak / 1024 / 1024

        logger.info("Final metrics calculated")

    def _validate_results(self):
        """Validate test results against thresholds"""
        self.metrics.passed = True

        # Check error rate
        if self.metrics.error_rate > self.config.error_threshold:
            self.metrics.passed = False
            self.metrics.failure_reasons.append(
                f"Error rate {self.metrics.error_rate:.3f} exceeds threshold {self.config.error_threshold}"
            )

        # Check P99 response time
        if self.metrics.p99_response_time > self.config.response_time_p99_threshold:
            self.metrics.passed = False
            self.metrics.failure_reasons.append(
                f"P99 response time {self.metrics.p99_response_time:.1f}ms exceeds threshold {self.config.response_time_p99_threshold}ms"
            )

        # Check memory usage
        if self.metrics.peak_memory_mb > self.config.memory_threshold_mb:
            self.metrics.passed = False
            self.metrics.failure_reasons.append(
                f"Peak memory {self.metrics.peak_memory_mb:.1f}MB exceeds threshold {self.config.memory_threshold_mb}MB"
            )

        # Check CPU usage
        if self.metrics.peak_cpu_percent > self.config.cpu_threshold_percent:
            self.metrics.passed = False
            self.metrics.failure_reasons.append(
                f"Peak CPU {self.metrics.peak_cpu_percent:.1f}% exceeds threshold {self.config.cpu_threshold_percent}%"
            )

        # Check minimum throughput
        min_rps = self.config.target_rps * 0.8  # Allow 20% deviation
        if self.metrics.requests_per_second < min_rps:
            self.metrics.passed = False
            self.metrics.failure_reasons.append(
                f"RPS {self.metrics.requests_per_second:.1f} below minimum {min_rps:.1f}"
            )

        logger.info(f"Test validation: {'PASSED' if self.metrics.passed else 'FAILED'}")
        if not self.metrics.passed:
            for reason in self.metrics.failure_reasons:
                logger.error(f"  - {reason}")


def create_test_profiles() -> dict[str, LoadTestConfig]:
    """Create predefined test profiles"""
    return {
        "basic": LoadTestConfig(
            test_profile="basic",
            duration_seconds=300,  # 5 minutes
            max_concurrent_users=50,
            target_rps=10.0,
            error_threshold=0.01,
            response_time_p99_threshold=2000.0,
        ),
        "soak": LoadTestConfig(
            test_profile="soak",
            duration_seconds=3600,  # 1 hour
            max_concurrent_users=100,
            ramp_up_seconds=300,
            ramp_down_seconds=300,
            target_rps=15.0,
            error_threshold=0.005,
            response_time_p99_threshold=1500.0,
        ),
        "stress": LoadTestConfig(
            test_profile="stress",
            duration_seconds=600,  # 10 minutes
            max_concurrent_users=200,
            ramp_up_seconds=120,
            target_rps=25.0,
            error_threshold=0.02,
            response_time_p99_threshold=3000.0,
            workload_stress=0.3,
        ),
        "scale": LoadTestConfig(
            test_profile="scale",
            duration_seconds=1200,  # 20 minutes
            max_concurrent_users=500,
            ramp_up_seconds=600,
            ramp_down_seconds=300,
            target_rps=50.0,
            error_threshold=0.01,
            response_time_p99_threshold=2500.0,
        ),
        "quick": LoadTestConfig(
            test_profile="quick",
            duration_seconds=60,  # 1 minute
            max_concurrent_users=20,
            ramp_up_seconds=10,
            target_rps=5.0,
            error_threshold=0.05,
            response_time_p99_threshold=5000.0,
        ),
    }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AIVillage Production Load Testing Suite")
    parser.add_argument(
        "--profile", choices=["basic", "soak", "stress", "scale", "quick"], default="basic", help="Test profile to use"
    )
    parser.add_argument("--duration", type=int, help="Test duration in seconds (overrides profile)")
    parser.add_argument("--max-users", type=int, help="Maximum concurrent users (overrides profile)")
    parser.add_argument("--target-rps", type=float, help="Target requests per second (overrides profile)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--output", type=Path, default="load_test_results.json", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get test configuration
    profiles = create_test_profiles()
    config = profiles[args.profile]

    # Apply CLI overrides
    if args.duration:
        config.duration_seconds = args.duration
    if args.max_users:
        config.max_concurrent_users = args.max_users
    if args.target_rps:
        config.target_rps = args.target_rps
    if args.base_url:
        config.base_url = args.base_url

    logger.info(f"Running {config.test_profile} load test profile")
    logger.info(f"Target: {config.max_concurrent_users} users, {config.target_rps} RPS for {config.duration_seconds}s")

    # Run the load test
    test_suite = ProductionLoadTestSuite(config)
    metrics = await test_suite.run_load_test()

    # Generate report
    report = {
        "config": asdict(config),
        "metrics": asdict(metrics),
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
    }

    # Add resource monitoring data
    if test_suite.resource_monitor.metrics:
        report["resource_monitoring"] = test_suite.resource_monitor.metrics

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print(f"LOAD TEST RESULTS - {config.test_profile.upper()} PROFILE")
    print("=" * 70)
    print(f"Duration: {metrics.duration_seconds:.1f}s")
    print(
        f"Requests: {metrics.total_requests} total, {metrics.successful_requests} success, {metrics.failed_requests} failed"
    )
    print(f"Error Rate: {metrics.error_rate:.3f} ({metrics.error_rate * 100:.1f}%)")
    print(f"Throughput: {metrics.requests_per_second:.1f} RPS")
    print(
        f"Response Times: min={metrics.min_response_time:.1f}ms, avg={metrics.avg_response_time:.1f}ms, p99={metrics.p99_response_time:.1f}ms"
    )
    print(f"Memory: Peak {metrics.peak_memory_mb:.1f}MB")
    print(f"CPU: Avg {metrics.avg_cpu_percent:.1f}%, Peak {metrics.peak_cpu_percent:.1f}%")
    print("\n" + "=" * 70)

    if metrics.passed:
        print("✅ LOAD TEST PASSED")
        return 0
    else:
        print("❌ LOAD TEST FAILED")
        print("\nFailure Reasons:")
        for reason in metrics.failure_reasons:
            print(f"  - {reason}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
