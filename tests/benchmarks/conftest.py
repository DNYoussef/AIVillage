"""Performance benchmark test configuration."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest


@dataclass
class BenchmarkResult:
    """Single benchmark measurement result."""

    test_name: str
    duration: float
    timestamp: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceBenchmark:
    """Track test performance over time and detect regressions."""

    def __init__(self):
        self.baselines_file = Path(__file__).parent / "baselines.json"
        self.results_file = Path(__file__).parent / "benchmark_results.json"
        self.baselines = self._load_baselines()
        self.current_results: list[BenchmarkResult] = []

        # Performance thresholds
        self.regression_threshold = 1.5  # 50% slower than baseline
        self.warning_threshold = 1.2  # 20% slower than baseline

    def _load_baselines(self) -> dict[str, float]:
        """Load performance baselines from file."""
        if not self.baselines_file.exists():
            return {}

        try:
            with open(self.baselines_file) as f:
                data = json.load(f)
            return data.get("baselines", {})
        except Exception as e:
            print(f"Warning: Could not load baselines: {e}")
            return {}

    def measure(self, test_name: str, metadata: dict[str, Any] | None = None):
        """Context manager for measuring test performance."""
        return BenchmarkContext(self, test_name, metadata or {})

    def record_result(self, result: BenchmarkResult):
        """Record a benchmark result."""
        self.current_results.append(result)

        # Check for regression
        baseline = self.baselines.get(result.test_name)
        if baseline and result.duration > baseline * self.regression_threshold:
            pytest.fail(
                f"Performance regression detected in {result.test_name}: "
                f"{result.duration:.3f}s > {baseline * self.regression_threshold:.3f}s "
                f"(baseline: {baseline:.3f}s, threshold: {self.regression_threshold}x)"
            )
        elif baseline and result.duration > baseline * self.warning_threshold:
            pytest.warns(
                UserWarning,
                match=f"Performance warning for {result.test_name}: "
                f"{result.duration:.3f}s vs baseline {baseline:.3f}s",
            )

    def save_results(self):
        """Save benchmark results to file."""
        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing results
            existing_results = []
            if self.results_file.exists():
                with open(self.results_file) as f:
                    existing_results = json.load(f)

            # Add new results
            all_results = existing_results + [r.to_dict() for r in self.current_results]

            # Keep only last 100 results per test
            results_by_test = {}
            for result in all_results:
                test_name = result["test_name"]
                if test_name not in results_by_test:
                    results_by_test[test_name] = []
                results_by_test[test_name].append(result)

            # Trim to last 100 per test
            trimmed_results = []
            for test_results in results_by_test.values():
                # Sort by timestamp and keep last 100
                sorted_results = sorted(test_results, key=lambda x: x["timestamp"])
                trimmed_results.extend(sorted_results[-100:])

            with open(self.results_file, "w") as f:
                json.dump(trimmed_results, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save benchmark results: {e}")


class BenchmarkContext:
    """Context manager for benchmark measurements."""

    def __init__(
        self, benchmark: PerformanceBenchmark, test_name: str, metadata: dict[str, Any]
    ):
        self.benchmark = benchmark
        self.test_name = test_name
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time

            from datetime import datetime

            result = BenchmarkResult(
                test_name=self.test_name,
                duration=duration,
                timestamp=datetime.now(UTC).isoformat(),
                metadata=self.metadata,
            )

            self.benchmark.record_result(result)


@pytest.fixture(scope="session")
def benchmark():
    """Provide performance benchmark fixture."""
    bench = PerformanceBenchmark()
    yield bench

    # Save results at end of session
    bench.save_results()


@pytest.fixture(scope="session")
def performance_config():
    """Performance test configuration."""
    return {
        "timeout_seconds": 30,
        "max_iterations": 1000,
        "min_duration": 0.001,
        "sample_sizes": [100, 1000, 10000],
    }


# Benchmark markers
def pytest_configure(config):
    """Configure benchmark markers."""
    config.addinivalue_line("markers", "benchmark: mark test as performance benchmark")
    config.addinivalue_line(
        "markers", "slow_benchmark: mark test as slow performance benchmark"
    )
    config.addinivalue_line(
        "markers", "memory_benchmark: mark test as memory performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Add benchmark markers based on test location."""
    for item in items:
        # Auto-mark tests in benchmarks directory
        if "benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)

        # Mark slow tests
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow_benchmark)

        # Mark memory tests
        if "memory" in item.name.lower():
            item.add_marker(pytest.mark.memory_benchmark)
