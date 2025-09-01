#!/usr/bin/env python3
"""
Baseline Performance Measurement Suite

Establishes performance baselines for AIVillage components to track
performance improvements/degradations over time. Provides:

- Consistent measurement methodology
- Historical performance tracking
- Regression detection
- Performance comparison reporting
- Automated baseline validation

This replaces unreliable mock measurements with real system baselines.
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.benchmarks.performance_benchmarker import PerformanceBenchmarker, PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Standard baseline metrics for system components."""

    component: str
    timestamp: str
    version: str

    # Throughput baselines
    baseline_throughput_per_second: float
    min_acceptable_throughput: float
    target_throughput: float

    # Latency baselines
    baseline_latency_avg_ms: float
    baseline_latency_p95_ms: float
    max_acceptable_latency_ms: float

    # Reliability baselines
    baseline_success_rate: float
    min_acceptable_success_rate: float

    # Resource baselines
    baseline_cpu_percent: float
    baseline_memory_mb: float
    max_acceptable_cpu: float
    max_acceptable_memory_mb: float

    # Scalability baselines
    max_concurrent_operations: int
    breaking_point_load: int | None = None

    # Test configuration
    test_parameters: dict[str, Any] = None

    def __post_init__(self):
        if self.test_parameters is None:
            self.test_parameters = {}


class BaselinePerformanceSuite:
    """Comprehensive baseline performance measurement and validation."""

    def __init__(self, baseline_file: str = "tools/benchmarks/performance_baselines.json"):
        self.baseline_file = baseline_file
        self.current_baselines = {}
        self.historical_baselines = []
        self.load_existing_baselines()

    def load_existing_baselines(self):
        """Load existing performance baselines from file."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file) as f:
                    data = json.load(f)

                self.current_baselines = data.get("current_baselines", {})
                self.historical_baselines = data.get("historical_baselines", [])

                logger.info(f"Loaded {len(self.current_baselines)} current baselines")
                logger.info(f"Loaded {len(self.historical_baselines)} historical baselines")

            except Exception as e:
                logger.warning(f"Failed to load existing baselines: {e}")

    def save_baselines(self):
        """Save baselines to file."""
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)

            data = {
                "current_baselines": self.current_baselines,
                "historical_baselines": self.historical_baselines,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.baseline_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Baselines saved to {self.baseline_file}")

        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    async def establish_component_baseline(self, component: str) -> BaselineMetrics:
        """Establish performance baseline for a specific component."""
        logger.info(f"Establishing baseline for {component}")

        # Run comprehensive performance tests
        benchmarker = PerformanceBenchmarker()

        if component == "P2P_Network":
            metrics_list = await self._baseline_p2p_network(benchmarker)
        elif component == "Agent_Forge":
            metrics_list = await self._baseline_agent_forge(benchmarker)
        elif component == "Gateway":
            metrics_list = await self._baseline_gateway(benchmarker)
        elif component == "Digital_Twin":
            metrics_list = await self._baseline_digital_twin(benchmarker)
        else:
            raise ValueError(f"Unknown component: {component}")

        # Calculate baseline metrics from test results
        baseline = self._calculate_baseline_from_metrics(component, metrics_list)

        # Store baseline
        self.current_baselines[component] = asdict(baseline)

        # Add to historical record
        self.historical_baselines.append(
            {"timestamp": baseline.timestamp, "component": component, "baseline": asdict(baseline)}
        )

        return baseline

    async def _baseline_p2p_network(self, benchmarker: PerformanceBenchmarker) -> list[PerformanceMetrics]:
        """Establish P2P network baselines."""
        metrics = []

        # Standard connection test
        try:
            from tools.benchmarks.performance_benchmarker import RealP2PBenchmark

            p2p_benchmark = RealP2PBenchmark()

            # Connection establishment baseline
            conn_metrics = await p2p_benchmark.benchmark_connection_establishment(25)
            metrics.append(conn_metrics)

            # Circuit creation baseline
            circuit_metrics = await p2p_benchmark.benchmark_circuit_creation(15)
            metrics.append(circuit_metrics)

        except Exception as e:
            logger.warning(f"P2P baseline tests failed: {e}")

        return metrics

    async def _baseline_agent_forge(self, benchmarker: PerformanceBenchmarker) -> list[PerformanceMetrics]:
        """Establish Agent Forge baselines."""
        metrics = []

        try:
            from tools.benchmarks.performance_benchmarker import RealAgentForgeBenchmark

            agent_benchmark = RealAgentForgeBenchmark()

            # Message processing baseline
            message_metrics = await agent_benchmark.benchmark_message_processing(100, 3)
            metrics.append(message_metrics)

            # Concurrent agents baseline
            concurrent_metrics = await agent_benchmark.benchmark_concurrent_agents(5, 25)
            metrics.append(concurrent_metrics)

        except Exception as e:
            logger.warning(f"Agent Forge baseline tests failed: {e}")

        return metrics

    async def _baseline_gateway(self, benchmarker: PerformanceBenchmarker) -> list[PerformanceMetrics]:
        """Establish Gateway baselines."""
        metrics = []

        try:
            from tools.benchmarks.performance_benchmarker import RealGatewayBenchmark

            gateway_benchmark = RealGatewayBenchmark()

            # Health endpoint baseline
            health_metrics = gateway_benchmark.benchmark_health_endpoint(50)
            metrics.append(health_metrics)

            # Concurrent requests baseline
            concurrent_metrics = gateway_benchmark.benchmark_concurrent_requests(10, 5)
            metrics.append(concurrent_metrics)

        except Exception as e:
            logger.warning(f"Gateway baseline tests failed (service may not be running): {e}")

        return metrics

    async def _baseline_digital_twin(self, benchmarker: PerformanceBenchmarker) -> list[PerformanceMetrics]:
        """Establish Digital Twin baselines."""
        metrics = []

        try:
            from tools.benchmarks.performance_benchmarker import RealDigitalTwinBenchmark

            twin_benchmark = RealDigitalTwinBenchmark()

            # Chat processing baseline
            chat_metrics = twin_benchmark.benchmark_chat_processing(25)
            metrics.append(chat_metrics)

        except Exception as e:
            logger.warning(f"Digital Twin baseline tests failed: {e}")

        return metrics

    def _calculate_baseline_from_metrics(
        self, component: str, metrics_list: list[PerformanceMetrics]
    ) -> BaselineMetrics:
        """Calculate baseline metrics from performance test results."""
        if not metrics_list:
            # Create minimal baseline if no metrics available
            return BaselineMetrics(
                component=component,
                timestamp=datetime.now(timezone.utc).isoformat(),
                version="unknown",
                baseline_throughput_per_second=1.0,
                min_acceptable_throughput=0.5,
                target_throughput=2.0,
                baseline_latency_avg_ms=1000.0,
                baseline_latency_p95_ms=2000.0,
                max_acceptable_latency_ms=5000.0,
                baseline_success_rate=0.90,
                min_acceptable_success_rate=0.80,
                baseline_cpu_percent=20.0,
                baseline_memory_mb=200.0,
                max_acceptable_cpu=80.0,
                max_acceptable_memory_mb=1024.0,
                max_concurrent_operations=10,
                test_parameters={"note": "minimal_baseline_no_metrics"},
            )

        # Calculate averages across all metrics
        avg_throughput = sum(m.throughput_per_second for m in metrics_list) / len(metrics_list)
        avg_latency = sum(m.latency_avg_ms for m in metrics_list) / len(metrics_list)
        avg_p95_latency = sum(m.latency_p95_ms for m in metrics_list) / len(metrics_list)
        avg_success_rate = sum(m.success_rate for m in metrics_list) / len(metrics_list)
        avg_cpu = sum(m.cpu_percent_avg for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_mb_peak for m in metrics_list) / len(metrics_list)
        max_items = max(m.items_processed for m in metrics_list)

        return BaselineMetrics(
            component=component,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",  # TODO: Get from git or version file
            # Throughput baselines (with safety margins)
            baseline_throughput_per_second=avg_throughput,
            min_acceptable_throughput=avg_throughput * 0.7,  # 30% degradation threshold
            target_throughput=avg_throughput * 1.5,  # 50% improvement target
            # Latency baselines
            baseline_latency_avg_ms=avg_latency,
            baseline_latency_p95_ms=avg_p95_latency,
            max_acceptable_latency_ms=avg_latency * 2.0,  # 2x latency degradation threshold
            # Reliability baselines
            baseline_success_rate=avg_success_rate,
            min_acceptable_success_rate=max(0.80, avg_success_rate - 0.10),  # Allow 10% degradation
            # Resource baselines
            baseline_cpu_percent=avg_cpu,
            baseline_memory_mb=avg_memory,
            max_acceptable_cpu=min(80.0, avg_cpu * 2.0),  # 2x CPU usage threshold
            max_acceptable_memory_mb=avg_memory * 2.0,  # 2x memory usage threshold
            # Scalability baselines
            max_concurrent_operations=max_items,
            test_parameters={
                "metrics_count": len(metrics_list),
                "test_date": datetime.now(timezone.utc).isoformat(),
                "test_types": [m.test_name for m in metrics_list],
            },
        )

    async def establish_all_baselines(self) -> dict[str, BaselineMetrics]:
        """Establish baselines for all system components."""
        logger.info("Establishing baselines for all components")

        components = ["P2P_Network", "Agent_Forge", "Gateway", "Digital_Twin"]
        baselines = {}

        for component in components:
            try:
                baseline = await self.establish_component_baseline(component)
                baselines[component] = baseline
                logger.info(f"Established baseline for {component}")

                # Brief pause between components
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Failed to establish baseline for {component}: {e}")

        # Save all baselines
        self.save_baselines()

        return baselines

    def compare_performance_to_baseline(self, component: str, metrics: PerformanceMetrics) -> dict[str, Any]:
        """Compare current performance against established baseline."""
        if component not in self.current_baselines:
            return {
                "status": "NO_BASELINE",
                "message": f"No baseline exists for {component}",
                "recommendation": "Establish baseline first",
            }

        baseline = BaselineMetrics(**self.current_baselines[component])

        comparisons = {
            "throughput": {
                "current": metrics.throughput_per_second,
                "baseline": baseline.baseline_throughput_per_second,
                "change_percent": (
                    (metrics.throughput_per_second - baseline.baseline_throughput_per_second)
                    / baseline.baseline_throughput_per_second
                )
                * 100,
                "status": "GOOD" if metrics.throughput_per_second >= baseline.min_acceptable_throughput else "DEGRADED",
            },
            "latency": {
                "current": metrics.latency_avg_ms,
                "baseline": baseline.baseline_latency_avg_ms,
                "change_percent": (
                    (metrics.latency_avg_ms - baseline.baseline_latency_avg_ms) / baseline.baseline_latency_avg_ms
                )
                * 100,
                "status": "GOOD" if metrics.latency_avg_ms <= baseline.max_acceptable_latency_ms else "DEGRADED",
            },
            "success_rate": {
                "current": metrics.success_rate,
                "baseline": baseline.baseline_success_rate,
                "change_percent": (
                    (metrics.success_rate - baseline.baseline_success_rate) / baseline.baseline_success_rate
                )
                * 100,
                "status": "GOOD" if metrics.success_rate >= baseline.min_acceptable_success_rate else "DEGRADED",
            },
            "cpu_usage": {
                "current": metrics.cpu_percent_avg,
                "baseline": baseline.baseline_cpu_percent,
                "status": "GOOD" if metrics.cpu_percent_avg <= baseline.max_acceptable_cpu else "HIGH",
            },
            "memory_usage": {
                "current": metrics.memory_mb_peak,
                "baseline": baseline.baseline_memory_mb,
                "status": "GOOD" if metrics.memory_mb_peak <= baseline.max_acceptable_memory_mb else "HIGH",
            },
        }

        # Overall assessment
        degraded_metrics = [k for k, v in comparisons.items() if v.get("status") == "DEGRADED"]
        high_resource_usage = [k for k, v in comparisons.items() if v.get("status") == "HIGH"]

        if degraded_metrics:
            overall_status = "PERFORMANCE_REGRESSION"
            message = f"Performance degradation detected in: {', '.join(degraded_metrics)}"
            recommendation = "Investigate performance regression and optimize affected components"
        elif high_resource_usage:
            overall_status = "RESOURCE_CONCERN"
            message = f"High resource usage detected: {', '.join(high_resource_usage)}"
            recommendation = "Monitor resource usage and consider optimization"
        else:
            overall_status = "WITHIN_BASELINE"
            message = "Performance is within acceptable baseline ranges"
            recommendation = "Continue monitoring"

        return {
            "component": component,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "message": message,
            "recommendation": recommendation,
            "detailed_comparisons": comparisons,
            "baseline_timestamp": baseline.timestamp,
        }

    def generate_baseline_report(self) -> dict[str, Any]:
        """Generate comprehensive baseline report."""
        return {
            "baseline_summary": {
                "components_with_baselines": len(self.current_baselines),
                "historical_measurements": len(self.historical_baselines),
                "last_updated": (
                    max([b["timestamp"] for b in self.historical_baselines]) if self.historical_baselines else None
                ),
            },
            "current_baselines": {
                component: {
                    "baseline_throughput": baseline["baseline_throughput_per_second"],
                    "baseline_latency_ms": baseline["baseline_latency_avg_ms"],
                    "baseline_success_rate": baseline["baseline_success_rate"],
                    "max_concurrent_operations": baseline["max_concurrent_operations"],
                    "established": baseline["timestamp"],
                }
                for component, baseline in self.current_baselines.items()
            },
            "performance_trends": self._analyze_performance_trends(),
            "baseline_validation": self._validate_baselines(),
        }

    def _analyze_performance_trends(self) -> dict[str, list[dict[str, Any]]]:
        """Analyze performance trends from historical data."""
        trends_by_component = {}

        for component in self.current_baselines.keys():
            component_history = [h for h in self.historical_baselines if h["component"] == component]

            if len(component_history) >= 2:
                # Sort by timestamp
                component_history.sort(key=lambda x: x["timestamp"])

                trends = []
                for i in range(1, len(component_history)):
                    prev = component_history[i - 1]["baseline"]
                    curr = component_history[i]["baseline"]

                    throughput_change = (
                        (curr["baseline_throughput_per_second"] - prev["baseline_throughput_per_second"])
                        / prev["baseline_throughput_per_second"]
                    ) * 100
                    latency_change = (
                        (curr["baseline_latency_avg_ms"] - prev["baseline_latency_avg_ms"])
                        / prev["baseline_latency_avg_ms"]
                    ) * 100

                    trends.append(
                        {
                            "from_date": prev["timestamp"],
                            "to_date": curr["timestamp"],
                            "throughput_change_percent": throughput_change,
                            "latency_change_percent": latency_change,
                            "trend": (
                                "IMPROVING"
                                if throughput_change > 0 and latency_change < 0
                                else "DEGRADING" if throughput_change < 0 or latency_change > 0 else "STABLE"
                            ),
                        }
                    )

                trends_by_component[component] = trends

        return trends_by_component

    def _validate_baselines(self) -> dict[str, Any]:
        """Validate current baselines for reasonableness."""
        validation_results = {}

        for component, baseline_data in self.current_baselines.items():
            baseline = BaselineMetrics(**baseline_data)
            issues = []

            # Validate throughput is reasonable
            if baseline.baseline_throughput_per_second <= 0:
                issues.append("Zero or negative baseline throughput")

            if baseline.min_acceptable_throughput >= baseline.baseline_throughput_per_second:
                issues.append("Minimum acceptable throughput is not below baseline")

            # Validate latency is reasonable
            if baseline.baseline_latency_avg_ms <= 0:
                issues.append("Zero or negative baseline latency")

            if baseline.max_acceptable_latency_ms <= baseline.baseline_latency_avg_ms:
                issues.append("Maximum acceptable latency is not above baseline")

            # Validate success rate
            if baseline.baseline_success_rate < 0.5 or baseline.baseline_success_rate > 1.0:
                issues.append("Unreasonable baseline success rate")

            validation_results[component] = {
                "valid": len(issues) == 0,
                "issues": issues,
                "last_updated": baseline.timestamp,
            }

        return validation_results


async def main():
    """Main entry point for baseline performance suite."""
    logger.info("Starting AIVillage Baseline Performance Measurement")

    suite = BaselinePerformanceSuite()

    try:
        # Establish baselines for all components
        baselines = await suite.establish_all_baselines()

        # Generate report
        report = suite.generate_baseline_report()

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"tools/benchmarks/baseline_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 50)
        print("AIVILLAGE BASELINE PERFORMANCE RESULTS")
        print("=" * 50)
        print(f"Components with baselines: {len(baselines)}")

        for component, baseline in baselines.items():
            print(f"\n{component}:")
            print(f"  Throughput: {baseline.baseline_throughput_per_second:.2f}/sec")
            print(f"  Latency: {baseline.baseline_latency_avg_ms:.0f}ms")
            print(f"  Success Rate: {baseline.baseline_success_rate:.1%}")
            print(f"  Max Concurrent: {baseline.max_concurrent_operations}")

        print(f"\nBaseline report saved to: {report_path}")
        print(f"Baseline data saved to: {suite.baseline_file}")

        return 0 if baselines else 1

    except Exception as e:
        logger.error(f"Baseline measurement failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
