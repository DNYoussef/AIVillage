#!/usr/bin/env python3
"""
Pre-commit Performance Monitor

Tracks and optimizes pre-commit hook performance to maintain <2 minute execution time.
Provides performance analytics, bottleneck detection, and optimization recommendations.
"""

from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


class PreCommitPerformanceMonitor:
    """Monitor and optimize pre-commit hook performance."""

    def __init__(self, metrics_file: str | None = None):
        """Initialize performance monitor."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.metrics_file = metrics_file or self.repo_root / ".pre-commit-metrics.json"
        self.target_time = 120.0  # 2 minutes target
        self.warning_time = 90.0  # Warning threshold

    def record_execution(self, hook_id: str, duration: float, exit_code: int, files_processed: int = 0) -> None:
        """Record hook execution metrics."""
        metrics = self._load_metrics()

        execution_data = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "exit_code": exit_code,
            "files_processed": files_processed,
            "success": exit_code == 0,
        }

        if hook_id not in metrics:
            metrics[hook_id] = {"executions": [], "statistics": {}}

        metrics[hook_id]["executions"].append(execution_data)

        # Keep only last 100 executions per hook
        if len(metrics[hook_id]["executions"]) > 100:
            metrics[hook_id]["executions"] = metrics[hook_id]["executions"][-100:]

        # Update statistics
        self._update_statistics(metrics[hook_id])

        self._save_metrics(metrics)

    def analyze_performance(self) -> dict:
        """Analyze current performance and provide recommendations."""
        metrics = self._load_metrics()
        total_time = 0.0
        analysis = {
            "total_estimated_time": 0.0,
            "hooks_over_threshold": [],
            "bottlenecks": [],
            "recommendations": [],
            "success_rate": 0.0,
            "trend": "stable",
        }

        for hook_id, hook_data in metrics.items():
            if not hook_data["executions"]:
                continue

            stats = hook_data["statistics"]
            avg_time = stats.get("avg_duration", 0.0)
            total_time += avg_time

            success_rate = stats.get("success_rate", 100.0)

            if avg_time > 30.0:  # Hooks taking more than 30 seconds
                analysis["bottlenecks"].append(
                    {"hook": hook_id, "avg_duration": avg_time, "success_rate": success_rate}
                )

            if avg_time > 15.0:  # Warning for hooks over 15 seconds
                analysis["hooks_over_threshold"].append(hook_id)

        analysis["total_estimated_time"] = total_time
        analysis["success_rate"] = self._calculate_overall_success_rate(metrics)
        analysis["trend"] = self._calculate_trend(metrics)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis, metrics)

        return analysis

    def _load_metrics(self) -> dict:
        """Load performance metrics from file."""
        if not self.metrics_file.exists():
            return {}

        try:
            with open(self.metrics_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_metrics(self, metrics: dict) -> None:
        """Save performance metrics to file."""
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except OSError as e:
            print(f"Warning: Could not save metrics: {e}", file=sys.stderr)

    def _update_statistics(self, hook_data: dict) -> None:
        """Update statistics for a hook."""
        executions = hook_data["executions"]
        if not executions:
            return

        durations = [e["duration"] for e in executions]
        successes = [e["success"] for e in executions]

        hook_data["statistics"] = {
            "avg_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "success_rate": (sum(successes) / len(successes)) * 100,
            "total_executions": len(executions),
            "last_execution": executions[-1]["timestamp"],
        }

    def _calculate_overall_success_rate(self, metrics: dict) -> float:
        """Calculate overall success rate across all hooks."""
        total_executions = 0
        total_successes = 0

        for hook_data in metrics.values():
            stats = hook_data.get("statistics", {})
            executions = stats.get("total_executions", 0)
            success_rate = stats.get("success_rate", 100.0) / 100.0

            total_executions += executions
            total_successes += executions * success_rate

        return (total_successes / total_executions * 100) if total_executions > 0 else 100.0

    def _calculate_trend(self, metrics: dict) -> str:
        """Calculate performance trend (improving/degrading/stable)."""
        trends = []

        for hook_data in metrics.values():
            executions = hook_data["executions"]
            if len(executions) < 10:
                continue

            recent = [e["duration"] for e in executions[-10:]]
            older = [e["duration"] for e in executions[-20:-10]] if len(executions) >= 20 else recent

            recent_avg = statistics.mean(recent)
            older_avg = statistics.mean(older)

            if recent_avg < older_avg * 0.9:
                trends.append("improving")
            elif recent_avg > older_avg * 1.1:
                trends.append("degrading")
            else:
                trends.append("stable")

        if not trends:
            return "stable"

        # Return most common trend
        return max(set(trends), key=trends.count)

    def _generate_recommendations(self, analysis: dict, metrics: dict) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if analysis["total_estimated_time"] > self.target_time:
            recommendations.append(
                f"Total execution time ({analysis['total_estimated_time']:.1f}s) exceeds "
                f"target ({self.target_time}s). Consider moving slow hooks to manual/push stages."
            )

        for bottleneck in analysis["bottlenecks"]:
            if bottleneck["success_rate"] < 90.0:
                recommendations.append(
                    f"Hook '{bottleneck['hook']}' has low success rate "
                    f"({bottleneck['success_rate']:.1f}%). Consider debugging or disabling."
                )

            if bottleneck["avg_duration"] > 60.0:
                recommendations.append(
                    f"Hook '{bottleneck['hook']}' is very slow "
                    f"({bottleneck['avg_duration']:.1f}s). Consider optimization or moving to push stage."
                )

        if analysis["trend"] == "degrading":
            recommendations.append(
                "Performance is degrading over time. Review recent changes and "
                "consider cleaning up temporary files or caches."
            )

        if analysis["success_rate"] < 95.0:
            recommendations.append(
                f"Overall success rate ({analysis['success_rate']:.1f}%) is below target (95%). "
                "Review failing hooks and fix configuration issues."
            )

        return recommendations

    def print_report(self) -> None:
        """Print performance analysis report."""
        analysis = self.analyze_performance()

        print("🚀 Pre-commit Performance Analysis")
        print("=" * 50)
        print(f"Estimated Total Time: {analysis['total_estimated_time']:.1f}s")
        print(f"Target Time: {self.target_time}s")
        print(f"Overall Success Rate: {analysis['success_rate']:.1f}%")
        print(f"Performance Trend: {analysis['trend']}")

        if analysis["total_estimated_time"] <= self.target_time:
            print("✅ Performance within target!")
        elif analysis["total_estimated_time"] <= self.warning_time:
            print("⚠️  Approaching performance limit")
        else:
            print("🔴 Performance exceeds target!")

        if analysis["bottlenecks"]:
            print("\n🐌 Performance Bottlenecks:")
            for bottleneck in analysis["bottlenecks"]:
                print(
                    f"  • {bottleneck['hook']}: {bottleneck['avg_duration']:.1f}s "
                    f"(success: {bottleneck['success_rate']:.1f}%)"
                )

        if analysis["recommendations"]:
            print("\n💡 Optimization Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"  • {rec}")

        print("\n" + "=" * 50)

    def benchmark_current_config(self) -> float:
        """Benchmark current pre-commit configuration."""
        print("🏃 Running pre-commit benchmark...")

        start_time = time.time()
        try:
            # Run pre-commit on a sample of files for quick benchmark
            result = subprocess.run(
                ["pre-commit", "run", "--all-files", "--verbose"], capture_output=True, text=True, cwd=self.repo_root
            )

            duration = time.time() - start_time

            print(f"Benchmark completed in {duration:.1f}s")
            if result.returncode == 0:
                print("✅ All hooks passed")
            else:
                print(f"❌ Some hooks failed (exit code: {result.returncode})")

            return duration

        except subprocess.SubprocessError as e:
            print(f"❌ Benchmark failed: {e}")
            return float("inf")


def main():
    """Main entry point."""
    monitor = PreCommitPerformanceMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        duration = monitor.benchmark_current_config()
        if duration > monitor.target_time:
            print(f"⚠️  Benchmark exceeds target time by {duration - monitor.target_time:.1f}s")
            sys.exit(1)
    else:
        monitor.print_report()


if __name__ == "__main__":
    main()
