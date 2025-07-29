#!/usr/bin/env python3
"""Analyze test trends over time

Provides trend analysis for test success rates, performance metrics,
and flaky test detection.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
from typing import Any


@dataclass
class TestRun:
    """Single test run data"""
    timestamp: datetime
    success_rate: float
    duration: float
    total_tests: int
    passed: int
    failed: int
    modules: dict[str, dict[str, Any]]

@dataclass
class TrendPoint:
    """Single point in a trend"""
    timestamp: datetime
    value: float

class TrendAnalyzer:
    """Analyze test trends over time"""

    def __init__(self, history_file: Path = None):
        self.history_file = history_file or Path(__file__).parent / "test_history.json"
        self.runs: list[TestRun] = []
        self._load_history()

    def _load_history(self):
        """Load test history from JSON file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file) as f:
                data = json.load(f)

            self.runs = []
            for item in data:
                timestamp = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                run = TestRun(
                    timestamp=timestamp,
                    success_rate=item["success_rate"],
                    duration=item["duration"],
                    total_tests=item["total_tests"],
                    passed=item["passed"],
                    failed=item["failed"],
                    modules=item["modules"]
                )
                self.runs.append(run)

            # Sort by timestamp
            self.runs.sort(key=lambda r: r.timestamp)

        except Exception as e:
            print(f"Error loading history: {e}")
            self.runs = []

    def generate_success_trend(self, days: int = 30) -> list[TrendPoint]:
        """Generate success rate trend data for graphing"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_runs = [run for run in self.runs if run.timestamp >= cutoff_date]

        return [TrendPoint(run.timestamp, run.success_rate) for run in recent_runs]

    def generate_performance_trend(self, days: int = 30) -> list[TrendPoint]:
        """Generate performance trend data"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_runs = [run for run in self.runs if run.timestamp >= cutoff_date]

        # Calculate average test duration
        points = []
        for run in recent_runs:
            avg_duration = run.duration / run.total_tests if run.total_tests > 0 else 0
            points.append(TrendPoint(run.timestamp, avg_duration))

        return points

    def identify_degrading_modules(self, threshold: float = 10.0, min_runs: int = 5) -> list[dict[str, Any]]:
        """Find modules with declining test success"""
        if len(self.runs) < min_runs:
            return []

        # Get recent runs for comparison
        recent_runs = self.runs[-min_runs:]
        if len(recent_runs) < 2:
            return []

        module_trends = defaultdict(list)

        # Collect module success rates over time
        for run in recent_runs:
            for module_name, stats in run.modules.items():
                success_rate = stats.get("success_rate", 0)
                module_trends[module_name].append(success_rate)

        degrading_modules = []

        for module_name, rates in module_trends.items():
            if len(rates) < 2:
                continue

            # Calculate trend - simple linear regression slope
            n = len(rates)
            x_values = list(range(n))

            # Calculate slope using least squares
            sum_x = sum(x_values)
            sum_y = sum(rates)
            sum_xy = sum(x * y for x, y in zip(x_values, rates, strict=False))
            sum_x2 = sum(x * x for x in x_values)

            if n * sum_x2 - sum_x * sum_x == 0:
                continue

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

            # Check if declining significantly
            if slope < -threshold / n:  # Negative slope indicates decline
                current_rate = rates[-1]
                initial_rate = rates[0]
                change = current_rate - initial_rate

                degrading_modules.append({
                    "module": module_name,
                    "current_rate": current_rate,
                    "initial_rate": initial_rate,
                    "change": change,
                    "trend_slope": slope,
                    "severity": "high" if change < -20 else "medium" if change < -10 else "low"
                })

        # Sort by severity and magnitude of decline
        degrading_modules.sort(key=lambda x: (x["severity"] == "high", abs(x["change"])), reverse=True)

        return degrading_modules

    def detect_flaky_tests(self, instability_threshold: float = 0.2, min_runs: int = 10) -> list[dict[str, Any]]:
        """Identify tests that intermittently fail"""
        if len(self.runs) < min_runs:
            return []

        # Track module stability over recent runs
        recent_runs = self.runs[-min_runs:]
        module_rates = defaultdict(list)

        for run in recent_runs:
            for module_name, stats in run.modules.items():
                success_rate = stats.get("success_rate", 0)
                module_rates[module_name].append(success_rate)

        flaky_modules = []

        for module_name, rates in module_rates.items():
            if len(rates) < 3:
                continue

            # Calculate standard deviation of success rates
            mean_rate = statistics.mean(rates)
            if mean_rate == 0 or mean_rate == 100:
                continue  # Skip always-failing or always-passing modules

            try:
                std_dev = statistics.stdev(rates)
                coefficient_of_variation = std_dev / mean_rate

                # Check for high variability
                if coefficient_of_variation > instability_threshold:
                    flaky_modules.append({
                        "module": module_name,
                        "mean_rate": mean_rate,
                        "std_dev": std_dev,
                        "coefficient_of_variation": coefficient_of_variation,
                        "instability_score": coefficient_of_variation,
                        "recent_rates": rates[-5:]  # Last 5 runs
                    })
            except statistics.StatisticsError:
                continue

        # Sort by instability score
        flaky_modules.sort(key=lambda x: x["instability_score"], reverse=True)

        return flaky_modules[:10]  # Top 10 most flaky

    def generate_ascii_trend_graph(self, data: list[float], width: int = 50, height: int = 10) -> str:
        """Generate ASCII art trend graph for markdown embedding"""
        if not data or len(data) < 2:
            return "Insufficient data for graph"

        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            middle = height // 2
            line = "─" * width
            graph_lines = [" " * width for _ in range(height)]
            graph_lines[middle] = line
            return f"{max_val:.1f} |" + "\n     |".join(graph_lines)

        # Normalize data to fit in height
        normalized = []
        for val in data:
            norm = int((val - min_val) / (max_val - min_val) * (height - 1))
            normalized.append(height - 1 - norm)  # Flip for display

        # Create graph
        graph_lines = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, y in enumerate(normalized):
            x = int(i * (width - 1) / (len(data) - 1))
            if 0 <= x < width and 0 <= y < height:
                graph_lines[y][x] = "●"

                # Connect with lines
                if i > 0:
                    prev_y = normalized[i-1]
                    prev_x = int((i-1) * (width - 1) / (len(data) - 1))

                    # Simple line drawing
                    if prev_x != x:
                        start_y, end_y = (prev_y, y) if prev_x < x else (y, prev_y)
                        for line_x in range(min(prev_x, x) + 1, max(prev_x, x)):
                            if 0 <= line_x < width:
                                # Linear interpolation for y
                                ratio = (line_x - prev_x) / (x - prev_x) if x != prev_x else 0
                                line_y = int(prev_y + ratio * (y - prev_y))
                                if 0 <= line_y < height:
                                    if graph_lines[line_y][line_x] == " ":
                                        graph_lines[line_y][line_x] = "·"

        # Format output
        result = []
        for i, line in enumerate(graph_lines):
            y_val = max_val - (i / (height - 1)) * (max_val - min_val)
            line_str = "".join(line)
            result.append(f"{y_val:6.1f} |{line_str}")

        # Add x-axis
        x_axis = " " * 7 + "├" + "─" * (width - 1) + "┤"
        result.append(x_axis)
        result.append(f"       {min(range(len(data))):<{width//2-2}}{max(range(len(data))):>{width//2-2}}")

        return "\n".join(result)

    def get_trend_summary(self, days: int = 30) -> dict[str, Any]:
        """Get comprehensive trend summary"""
        success_trend = self.generate_success_trend(days)
        performance_trend = self.generate_performance_trend(days)
        degrading_modules = self.identify_degrading_modules()
        flaky_modules = self.detect_flaky_tests()

        summary = {
            "period_days": days,
            "total_runs": len([p for p in success_trend]),
            "success_trend": {
                "current": success_trend[-1].value if success_trend else 0,
                "min": min(p.value for p in success_trend) if success_trend else 0,
                "max": max(p.value for p in success_trend) if success_trend else 0,
                "trend_direction": self._get_trend_direction(success_trend)
            },
            "performance_trend": {
                "current_avg_duration": performance_trend[-1].value if performance_trend else 0,
                "trend_direction": self._get_trend_direction(performance_trend, lower_is_better=True)
            },
            "degrading_modules_count": len(degrading_modules),
            "flaky_modules_count": len(flaky_modules),
            "degrading_modules": degrading_modules[:3],  # Top 3
            "flaky_modules": flaky_modules[:3]  # Top 3
        }

        return summary

    def _get_trend_direction(self, trend_points: list[TrendPoint], lower_is_better: bool = False) -> str:
        """Determine trend direction from points"""
        if len(trend_points) < 2:
            return "stable"

        # Simple comparison of first and last
        first_val = trend_points[0].value
        last_val = trend_points[-1].value

        threshold = 0.05 * first_val  # 5% change threshold

        if abs(last_val - first_val) < threshold:
            return "stable"
        if last_val > first_val:
            return "declining" if lower_is_better else "improving"
        return "improving" if lower_is_better else "declining"

def main():
    """CLI interface for trend analysis"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Trend Analyzer")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--summary", action="store_true", help="Show trend summary")
    parser.add_argument("--flaky", action="store_true", help="Show flaky tests")
    parser.add_argument("--degrading", action="store_true", help="Show degrading modules")

    args = parser.parse_args()

    analyzer = TrendAnalyzer()

    if args.summary:
        summary = analyzer.get_trend_summary(args.days)
        print(json.dumps(summary, indent=2))

    if args.flaky:
        flaky = analyzer.detect_flaky_tests()
        print("Flaky Modules:")
        for module in flaky:
            print(f"- {module['module']}: {module['instability_score']:.3f} instability score")

    if args.degrading:
        degrading = analyzer.identify_degrading_modules()
        print("Degrading Modules:")
        for module in degrading:
            print(f"- {module['module']}: {module['change']:.1f}% change ({module['severity']})")

if __name__ == "__main__":
    main()
