#!/usr/bin/env python3
"""Generate performance tracking dashboard

Creates a comprehensive dashboard showing performance trends,
benchmark results, and regression detection.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
from typing import Any


class PerformanceDashboard:
    """Generate performance tracking dashboard."""

    def __init__(self, benchmarks_dir: Path = None):
        self.benchmarks_dir = benchmarks_dir or Path(__file__).parent.parent / "tests" / "benchmarks"
        self.baselines_file = self.benchmarks_dir / "baselines.json"
        self.results_file = self.benchmarks_dir / "benchmark_results.json"
        self.dashboard_file = Path("performance_dashboard.md")

        self.baselines = self._load_baselines()
        self.results = self._load_results()

    def _load_baselines(self) -> dict[str, float]:
        """Load performance baselines."""
        if not self.baselines_file.exists():
            return {}

        try:
            with open(self.baselines_file) as f:
                data = json.load(f)
            return data.get("baselines", {})
        except Exception as e:
            print(f"Warning: Could not load baselines: {e}")
            return {}

    def _load_results(self) -> list[dict[str, Any]]:
        """Load benchmark results."""
        if not self.results_file.exists():
            return []

        try:
            with open(self.results_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load results: {e}")
            return []

    def _get_recent_results(self, days: int = 30) -> list[dict[str, Any]]:
        """Get results from the last N days."""
        if not self.results:
            return []

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        recent_results = []
        for result in self.results:
            try:
                timestamp = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
                if timestamp >= cutoff_date:
                    recent_results.append(result)
            except Exception:
                continue  # Skip malformed timestamps

        return recent_results

    def _analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends over time."""
        recent_results = self._get_recent_results()

        if not recent_results:
            return {}

        # Group results by test name
        results_by_test = {}
        for result in recent_results:
            test_name = result["test_name"]
            if test_name not in results_by_test:
                results_by_test[test_name] = []
            results_by_test[test_name].append(result)

        trends = {}
        for test_name, test_results in results_by_test.items():
            if len(test_results) < 2:
                continue

            # Sort by timestamp
            sorted_results = sorted(test_results, key=lambda x: x["timestamp"])
            durations = [r["duration"] for r in sorted_results]

            # Calculate trend
            recent_avg = statistics.mean(durations[-5:]) if len(durations) >= 5 else statistics.mean(durations)
            overall_avg = statistics.mean(durations)

            trend_direction = "stable"
            if recent_avg > overall_avg * 1.1:
                trend_direction = "degrading"
            elif recent_avg < overall_avg * 0.9:
                trend_direction = "improving"

            trends[test_name] = {
                "direction": trend_direction,
                "recent_avg": recent_avg,
                "overall_avg": overall_avg,
                "samples": len(durations),
                "latest": durations[-1] if durations else 0
            }

        return trends if isinstance(trends, dict) else {}

    def _generate_ascii_sparkline(self, values: list[float], width: int = 20) -> str:
        """Generate ASCII sparkline for performance trends."""
        if not values or len(values) < 2:
            return "─" * width

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return "─" * width

        # Normalize values to 0-7 range for ASCII chars
        chars = " ▁▂▃▄▅▆▇█"
        normalized = []

        for val in values:
            norm = int((val - min_val) / (max_val - min_val) * 8)
            normalized.append(chars[min(norm, 8)])

        # Fit to desired width
        if len(normalized) > width:
            # Sample values evenly
            indices = [int(i * (len(normalized) - 1) / (width - 1)) for i in range(width)]
            return "".join(normalized[i] for i in indices)
        return "".join(normalized)

    def _detect_regressions(self) -> list[dict[str, Any]]:
        """Detect performance regressions."""
        regressions = []
        recent_results = self._get_recent_results(7)  # Last week

        # Group by test name
        results_by_test = {}
        for result in recent_results:
            test_name = result["test_name"]
            if test_name not in results_by_test:
                results_by_test[test_name] = []
            results_by_test[test_name].append(result["duration"])

        for test_name, durations in results_by_test.items():
            if not durations:
                continue

            current_avg = statistics.mean(durations)
            baseline = self.baselines.get(test_name)

            if baseline and current_avg > baseline * 1.2:  # 20% slower
                severity = "critical" if current_avg > baseline * 1.5 else "warning"

                regressions.append({
                    "test_name": test_name,
                    "baseline": baseline,
                    "current": current_avg,
                    "slowdown": (current_avg / baseline) if baseline > 0 else float("inf"),
                    "severity": severity
                })

        return sorted(regressions, key=lambda x: x.get("slowdown", 0), reverse=True)

    def generate_dashboard(self) -> str:
        """Generate the performance dashboard content."""
        trends = self._analyze_performance_trends()
        regressions = self._detect_regressions()
        recent_results = self._get_recent_results()

        # Get summary statistics
        total_tests = len(set(r["test_name"] for r in recent_results)) if recent_results else 0
        total_runs = len(recent_results)

        dashboard_content = f"""# Performance Benchmark Dashboard

Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Auto-Generated by Performance Monitor

## 📊 Overview

- **Total Benchmark Tests**: {total_tests}
- **Recent Runs (30 days)**: {total_runs}
- **Baselines Established**: {len(self.baselines)}
- **Active Regressions**: {len(regressions)}

## 🚨 Performance Regressions

"""

        if regressions:
            dashboard_content += "| Test | Baseline | Current | Slowdown | Severity |\n"
            dashboard_content += "|------|----------|---------|----------|----------|\n"

            for regression in regressions[:10]:  # Top 10 regressions
                severity_emoji = "CRITICAL" if regression["severity"] == "critical" else "WARNING"
                dashboard_content += f"| {regression['test_name']} | {regression['baseline']:.3f}s | {regression['current']:.3f}s | {regression['slowdown']:.1f}x | {severity_emoji} |\n"
        else:
            dashboard_content += "No performance regressions detected in the last 7 days.\n"

        dashboard_content += """

## 📈 Performance Trends (30 days)

"""

        if trends:
            dashboard_content += "| Test | Trend | Latest | Recent Avg | Overall Avg | Sparkline |\n"
            dashboard_content += "|------|-------|--------|------------|-------------|----------|\n"

            for test_name, trend_data in sorted(trends.items()):
                if isinstance(trend_data, dict):
                    trend_emoji = {
                        "improving": "UP",
                        "degrading": "DOWN",
                        "stable": "STABLE"
                    }.get(trend_data.get("direction", "stable"), "UNKNOWN")

                    # Get sparkline data
                    test_results = [r for r in recent_results if r["test_name"] == test_name]
                    durations = [r["duration"] for r in sorted(test_results, key=lambda x: x["timestamp"])]
                    sparkline = self._generate_ascii_sparkline(durations[-20:])  # Last 20 runs

                    dashboard_content += f"| {test_name} | {trend_emoji} {trend_data['direction']} | {trend_data['latest']:.3f}s | {trend_data['recent_avg']:.3f}s | {trend_data['overall_avg']:.3f}s | `{sparkline}` |\n"
        else:
            dashboard_content += "No trend data available.\n"

        dashboard_content += """

## 🎯 Baseline Comparison

"""

        if self.baselines:
            dashboard_content += "| Test | Baseline | Status |\n"
            dashboard_content += "|------|----------|--------|\n"

            for test_name, baseline in sorted(self.baselines.items()):
                # Get recent average for this test
                test_results = [r for r in recent_results if r["test_name"] == test_name]

                if test_results:
                    recent_durations = [r["duration"] for r in test_results[-5:]]  # Last 5 runs
                    recent_avg = statistics.mean(recent_durations)

                    if recent_avg <= baseline:
                        status = "OK - Within baseline"
                    elif recent_avg <= baseline * 1.2:
                        status = "WARN - Slightly slower"
                    else:
                        status = "SLOW - Significantly slower"

                    dashboard_content += f"| {test_name} | {baseline:.3f}s | {status} (current: {recent_avg:.3f}s) |\n"
                else:
                    dashboard_content += f"| {test_name} | {baseline:.3f}s | No recent data |\n"
        else:
            dashboard_content += "No baselines established. Run `python scripts/collect_baselines.py` to create baselines.\n"

        dashboard_content += """

## 📊 Performance Statistics

### Recent Activity (Last 7 days)
"""

        week_results = self._get_recent_results(7)
        if week_results:
            dashboard_content += f"- **Total Runs**: {len(week_results)}\n"
            dashboard_content += f"- **Average Duration**: {statistics.mean([r['duration'] for r in week_results]):.3f}s\n"
            dashboard_content += f"- **Fastest Test**: {min([r['duration'] for r in week_results]):.3f}s\n"
            dashboard_content += f"- **Slowest Test**: {max([r['duration'] for r in week_results]):.3f}s\n"

            # Group by test type
            test_types = {}
            for result in week_results:
                test_name = result["test_name"]
                if "simulation" in test_name:
                    test_type = "Simulation"
                elif "processing" in test_name:
                    test_type = "Processing"
                elif "io" in test_name:
                    test_type = "I/O"
                else:
                    test_type = "Other"

                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(result["duration"])

            dashboard_content += "\n### Performance by Category\n"
            for test_type, durations in test_types.items():
                avg_duration = statistics.mean(durations)
                dashboard_content += f"- **{test_type}**: {avg_duration:.3f}s avg ({len(durations)} runs)\n"
        else:
            dashboard_content += "No runs in the last 7 days.\n"

        dashboard_content += """

## 🔧 Recommendations

"""

        if regressions:
            dashboard_content += "### Performance Issues\n"
            for regression in regressions[:3]:  # Top 3
                dashboard_content += f"- **{regression['test_name']}**: Investigate {regression['slowdown']:.1f}x slowdown\n"

        if trends:
            degrading_tests = [name for name, data in trends.items() if data["direction"] == "degrading"]
            if degrading_tests:
                dashboard_content += "\n### Degrading Performance\n"
                for test_name in degrading_tests[:3]:
                    dashboard_content += f"- **{test_name}**: Monitor for continued degradation\n"

        if not self.baselines:
            dashboard_content += "\n### Setup\n"
            dashboard_content += "- Run `python scripts/collect_baselines.py` to establish performance baselines\n"
            dashboard_content += "- Add more benchmark tests to increase coverage\n"

        dashboard_content += """

## 📚 Usage

### Running Benchmarks
```bash
# Run all benchmarks
pytest tests/benchmarks/ -m benchmark

# Run only fast benchmarks
pytest tests/benchmarks/ -m "benchmark and not slow_benchmark"

# Collect new baselines
python scripts/collect_baselines.py --iterations 5
```

### Interpreting Results
- **Sparklines**: Show performance trend over time (higher = slower)
- **Regressions**: Tests running >20% slower than baseline
- **Trends**: Direction of performance change over last 30 days

---
*This dashboard is automatically updated when benchmarks are run*
"""

        return dashboard_content

    def save_dashboard(self) -> None:
        """Save the dashboard to file."""
        try:
            content = self.generate_dashboard()
            with open(self.dashboard_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Performance dashboard saved to {self.dashboard_file}")
        except Exception as e:
            print(f"Failed to save dashboard: {e}")

def main():
    """CLI interface for performance dashboard."""
    parser = argparse.ArgumentParser(description="Generate Performance Dashboard")
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("performance_dashboard.md"),
        help="Output file for dashboard"
    )
    parser.add_argument(
        "--benchmarks-dir", type=Path,
        help="Directory containing benchmark data"
    )

    args = parser.parse_args()

    dashboard = PerformanceDashboard(args.benchmarks_dir)
    dashboard.dashboard_file = args.output
    dashboard.save_dashboard()

if __name__ == "__main__":
    main()
