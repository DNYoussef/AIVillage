#!/usr/bin/env python3
"""Automated test health monitoring system

Captures test results after each run, stores historical data,
triggers dashboard updates, and sends alerts on degradation.
"""

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


@dataclass
class MonitorStats:
    """Test statistics for a single run"""

    timestamp: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    success_rate: float
    duration: float
    modules: dict[str, dict[str, Any]]

    @classmethod
    def from_pytest_json(cls, report_data: dict[str, Any]) -> "MonitorStats":
        """Create MonitorStats from pytest JSON report"""
        summary = report_data.get("summary", {})

        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        skipped = summary.get("skipped", 0)
        errors = summary.get("error", 0)

        success_rate = (passed / total * 100) if total > 0 else 0
        duration = report_data.get("duration", 0)

        # Parse module-level stats
        modules = {}
        for test in report_data.get("tests", []):
            module_name = test.get("nodeid", "").split("::")[0]
            if "/" in module_name:
                module_name = module_name.split("/")[-1].replace(".py", "")

            if module_name not in modules:
                modules[module_name] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                }

            modules[module_name]["total"] += 1
            outcome = test.get("outcome", "unknown")
            if outcome in modules[module_name]:
                modules[module_name][outcome] += 1

        # Calculate success rates for modules
        for module_stats in modules.values():
            total_module = module_stats["total"]
            passed_module = module_stats["passed"]
            module_stats["success_rate"] = (
                (passed_module / total_module * 100) if total_module > 0 else 0
            )

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_tests=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            success_rate=success_rate,
            duration=duration,
            modules=modules,
        )


class HealthMonitor:
    """Automated test health monitoring system"""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.history_file = self.base_dir / "test_history.json"
        self.dashboard_path = Path("test_health_dashboard.md")
        self.alert_threshold = 95.0  # 95% success rate
        self.history: list[MonitorStats] = []

        # Load existing history
        self._load_history()

    def _load_history(self):
        """Load historical test data"""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    self.history = [MonitorStats(**item) for item in data]
                logger.info(f"Loaded {len(self.history)} historical test runs")
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
                self.history = []
        else:
            self.history = []

    def _save_history(self):
        """Save historical test data"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump([asdict(stat) for stat in self.history], f, indent=2)
            logger.info(f"Saved {len(self.history)} test runs to history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    async def capture_test_results(self, pytest_json_report: str):
        """Parse pytest JSON report and store results"""
        try:
            with open(pytest_json_report) as f:
                report_data = json.load(f)

            stats = MonitorStats.from_pytest_json(report_data)
            self.history.append(stats)

            # Keep last 100 runs
            if len(self.history) > 100:
                self.history = self.history[-100:]

            self._save_history()

            logger.info(
                f"Captured test results: {stats.success_rate:.1f}% success rate"
            )

            # Update dashboard and check alerts
            await self.update_dashboard()
            await self.check_alert_conditions()

        except Exception as e:
            logger.error(f"Failed to capture test results: {e}")

    def get_trend_arrow(self) -> str:
        """Get trend direction arrow"""
        if len(self.history) < 2:
            return "→"

        current = self.history[-1].success_rate
        previous = self.history[-2].success_rate

        if current > previous + 1:
            return "↗️"
        if current < previous - 1:
            return "↘️"
        return "→"

    def generate_ascii_trend_graph(self, days: int = 30) -> str:
        """Generate ASCII trend graph"""
        if len(self.history) < 2:
            return "Insufficient data for trend graph"

        # Get last N runs
        recent_runs = self.history[-min(days, len(self.history)) :]
        success_rates = [run.success_rate for run in recent_runs]

        if not success_rates:
            return "No data available"

        # Simple ASCII sparkline
        min_rate = min(success_rates)
        max_rate = max(success_rates)

        if max_rate == min_rate:
            return f"Stable at {min_rate:.1f}%: " + "─" * len(success_rates)

        # Normalize to 0-7 range for ASCII chars
        chars = " ▁▂▃▄▅▆▇█"
        normalized = []
        for rate in success_rates:
            norm = int((rate - min_rate) / (max_rate - min_rate) * 8)
            normalized.append(chars[min(norm, 8)])

        sparkline = "".join(normalized)
        return f"{min_rate:.1f}%-{max_rate:.1f}%: {sparkline}"

    def identify_hot_issues(self) -> list[dict[str, Any]]:
        """Identify current hot issues"""
        if not self.history:
            return []

        current_stats = self.history[-1]
        issues = []

        # Find modules with high failure rates
        for module_name, module_stats in current_stats.modules.items():
            success_rate = module_stats.get("success_rate", 0)
            if success_rate < 80 and module_stats["total"] > 0:  # Modules below 80%
                issues.append(
                    {
                        "module": module_name,
                        "description": "Low success rate",
                        "failure_rate": 100 - success_rate,
                        "failed_tests": module_stats.get("failed", 0),
                    }
                )

        # Check for trend degradation
        if len(self.history) >= 3:
            recent_rates = [run.success_rate for run in self.history[-3:]]
            if all(
                recent_rates[i] > recent_rates[i + 1]
                for i in range(len(recent_rates) - 1)
            ):
                issues.append(
                    {
                        "module": "Overall",
                        "description": "Declining trend over last 3 runs",
                        "failure_rate": recent_rates[0] - recent_rates[-1],
                        "trend": "declining",
                    }
                )

        return sorted(issues, key=lambda x: x["failure_rate"], reverse=True)[:5]

    async def update_dashboard(self):
        """Regenerate dashboard with latest data and trends"""
        if not self.history:
            logger.warning("No test history available for dashboard")
            return

        current_stats = self.history[-1]

        # Determine status emoji
        success_rate = current_stats.success_rate
        if success_rate >= 95:
            status_emoji = "✅"
        elif success_rate >= 85:
            status_emoji = "⚠️"
        else:
            status_emoji = "❌"

        # Get trend info
        trend_arrow = self.get_trend_arrow()
        if "↗️" in trend_arrow:
            trend_description = "Improving"
        elif "↘️" in trend_arrow:
            trend_description = "Declining"
        else:
            trend_description = "Stable"

        # Generate dashboard content
        dashboard_content = f"""# AI Village Test Health Dashboard
Last Updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
Auto-Generated by Test Monitor v{__version__}

## 📊 Current Health Status
Overall Success Rate: {success_rate:.1f}% {status_emoji}
Trend: {trend_arrow} {trend_description}

## 📈 30-Day Success Rate Trend
```
{self.generate_ascii_trend_graph()}
```

## 🔥 Hot Issues
"""

        hot_issues = self.identify_hot_issues()
        if hot_issues:
            for issue in hot_issues:
                dashboard_content += f"- **{issue['module']}**: {issue['description']} ({issue['failure_rate']:.1f}% failure rate)\n"
        else:
            dashboard_content += "✅ No major issues detected\n"

        dashboard_content += """

## 📊 Module Health Matrix
| Module | Tests | Pass | Fail | Skip | Success% | Status |
|--------|-------|------|------|------|----------|--------|
"""

        # Sort modules by success rate
        sorted_modules = sorted(
            current_stats.modules.items(),
            key=lambda x: x[1].get("success_rate", 0),
            reverse=True,
        )

        for module_name, module_stats in sorted_modules:
            success_rate_mod = module_stats.get("success_rate", 0)
            status = (
                "✅"
                if success_rate_mod >= 95
                else "⚠️"
                if success_rate_mod >= 80
                else "❌"
            )

            dashboard_content += f"| {module_name} | {module_stats['total']} | {module_stats['passed']} | {module_stats['failed']} | {module_stats['skipped']} | {success_rate_mod:.1f}% | {status} |\n"

        dashboard_content += f"""

## ⚡ Performance Metrics
- Total Tests Run: {current_stats.total_tests}
- Test Suite Duration: {current_stats.duration:.1f}s
- Average Test Time: {(current_stats.duration / current_stats.total_tests):.2f}s per test

## 🔄 Recent Changes
"""

        if len(self.history) >= 3:
            for _i, run in enumerate(self.history[-3:], 1):
                time_str = datetime.fromisoformat(
                    run.timestamp.replace("Z", "+00:00")
                ).strftime("%m-%d %H:%M")
                dashboard_content += f"- {time_str}: {run.success_rate:.1f}% success rate ({run.total_tests} tests)\n"

        dashboard_content += """

## 🚨 Alert Status
"""

        alerts = await self._check_alert_conditions_internal()
        if alerts:
            for alert in alerts:
                dashboard_content += (
                    f"> ⚠️ **{alert['severity']}**: {alert['message']}\n"
                )
        else:
            dashboard_content += "✅ No active alerts\n"

        dashboard_content += """

---
*Dashboard automatically updated after each test run*
"""

        # Write dashboard
        try:
            with open(self.dashboard_path, "w", encoding="utf-8") as f:
                f.write(dashboard_content)
            logger.info(f"Updated dashboard: {self.dashboard_path}")
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")

    async def _check_alert_conditions_internal(self) -> list[dict[str, str]]:
        """Internal method to check alert conditions"""
        alerts = []

        if not self.history:
            return alerts

        current_stats = self.history[-1]

        # Check success rate threshold
        if current_stats.success_rate < self.alert_threshold:
            alerts.append(
                {
                    "severity": "HIGH",
                    "message": f"Success rate {current_stats.success_rate:.1f}% below threshold {self.alert_threshold}%",
                }
            )

        # Check for consecutive failures
        if len(self.history) >= 3:
            recent_rates = [run.success_rate for run in self.history[-3:]]
            if all(rate < self.alert_threshold for rate in recent_rates):
                alerts.append(
                    {
                        "severity": "CRITICAL",
                        "message": "Success rate below threshold for 3 consecutive runs",
                    }
                )

        return alerts

    async def check_alert_conditions(self):
        """Check if alerts should be triggered"""
        alerts = await self._check_alert_conditions_internal()

        for alert in alerts:
            logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")

            # Write to alert log
            alert_log = self.base_dir / "alerts.log"
            alert_log.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).isoformat()
            with open(alert_log, "a") as f:
                f.write(f"{timestamp} [{alert['severity']}] {alert['message']}\n")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Test Health Monitor")
    parser.add_argument("--capture", help="Capture results from pytest JSON report")
    parser.add_argument(
        "--update-dashboard", action="store_true", help="Update dashboard"
    )
    parser.add_argument(
        "--check-thresholds", action="store_true", help="Check alert thresholds"
    )

    args = parser.parse_args()

    monitor = HealthMonitor()

    if args.capture:
        await monitor.capture_test_results(args.capture)
    elif args.update_dashboard:
        await monitor.update_dashboard()
    elif args.check_thresholds:
        await monitor.check_alert_conditions()
    else:
        # Default: try to find recent test results
        test_results = Path("test-results.json")
        if test_results.exists():
            await monitor.capture_test_results(str(test_results))
        else:
            logger.info(
                "No test results found. Use --capture <report.json> to specify report file."
            )


if __name__ == "__main__":
    asyncio.run(main())
