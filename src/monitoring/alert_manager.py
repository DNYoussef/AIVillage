#!/usr/bin/env python3
"""Manage test health alerts.

Handles threshold checking, alert dispatch through multiple channels,
and automated issue creation for test degradation.
"""

import asyncio
import json
import logging
import os
import smtplib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.mime.text import MIMEMultipart, MIMEText
from pathlib import Path
from typing import Any

import aiohttp
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Test health alert."""

    timestamp: str
    severity: str  # low, medium, high, critical
    message: str
    category: str  # success_rate, performance, flaky_tests, module_degradation
    details: dict[str, Any]
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlertConfig:
    """Alert configuration."""

    success_rate_threshold: float = 95.0
    performance_degradation_threshold: float = 1.5  # 50% slower
    flaky_test_threshold: float = 0.2
    consecutive_failures_threshold: int = 3

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AlertConfig":
        """Load config from YAML file."""
        if not config_path.exists():
            return cls()

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)

            alerts_config = data.get("alerts", {})
            return cls(
                success_rate_threshold=alerts_config.get(
                    "success_rate_threshold", 95.0
                ),
                performance_degradation_threshold=alerts_config.get(
                    "performance_degradation", 1.5
                ),
                flaky_test_threshold=alerts_config.get("flaky_test_threshold", 0.2),
                consecutive_failures_threshold=alerts_config.get(
                    "consecutive_failures", 3
                ),
            )
        except Exception as e:
            logger.exception(f"Failed to load alert config: {e}")
            return cls()


class AlertManager:
    """Manage test health alerts."""

    def __init__(
        self, config_path: Path | None = None, base_dir: Path | None = None
    ) -> None:
        self.base_dir = base_dir or Path(__file__).parent
        self.config_path = config_path or self.base_dir / "alert_config.yaml"
        self.alerts_log = self.base_dir / "alerts.log"
        self.active_alerts_file = self.base_dir / "active_alerts.json"

        self.config = AlertConfig.from_yaml(self.config_path)
        self.active_alerts: list[Alert] = []
        self.channels = []

        self._load_config()
        self._load_active_alerts()

    def _load_config(self) -> None:
        """Load alert channels configuration."""
        if not self.config_path.exists():
            # Create default config
            self._create_default_config()

        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)

            self.channels = data.get("channels", [])
            logger.info(f"Loaded {len(self.channels)} alert channels")

        except Exception as e:
            logger.exception(f"Failed to load alert channels config: {e}")
            self.channels = []

    def _create_default_config(self) -> None:
        """Create default alert configuration."""
        default_config = {
            "alerts": {
                "success_rate_threshold": 95.0,
                "performance_degradation": 1.5,
                "flaky_test_threshold": 0.2,
                "consecutive_failures": 3,
            },
            "channels": [
                {"type": "log", "path": "monitoring/alerts.log"}
                # Add more channels as needed
            ],
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default alert config: {self.config_path}")
        except Exception as e:
            logger.exception(f"Failed to create default config: {e}")

    def _load_active_alerts(self) -> None:
        """Load active alerts from file."""
        if not self.active_alerts_file.exists():
            self.active_alerts = []
            return

        try:
            with open(self.active_alerts_file) as f:
                data = json.load(f)

            self.active_alerts = [Alert(**alert_data) for alert_data in data]
            logger.info(f"Loaded {len(self.active_alerts)} active alerts")

        except Exception as e:
            logger.exception(f"Failed to load active alerts: {e}")
            self.active_alerts = []

    def _save_active_alerts(self) -> None:
        """Save active alerts to file."""
        try:
            self.active_alerts_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.active_alerts_file, "w") as f:
                json.dump(
                    [alert.to_dict() for alert in self.active_alerts], f, indent=2
                )
        except Exception as e:
            logger.exception(f"Failed to save active alerts: {e}")

    def check_thresholds(
        self, current_stats: dict[str, Any], history: list[dict[str, Any]] | None = None
    ) -> list[Alert]:
        """Check if any thresholds are breached."""
        alerts = []
        timestamp = datetime.now(UTC).isoformat()

        # Check success rate threshold
        success_rate = current_stats.get("success_rate", 0)
        if success_rate < self.config.success_rate_threshold:
            severity = (
                "critical"
                if success_rate < 80
                else "high"
                if success_rate < 90
                else "medium"
            )

            alert = Alert(
                timestamp=timestamp,
                severity=severity,
                message=f"Success rate {success_rate:.1f}% below threshold {self.config.success_rate_threshold}%",
                category="success_rate",
                details={
                    "current_rate": success_rate,
                    "threshold": self.config.success_rate_threshold,
                    "failed_tests": current_stats.get("failed", 0),
                    "total_tests": current_stats.get("total_tests", 0),
                },
            )
            alerts.append(alert)

        # Check consecutive failures
        if history and len(history) >= self.config.consecutive_failures_threshold:
            recent_runs = history[-self.config.consecutive_failures_threshold :]
            if all(
                run.get("success_rate", 100) < self.config.success_rate_threshold
                for run in recent_runs
            ):
                alert = Alert(
                    timestamp=timestamp,
                    severity="critical",
                    message=f"Success rate below threshold for {self.config.consecutive_failures_threshold} consecutive runs",
                    category="success_rate",
                    details={
                        "consecutive_failures": self.config.consecutive_failures_threshold,
                        "recent_rates": [
                            run.get("success_rate", 0) for run in recent_runs
                        ],
                    },
                )
                alerts.append(alert)

        # Check performance degradation
        if history and len(history) >= 2:
            current_duration = current_stats.get("duration", 0)
            current_tests = current_stats.get("total_tests", 1)
            current_avg = current_duration / current_tests if current_tests > 0 else 0

            # Compare with average of last 5 runs
            recent_runs = history[-6:-1]  # Exclude current run
            if recent_runs:
                baseline_avgs = []
                for run in recent_runs:
                    run_duration = run.get("duration", 0)
                    run_tests = run.get("total_tests", 1)
                    avg = run_duration / run_tests if run_tests > 0 else 0
                    if avg > 0:
                        baseline_avgs.append(avg)

                if baseline_avgs:
                    baseline_avg = sum(baseline_avgs) / len(baseline_avgs)
                    if (
                        current_avg
                        > baseline_avg * self.config.performance_degradation_threshold
                    ):
                        slowdown_factor = current_avg / baseline_avg

                        alert = Alert(
                            timestamp=timestamp,
                            severity="medium",
                            message=f"Performance degraded by {slowdown_factor:.1f}x (avg test time: {current_avg:.2f}s vs baseline {baseline_avg:.2f}s)",
                            category="performance",
                            details={
                                "current_avg": current_avg,
                                "baseline_avg": baseline_avg,
                                "slowdown_factor": slowdown_factor,
                                "threshold": self.config.performance_degradation_threshold,
                            },
                        )
                        alerts.append(alert)

        # Check module-specific issues
        modules = current_stats.get("modules", {})
        for module_name, module_stats in modules.items():
            module_success_rate = module_stats.get("success_rate", 0)
            module_tests = module_stats.get("total", 0)

            # Alert for modules with significant failures
            if (
                module_tests >= 3 and module_success_rate < 70
            ):  # At least 3 tests and <70% success
                alert = Alert(
                    timestamp=timestamp,
                    severity="high" if module_success_rate < 50 else "medium",
                    message=f"Module '{module_name}' has low success rate: {module_success_rate:.1f}%",
                    category="module_degradation",
                    details={
                        "module": module_name,
                        "success_rate": module_success_rate,
                        "failed_tests": module_stats.get("failed", 0),
                        "total_tests": module_tests,
                    },
                )
                alerts.append(alert)

        return alerts

    async def send_alert(self, alert: Alert) -> None:
        """Dispatch alert through configured channels."""
        logger.info(f"Sending alert: [{alert.severity}] {alert.message}")

        # Add to active alerts
        self.active_alerts.append(alert)
        self._save_active_alerts()

        # Log to file
        await self._log_alert(alert)

        # Send through configured channels
        for channel in self.channels:
            try:
                if channel["type"] == "log":
                    await self._log_alert(alert)
                elif channel["type"] == "webhook":
                    await self._send_webhook_alert(alert, channel)
                elif channel["type"] == "github":
                    await self._create_github_issue(alert, channel)
                elif channel["type"] == "email":
                    await self._send_email_alert(alert, channel)
            except Exception as e:
                logger.exception(f"Failed to send alert via {channel['type']}: {e}")

    async def _log_alert(self, alert: Alert) -> None:
        """Log alert to file."""
        try:
            self.alerts_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alerts_log, "a", encoding="utf-8") as f:
                f.write(
                    f"{alert.timestamp} [{alert.severity.upper()}] {alert.category}: {alert.message}\n"
                )
        except Exception as e:
            logger.exception(f"Failed to log alert: {e}")

    async def _send_webhook_alert(self, alert: Alert, channel: dict[str, Any]) -> None:
        """Send alert via webhook."""
        webhook_url = channel.get("url")
        if not webhook_url:
            logger.error("Webhook URL not configured")
            return

        # Replace environment variables
        webhook_url = os.path.expandvars(webhook_url)

        payload = {
            "timestamp": alert.timestamp,
            "severity": alert.severity,
            "message": alert.message,
            "category": alert.category,
            "details": alert.details,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Webhook alert sent successfully")
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
        except Exception as e:
            logger.exception(f"Failed to send webhook alert: {e}")

    async def _create_github_issue(self, alert: Alert, channel: dict[str, Any]) -> None:
        """Auto-create GitHub issue for test degradation."""
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            logger.warning("GITHUB_TOKEN not set, skipping GitHub issue creation")
            return

        repo = channel.get("repo", "ai-village")
        labels = channel.get("labels", ["test-degradation", "automated"])

        # Create issue title and body
        title = f"Test Alert: {alert.message}"
        body = f"""# Test Health Alert

**Severity**: {alert.severity.upper()}
**Category**: {alert.category}
**Timestamp**: {alert.timestamp}

## Message
{alert.message}

## Details
```json
{json.dumps(alert.details, indent=2)}
```

## Next Steps
- [ ] Investigate the root cause
- [ ] Fix failing tests or infrastructure
- [ ] Verify resolution with next test run

*This issue was automatically created by the test monitoring system.*
"""

        payload = {"title": title, "body": body, "labels": labels}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"token {github_token}",
                    "Accept": "application/vnd.github.v3+json",
                }

                async with session.post(
                    f"https://api.github.com/repos/{repo}/issues",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status == 201:
                        issue_data = await response.json()
                        logger.info(f"Created GitHub issue: {issue_data['html_url']}")
                    else:
                        logger.error(
                            f"GitHub issue creation failed with status {response.status}"
                        )
        except Exception as e:
            logger.exception(f"Failed to create GitHub issue: {e}")

    async def _send_email_alert(self, alert: Alert, channel: dict[str, Any]) -> None:
        """Send email alert."""
        smtp_server = channel.get("smtp_server")
        smtp_port = channel.get("smtp_port", 587)
        username = channel.get("username")
        password = os.getenv("ALERT_EMAIL_PASSWORD")
        to_emails = channel.get("to_emails", [])

        if not all([smtp_server, username, password, to_emails]):
            logger.warning("Email configuration incomplete, skipping email alert")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = username
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"Test Alert: {alert.severity.upper()} - {alert.message}"

            body = f"""
Test Health Alert

Severity: {alert.severity.upper()}
Category: {alert.category}
Timestamp: {alert.timestamp}

Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

Please investigate and resolve the issue.
"""

            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(username, to_emails, text)
            server.quit()

            logger.info("Email alert sent successfully")

        except Exception as e:
            logger.exception(f"Failed to send email alert: {e}")

    def resolve_alert(self, alert_id: str) -> None:
        """Mark an alert as resolved."""
        for alert in self.active_alerts:
            if alert.timestamp == alert_id:
                alert.resolved = True
                self._save_active_alerts()
                logger.info(f"Resolved alert: {alert.message}")
                break

    def get_active_alerts(self) -> list[Alert]:
        """Get list of active (unresolved) alerts."""
        return [alert for alert in self.active_alerts if not alert.resolved]


async def main() -> None:
    """CLI interface for alert manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Alert Manager")
    parser.add_argument("--check", help="Check current stats JSON file for alerts")
    parser.add_argument("--test-webhook", help="Test webhook with URL")
    parser.add_argument("--list-active", action="store_true", help="List active alerts")

    args = parser.parse_args()

    manager = AlertManager()

    if args.check:
        try:
            with open(args.check) as f:
                stats = json.load(f)

            alerts = manager.check_thresholds(stats)
            if alerts:
                print(f"Found {len(alerts)} alerts:")
                for alert in alerts:
                    print(f"- [{alert.severity}] {alert.message}")
                    await manager.send_alert(alert)
            else:
                print("No alerts triggered")
        except Exception as e:
            print(f"Error checking stats: {e}")

    elif args.test_webhook:
        test_alert = Alert(
            timestamp=datetime.now(UTC).isoformat(),
            severity="medium",
            message="Test webhook alert",
            category="test",
            details={"test": True},
        )

        channel = {"type": "webhook", "url": args.test_webhook}
        await manager._send_webhook_alert(test_alert, channel)

    elif args.list_active:
        active = manager.get_active_alerts()
        if active:
            print(f"Active alerts ({len(active)}):")
            for alert in active:
                print(f"- {alert.timestamp} [{alert.severity}] {alert.message}")
        else:
            print("No active alerts")


if __name__ == "__main__":
    asyncio.run(main())
