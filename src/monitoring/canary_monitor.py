#!/usr/bin/env python3
"""Monitor canary tests for unexpected changes.

Canary tests are expected to fail but serve as indicators of
architectural changes when they start passing or fail differently.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CanaryTest:
    """Represents a canary test and its expected behavior."""

    name: str
    path: str
    expected_status: str  # 'xfail', 'skip', 'fail'
    reason: str
    last_seen_status: str | None = None
    last_change: str | None = None
    monitored_since: str | None = None


@dataclass
class CanaryChange:
    """Represents a change in canary test behavior."""

    timestamp: str
    test_name: str
    old_status: str
    new_status: str
    reason: str
    alert_level: str  # 'info', 'warning', 'critical'


class CanaryMonitor:
    """Monitor canary tests for unexpected changes."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).parent
        self.canary_config_file = self.base_dir / "canary_tests.json"
        self.canary_history_file = self.base_dir / "canary_history.json"

        self.known_canaries: dict[str, CanaryTest] = {}
        self.change_history: list[CanaryChange] = []

        self._load_canary_config()
        self._load_change_history()

    def _load_canary_config(self) -> None:
        """Load known canary tests configuration."""
        if not self.canary_config_file.exists():
            self._create_default_canary_config()
            return

        try:
            with open(self.canary_config_file) as f:
                data = json.load(f)

            self.known_canaries = {}
            for canary_data in data.get("canary_tests", []):
                canary = CanaryTest(**canary_data)
                self.known_canaries[canary.name] = canary

            logger.info(f"Loaded {len(self.known_canaries)} canary tests")

        except Exception as e:
            logger.exception(f"Failed to load canary config: {e}")
            self.known_canaries = {}

    def _create_default_canary_config(self) -> None:
        """Create default canary test configuration."""
        default_canaries = [
            {
                "name": "test_bitnet_quantization_accuracy",
                "path": "tests/compression/test_stage1.py::test_bitnet_quantization_accuracy",
                "expected_status": "xfail",
                "reason": "BitNet quantization not implemented - architectural canary",
                "monitored_since": datetime.now(timezone.utc).isoformat(),
            },
            {
                "name": "evo_merge_tests",
                "path": "tests/evo_merge/test_*.py",
                "expected_status": "collection_error",
                "reason": "EvoMerge architecture not available - architectural canary",
                "monitored_since": datetime.now(timezone.utc).isoformat(),
            },
        ]

        config = {
            "canary_tests": default_canaries,
            "monitoring_enabled": True,
            "alert_on_unexpected_pass": True,
            "alert_on_status_change": True,
        }

        try:
            self.canary_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.canary_config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default canary config: {self.canary_config_file}")
        except Exception as e:
            logger.exception(f"Failed to create canary config: {e}")

    def _load_change_history(self) -> None:
        """Load canary change history."""
        if not self.canary_history_file.exists():
            self.change_history = []
            return

        try:
            with open(self.canary_history_file) as f:
                data = json.load(f)

            self.change_history = [CanaryChange(**change) for change in data]
            logger.info(f"Loaded {len(self.change_history)} canary changes")

        except Exception as e:
            logger.exception(f"Failed to load canary history: {e}")
            self.change_history = []

    def _save_change_history(self) -> None:
        """Save canary change history."""
        try:
            self.canary_history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.canary_history_file, "w") as f:
                data = [
                    {
                        "timestamp": change.timestamp,
                        "test_name": change.test_name,
                        "old_status": change.old_status,
                        "new_status": change.new_status,
                        "reason": change.reason,
                        "alert_level": change.alert_level,
                    }
                    for change in self.change_history
                ]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.exception(f"Failed to save change history: {e}")

    def check_canary_status(self, test_results: dict[str, Any]) -> list[CanaryChange]:
        """Check canary tests for unexpected changes."""
        changes = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract test results
        tests = test_results.get("tests", [])

        for test in tests:
            test_id = test.get("nodeid", "")
            outcome = test.get("outcome", "unknown")

            # Check if this is a known canary test
            canary = self._find_canary_for_test(test_id)
            if not canary:
                continue

            # Determine current status
            current_status = self._normalize_test_status(outcome, test)

            # Check for status changes
            if canary.last_seen_status and canary.last_seen_status != current_status:
                alert_level = self._determine_alert_level(
                    canary, canary.last_seen_status, current_status
                )

                change = CanaryChange(
                    timestamp=timestamp,
                    test_name=canary.name,
                    old_status=canary.last_seen_status,
                    new_status=current_status,
                    reason=f"Canary test status changed from {canary.last_seen_status} to {current_status}",
                    alert_level=alert_level,
                )

                changes.append(change)
                self.change_history.append(change)

                # Update canary status
                canary.last_seen_status = current_status
                canary.last_change = timestamp
            else:
                # First time seeing this status
                canary.last_seen_status = current_status

        # Save changes
        if changes:
            self._save_change_history()
            self._update_canary_config()

        return changes

    def _find_canary_for_test(self, test_id: str) -> CanaryTest | None:
        """Find canary test configuration for a given test ID."""
        for canary in self.known_canaries.values():
            if canary.path in test_id or canary.name in test_id:
                return canary

            # Handle wildcard patterns
            if "*" in canary.path:
                path_pattern = canary.path.replace("*", "")
                if path_pattern.rstrip("/") in test_id:
                    return canary

        return None

    def _normalize_test_status(self, outcome: str, test_data: dict[str, Any]) -> str:
        """Normalize pytest outcomes to consistent status strings."""
        if outcome == "passed":
            return "passed"
        if outcome == "failed":
            return "failed"
        if outcome == "skipped":
            return "skipped"
        if outcome == "xfail":
            return "xfail"
        if outcome == "xpass":
            return "xpass"  # Expected fail but passed - important!
        return outcome

    def _determine_alert_level(
        self, canary: CanaryTest, old_status: str, new_status: str
    ) -> str:
        """Determine alert level for status changes."""
        # Expected fail → Pass = Critical (architecture changed!)
        if canary.expected_status == "xfail" and new_status == "passed":
            return "critical"

        # Expected fail → xpass = Critical (architecture changed!)
        if canary.expected_status == "xfail" and new_status == "xpass":
            return "critical"

        # Collection error → Success = Critical (dependencies resolved!)
        if canary.expected_status == "collection_error" and new_status == "passed":
            return "critical"

        # Any pass → fail = Warning (regression?)
        if old_status == "passed" and new_status in ["failed", "error"]:
            return "warning"

        # Other changes = Info
        return "info"

    def _update_canary_config(self) -> None:
        """Update canary configuration with latest status."""
        try:
            config = {
                "canary_tests": [
                    {
                        "name": canary.name,
                        "path": canary.path,
                        "expected_status": canary.expected_status,
                        "reason": canary.reason,
                        "last_seen_status": canary.last_seen_status,
                        "last_change": canary.last_change,
                        "monitored_since": canary.monitored_since,
                    }
                    for canary in self.known_canaries.values()
                ],
                "monitoring_enabled": True,
                "alert_on_unexpected_pass": True,
                "alert_on_status_change": True,
            }

            with open(self.canary_config_file, "w") as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.exception(f"Failed to update canary config: {e}")

    def alert_canary_change(self, change: CanaryChange) -> None:
        """Send alert for canary test change."""
        if change.alert_level == "critical":
            logger.critical(f"CANARY ALERT: {change.test_name} - {change.reason}")
            print("\n🚨 CRITICAL CANARY ALERT 🚨")
            print(f"Test: {change.test_name}")
            print(f"Change: {change.old_status} → {change.new_status}")
            print("This may indicate significant architectural changes!")
            print("Please investigate immediately.\n")

        elif change.alert_level == "warning":
            logger.warning(f"CANARY WARNING: {change.test_name} - {change.reason}")
            print(f"\n⚠️  CANARY WARNING: {change.test_name}")
            print(f"Status changed: {change.old_status} → {change.new_status}\n")

        else:
            logger.info(f"CANARY INFO: {change.test_name} - {change.reason}")

    def get_canary_summary(self) -> dict[str, Any]:
        """Get summary of canary test status."""
        summary = {
            "total_canaries": len(self.known_canaries),
            "recent_changes": len(
                [
                    c
                    for c in self.change_history
                    if c.alert_level in ["warning", "critical"]
                ]
            ),
            "critical_alerts": len(
                [c for c in self.change_history if c.alert_level == "critical"]
            ),
            "canary_status": {},
        }

        for name, canary in self.known_canaries.items():
            status_emoji = (
                "🔴"
                if canary.last_seen_status in ["failed", "error"]
                else (
                    "🟡"
                    if canary.last_seen_status in ["skipped", "xfail"]
                    else "🟢"
                    if canary.last_seen_status == "passed"
                    else "⚪"
                )
            )

            summary["canary_status"][name] = {
                "status": canary.last_seen_status or "unknown",
                "expected": canary.expected_status,
                "emoji": status_emoji,
                "last_change": canary.last_change,
            }

        return summary


def main() -> None:
    """CLI interface for canary monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Canary Test Monitor")
    parser.add_argument("--check", help="Check test results JSON for canary changes")
    parser.add_argument("--summary", action="store_true", help="Show canary summary")
    parser.add_argument("--history", action="store_true", help="Show change history")

    args = parser.parse_args()

    monitor = CanaryMonitor()

    if args.check:
        try:
            with open(args.check) as f:
                test_results = json.load(f)

            changes = monitor.check_canary_status(test_results)
            if changes:
                print(f"Detected {len(changes)} canary changes:")
                for change in changes:
                    monitor.alert_canary_change(change)
            else:
                print("No canary changes detected")

        except Exception as e:
            print(f"Error checking canary status: {e}")

    elif args.summary:
        summary = monitor.get_canary_summary()
        print("Canary Test Summary:")
        print(f"  Total canaries: {summary['total_canaries']}")
        print(f"  Recent changes: {summary['recent_changes']}")
        print(f"  Critical alerts: {summary['critical_alerts']}")
        print("\nCanary Status:")
        for name, status in summary["canary_status"].items():
            print(
                f"  {status['emoji']} {name}: {status['status']} (expected: {status['expected']})"
            )

    elif args.history:
        if monitor.change_history:
            print("Canary Change History:")
            for change in monitor.change_history[-10:]:  # Last 10 changes
                print(
                    f"  {change.timestamp} [{change.alert_level.upper()}] {change.test_name}: {change.old_status} → {change.new_status}"
                )
        else:
            print("No canary changes recorded")


if __name__ == "__main__":
    main()
