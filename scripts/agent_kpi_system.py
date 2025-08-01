#!/usr/bin/env python3
"""Implement comprehensive KPI system for agent performance tracking and evolution.

Enables automatic retirement of underperforming agents.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Constants
MIN_VALUES_FOR_TREND = 2
SIGNIFICANT_NEGATIVE_TREND_THRESHOLD = -0.02


class KPIMetric(Enum):
    """Standard KPI metrics across all agents."""

    # Efficiency metrics
    TASK_COMPLETION_RATE = "task_completion_rate"
    AVERAGE_RESPONSE_TIME = "average_response_time"
    RESOURCE_EFFICIENCY = "resource_efficiency"

    # Quality metrics
    OUTPUT_QUALITY_SCORE = "output_quality_score"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"

    # Learning metrics
    IMPROVEMENT_RATE = "improvement_rate"
    ADAPTATION_SPEED = "adaptation_speed"
    KNOWLEDGE_RETENTION = "knowledge_retention"

    # Collaboration metrics
    INTER_AGENT_COOPERATION = "inter_agent_cooperation"
    COMMUNICATION_CLARITY = "communication_clarity"
    HANDOFF_SUCCESS_RATE = "handoff_success_rate"


@dataclass
class KPIThresholds:
    """Thresholds for agent performance evaluation."""

    critical: float = 0.3  # Below this = immediate retirement consideration
    warning: float = 0.5  # Below this = performance improvement needed
    good: float = 0.7  # Above this = performing well
    excellent: float = 0.9  # Above this = exceptional performance


@dataclass
class PerformanceRecord:
    """Single performance measurement."""

    timestamp: datetime
    task_id: str
    metrics: dict[KPIMetric, float]
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "metrics": {k.value: v for k, v in self.metrics.items()},
            "context": self.context,
        }


class AgentKPITracker:
    """Tracks KPIs for individual agents."""

    def __init__(self, agent_id: str, role: str) -> None:
        """Initialize agent KPI tracker."""
        self.agent_id = agent_id
        self.role = role
        self.performance_history: list[PerformanceRecord] = []
        self.kpi_weights = self._initialize_weights()
        self.thresholds = KPIThresholds()

    def _initialize_weights(self) -> dict[KPIMetric, float]:
        """Initialize KPI weights based on agent role."""
        # Default weights
        weights = dict.fromkeys(KPIMetric, 1.0)

        # Adjust weights based on role
        if self.role == "king":
            weights[KPIMetric.INTER_AGENT_COOPERATION] = 2.0
            weights[KPIMetric.TASK_COMPLETION_RATE] = 1.5
        elif self.role == "magi":
            weights[KPIMetric.OUTPUT_QUALITY_SCORE] = 2.0
            weights[KPIMetric.ERROR_RATE] = 1.5
        elif self.role == "sage":
            weights[KPIMetric.KNOWLEDGE_RETENTION] = 2.0
            weights[KPIMetric.ADAPTATION_SPEED] = 1.5
        elif self.role == "polyglot":
            weights[KPIMetric.OUTPUT_QUALITY_SCORE] = 2.0
            weights[KPIMetric.COMMUNICATION_CLARITY] = 1.5
        elif self.role == "medic":
            weights[KPIMetric.USER_SATISFACTION] = 2.0
            weights[KPIMetric.ERROR_RATE] = 2.5  # Critical for medical
        elif self.role == "oracle":
            weights[KPIMetric.OUTPUT_QUALITY_SCORE] = 2.0
            weights[KPIMetric.RESOURCE_EFFICIENCY] = 1.5
        elif self.role == "sustainer":
            weights[KPIMetric.RESOURCE_EFFICIENCY] = 2.5
            weights[KPIMetric.IMPROVEMENT_RATE] = 1.5

        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def record_performance(
        self, task_id: str, metrics: dict[KPIMetric, float], context: dict | None = None
    ) -> None:
        """Record a performance measurement."""
        record = PerformanceRecord(
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            metrics=metrics,
            context=context or {},
        )
        self.performance_history.append(record)

    def calculate_overall_kpi(self, time_window: timedelta | None = None) -> float:
        """Calculate weighted overall KPI score."""
        if not self.performance_history:
            return 0.5  # Neutral starting score

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now(timezone.utc) - time_window
            records = [r for r in self.performance_history if r.timestamp > cutoff]
        else:
            records = self.performance_history

        if not records:
            return 0.5

        # Calculate average for each metric
        metric_averages = {}
        for metric in KPIMetric:
            values = [
                r.metrics.get(metric, 0.5) for r in records if metric in r.metrics
            ]
            if values:
                metric_averages[metric] = np.mean(values)

        # Calculate weighted overall score
        overall_score = 0.0
        for metric, avg_value in metric_averages.items():
            overall_score += avg_value * self.kpi_weights.get(metric, 1.0)

        return overall_score

    def get_trend(self, metric: KPIMetric, window_size: int = 10) -> float:
        """Calculate trend for a specific metric."""
        values = [
            r.metrics.get(metric, 0.5)
            for r in self.performance_history[-window_size:]
            if metric in r.metrics
        ]

        if len(values) < MIN_VALUES_FOR_TREND:
            return 0.0

        # Simple linear regression for trend
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]  # Slope indicates trend

    def should_retire(self) -> tuple[bool, str]:
        """Determine if agent should retire based on KPIs."""
        overall_kpi = self.calculate_overall_kpi()

        # Check critical threshold
        if overall_kpi < self.thresholds.critical:
            return True, f"Overall KPI {overall_kpi:.2f} below critical threshold"

        # Check declining trend
        declining_metrics = []
        for metric in KPIMetric:
            trend = self.get_trend(metric)
            if (
                trend < SIGNIFICANT_NEGATIVE_TREND_THRESHOLD
            ):  # Significant negative trend
                declining_metrics.append(metric)

        if len(declining_metrics) > len(KPIMetric) / 2:
            return True, f"Declining trend in {len(declining_metrics)} metrics"

        # Check sustained poor performance
        recent_kpi = self.calculate_overall_kpi(time_window=timedelta(days=7))
        if recent_kpi < self.thresholds.warning:
            older_kpi = self.calculate_overall_kpi(time_window=timedelta(days=14))
            if older_kpi < self.thresholds.warning:
                return True, "Sustained poor performance over 2 weeks"

        return False, "Performance acceptable"

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        overall_kpi = self.calculate_overall_kpi()
        recent_kpi = self.calculate_overall_kpi(time_window=timedelta(days=1))

        summary = {
            "agent_id": self.agent_id,
            "role": self.role,
            "overall_kpi": overall_kpi,
            "recent_kpi": recent_kpi,
            "total_tasks": len(self.performance_history),
            "performance_status": self._get_status(overall_kpi),
            "metrics": {},
        }

        # Add per-metric summaries
        for metric in KPIMetric:
            values = [
                r.metrics.get(metric, 0.5)
                for r in self.performance_history
                if metric in r.metrics
            ]
            if values:
                summary["metrics"][metric.value] = {
                    "average": np.mean(values),
                    "trend": self.get_trend(metric),
                    "latest": values[-1] if values else None,
                }

        return summary

    def _get_status(self, kpi_score: float) -> str:
        """Get performance status based on KPI score."""
        if kpi_score >= self.thresholds.excellent:
            return "excellent"
        if kpi_score >= self.thresholds.good:
            return "good"
        if kpi_score >= self.thresholds.warning:
            return "warning"
        return "critical"


class AgentPerformanceManager:
    """Manages performance tracking for all agents."""

    def __init__(self, data_dir: str = "agent_performance_data") -> None:
        """Initialize performance manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.trackers: dict[str, AgentKPITracker] = {}
        self.load_existing_data()

    def register_agent(self, agent_id: str, role: str) -> None:
        """Register a new agent for performance tracking."""
        if agent_id not in self.trackers:
            self.trackers[agent_id] = AgentKPITracker(agent_id, role)
            print(f"Registered agent {agent_id} with role {role}")

    def record_task_performance(
        self, agent_id: str, task_id: str, performance_data: dict[str, float]
    ) -> None:
        """Record performance for a completed task."""
        if agent_id not in self.trackers:
            raise ValueError(f"Agent {agent_id} not registered")

        # Convert performance data to KPI metrics
        metrics = {}

        # Map raw performance data to KPI metrics
        if "completion_time" in performance_data:
            # Normalize to 0-1 scale (assuming 10s is target)
            metrics[KPIMetric.AVERAGE_RESPONSE_TIME] = min(
                1.0, 10.0 / performance_data["completion_time"]
            )

        if "success" in performance_data:
            metrics[KPIMetric.TASK_COMPLETION_RATE] = float(performance_data["success"])

        if "quality_score" in performance_data:
            metrics[KPIMetric.OUTPUT_QUALITY_SCORE] = performance_data["quality_score"]

        if "errors" in performance_data:
            metrics[KPIMetric.ERROR_RATE] = 1.0 - min(
                1.0, performance_data["errors"] / 10.0
            )

        if "resource_usage" in performance_data:
            # Normalize resource usage (lower is better)
            metrics[KPIMetric.RESOURCE_EFFICIENCY] = 1.0 - min(
                1.0, performance_data["resource_usage"]
            )

        if "user_satisfaction" in performance_data:
            metrics[KPIMetric.USER_SATISFACTION] = performance_data["user_satisfaction"]

        if "cooperation_score" in performance_data:
            metrics[KPIMetric.INTER_AGENT_COOPERATION] = performance_data[
                "cooperation_score"
            ]

        if "communication_clarity" in performance_data:
            metrics[KPIMetric.COMMUNICATION_CLARITY] = performance_data[
                "communication_clarity"
            ]

        self.trackers[agent_id].record_performance(task_id, metrics, performance_data)

    def evaluate_all_agents(self) -> list[tuple[str, bool, str]]:
        """Evaluate all agents and return retirement recommendations."""
        recommendations = []

        for agent_id, tracker in self.trackers.items():
            should_retire, reason = tracker.should_retire()
            recommendations.append((agent_id, should_retire, reason))

        return recommendations

    def get_leaderboard(self, metric: KPIMetric | None = None) -> pd.DataFrame:
        """Get agent leaderboard sorted by performance."""
        data = []

        for agent_id, tracker in self.trackers.items():
            summary = tracker.get_performance_summary()

            if metric:
                # Sort by specific metric
                metric_data = summary["metrics"].get(metric.value, {})
                score = metric_data.get("average", 0.0)
            else:
                # Sort by overall KPI
                score = summary["overall_kpi"]

            data.append(
                {
                    "agent_id": agent_id,
                    "role": summary["role"],
                    "score": score,
                    "status": summary["performance_status"],
                    "total_tasks": summary["total_tasks"],
                }
            )

        df = pd.DataFrame(data)
        return df.sort_values("score", ascending=False)

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report = "# Agent Performance Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Overall statistics
        report += "## Overall Statistics\n\n"
        report += f"- Total agents: {len(self.trackers)}\n"

        status_counts = {}
        for tracker in self.trackers.values():
            status = tracker._get_status(tracker.calculate_overall_kpi())
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in status_counts.items():
            report += f"- {status.title()}: {count}\n"

        # Leaderboard
        report += "\n## Overall Leaderboard\n\n"
        leaderboard = self.get_leaderboard()
        if not leaderboard.empty:
            report += leaderboard.to_string(index=False)
        else:
            report += "No agents registered yet.\n"

        # Agents needing attention
        report += "\n\n## Agents Needing Attention\n\n"
        recommendations = self.evaluate_all_agents()

        retirement_needed = False
        for agent_id, should_retire, reason in recommendations:
            if should_retire:
                report += f"- **{agent_id}**: {reason}\n"
                retirement_needed = True

        if not retirement_needed:
            report += "No agents currently need retirement.\n"

        # Per-metric analysis
        report += "\n## Per-Metric Analysis\n\n"
        for metric in KPIMetric:
            report += f"### {metric.value.replace('_', ' ').title()}\n\n"
            metric_leaderboard = self.get_leaderboard(metric)
            if not metric_leaderboard.empty:
                report += metric_leaderboard.head(5).to_string(index=False)
                report += "\n\n"
            else:
                report += "No data available.\n\n"

        return report

    def save_data(self) -> None:
        """Save performance data to disk."""
        for agent_id, tracker in self.trackers.items():
            data = {
                "agent_id": agent_id,
                "role": tracker.role,
                "history": [record.to_dict() for record in tracker.performance_history],
            }

            file_path = self.data_dir / f"{agent_id}_performance.json"
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

    def load_existing_data(self) -> None:
        """Load existing performance data from disk."""
        for file_path in self.data_dir.glob("*_performance.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                agent_id = data["agent_id"]
                tracker = AgentKPITracker(agent_id, data["role"])

                # Reconstruct performance history
                for record_data in data["history"]:
                    record = PerformanceRecord(
                        timestamp=datetime.fromisoformat(record_data["timestamp"]),
                        task_id=record_data["task_id"],
                        metrics={
                            KPIMetric(k): v for k, v in record_data["metrics"].items()
                        },
                        context=record_data.get("context", {}),
                    )
                    tracker.performance_history.append(record)

                self.trackers[agent_id] = tracker
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")


def test_kpi_system() -> None:
    """Test the KPI system with simulated data."""
    print("Testing Agent KPI System...")

    # Initialize performance manager
    manager = AgentPerformanceManager()

    # Register agents
    agents = [
        ("king_001", "king"),
        ("magi_001", "magi"),
        ("sage_001", "sage"),
        ("polyglot_001", "polyglot"),
        ("magi_002", "magi"),
        ("medic_001", "medic"),
        ("oracle_001", "oracle"),
        ("sustainer_001", "sustainer"),
    ]

    for agent_id, role in agents:
        manager.register_agent(agent_id, role)

    # Simulate performance data
    print("\n=== Simulating Performance Data ===")

    for day in range(30):  # 30 days of data
        for agent_id, role in agents:
            # Simulate 5-10 tasks per day
            for task_num in range(random.randint(5, 10)):
                task_id = f"task_{day}_{task_num}"

                # Generate performance based on agent "personality"
                if agent_id == "king_001":  # High performer
                    base_performance = 0.8
                elif agent_id == "magi_002":  # Declining performer
                    base_performance = 0.7 - (day * 0.01)  # Declining over time
                elif agent_id == "medic_001":  # Consistent high performer
                    base_performance = 0.85
                else:  # Average performers
                    base_performance = 0.6

                performance_data = {
                    "completion_time": random.uniform(5, 15) / base_performance,
                    "success": random.random() < base_performance,
                    "quality_score": min(
                        1.0, base_performance + random.uniform(-0.1, 0.1)
                    ),
                    "errors": (
                        random.randint(0, 5)
                        if random.random() > base_performance
                        else 0
                    ),
                    "resource_usage": random.uniform(0.3, 0.9),
                    "user_satisfaction": min(
                        1.0, base_performance + random.uniform(-0.15, 0.1)
                    ),
                    "cooperation_score": random.uniform(0.4, 1.0),
                    "communication_clarity": min(
                        1.0, base_performance + random.uniform(-0.1, 0.1)
                    ),
                }

                manager.record_task_performance(agent_id, task_id, performance_data)

    # Generate performance report
    print("\n=== Performance Report ===")
    report = manager.generate_performance_report()
    print(report)

    # Save report to file
    with open("agent_performance_report.md", "w") as f:
        f.write(report)

    # Check retirement recommendations
    print("\n=== Retirement Recommendations ===")
    recommendations = manager.evaluate_all_agents()
    for agent_id, should_retire, reason in recommendations:
        if should_retire:
            print(f"❌ {agent_id}: {reason}")
        else:
            print(f"✅ {agent_id}: {reason}")

    # Save data
    manager.save_data()
    print("\n✓ Performance data saved")

    print("\n✅ KPI system testing complete!")


if __name__ == "__main__":
    test_kpi_system()
