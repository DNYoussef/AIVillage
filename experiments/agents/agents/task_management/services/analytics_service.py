"""
Analytics Service - Handles performance tracking and reporting.
Extracted from UnifiedManagement god class.
"""

import logging
from typing import Any
from datetime import datetime

from AIVillage.experimental.agents.agents.analytics.unified_analytics import UnifiedAnalytics
from core.error_handling import AIVillageException

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service responsible for analytics, metrics tracking, and reporting."""

    def __init__(self, unified_analytics: UnifiedAnalytics) -> None:
        """Initialize with dependencies."""
        self._unified_analytics = unified_analytics
        self._task_metrics: dict[str, dict[str, Any]] = {}
        self._system_metrics: dict[str, Any] = {}

    async def record_task_completion(self, task_id: str, completion_time: float, success: bool) -> None:
        """Record task completion metrics."""
        try:
            # Record in unified analytics
            self._unified_analytics.record_task_completion(task_id, completion_time, success)

            # Store local metrics
            self._task_metrics[task_id] = {
                "completion_time": completion_time,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }

            logger.debug("Recorded task completion: %s (success: %s, time: %fs)", task_id, success, completion_time)

        except Exception as e:
            logger.exception("Error recording task completion: %s", e)
            msg = f"Error recording task completion: {e!s}"
            raise AIVillageException(msg) from e

    async def record_system_metric(self, metric_name: str, value: Any) -> None:
        """Record a system-level metric."""
        try:
            self._system_metrics[metric_name] = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }

            logger.debug("Recorded system metric: %s = %s", metric_name, value)

        except Exception as e:
            logger.exception("Error recording system metric: %s", e)
            msg = f"Error recording system metric: {e!s}"
            raise AIVillageException(msg) from e

    async def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Get unified analytics report
            unified_report = self._unified_analytics.generate_summary_report()

            # Calculate additional metrics
            task_success_rate = self._calculate_task_success_rate()
            average_completion_time = self._calculate_average_completion_time()
            throughput_metrics = self._calculate_throughput_metrics()

            report = {
                "unified_analytics": unified_report,
                "task_metrics": {
                    "total_tasks": len(self._task_metrics),
                    "success_rate": task_success_rate,
                    "average_completion_time": average_completion_time,
                    "throughput": throughput_metrics,
                },
                "system_metrics": self._system_metrics,
                "report_generated_at": datetime.now().isoformat(),
            }

            logger.info("Generated performance report with %d tasks", len(self._task_metrics))
            return report

        except Exception as e:
            logger.exception("Error generating performance report: %s", e)
            msg = f"Error generating performance report: {e!s}"
            raise AIVillageException(msg) from e

    def _calculate_task_success_rate(self) -> float:
        """Calculate overall task success rate."""
        if not self._task_metrics:
            return 0.0

        successful_tasks = sum(1 for metrics in self._task_metrics.values() if metrics.get("success", False))

        return successful_tasks / len(self._task_metrics)

    def _calculate_average_completion_time(self) -> float:
        """Calculate average task completion time."""
        if not self._task_metrics:
            return 0.0

        total_time = sum(metrics.get("completion_time", 0) for metrics in self._task_metrics.values())

        return total_time / len(self._task_metrics)

    def _calculate_throughput_metrics(self) -> dict[str, float]:
        """Calculate throughput metrics."""
        if not self._task_metrics:
            return {"tasks_per_hour": 0.0, "tasks_per_minute": 0.0}

        # Simple approximation - in real implementation,
        # we'd track timing more precisely
        total_tasks = len(self._task_metrics)

        return {
            "tasks_per_hour": total_tasks,  # Simplified
            "tasks_per_minute": total_tasks / 60,
        }

    async def get_task_metrics(self, task_id: str) -> dict[str, Any] | None:
        """Get metrics for a specific task."""
        return self._task_metrics.get(task_id)

    async def get_all_task_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all tasks."""
        return self._task_metrics.copy()

    async def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        try:
            self._task_metrics.clear()
            self._system_metrics.clear()
            logger.info("Cleared all metrics")
        except Exception as e:
            logger.exception("Error clearing metrics: %s", e)
            msg = f"Error clearing metrics: {e!s}"
            raise AIVillageException(msg) from e

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a quick summary of current metrics."""
        try:
            return {
                "total_task_metrics": len(self._task_metrics),
                "total_system_metrics": len(self._system_metrics),
                "success_rate": self._calculate_task_success_rate(),
                "average_completion_time": self._calculate_average_completion_time(),
            }
        except Exception as e:
            logger.exception("Error getting metrics summary: %s", e)
            return {"error": str(e)}
