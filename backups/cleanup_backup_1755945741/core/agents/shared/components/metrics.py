"""Agent Metrics Component.

Handles performance monitoring, memory management, and analytics.
Provides centralized metrics collection and reporting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any, Deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""

    PERFORMANCE = "performance"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    TASK = "task"
    HEALTH = "health"
    CUSTOM = "custom"


@dataclass
class MetricEntry:
    """Single metric measurement entry."""

    timestamp: datetime
    metric_type: MetricType
    metric_name: str
    value: float
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""

    timestamp: datetime
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    task_completion_rate: float = 0.0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0


@dataclass
class MemoryStats:
    """Memory and storage statistics."""

    journal_entries: int = 0
    memory_entries: int = 0
    geometric_states: int = 0
    task_history_size: int = 0
    total_memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    collection_interval_seconds: int = 60
    max_history_entries: int = 1000
    enable_performance_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_real_time_metrics: bool = True
    metric_retention_hours: int = 24


class AgentMetrics:
    """Centralized metrics collection and analysis for agents.

    This component handles all performance monitoring, memory tracking,
    and analytics in a focused, single-responsibility manner.
    """

    def __init__(self, agent_id: str, config: MetricsConfig | None = None):
        """Initialize metrics collection.

        Args:
            agent_id: Unique identifier for the agent
            config: Metrics configuration, uses defaults if None
        """
        self.agent_id = agent_id
        self.config = config or MetricsConfig()

        # Metric storage with efficient access patterns
        self._metrics: Deque[MetricEntry] = deque(maxlen=self.config.max_history_entries)
        self._performance_history: Deque[PerformanceSnapshot] = deque(maxlen=100)

        # Current aggregated metrics (CoI - Connascence of Identity for metric objects)
        self._current_performance = PerformanceSnapshot(timestamp=datetime.now())
        self._memory_stats = MemoryStats()

        # Counters for rate calculations
        self._task_counters = {"completed": 0, "failed": 0, "total_processing_time_ms": 0.0}

        # Real-time metrics cache
        self._metric_cache: dict[str, Any] = {}
        self._cache_updated = datetime.now()

        # Start time for uptime calculations
        self._start_time = datetime.now()

        logger.debug(f"Metrics component initialized for agent {agent_id}")

    def record_metric(
        self,
        metric_type: MetricType,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: list[str] | None = None,
        **metadata,
    ) -> None:
        """Record a metric measurement.

        Args:
            metric_type: Category of metric
            metric_name: Name/identifier for the metric
            value: Numeric value of the measurement
            unit: Unit of measurement (e.g., "ms", "bytes", "%")
            tags: Optional tags for filtering/grouping
            **metadata: Additional metadata for the metric
        """
        entry = MetricEntry(
            timestamp=datetime.now(),
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata,
            tags=tags or [],
        )

        self._metrics.append(entry)
        self._invalidate_cache()

        logger.debug(f"Recorded metric {metric_name}: {value} {unit}")

    def record_task_completion(
        self, task_id: str, processing_time_ms: float, success: bool = True, accuracy: float = 1.0, **metadata
    ) -> None:
        """Record task completion metrics.

        Args:
            task_id: Unique task identifier
            processing_time_ms: Time taken to complete task
            success: Whether task completed successfully
            accuracy: Task accuracy score (0.0 to 1.0)
            **metadata: Additional task metadata
        """
        # Update counters
        if success:
            self._task_counters["completed"] += 1
        else:
            self._task_counters["failed"] += 1

        self._task_counters["total_processing_time_ms"] += processing_time_ms

        # Record individual metrics
        self.record_metric(
            MetricType.TASK,
            "processing_time",
            processing_time_ms,
            "ms",
            tags=["task_completion"],
            task_id=task_id,
            success=success,
            **metadata,
        )

        if not success:
            self.record_metric(MetricType.TASK, "task_failure", 1.0, "count", tags=["task_failure"], task_id=task_id)

        self.record_metric(MetricType.TASK, "task_accuracy", accuracy, "score", tags=["accuracy"], task_id=task_id)

        logger.debug(f"Task metrics recorded for {task_id}: {processing_time_ms}ms, success={success}")

    def record_communication_event(
        self, event_type: str, recipient: str | None = None, latency_ms: float = 0.0, success: bool = True, **metadata
    ) -> None:
        """Record communication event metrics.

        Args:
            event_type: Type of communication (send, receive, broadcast)
            recipient: Target of communication (if applicable)
            latency_ms: Communication latency
            success: Whether communication succeeded
            **metadata: Additional event metadata
        """
        self.record_metric(
            MetricType.COMMUNICATION,
            f"communication_{event_type}",
            1.0,
            "count",
            tags=["communication", event_type],
            recipient=recipient,
            success=success,
            **metadata,
        )

        if latency_ms > 0:
            self.record_metric(
                MetricType.COMMUNICATION,
                "communication_latency",
                latency_ms,
                "ms",
                tags=["latency", event_type],
                recipient=recipient,
            )

    def update_memory_stats(
        self, journal_entries: int = 0, memory_entries: int = 0, geometric_states: int = 0, task_history_size: int = 0
    ) -> None:
        """Update memory and storage statistics.

        Args:
            journal_entries: Number of journal entries
            memory_entries: Number of memory entries
            geometric_states: Number of geometric state entries
            task_history_size: Size of task history
        """
        self._memory_stats.journal_entries = journal_entries
        self._memory_stats.memory_entries = memory_entries
        self._memory_stats.geometric_states = geometric_states
        self._memory_stats.task_history_size = task_history_size

        # Record as metrics for trending
        self.record_metric(MetricType.MEMORY, "journal_entries", journal_entries, "count")
        self.record_metric(MetricType.MEMORY, "memory_entries", memory_entries, "count")
        self.record_metric(MetricType.MEMORY, "geometric_states", geometric_states, "count")

    def update_performance_snapshot(
        self, cpu_utilization: float = 0.0, memory_utilization: float = 0.0, **kwargs
    ) -> None:
        """Update current performance snapshot.

        Args:
            cpu_utilization: Current CPU usage (0.0 to 1.0)
            memory_utilization: Current memory usage (0.0 to 1.0)
            **kwargs: Additional performance metrics
        """
        # Calculate derived metrics
        total_tasks = self._task_counters["completed"] + self._task_counters["failed"]
        error_rate = self._task_counters["failed"] / total_tasks if total_tasks > 0 else 0.0

        avg_response_time = self._task_counters["total_processing_time_ms"] / max(total_tasks, 1)

        # Calculate throughput (tasks per minute in recent period)
        recent_time = datetime.now() - timedelta(minutes=1)
        recent_tasks = [
            entry
            for entry in self._metrics
            if (
                entry.metric_type == MetricType.TASK
                and entry.metric_name == "processing_time"
                and entry.timestamp >= recent_time
            )
        ]
        throughput = len(recent_tasks)

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization,
            task_completion_rate=self._task_counters["completed"] / max(total_tasks, 1),
            average_response_time_ms=avg_response_time,
            error_rate=error_rate,
            throughput_per_minute=throughput,
            **kwargs,
        )

        self._current_performance = snapshot
        self._performance_history.append(snapshot)

        # Record as individual metrics
        self.record_metric(MetricType.PERFORMANCE, "cpu_utilization", cpu_utilization, "%")
        self.record_metric(MetricType.PERFORMANCE, "memory_utilization", memory_utilization, "%")
        self.record_metric(MetricType.PERFORMANCE, "error_rate", error_rate, "%")
        self.record_metric(MetricType.PERFORMANCE, "throughput", throughput, "tasks/min")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current aggregated metrics.

        Returns:
            Dict containing current performance and memory metrics
        """
        # Use cached metrics if recent
        cache_age = datetime.now() - self._cache_updated
        if cache_age.total_seconds() < 30 and self._metric_cache:
            return self._metric_cache.copy()

        # Calculate fresh metrics
        uptime = datetime.now() - self._start_time
        total_tasks = self._task_counters["completed"] + self._task_counters["failed"]

        metrics = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "performance": {
                "cpu_utilization": self._current_performance.cpu_utilization,
                "memory_utilization": self._current_performance.memory_utilization,
                "task_completion_rate": self._current_performance.task_completion_rate,
                "average_response_time_ms": self._current_performance.average_response_time_ms,
                "error_rate": self._current_performance.error_rate,
                "throughput_per_minute": self._current_performance.throughput_per_minute,
            },
            "task_statistics": {
                "total_tasks": total_tasks,
                "completed_tasks": self._task_counters["completed"],
                "failed_tasks": self._task_counters["failed"],
                "success_rate": (self._task_counters["completed"] / max(total_tasks, 1)),
            },
            "memory_statistics": {
                "journal_entries": self._memory_stats.journal_entries,
                "memory_entries": self._memory_stats.memory_entries,
                "geometric_states": self._memory_stats.geometric_states,
                "task_history_size": self._memory_stats.task_history_size,
                "total_memory_usage_mb": self._memory_stats.total_memory_usage_mb,
            },
            "metric_collection": {
                "total_metrics": len(self._metrics),
                "collection_enabled": self.config.enable_performance_monitoring,
                "retention_hours": self.config.metric_retention_hours,
            },
        }

        # Cache the results
        self._metric_cache = metrics
        self._cache_updated = datetime.now()

        return metrics

    def get_metric_history(
        self,
        metric_name: str | None = None,
        metric_type: MetricType | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[MetricEntry]:
        """Get historical metrics with filtering.

        Args:
            metric_name: Filter by specific metric name
            metric_type: Filter by metric type
            since: Return metrics since this timestamp
            limit: Maximum number of entries to return

        Returns:
            List of metric entries matching filters
        """
        filtered_metrics = []

        for entry in reversed(self._metrics):  # Most recent first
            # Apply filters
            if metric_name and entry.metric_name != metric_name:
                continue
            if metric_type and entry.metric_type != metric_type:
                continue
            if since and entry.timestamp < since:
                break  # Entries are ordered, so we can stop here

            filtered_metrics.append(entry)

            # Apply limit
            if limit and len(filtered_metrics) >= limit:
                break

        return filtered_metrics

    def get_performance_trends(self, hours: int = 1) -> dict[str, Any]:
        """Get performance trends over specified time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dict with trend analysis and statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [snapshot for snapshot in self._performance_history if snapshot.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {"error": "No performance data available for specified period"}

        # Calculate trends
        cpu_values = [s.cpu_utilization for s in recent_snapshots]
        memory_values = [s.memory_utilization for s in recent_snapshots]
        response_times = [s.average_response_time_ms for s in recent_snapshots]

        return {
            "period_hours": hours,
            "data_points": len(recent_snapshots),
            "cpu_utilization": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "trend": self._calculate_trend(cpu_values),
            },
            "memory_utilization": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "trend": self._calculate_trend(memory_values),
            },
            "response_time": {
                "current": response_times[-1] if response_times else 0,
                "average": sum(response_times) / len(response_times),
                "max": max(response_times),
                "min": min(response_times),
                "trend": self._calculate_trend(response_times),
            },
        }

    def cleanup_old_metrics(self) -> int:
        """Remove metrics older than retention period.

        Returns:
            Number of metrics removed
        """
        cutoff_time = datetime.now() - timedelta(hours=self.config.metric_retention_hours)
        initial_count = len(self._metrics)

        # Filter out old metrics
        self._metrics = deque(
            (entry for entry in self._metrics if entry.timestamp >= cutoff_time), maxlen=self.config.max_history_entries
        )

        removed_count = initial_count - len(self._metrics)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old metrics")
            self._invalidate_cache()

        return removed_count

    def export_metrics(self, format: str = "json") -> dict[str, Any] | str:
        """Export metrics in specified format.

        Args:
            format: Export format ("json", "csv", "summary")

        Returns:
            Exported metrics data
        """
        if format == "summary":
            return {
                "summary": self.get_current_metrics(),
                "trends": self.get_performance_trends(hours=24),
                "memory_stats": self._memory_stats,
                "export_timestamp": datetime.now().isoformat(),
            }
        elif format == "json":
            return {
                "metrics": [
                    {
                        "timestamp": entry.timestamp.isoformat(),
                        "type": entry.metric_type.value,
                        "name": entry.metric_name,
                        "value": entry.value,
                        "unit": entry.unit,
                        "tags": entry.tags,
                        "metadata": entry.metadata,
                    }
                    for entry in self._metrics
                ],
                "performance_history": [
                    {
                        "timestamp": snapshot.timestamp.isoformat(),
                        "cpu_utilization": snapshot.cpu_utilization,
                        "memory_utilization": snapshot.memory_utilization,
                        "task_completion_rate": snapshot.task_completion_rate,
                        "average_response_time_ms": snapshot.average_response_time_ms,
                        "error_rate": snapshot.error_rate,
                        "throughput_per_minute": snapshot.throughput_per_minute,
                    }
                    for snapshot in self._performance_history
                ],
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple trend calculation - compare first and last quartile
        q1_end = len(values) // 4
        q4_start = (3 * len(values)) // 4

        if q1_end >= q4_start:
            return "stable"

        q1_avg = sum(values[:q1_end]) / q1_end if q1_end > 0 else values[0]
        q4_avg = sum(values[q4_start:]) / (len(values) - q4_start)

        change_pct = ((q4_avg - q1_avg) / max(abs(q1_avg), 0.001)) * 100

        if change_pct > 10:
            return "increasing"
        elif change_pct < -10:
            return "decreasing"
        else:
            return "stable"

    def _invalidate_cache(self) -> None:
        """Invalidate the metrics cache to force recalculation."""
        self._metric_cache.clear()
        self._cache_updated = datetime.min
