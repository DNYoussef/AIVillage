"""
Performance Monitor for Phase 4 Validation

Real-time performance monitoring and metrics collection.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics
import json
from pathlib import Path


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot"""

    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    active_threads: int
    open_files: int


@dataclass
class SystemResourceUsage:
    """System resource usage metrics"""

    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    disk_usage: Dict[str, List[float]] = field(default_factory=dict)
    network_usage: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class ProcessMetrics:
    """Process-specific metrics"""

    pid: int
    name: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    num_threads: int
    status: str
    create_time: float


class PerformanceMonitor:
    """
    Real-time performance monitoring for Phase 4 validation
    """

    def __init__(self, max_snapshots: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_snapshots = max_snapshots

        # Performance data storage
        self.snapshots = deque(maxlen=max_snapshots)
        self.process_metrics = {}

        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds

        # Alert thresholds
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 90.0,
            "disk_warning": 85.0,
            "disk_critical": 95.0,
        }

        # Performance alerts
        self.alerts = []
        self.alert_callbacks = []

        # Baseline metrics for comparison
        self.baseline_metrics = None

    async def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous performance monitoring"""
        if self.monitoring:
            self.logger.warning("Performance monitoring already running")
            return

        self.monitor_interval = interval
        self.monitoring = True

        # Start monitoring in background thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info(f"Performance monitoring started (interval: {interval}s)")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        self.logger.info("Performance monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)"""
        while self.monitoring:
            try:
                # Capture performance snapshot
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)

                # Check for alerts
                self._check_alerts(snapshot)

                # Update process metrics
                self._update_process_metrics()

                time.sleep(self.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)

    def _capture_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance snapshot"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0
        except Exception:
            disk_read = disk_write = 0

        # Network I/O
        try:
            network_io = psutil.net_io_counters()
            network_sent = network_io.bytes_sent if network_io else 0
            network_recv = network_io.bytes_recv if network_io else 0
        except Exception:
            network_sent = network_recv = 0

        # Process info
        try:
            process = psutil.Process()
            active_threads = process.num_threads()
            open_files = len(process.open_files())
        except Exception:
            active_threads = 0
            open_files = 0

        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory.used / 1024 / 1024,
            memory_percent=memory.percent,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_io_sent=network_sent,
            network_io_recv=network_recv,
            active_threads=active_threads,
            open_files=open_files,
        )

    def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for performance alerts"""
        alerts_triggered = []

        # CPU alerts
        if snapshot.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts_triggered.append(
                {
                    "type": "cpu",
                    "level": "critical",
                    "value": snapshot.cpu_percent,
                    "threshold": self.thresholds["cpu_critical"],
                    "timestamp": snapshot.timestamp,
                }
            )
        elif snapshot.cpu_percent >= self.thresholds["cpu_warning"]:
            alerts_triggered.append(
                {
                    "type": "cpu",
                    "level": "warning",
                    "value": snapshot.cpu_percent,
                    "threshold": self.thresholds["cpu_warning"],
                    "timestamp": snapshot.timestamp,
                }
            )

        # Memory alerts
        if snapshot.memory_percent >= self.thresholds["memory_critical"]:
            alerts_triggered.append(
                {
                    "type": "memory",
                    "level": "critical",
                    "value": snapshot.memory_percent,
                    "threshold": self.thresholds["memory_critical"],
                    "timestamp": snapshot.timestamp,
                }
            )
        elif snapshot.memory_percent >= self.thresholds["memory_warning"]:
            alerts_triggered.append(
                {
                    "type": "memory",
                    "level": "warning",
                    "value": snapshot.memory_percent,
                    "threshold": self.thresholds["memory_warning"],
                    "timestamp": snapshot.timestamp,
                }
            )

        # Store alerts and trigger callbacks
        for alert in alerts_triggered:
            self.alerts.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")

    def _update_process_metrics(self) -> None:
        """Update process-specific metrics"""
        try:
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_info", "memory_percent", "num_threads", "status", "create_time"]
            ):
                try:
                    pinfo = proc.info
                    if pinfo["name"] and "python" in pinfo["name"].lower():  # Focus on Python processes
                        metrics = ProcessMetrics(
                            pid=pinfo["pid"],
                            name=pinfo["name"],
                            cpu_percent=pinfo["cpu_percent"] or 0,
                            memory_mb=pinfo["memory_info"].rss / 1024 / 1024 if pinfo["memory_info"] else 0,
                            memory_percent=pinfo["memory_percent"] or 0,
                            num_threads=pinfo["num_threads"] or 0,
                            status=pinfo["status"],
                            create_time=pinfo["create_time"] or 0,
                        )
                        self.process_metrics[pinfo["pid"]] = metrics

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            self.logger.debug(f"Error updating process metrics: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.snapshots:
            return {"error": "No performance data available"}

        latest = self.snapshots[-1]

        # Calculate recent trends (last 10 snapshots)
        recent_snapshots = list(self.snapshots)[-10:]

        cpu_trend = [s.cpu_percent for s in recent_snapshots]
        memory_trend = [s.memory_percent for s in recent_snapshots]

        return {
            "current": {
                "timestamp": latest.timestamp,
                "cpu_percent": latest.cpu_percent,
                "memory_mb": latest.memory_mb,
                "memory_percent": latest.memory_percent,
                "active_threads": latest.active_threads,
                "open_files": latest.open_files,
            },
            "trends": {
                "cpu_avg": statistics.mean(cpu_trend) if cpu_trend else 0,
                "cpu_max": max(cpu_trend) if cpu_trend else 0,
                "memory_avg": statistics.mean(memory_trend) if memory_trend else 0,
                "memory_max": max(memory_trend) if memory_trend else 0,
            },
            "alerts": {
                "total_alerts": len(self.alerts),
                "recent_alerts": [a for a in self.alerts if time.time() - a["timestamp"] < 300],  # Last 5 minutes
            },
        }

    def get_performance_summary(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary for specified duration"""
        if not self.snapshots:
            return {"error": "No performance data available"}

        # Filter snapshots by duration
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {"error": f"No data available for last {duration_minutes} minutes"}

        # Calculate statistics
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        memory_mb_values = [s.memory_mb for s in recent_snapshots]
        thread_values = [s.active_threads for s in recent_snapshots]

        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_snapshots),
            "cpu": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": statistics.mean(cpu_values),
                "median": statistics.median(cpu_values),
            },
            "memory_percent": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": statistics.mean(memory_values),
                "median": statistics.median(memory_values),
            },
            "memory_mb": {
                "min": min(memory_mb_values),
                "max": max(memory_mb_values),
                "avg": statistics.mean(memory_mb_values),
                "median": statistics.median(memory_mb_values),
            },
            "threads": {"min": min(thread_values), "max": max(thread_values), "avg": statistics.mean(thread_values)},
            "alerts_in_period": len([a for a in self.alerts if a["timestamp"] >= cutoff_time]),
        }

    def detect_performance_issues(self) -> Dict[str, Any]:
        """Detect potential performance issues"""
        issues = []
        recommendations = []

        if not self.snapshots:
            return {"issues": issues, "recommendations": recommendations}

        # Analyze recent performance
        recent = list(self.snapshots)[-60:]  # Last 60 snapshots

        if len(recent) < 10:
            return {"issues": issues, "recommendations": ["Need more monitoring data for analysis"]}

        cpu_values = [s.cpu_percent for s in recent]
        memory_values = [s.memory_percent for s in recent]
        memory_mb_values = [s.memory_mb for s in recent]

        # High CPU usage
        avg_cpu = statistics.mean(cpu_values)
        max_cpu = max(cpu_values)

        if avg_cpu > 70:
            issues.append(
                {
                    "type": "high_cpu_usage",
                    "severity": "high" if avg_cpu > 85 else "medium",
                    "description": f"Average CPU usage is {avg_cpu:.1f}%",
                    "value": avg_cpu,
                }
            )
            recommendations.append("Consider optimizing CPU-intensive operations")

        if max_cpu > 95:
            issues.append(
                {
                    "type": "cpu_spikes",
                    "severity": "high",
                    "description": f"CPU usage peaked at {max_cpu:.1f}%",
                    "value": max_cpu,
                }
            )
            recommendations.append("Investigate CPU spikes and consider load balancing")

        # High memory usage
        avg_memory = statistics.mean(memory_values)
        memory_trend = self._calculate_trend(memory_mb_values)

        if avg_memory > 75:
            issues.append(
                {
                    "type": "high_memory_usage",
                    "severity": "high" if avg_memory > 90 else "medium",
                    "description": f"Average memory usage is {avg_memory:.1f}%",
                    "value": avg_memory,
                }
            )
            recommendations.append("Monitor memory usage and check for memory leaks")

        # Memory leak detection
        if memory_trend > 1.0:  # Increasing trend
            issues.append(
                {
                    "type": "memory_leak_suspected",
                    "severity": "high",
                    "description": f"Memory usage trending upward ({memory_trend:.2f} MB/min)",
                    "value": memory_trend,
                }
            )
            recommendations.append("Investigate potential memory leaks")

        # Thread count issues
        thread_values = [s.active_threads for s in recent]
        statistics.mean(thread_values)
        max_threads = max(thread_values)

        if max_threads > 100:
            issues.append(
                {
                    "type": "high_thread_count",
                    "severity": "medium",
                    "description": f"Maximum thread count reached {max_threads}",
                    "value": max_threads,
                }
            )
            recommendations.append("Review thread usage and consider connection pooling")

        return {
            "issues": issues,
            "recommendations": recommendations,
            "analysis_period_minutes": len(recent) * self.monitor_interval / 60,
        }

    def _calculate_trend(self, values: List[float], window_size: int = 10) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)"""
        if len(values) < window_size:
            return 0.0

        # Use linear regression to calculate trend
        recent_values = values[-window_size:]
        x_values = list(range(len(recent_values)))

        # Simple linear regression
        n = len(recent_values)
        sum_x = sum(x_values)
        sum_y = sum(recent_values)
        sum_xy = sum(x * y for x, y in zip(x_values, recent_values))
        sum_x_squared = sum(x * x for x in x_values)

        # Calculate slope (trend)
        denominator = n * sum_x_squared - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Convert to per-minute rate
        return slope * (60 / self.monitor_interval)

    def set_baseline(self) -> None:
        """Set current performance as baseline for comparisons"""
        if not self.snapshots:
            self.logger.warning("No performance data available to set as baseline")
            return

        # Use last 10 snapshots as baseline
        recent = list(self.snapshots)[-10:]

        self.baseline_metrics = {
            "cpu_avg": statistics.mean(s.cpu_percent for s in recent),
            "memory_avg": statistics.mean(s.memory_percent for s in recent),
            "memory_mb_avg": statistics.mean(s.memory_mb for s in recent),
            "threads_avg": statistics.mean(s.active_threads for s in recent),
            "timestamp": time.time(),
        }

        self.logger.info("Performance baseline set")

    def compare_to_baseline(self) -> Dict[str, Any]:
        """Compare current performance to baseline"""
        if not self.baseline_metrics:
            return {"error": "No baseline metrics set"}

        if not self.snapshots:
            return {"error": "No current performance data"}

        # Get current metrics (last 10 snapshots)
        recent = list(self.snapshots)[-10:]

        current_metrics = {
            "cpu_avg": statistics.mean(s.cpu_percent for s in recent),
            "memory_avg": statistics.mean(s.memory_percent for s in recent),
            "memory_mb_avg": statistics.mean(s.memory_mb for s in recent),
            "threads_avg": statistics.mean(s.active_threads for s in recent),
        }

        # Calculate differences
        comparison = {}
        for metric in current_metrics:
            baseline_value = self.baseline_metrics[metric]
            current_value = current_metrics[metric]

            difference = current_value - baseline_value
            percentage_change = (difference / baseline_value * 100) if baseline_value != 0 else 0

            comparison[metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "difference": difference,
                "percentage_change": percentage_change,
            }

        return {
            "baseline_timestamp": self.baseline_metrics["timestamp"],
            "comparison_timestamp": time.time(),
            "metrics": comparison,
            "overall_impact": self._calculate_overall_impact(comparison),
        }

    def _calculate_overall_impact(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance impact"""
        # Weight different metrics
        weights = {"cpu_avg": 0.4, "memory_avg": 0.3, "memory_mb_avg": 0.2, "threads_avg": 0.1}

        weighted_changes = []
        for metric, weight in weights.items():
            if metric in comparison:
                change = comparison[metric]["percentage_change"]
                weighted_changes.append(change * weight)

        overall_change = sum(weighted_changes)

        # Classify impact
        if overall_change > 10:
            impact_level = "significant_degradation"
        elif overall_change > 5:
            impact_level = "moderate_degradation"
        elif overall_change < -5:
            impact_level = "improvement"
        else:
            impact_level = "minimal_impact"

        return {
            "overall_percentage_change": overall_change,
            "impact_level": impact_level,
            "description": self._get_impact_description(impact_level, overall_change),
        }

    def _get_impact_description(self, impact_level: str, change: float) -> str:
        """Get description for impact level"""
        descriptions = {
            "significant_degradation": f"Performance significantly degraded ({change:+.1f}%)",
            "moderate_degradation": f"Performance moderately degraded ({change:+.1f}%)",
            "improvement": f"Performance improved ({change:+.1f}%)",
            "minimal_impact": f"Minimal performance impact ({change:+.1f}%)",
        }
        return descriptions.get(impact_level, f"Unknown impact ({change:+.1f}%)")

    def export_metrics(self, file_path: Path) -> None:
        """Export performance metrics to JSON file"""
        export_data = {
            "export_timestamp": time.time(),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "cpu_percent": s.cpu_percent,
                    "memory_mb": s.memory_mb,
                    "memory_percent": s.memory_percent,
                    "active_threads": s.active_threads,
                    "open_files": s.open_files,
                }
                for s in self.snapshots
            ],
            "alerts": self.alerts,
            "baseline_metrics": self.baseline_metrics,
            "monitoring_config": {
                "interval": self.monitor_interval,
                "max_snapshots": self.max_snapshots,
                "thresholds": self.thresholds,
            },
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Performance metrics exported to {file_path}")

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback function for performance alerts"""
        self.alert_callbacks.append(callback)

    def clear_alerts(self) -> None:
        """Clear all stored alerts"""
        self.alerts.clear()
        self.logger.info("Performance alerts cleared")
