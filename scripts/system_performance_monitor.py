#!/usr/bin/env python3
"""System Performance Monitor for AIVillage.

Comprehensive performance monitoring across all system components including:
- Memory usage tracking
- CPU utilization monitoring
- I/O performance metrics
- Process-specific monitoring
- Automated alerting and reporting

Usage:
    python system_performance_monitor.py [--interval SECONDS] [--duration MINUTES]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    print("Error: 'psutil' package required. Install with: pip install psutil")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("system_monitor.log"),
    ],
)
logger = logging.getLogger(__name__)


class SystemMetrics:
    """Container for system performance metrics."""

    def __init__(self) -> None:
        """Initialize system metrics."""
        self.timestamp: str = datetime.now().isoformat()
        self.cpu_percent: float = 0.0
        self.memory_percent: float = 0.0
        self.memory_available_gb: float = 0.0
        self.memory_used_gb: float = 0.0
        self.disk_usage_percent: float = 0.0
        self.disk_io_read_mb: float = 0.0
        self.disk_io_write_mb: float = 0.0
        self.network_sent_mb: float = 0.0
        self.network_recv_mb: float = 0.0
        self.load_average: list[float] = []
        self.process_count: int = 0
        self.python_processes: list[dict[str, Any]] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the metrics
        """
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_gb": self.memory_available_gb,
            "memory_used_gb": self.memory_used_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "network_sent_mb": self.network_sent_mb,
            "network_recv_mb": self.network_recv_mb,
            "load_average": self.load_average,
            "process_count": self.process_count,
            "python_processes": self.python_processes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary containing metric data

        Returns:
            SystemMetrics instance
        """
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics


class PerformanceMonitor:
    """System performance monitor."""

    def __init__(
        self, interval: float = 5.0, data_file: str = "system_metrics.json"
    ) -> None:
        """Initialize the performance monitor.

        Args:
            interval: Monitoring interval in seconds
            data_file: File to store metrics history
        """
        self.interval = interval
        self.data_file = Path(data_file)
        self.metrics_history: list[SystemMetrics] = []
        self.is_monitoring = False

        # Load existing metrics
        self.load_history()

        # Performance thresholds for alerting
        self.thresholds = {
            "cpu_percent_high": 80.0,
            "memory_percent_high": 85.0,
            "disk_usage_high": 90.0,
            "load_average_high": 8.0,
        }

    def load_history(self) -> None:
        """Load metrics history from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics_history = [
                        SystemMetrics.from_dict(item) for item in data
                    ]
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
            except Exception as e:
                logger.exception(f"Failed to load metrics history: {e}")
                self.metrics_history = []

    def save_history(self) -> None:
        """Save metrics history to file."""
        try:
            # Keep only recent metrics to prevent file growth
            recent_metrics = self.metrics_history[-1000:]
            data = [metrics.to_dict() for metrics in recent_metrics]

            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(recent_metrics)} metrics to {self.data_file}")
        except Exception as e:
            logger.exception(f"Failed to save metrics history: {e}")

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics.

        Returns:
            SystemMetrics object with current system state
        """
        metrics = SystemMetrics()

        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            metrics.memory_used_gb = memory.used / (1024**3)

            # Disk metrics
            disk_usage = psutil.disk_usage("/")
            metrics.disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.disk_io_read_mb = disk_io.read_bytes / (1024**2)
                metrics.disk_io_write_mb = disk_io.write_bytes / (1024**2)

            # Network metrics
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.network_sent_mb = network_io.bytes_sent / (1024**2)
                metrics.network_recv_mb = network_io.bytes_recv / (1024**2)

            # Load average (Unix-like systems)
            try:
                metrics.load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                # Not available on Windows
                metrics.load_average = []

            # Process metrics
            metrics.process_count = len(psutil.pids())

            # Python process details
            python_processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    if "python" in proc.info["name"].lower():
                        python_processes.append(
                            {
                                "pid": proc.info["pid"],
                                "name": proc.info["name"],
                                "cpu_percent": proc.info["cpu_percent"] or 0.0,
                                "memory_percent": proc.info["memory_percent"] or 0.0,
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            metrics.python_processes = python_processes[:10]  # Top 10 Python processes

        except Exception as e:
            logger.exception(f"Failed to collect system metrics: {e}")

        return metrics

    def check_alerts(self, metrics: SystemMetrics) -> list[str]:
        """Check for performance alerts.

        Args:
            metrics: Current system metrics

        Returns:
            List of alert messages
        """
        alerts = []

        # CPU alert
        if metrics.cpu_percent > self.thresholds["cpu_percent_high"]:
            alerts.append(
                f"High CPU usage: {metrics.cpu_percent:.1f}% "
                f"(threshold: {self.thresholds['cpu_percent_high']:.1f}%)"
            )

        # Memory alert
        if metrics.memory_percent > self.thresholds["memory_percent_high"]:
            alerts.append(
                f"High memory usage: {metrics.memory_percent:.1f}% "
                f"(threshold: {self.thresholds['memory_percent_high']:.1f}%)"
            )

        # Disk alert
        if metrics.disk_usage_percent > self.thresholds["disk_usage_high"]:
            alerts.append(
                f"High disk usage: {metrics.disk_usage_percent:.1f}% "
                f"(threshold: {self.thresholds['disk_usage_high']:.1f}%)"
            )

        # Load average alert (if available)
        if metrics.load_average and len(metrics.load_average) > 0:
            load_1min = metrics.load_average[0]
            if load_1min > self.thresholds["load_average_high"]:
                alerts.append(
                    f"High load average: {load_1min:.2f} "
                    f"(threshold: {self.thresholds['load_average_high']:.2f})"
                )

        return alerts

    def monitor(self, duration_minutes: float | None = None) -> None:
        """Start monitoring system performance.

        Args:
            duration_minutes: How long to monitor (None for indefinite)
        """
        logger.info(
            f"Starting system monitoring (interval: {self.interval}s, "
            f"duration: {duration_minutes or 'indefinite'} minutes)"
        )

        self.is_monitoring = True
        start_time = time.time()

        try:
            while self.is_monitoring:
                # Collect metrics
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)

                # Check for alerts
                alerts = self.check_alerts(metrics)
                if alerts:
                    logger.warning("Performance alerts:")
                    for alert in alerts:
                        logger.warning(f"  - {alert}")

                # Log current status
                logger.info(
                    f"CPU: {metrics.cpu_percent:.1f}%, "
                    f"Memory: {metrics.memory_percent:.1f}%, "
                    f"Disk: {metrics.disk_usage_percent:.1f}%, "
                    f"Processes: {metrics.process_count}"
                )

                # Save metrics periodically
                if len(self.metrics_history) % 10 == 0:
                    self.save_history()

                # Check duration
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        logger.info(
                            f"Monitoring completed ({elapsed_minutes:.1f} minutes)"
                        )
                        break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self.is_monitoring = False
            self.save_history()

    def generate_report(self, hours: int = 24) -> str:
        """Generate performance report.

        Args:
            hours: Number of hours to include in report

        Returns:
            Formatted report string
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        if not recent_metrics:
            return "No metrics available for the specified time period."

        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]

        report = f"""
# System Performance Report ({hours} hours)

## Summary Statistics
- **Total Samples**: {len(recent_metrics)}
- **Time Range**: {recent_metrics[0].timestamp} to {recent_metrics[-1].timestamp}

## CPU Performance
- **Average CPU Usage**: {sum(cpu_values) / len(cpu_values):.1f}%
- **Peak CPU Usage**: {max(cpu_values):.1f}%
- **Minimum CPU Usage**: {min(cpu_values):.1f}%

## Memory Performance
- **Average Memory Usage**: {sum(memory_values) / len(memory_values):.1f}%
- **Peak Memory Usage**: {max(memory_values):.1f}%
- **Available Memory (Latest)**: {recent_metrics[-1].memory_available_gb:.1f} GB

## Current Status
- **Process Count**: {recent_metrics[-1].process_count}
- **Python Processes**: {len(recent_metrics[-1].python_processes)}
- **Disk Usage**: {recent_metrics[-1].disk_usage_percent:.1f}%

## Top Python Processes
"""

        # Add top Python processes
        if recent_metrics[-1].python_processes:
            for proc in recent_metrics[-1].python_processes[:5]:
                report += f"- PID {proc['pid']}: {proc['name']} "
                report += f"(CPU: {proc['cpu_percent']:.1f}%, Memory: {proc['memory_percent']:.1f}%)\n"
        else:
            report += "No Python processes found\n"

        return report

    def cleanup_old_metrics(self, days_to_keep: int = 30) -> int:
        """Clean up old metrics to prevent unbounded growth.

        Args:
            days_to_keep: Number of days of metrics to retain

        Returns:
            Number of metrics removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        original_count = len(self.metrics_history)

        self.metrics_history = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_date
        ]

        removed_count = original_count - len(self.metrics_history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old metrics")
            self.save_history()

        return removed_count


def main() -> int:
    """Main monitoring function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="System Performance Monitor for AIVillage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python system_performance_monitor.py
  python system_performance_monitor.py --interval 10 --duration 60
  python system_performance_monitor.py --report 48
""",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Monitoring interval in seconds (default: 5.0)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Monitoring duration in minutes (default: indefinite)",
    )

    parser.add_argument(
        "--report",
        type=int,
        metavar="HOURS",
        help="Generate performance report for N hours",
    )

    parser.add_argument(
        "--data-file",
        default="system_metrics.json",
        help="Path to metrics data file",
    )

    args = parser.parse_args()

    try:
        monitor = PerformanceMonitor(interval=args.interval, data_file=args.data_file)

        if args.report:
            logger.info(f"Generating performance report for {args.report} hours...")
            report = monitor.generate_report(hours=args.report)

            # Save report
            report_file = f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)

            print(report)
            logger.info(f"Report saved to {report_file}")
        else:
            # Start monitoring
            monitor.monitor(duration_minutes=args.duration)

        # Cleanup old metrics
        monitor.cleanup_old_metrics()

        return 0

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
