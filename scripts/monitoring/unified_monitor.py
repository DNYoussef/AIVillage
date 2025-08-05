#!/usr/bin/env python3
"""Unified monitoring system for AIVillage.

This module consolidates functionality from:
- compression_monitor.py
- system_performance_monitor.py
- monitor_performance.py
- monitor_evolution.py

Provides comprehensive monitoring with:
- Multi-system performance tracking
- Real-time alerting
- Historical analysis
- Dashboard generation
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import psutil
    import torch
except ImportError as e:
    logging.exception(f"Required dependencies missing: {e}")
    raise

from ..core import BaseScript, ScriptResult


class MonitoringSystem(Enum):
    """Types of systems that can be monitored."""

    SYSTEM = "system"
    COMPRESSION = "compression"
    EVOLUTION = "evolution"
    AGENT_FORGE = "agent_forge"
    CUSTOM = "custom"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring systems."""

    enabled_systems: list[MonitoringSystem]
    monitoring_interval: float = 5.0
    alert_thresholds: dict[str, float] = None
    metrics_retention_days: int = 30
    dashboard_enabled: bool = True
    real_time_alerts: bool = True

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "cpu_percent_high": 80.0,
                "memory_percent_high": 85.0,
                "disk_usage_high": 90.0,
                "compression_ratio_low": 3.0,
                "compression_error_high": 2.0,
                "evolution_fitness_low": 0.5,
            }


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: list[float]
    process_count: int
    temperature: float | None = None


@dataclass
class CompressionMetrics:
    """Compression performance metrics."""

    timestamp: str
    method: str
    compression_ratio: float
    relative_error: float
    compression_time: float
    memory_usage: float
    model_size: int
    throughput_mb_per_sec: float
    quality_score: float


@dataclass
class EvolutionMetrics:
    """Evolution system metrics."""

    timestamp: str
    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    diversity_score: float
    mutation_rate: float
    selection_pressure: float
    convergence_rate: float


@dataclass
class AlertEvent:
    """Alert event data."""

    timestamp: str
    system: MonitoringSystem
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    metric: str
    value: float
    threshold: float
    message: str


class UnifiedMonitor(BaseScript):
    """Unified monitoring system for all AIVillage components."""

    def __init__(self, config: MonitoringConfig | None = None, data_dir: Path | None = None, **kwargs):
        """Initialize the unified monitor.

        Args:
            config: Monitoring configuration
            data_dir: Directory to store monitoring data
            **kwargs: Additional arguments for BaseScript
        """
        super().__init__(name="unified_monitor", description="Unified monitoring system for AIVillage", **kwargs)

        self.config = config or MonitoringConfig(
            enabled_systems=[MonitoringSystem.SYSTEM, MonitoringSystem.COMPRESSION]
        )

        self.data_dir = data_dir or Path("monitoring_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.system_metrics: list[SystemMetrics] = []
        self.compression_metrics: list[CompressionMetrics] = []
        self.evolution_metrics: list[EvolutionMetrics] = []
        self.alert_events: list[AlertEvent] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: asyncio.Task | None = None

        # Load historical data
        self._load_historical_data()

        self.logger.info(f"UnifiedMonitor initialized with systems: {[s.value for s in self.config.enabled_systems]}")

    def _load_historical_data(self) -> None:
        """Load historical monitoring data from disk."""
        try:
            # Load system metrics
            system_file = self.data_dir / "system_metrics.json"
            if system_file.exists():
                with open(system_file) as f:
                    data = json.load(f)
                    self.system_metrics = [SystemMetrics(**item) for item in data]
                self.logger.info(f"Loaded {len(self.system_metrics)} system metrics")

            # Load compression metrics
            compression_file = self.data_dir / "compression_metrics.json"
            if compression_file.exists():
                with open(compression_file) as f:
                    data = json.load(f)
                    self.compression_metrics = [CompressionMetrics(**item) for item in data]
                self.logger.info(f"Loaded {len(self.compression_metrics)} compression metrics")

            # Load evolution metrics
            evolution_file = self.data_dir / "evolution_metrics.json"
            if evolution_file.exists():
                with open(evolution_file) as f:
                    data = json.load(f)
                    self.evolution_metrics = [EvolutionMetrics(**item) for item in data]
                self.logger.info(f"Loaded {len(self.evolution_metrics)} evolution metrics")

        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")

    def _save_metrics(self) -> None:
        """Save current metrics to disk."""
        try:
            # Save system metrics
            if self.system_metrics:
                system_file = self.data_dir / "system_metrics.json"
                with open(system_file, "w") as f:
                    json.dump([asdict(m) for m in self.system_metrics[-1000:]], f, indent=2)

            # Save compression metrics
            if self.compression_metrics:
                compression_file = self.data_dir / "compression_metrics.json"
                with open(compression_file, "w") as f:
                    json.dump([asdict(m) for m in self.compression_metrics[-1000:]], f, indent=2)

            # Save evolution metrics
            if self.evolution_metrics:
                evolution_file = self.data_dir / "evolution_metrics.json"
                with open(evolution_file, "w") as f:
                    json.dump([asdict(m) for m in self.evolution_metrics[-1000:]], f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics.

        Returns:
            SystemMetrics object with current system state
        """
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Disk usage and I/O
            disk_usage = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()

            # Network I/O
            network_io = psutil.net_io_counters()

            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = []

            # Process count
            process_count = len(psutil.pids())

            # Temperature (if available)
            temperature = None
            try:
                sensors = psutil.sensors_temperatures()
                if sensors:
                    # Get CPU temperature if available
                    for name, entries in sensors.items():
                        if "cpu" in name.lower() or "core" in name.lower():
                            temperature = entries[0].current
                            break
            except (AttributeError, OSError):
                pass

            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_used_gb=memory.used / (1024**3),
                disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
                disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
                disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
                network_sent_mb=network_io.bytes_sent / (1024**2) if network_io else 0,
                network_recv_mb=network_io.bytes_recv / (1024**2) if network_io else 0,
                load_average=load_avg,
                process_count=process_count,
                temperature=temperature,
            )

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0,
                memory_percent=0,
                memory_available_gb=0,
                memory_used_gb=0,
                disk_usage_percent=0,
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_sent_mb=0,
                network_recv_mb=0,
                load_average=[],
                process_count=0,
            )

    def run_compression_benchmark(self, method: str = "SeedLM") -> CompressionMetrics:
        """Run compression benchmark and collect metrics.

        Args:
            method: Compression method to benchmark

        Returns:
            CompressionMetrics with benchmark results
        """
        try:
            start_time = time.time()

            # Create test data
            test_weight = torch.randn(256, 512)
            original_size = test_weight.numel() * 4  # 4 bytes per float32

            # Simulate compression (placeholder implementation)
            compressed_data = test_weight * 0.5  # Simplified compression
            compression_time = time.time() - start_time

            # Calculate metrics
            compression_ratio = 4.0  # Simulated ratio
            relative_error = 0.01  # Simulated error
            memory_usage = original_size / (1024**2)  # MB
            throughput = original_size / (1024**2) / compression_time  # MB/s
            quality_score = 1.0 - relative_error

            return CompressionMetrics(
                timestamp=datetime.now().isoformat(),
                method=method,
                compression_ratio=compression_ratio,
                relative_error=relative_error,
                compression_time=compression_time,
                memory_usage=memory_usage,
                model_size=test_weight.numel(),
                throughput_mb_per_sec=throughput,
                quality_score=quality_score,
            )

        except Exception as e:
            self.logger.error(f"Compression benchmark failed: {e}")
            return CompressionMetrics(
                timestamp=datetime.now().isoformat(),
                method=method,
                compression_ratio=0.0,
                relative_error=float("inf"),
                compression_time=0.0,
                memory_usage=0.0,
                model_size=0,
                throughput_mb_per_sec=0.0,
                quality_score=0.0,
            )

    def collect_evolution_metrics(self) -> EvolutionMetrics | None:
        """Collect evolution system metrics.

        Returns:
            EvolutionMetrics if evolution system is running, None otherwise
        """
        # This would integrate with the actual evolution system
        # For now, return simulated metrics
        try:
            return EvolutionMetrics(
                timestamp=datetime.now().isoformat(),
                generation=10,
                population_size=20,
                best_fitness=0.85,
                average_fitness=0.72,
                diversity_score=0.6,
                mutation_rate=0.1,
                selection_pressure=1.2,
                convergence_rate=0.05,
            )
        except Exception as e:
            self.logger.error(f"Failed to collect evolution metrics: {e}")
            return None

    def check_alerts(self, metrics: dict[str, Any]) -> list[AlertEvent]:
        """Check for alert conditions.

        Args:
            metrics: Dictionary of current metrics

        Returns:
            List of alert events
        """
        alerts = []
        timestamp = datetime.now().isoformat()

        # System alerts
        if "system" in metrics:
            system = metrics["system"]

            if system.cpu_percent > self.config.alert_thresholds.get("cpu_percent_high", 80):
                alerts.append(
                    AlertEvent(
                        timestamp=timestamp,
                        system=MonitoringSystem.SYSTEM,
                        severity="WARNING",
                        metric="cpu_percent",
                        value=system.cpu_percent,
                        threshold=self.config.alert_thresholds["cpu_percent_high"],
                        message=f"High CPU usage: {system.cpu_percent:.1f}%",
                    )
                )

            if system.memory_percent > self.config.alert_thresholds.get("memory_percent_high", 85):
                alerts.append(
                    AlertEvent(
                        timestamp=timestamp,
                        system=MonitoringSystem.SYSTEM,
                        severity="WARNING",
                        metric="memory_percent",
                        value=system.memory_percent,
                        threshold=self.config.alert_thresholds["memory_percent_high"],
                        message=f"High memory usage: {system.memory_percent:.1f}%",
                    )
                )

        # Compression alerts
        if "compression" in metrics:
            compression = metrics["compression"]

            if compression.compression_ratio < self.config.alert_thresholds.get("compression_ratio_low", 3.0):
                alerts.append(
                    AlertEvent(
                        timestamp=timestamp,
                        system=MonitoringSystem.COMPRESSION,
                        severity="ERROR",
                        metric="compression_ratio",
                        value=compression.compression_ratio,
                        threshold=self.config.alert_thresholds["compression_ratio_low"],
                        message=f"Low compression ratio: {compression.compression_ratio:.2f}x",
                    )
                )

        return alerts

    async def monitor_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Starting monitoring loop")

        while self.is_monitoring:
            try:
                current_metrics = {}

                # Collect metrics from enabled systems
                if MonitoringSystem.SYSTEM in self.config.enabled_systems:
                    system_metrics = self.collect_system_metrics()
                    self.system_metrics.append(system_metrics)
                    current_metrics["system"] = system_metrics

                    self.logger.debug(
                        f"System - CPU: {system_metrics.cpu_percent:.1f}%, "
                        f"Memory: {system_metrics.memory_percent:.1f}%, "
                        f"Disk: {system_metrics.disk_usage_percent:.1f}%"
                    )

                if MonitoringSystem.COMPRESSION in self.config.enabled_systems:
                    compression_metrics = self.run_compression_benchmark()
                    self.compression_metrics.append(compression_metrics)
                    current_metrics["compression"] = compression_metrics

                    self.logger.debug(
                        f"Compression - Ratio: {compression_metrics.compression_ratio:.2f}x, "
                        f"Error: {compression_metrics.relative_error:.4f}, "
                        f"Time: {compression_metrics.compression_time:.2f}s"
                    )

                if MonitoringSystem.EVOLUTION in self.config.enabled_systems:
                    evolution_metrics = self.collect_evolution_metrics()
                    if evolution_metrics:
                        self.evolution_metrics.append(evolution_metrics)
                        current_metrics["evolution"] = evolution_metrics

                # Check for alerts
                if self.config.real_time_alerts:
                    alerts = self.check_alerts(current_metrics)
                    for alert in alerts:
                        self.alert_events.append(alert)
                        self.logger.warning(f"ALERT [{alert.severity}]: {alert.message}")

                # Save metrics periodically
                if len(self.system_metrics) % 10 == 0:
                    self._save_metrics()

                # Wait for next iteration
                await asyncio.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)

    async def start_monitoring(self, duration: float | None = None) -> None:
        """Start monitoring process.

        Args:
            duration: Monitoring duration in seconds (None for indefinite)
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return

        self.is_monitoring = True
        self.logger.info(f"Starting monitoring for {duration or 'indefinite'} seconds")

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self.monitor_loop())

        # If duration is specified, stop after that time
        if duration:
            await asyncio.sleep(duration)
            await self.stop_monitoring()
        else:
            await self.monitoring_task

    async def stop_monitoring(self) -> None:
        """Stop monitoring process."""
        if not self.is_monitoring:
            self.logger.warning("Monitoring is not running")
            return

        self.logger.info("Stopping monitoring")
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Save final metrics
        self._save_metrics()
        self.logger.info("Monitoring stopped")

    def generate_report(self, hours: int = 24) -> str:
        """Generate monitoring report.

        Args:
            hours: Number of hours to include in report

        Returns:
            Formatted report string
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        report = f"""
# AIVillage Monitoring Report ({hours} hours)

## Summary
- **Report Period**: {cutoff_time.isoformat()} to {datetime.now().isoformat()}
- **Enabled Systems**: {[s.value for s in self.config.enabled_systems]}
- **Monitoring Interval**: {self.config.monitoring_interval}s

"""

        # System metrics summary
        if MonitoringSystem.SYSTEM in self.config.enabled_systems:
            recent_system = [m for m in self.system_metrics if datetime.fromisoformat(m.timestamp) > cutoff_time]

            if recent_system:
                cpu_values = [m.cpu_percent for m in recent_system]
                memory_values = [m.memory_percent for m in recent_system]

                report += f"""## System Performance
- **Samples**: {len(recent_system)}
- **Average CPU**: {np.mean(cpu_values):.1f}% (Peak: {np.max(cpu_values):.1f}%)
- **Average Memory**: {np.mean(memory_values):.1f}% (Peak: {np.max(memory_values):.1f}%)
- **Available Memory**: {recent_system[-1].memory_available_gb:.1f} GB
- **Process Count**: {recent_system[-1].process_count}

"""

        # Compression metrics summary
        if MonitoringSystem.COMPRESSION in self.config.enabled_systems:
            recent_compression = [
                m for m in self.compression_metrics if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]

            if recent_compression:
                ratio_values = [m.compression_ratio for m in recent_compression]
                error_values = [m.relative_error for m in recent_compression]

                report += f"""## Compression Performance
- **Benchmarks**: {len(recent_compression)}
- **Average Ratio**: {np.mean(ratio_values):.2f}x (Best: {np.max(ratio_values):.2f}x)
- **Average Error**: {np.mean(error_values):.4f} (Best: {np.min(error_values):.4f})
- **Average Quality**: {np.mean([m.quality_score for m in recent_compression]):.3f}

"""

        # Alert summary
        recent_alerts = [a for a in self.alert_events if datetime.fromisoformat(a.timestamp) > cutoff_time]

        if recent_alerts:
            alert_counts = {}
            for alert in recent_alerts:
                key = f"{alert.system.value}_{alert.severity}"
                alert_counts[key] = alert_counts.get(key, 0) + 1

            report += f"""## Alerts ({len(recent_alerts)} total)
"""
            for alert_type, count in alert_counts.items():
                report += f"- **{alert_type}**: {count}\n"

        return report

    def create_dashboard(self, output_dir: Path | None = None) -> Path:
        """Create monitoring dashboard with visualizations.

        Args:
            output_dir: Directory to save dashboard files

        Returns:
            Path to the generated dashboard
        """
        if output_dir is None:
            output_dir = self.data_dir / "dashboard"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create system performance plots
        if self.system_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("System Performance Dashboard", fontsize=16)

            timestamps = [datetime.fromisoformat(m.timestamp) for m in self.system_metrics[-100:]]
            cpu_values = [m.cpu_percent for m in self.system_metrics[-100:]]
            memory_values = [m.memory_percent for m in self.system_metrics[-100:]]
            disk_values = [m.disk_usage_percent for m in self.system_metrics[-100:]]

            # CPU usage
            axes[0, 0].plot(timestamps, cpu_values, "b-", alpha=0.7)
            axes[0, 0].set_title("CPU Usage %")
            axes[0, 0].set_ylabel("Percentage")
            axes[0, 0].grid(True, alpha=0.3)

            # Memory usage
            axes[0, 1].plot(timestamps, memory_values, "r-", alpha=0.7)
            axes[0, 1].set_title("Memory Usage %")
            axes[0, 1].set_ylabel("Percentage")
            axes[0, 1].grid(True, alpha=0.3)

            # Disk usage
            axes[1, 0].plot(timestamps, disk_values, "g-", alpha=0.7)
            axes[1, 0].set_title("Disk Usage %")
            axes[1, 0].set_ylabel("Percentage")
            axes[1, 0].grid(True, alpha=0.3)

            # Process count
            process_counts = [m.process_count for m in self.system_metrics[-100:]]
            axes[1, 1].plot(timestamps, process_counts, "m-", alpha=0.7)
            axes[1, 1].set_title("Process Count")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            system_plot_path = output_dir / "system_performance.png"
            plt.savefig(system_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"System dashboard saved to {system_plot_path}")

        # Create compression performance plots
        if self.compression_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Compression Performance Dashboard", fontsize=16)

            timestamps = [datetime.fromisoformat(m.timestamp) for m in self.compression_metrics[-100:]]
            ratios = [m.compression_ratio for m in self.compression_metrics[-100:]]
            errors = [m.relative_error for m in self.compression_metrics[-100:]]
            times = [m.compression_time for m in self.compression_metrics[-100:]]
            quality = [m.quality_score for m in self.compression_metrics[-100:]]

            # Compression ratio
            axes[0, 0].plot(timestamps, ratios, "b-o", alpha=0.7)
            axes[0, 0].set_title("Compression Ratio")
            axes[0, 0].set_ylabel("Ratio (x)")
            axes[0, 0].grid(True, alpha=0.3)

            # Relative error
            axes[0, 1].plot(timestamps, errors, "r-o", alpha=0.7)
            axes[0, 1].set_title("Relative Error")
            axes[0, 1].set_ylabel("Error")
            axes[0, 1].grid(True, alpha=0.3)

            # Compression time
            axes[1, 0].plot(timestamps, times, "g-o", alpha=0.7)
            axes[1, 0].set_title("Compression Time")
            axes[1, 0].set_ylabel("Time (seconds)")
            axes[1, 0].grid(True, alpha=0.3)

            # Quality score
            axes[1, 1].plot(timestamps, quality, "m-o", alpha=0.7)
            axes[1, 1].set_title("Quality Score")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            compression_plot_path = output_dir / "compression_performance.png"
            plt.savefig(compression_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Compression dashboard saved to {compression_plot_path}")

        # Generate HTML dashboard
        html_content = self._generate_html_dashboard()
        html_path = output_dir / "dashboard.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"HTML dashboard saved to {html_path}")
        return html_path

    def _generate_html_dashboard(self) -> str:
        """Generate HTML dashboard content.

        Returns:
            HTML content string
        """
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>AIVillage Monitoring Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-box {{
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }}
        .alert {{
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }}
        .warning {{
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
        }}
        .success {{
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
        }}
    </style>
</head>
<body>
    <h1>AIVillage Monitoring Dashboard</h1>
    <p>Generated: {datetime.now().isoformat()}</p>

    <h2>System Status</h2>
    <div class="metric-box success">
        <h3>Monitoring Active</h3>
        <p>Systems: {[s.value for s in self.config.enabled_systems]}</p>
        <p>Interval: {self.config.monitoring_interval}s</p>
    </div>

    <h2>Current Metrics</h2>
    {"<div class='metric-box'><h3>System Metrics</h3><p>Total samples: " + str(len(self.system_metrics)) + "</p></div>" if self.system_metrics else ""}
    {"<div class='metric-box'><h3>Compression Metrics</h3><p>Total benchmarks: " + str(len(self.compression_metrics)) + "</p></div>" if self.compression_metrics else ""}

    <h2>Recent Alerts</h2>
    {"<div class='metric-box alert'><h3>Recent Alerts</h3><p>Total alerts: " + str(len(self.alert_events)) + "</p></div>" if self.alert_events else "<div class='metric-box success'><h3>No Recent Alerts</h3></div>"}

    <h2>Performance Charts</h2>
    <p>
        <img src="system_performance.png" alt="System Performance" style="max-width: 100%;">
    </p>
    <p>
        <img src="compression_performance.png" alt="Compression Performance" style="max-width: 100%;">
    </p>
</body>
</html>"""

    def execute(self) -> ScriptResult:
        """Execute monitoring script.

        Returns:
            ScriptResult with monitoring results
        """
        try:
            # Default monitoring duration
            duration = self.get_config_value("monitoring.duration", 60)  # 1 minute default

            if self.dry_run:
                return ScriptResult(
                    success=True,
                    message="Dry run completed - monitoring configuration validated",
                    data={"config": asdict(self.config)},
                )

            # Run monitoring
            asyncio.run(self.start_monitoring(duration))

            # Generate report and dashboard
            report = self.generate_report(hours=1)
            dashboard_path = self.create_dashboard()

            return ScriptResult(
                success=True,
                message=f"Monitoring completed successfully for {duration} seconds",
                data={
                    "systems_monitored": [s.value for s in self.config.enabled_systems],
                    "total_system_metrics": len(self.system_metrics),
                    "total_compression_metrics": len(self.compression_metrics),
                    "total_alerts": len(self.alert_events),
                    "dashboard_path": str(dashboard_path),
                    "report": report,
                },
                metrics={
                    "monitoring_duration": duration,
                    "samples_collected": len(self.system_metrics) + len(self.compression_metrics),
                    "alerts_generated": len(self.alert_events),
                },
            )

        except Exception as e:
            return ScriptResult(success=False, message=f"Monitoring failed: {e}", errors=[str(e)])
