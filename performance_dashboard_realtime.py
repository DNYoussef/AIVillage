#!/usr/bin/env python3
"""Real-time Performance Dashboard for AIVillage

Provides live monitoring of:
- System resources (CPU, Memory, Disk, Network)
- Mesh network performance
- AI model performance metrics
- Performance alerts and notifications
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import queue

try:
    import psutil
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import tkinter as tk
    from tkinter import ttk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceDataCollector:
    """Collects real-time performance data."""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.data_history = {
            'timestamp': deque(maxlen=history_size),
            'cpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'memory_available_gb': deque(maxlen=history_size),
            'disk_io_read_mb': deque(maxlen=history_size),
            'disk_io_write_mb': deque(maxlen=history_size),
            'network_sent_mb': deque(maxlen=history_size),
            'network_recv_mb': deque(maxlen=history_size),
            'mesh_delivery_rate': deque(maxlen=history_size),
            'active_connections': deque(maxlen=history_size)
        }

        self.alerts = queue.Queue()
        self.alert_thresholds = {
            'memory_critical': 90.0,
            'memory_warning': 80.0,
            'cpu_critical': 95.0,
            'cpu_warning': 85.0,
            'mesh_delivery_critical': 30.0,
            'mesh_delivery_warning': 70.0
        }

        self.last_disk_io = None
        self.last_network_io = None

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            timestamp = time.time()

            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = 0
            disk_write_mb = 0

            if disk_io and self.last_disk_io:
                disk_read_mb = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024*1024)
                disk_write_mb = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024*1024)

            if disk_io:
                self.last_disk_io = disk_io

            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = 0
            network_recv_mb = 0

            if network_io and self.last_network_io:
                network_sent_mb = (network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024*1024)
                network_recv_mb = (network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024*1024)

            if network_io:
                self.last_network_io = network_io

            # Mesh network performance (from latest results)
            mesh_delivery_rate = self._get_mesh_performance()
            active_connections = self._get_active_connections()

            metrics = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_io_read_mb': max(0, disk_read_mb),
                'disk_io_write_mb': max(0, disk_write_mb),
                'network_sent_mb': max(0, network_sent_mb),
                'network_recv_mb': max(0, network_recv_mb),
                'mesh_delivery_rate': mesh_delivery_rate,
                'active_connections': active_connections
            }

            # Store in history
            for key, value in metrics.items():
                if key in self.data_history:
                    self.data_history[key].append(value)

            # Check for alerts
            self._check_alerts(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}

    def _get_mesh_performance(self) -> float:
        """Get latest mesh network performance."""
        try:
            mesh_file = Path("mesh_network_fix_results.json")
            if mesh_file.exists():
                with open(mesh_file, 'r') as f:
                    data = json.load(f)
                return data.get('delivery_results', {}).get('delivery_rate', 0.0)
        except Exception:
            pass
        return 0.0

    def _get_active_connections(self) -> int:
        """Get number of active network connections."""
        try:
            connections = psutil.net_connections()
            return len([c for c in connections if c.status == 'ESTABLISHED'])
        except Exception:
            return 0

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds."""
        alerts = []

        # Memory alerts
        memory_percent = metrics.get('memory_percent', 0)
        if memory_percent > self.alert_thresholds['memory_critical']:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'Memory',
                'message': f'Memory usage at {memory_percent:.1f}% - Risk of crash',
                'value': memory_percent,
                'threshold': self.alert_thresholds['memory_critical']
            })
        elif memory_percent > self.alert_thresholds['memory_warning']:
            alerts.append({
                'type': 'WARNING',
                'component': 'Memory',
                'message': f'Memory usage at {memory_percent:.1f}% - Monitor closely',
                'value': memory_percent,
                'threshold': self.alert_thresholds['memory_warning']
            })

        # CPU alerts
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent > self.alert_thresholds['cpu_critical']:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'CPU',
                'message': f'CPU usage at {cpu_percent:.1f}% - Performance degraded',
                'value': cpu_percent,
                'threshold': self.alert_thresholds['cpu_critical']
            })

        # Mesh network alerts
        mesh_delivery = metrics.get('mesh_delivery_rate', 0)
        if mesh_delivery < self.alert_thresholds['mesh_delivery_critical']:
            alerts.append({
                'type': 'CRITICAL',
                'component': 'Mesh Network',
                'message': f'Message delivery at {mesh_delivery:.1f}% - Communication failing',
                'value': mesh_delivery,
                'threshold': self.alert_thresholds['mesh_delivery_critical']
            })
        elif mesh_delivery < self.alert_thresholds['mesh_delivery_warning']:
            alerts.append({
                'type': 'WARNING',
                'component': 'Mesh Network',
                'message': f'Message delivery at {mesh_delivery:.1f}% - Below target',
                'value': mesh_delivery,
                'threshold': self.alert_thresholds['mesh_delivery_warning']
            })

        # Queue alerts for processing
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            try:
                self.alerts.put_nowait(alert)
                logger.warning(f"ALERT: {alert['type']} - {alert['message']}")
            except queue.Full:
                pass


class ConsoleMonitor:
    """Console-based performance monitor."""

    def __init__(self, collector: PerformanceDataCollector):
        self.collector = collector
        self.running = False

    def start(self, update_interval: float = 5.0):
        """Start console monitoring."""
        self.running = True

        print("=== AIVillage Real-time Performance Monitor ===")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                metrics = self.collector.collect_metrics()
                if metrics:
                    self._display_metrics(metrics)
                    self._display_alerts()

                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.running = False
        except Exception as e:
            print(f"Monitoring error: {e}")
            self.running = False

    def _display_metrics(self, metrics: Dict[str, Any]):
        """Display current metrics."""
        timestamp = datetime.fromtimestamp(metrics['timestamp']).strftime('%H:%M:%S')

        # Clear screen (works on Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=== AIVillage Performance Dashboard ===")
        print(f"Last Update: {timestamp}\n")

        # System Resources
        print("📊 SYSTEM RESOURCES")
        print("-" * 40)

        memory_status = self._get_status_icon(metrics['memory_percent'], 80, 90)
        cpu_status = self._get_status_icon(metrics['cpu_percent'], 70, 85)

        print(f"Memory:     {memory_status} {metrics['memory_percent']:.1f}% used")
        print(f"Available:  💾 {metrics['memory_available_gb']:.1f} GB")
        print(f"CPU:        {cpu_status} {metrics['cpu_percent']:.1f}%")
        print(f"Disk Read:  📖 {metrics['disk_io_read_mb']:.1f} MB/s")
        print(f"Disk Write: 📝 {metrics['disk_io_write_mb']:.1f} MB/s")
        print(f"Net Sent:   📤 {metrics['network_sent_mb']:.2f} MB/s")
        print(f"Net Recv:   📥 {metrics['network_recv_mb']:.2f} MB/s")

        # Mesh Network
        print(f"\n🌐 MESH NETWORK")
        print("-" * 40)

        mesh_status = "🔴" if metrics['mesh_delivery_rate'] < 30 else "🟡" if metrics['mesh_delivery_rate'] < 70 else "🟢"
        print(f"Delivery Rate: {mesh_status} {metrics['mesh_delivery_rate']:.1f}%")
        print(f"Connections:   📡 {metrics['active_connections']}")

        # Performance Trends
        print(f"\n📈 TRENDS (Last {len(self.collector.data_history['cpu_percent'])} samples)")
        print("-" * 40)

        if len(self.collector.data_history['cpu_percent']) > 1:
            cpu_trend = self._calculate_trend(self.collector.data_history['cpu_percent'])
            memory_trend = self._calculate_trend(self.collector.data_history['memory_percent'])
            mesh_trend = self._calculate_trend(self.collector.data_history['mesh_delivery_rate'])

            print(f"CPU Trend:    {self._format_trend(cpu_trend)}")
            print(f"Memory Trend: {self._format_trend(memory_trend)}")
            print(f"Mesh Trend:   {self._format_trend(mesh_trend)}")

    def _display_alerts(self):
        """Display recent alerts."""
        alerts = []
        try:
            while True:
                alert = self.collector.alerts.get_nowait()
                alerts.append(alert)
        except queue.Empty:
            pass

        if alerts:
            print(f"\n🚨 RECENT ALERTS")
            print("-" * 40)

            for alert in alerts[-5:]:  # Show last 5 alerts
                icon = "🚨" if alert['type'] == 'CRITICAL' else "⚠️"
                time_str = alert['timestamp'].split('T')[1][:8]
                print(f"{icon} {time_str} - {alert['message']}")

    def _get_status_icon(self, value: float, warning_threshold: float, critical_threshold: float) -> str:
        """Get status icon based on thresholds."""
        if value >= critical_threshold:
            return "🔴"
        elif value >= warning_threshold:
            return "🟡"
        else:
            return "🟢"

    def _calculate_trend(self, data: deque) -> float:
        """Calculate trend (positive = increasing)."""
        if len(data) < 2:
            return 0.0

        recent = list(data)[-10:]  # Last 10 samples
        if len(recent) < 2:
            return 0.0

        return recent[-1] - recent[0]

    def _format_trend(self, trend: float) -> str:
        """Format trend for display."""
        if abs(trend) < 0.1:
            return "➡️ Stable"
        elif trend > 0:
            return f"📈 +{trend:.1f}"
        else:
            return f"📉 {trend:.1f}"


class PerformanceDashboard:
    """Main performance dashboard."""

    def __init__(self):
        self.collector = PerformanceDataCollector()
        self.console_monitor = ConsoleMonitor(self.collector)

    def start_console_monitor(self, update_interval: float = 5.0):
        """Start console-based monitoring."""
        print("Starting console performance monitor...")
        self.console_monitor.start(update_interval)

    def generate_performance_report(self) -> str:
        """Generate current performance report."""
        metrics = self.collector.collect_metrics()
        if not metrics:
            return "No metrics available"

        timestamp = datetime.fromtimestamp(metrics['timestamp'])

        report = f"""# AIVillage Performance Status Report
Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Current System Status

### Resource Utilization
- **Memory Usage**: {metrics['memory_percent']:.1f}% ({metrics['memory_available_gb']:.1f} GB available)
- **CPU Usage**: {metrics['cpu_percent']:.1f}%
- **Disk I/O**: {metrics['disk_io_read_mb']:.1f} MB/s read, {metrics['disk_io_write_mb']:.1f} MB/s write
- **Network I/O**: {metrics['network_sent_mb']:.2f} MB/s sent, {metrics['network_recv_mb']:.2f} MB/s received

### Mesh Network Performance
- **Message Delivery Rate**: {metrics['mesh_delivery_rate']:.1f}%
- **Active Connections**: {metrics['active_connections']}

### Status Assessment
"""

        # Status assessment
        if metrics['memory_percent'] > 90:
            report += "🚨 **CRITICAL**: Memory usage above 90% - immediate action required\n"
        elif metrics['memory_percent'] > 80:
            report += "⚠️ **WARNING**: High memory usage - monitor closely\n"
        else:
            report += "✅ **OK**: Memory usage within acceptable range\n"

        if metrics['mesh_delivery_rate'] < 30:
            report += "🚨 **CRITICAL**: Mesh network delivery rate critically low\n"
        elif metrics['mesh_delivery_rate'] < 70:
            report += "⚠️ **WARNING**: Mesh network delivery rate below target\n"
        else:
            report += "✅ **OK**: Mesh network performing adequately\n"

        # Recommendations
        report += "\n### Recommendations\n"

        if metrics['memory_percent'] > 85:
            report += "- Run memory optimization: `python memory_optimizer.py`\n"

        if metrics['mesh_delivery_rate'] < 70:
            report += "- Check mesh network: `python mesh_network_performance_fixer.py`\n"

        if metrics['cpu_percent'] > 80:
            report += "- Investigate high CPU usage processes\n"

        return report

    def save_metrics(self, filename: str = "performance_metrics.json"):
        """Save current metrics to file."""
        try:
            metrics = self.collector.collect_metrics()
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


def main():
    """Main entry point."""
    dashboard = PerformanceDashboard()

    print("AIVillage Real-time Performance Dashboard")
    print("=" * 50)

    try:
        if len(os.sys.argv) > 1:
            command = os.sys.argv[1]

            if command == "report":
                # Generate single report
                report = dashboard.generate_performance_report()
                print(report)

                # Save to file
                with open("performance_status_report.md", "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"\nReport saved to: performance_status_report.md")

            elif command == "save":
                # Save metrics
                dashboard.save_metrics()
                print("Metrics saved to performance_metrics.json")

            else:
                print(f"Unknown command: {command}")
                print("Available commands: report, save")

        else:
            # Start real-time monitoring
            dashboard.start_console_monitor()

    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")


if __name__ == "__main__":
    main()
