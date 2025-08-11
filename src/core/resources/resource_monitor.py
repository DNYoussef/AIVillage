"""Resource Monitoring - Essential for adaptive resource management
Currently returns None for everything!
"""

import logging
import socket
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Real-time system metrics for adaptive resource management."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.max_history = 100
        self.cpu_history: list[float] = []
        self.memory_history: list[float] = []

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        value = psutil.cpu_percent(interval=0.1)
        self.cpu_history.append(value)
        if len(self.cpu_history) > 60:
            self.cpu_history.pop(0)
        return value

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage statistics."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }

    def get_disk_usage(self) -> dict[str, float]:
        """Get disk usage statistics."""
        disk = psutil.disk_usage("/")
        return {
            "total_gb": disk.total / (1024**3),
            "free_gb": disk.free / (1024**3),
            "used_gb": disk.used / (1024**3),
            "percent": disk.percent,
        }

    def get_network_usage(self) -> dict[str, float]:
        """Get network usage statistics and latency."""
        net = psutil.net_io_counters()

        # Simple latency check
        latency_ms = None
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1)
            sock.sendto(b"ping", ("8.8.8.8", 53))
            latency_ms = (time.time() - start) * 1000
            sock.close()
        except Exception:
            latency_ms = None

        return {
            "bytes_sent_mb": net.bytes_sent / (1024**2),
            "bytes_recv_mb": net.bytes_recv / (1024**2),
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
            "latency_ms": latency_ms,
        }

    def get_gpu_usage(self) -> dict[str, Any] | None:
        """Get GPU usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                }
        except:
            pass
        return None

    def get_battery(self) -> dict[str, Any] | None:
        """Get battery information if available."""
        try:
            batt = psutil.sensors_battery()
            if batt is None:
                return None
            return {
                "percent": batt.percent,
                "secsleft": batt.secsleft,
                "power_plugged": batt.power_plugged,
            }
        except Exception:
            return None

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all system metrics - NO LONGER RETURNS NONE!"""
        cpu = self.get_cpu_usage()
        mem = self.get_memory_usage()
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": cpu,
            "cpu_avg_percent": sum(self.cpu_history) / len(self.cpu_history),
            "memory": mem,
            "disk": self.get_disk_usage(),
            "network": self.get_network_usage(),
            "battery": self.get_battery(),
            "gpu": self.get_gpu_usage(),
        }

        self.memory_history.append(mem["percent"])
        if len(self.memory_history) > 60:
            self.memory_history.pop(0)

        # Threshold warnings
        if cpu > 90:
            logger.warning("High CPU usage: %.2f%%", cpu)
        if mem["percent"] > 80:
            logger.warning("High memory usage: %.2f%%", mem["percent"])

        # Add to history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return metrics

    def can_allocate(self, memory_gb: float) -> bool:
        """Check if we can allocate specified memory."""
        mem = self.get_memory_usage()
        return mem["available_gb"] >= memory_gb * 1.2  # 20% buffer

    def can_run_model(self, size_mb: float) -> bool:
        """Determine if a model of given size can run based on resources."""
        mem = self.get_memory_usage()
        disk = self.get_disk_usage()
        battery = self.get_battery()

        if mem["available_gb"] * 1024 < size_mb * 1.2:
            return False
        if disk["free_gb"] * 1024 < size_mb * 2:
            return False
        return not (battery and battery["percent"] is not None and battery["percent"] < 20)


# Module-level functions
_monitor_instance = None


def get_monitor_instance():
    """Get or create monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ResourceMonitor()
    return _monitor_instance


def get_cpu_usage() -> float:
    """Get CPU usage - NO LONGER RETURNS NONE!"""
    return get_monitor_instance().get_cpu_usage()


def get_memory_usage() -> dict[str, float]:
    """Get memory usage - NO LONGER RETURNS NONE!"""
    return get_monitor_instance().get_memory_usage()


def get_all_metrics() -> dict[str, Any]:
    """Get all metrics - NO LONGER RETURNS NONE!"""
    return get_monitor_instance().get_all_metrics()
