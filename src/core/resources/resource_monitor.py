"""Resource Monitoring - Essential for adaptive resource management
Currently returns None for everything!
"""

import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Real-time system metrics for adaptive resource management."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.max_history = 100

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

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
        """Get network usage statistics."""
        net = psutil.net_io_counters()
        return {
            "bytes_sent_mb": net.bytes_sent / (1024**2),
            "bytes_recv_mb": net.bytes_recv / (1024**2),
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
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

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all system metrics - NO LONGER RETURNS NONE!"""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "disk": self.get_disk_usage(),
            "network": self.get_network_usage(),
            "gpu": self.get_gpu_usage(),
        }

        # Add to history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return metrics

    def can_allocate(self, memory_gb: float) -> bool:
        """Check if we can allocate specified memory."""
        mem = self.get_memory_usage()
        return mem["available_gb"] >= memory_gb * 1.2  # 20% buffer


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
