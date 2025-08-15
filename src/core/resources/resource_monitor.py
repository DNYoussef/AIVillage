"""Resource Monitoring - Essential for adaptive resource management
Currently returns None for everything!
"""

import logging
import os
import socket
import time
from typing import Any

try:  # pragma: no cover
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

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
        if psutil:
            value = psutil.cpu_percent(interval=0.1)
        else:  # Fallback to load average
            try:
                load, _, _ = os.getloadavg()
                value = min(100.0, load * 100 / (os.cpu_count() or 1))
            except Exception:  # pragma: no cover
                value = 0.0
        self.cpu_history.append(value)
        if len(self.cpu_history) > 60:
            self.cpu_history.pop(0)
        return value

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage statistics."""
        if psutil:
            mem = psutil.virtual_memory()
            total = mem.total
            available = mem.available
            used = mem.used
            percent = mem.percent
        else:  # pragma: no cover
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                    info: dict[str, int] = {}
                    for line in fh:
                        if ":" in line:
                            key, val = line.split(":", 1)
                            info[key] = int(val.strip().split()[0]) * 1024
                total = info.get("MemTotal", 0)
                available = info.get("MemAvailable", info.get("MemFree", 0))
                used = total - available
                percent = (used / total * 100) if total else 0.0
            except Exception:  # pragma: no cover
                total = available = used = 0
                percent = 0.0
        return {
            "total_gb": total / (1024**3),
            "available_gb": available / (1024**3),
            "used_gb": used / (1024**3),
            "percent": percent,
        }

    def get_disk_usage(self) -> dict[str, float]:
        """Get disk usage statistics."""
        if psutil:
            disk = psutil.disk_usage("/")
            total = disk.total
            free = disk.free
            used = disk.used
            percent = disk.percent
        else:  # pragma: no cover
            try:
                stat = os.statvfs("/")
                total = stat.f_frsize * stat.f_blocks
                free = stat.f_frsize * stat.f_bfree
                used = total - free
                percent = (used / total * 100) if total else 0.0
            except Exception:
                total = free = used = 0
                percent = 0.0
        return {
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "used_gb": used / (1024**3),
            "percent": percent,
        }

    def get_network_usage(self) -> dict[str, float]:
        """Get network usage statistics and latency."""
        if psutil:
            net = psutil.net_io_counters()
            bytes_sent = net.bytes_sent
            bytes_recv = net.bytes_recv
            packets_sent = net.packets_sent
            packets_recv = net.packets_recv
        else:  # pragma: no cover
            bytes_sent = bytes_recv = packets_sent = packets_recv = 0

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
            "bytes_sent_mb": bytes_sent / (1024**2),
            "bytes_recv_mb": bytes_recv / (1024**2),
            "packets_sent": packets_sent,
            "packets_recv": packets_recv,
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
        except Exception as e:
            logger.debug(f"Failed to get GPU memory info: {e}")
        return None

    def get_battery(self) -> dict[str, Any] | None:
        """Get battery information if available."""
        env_batt = os.getenv("BATTERY")
        if env_batt is not None:
            try:
                percent = float(env_batt)
            except ValueError:
                percent = None
            return {"percent": percent, "secsleft": None, "power_plugged": False}
        if not psutil:  # pragma: no cover
            return None
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
        return not (
            battery and battery["percent"] is not None and battery["percent"] < 20
        )


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
