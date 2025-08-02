import argparse
import psutil
import pytest
from .device_profiler import DeviceProfiler, ResourceSnapshot, PowerState, ThermalState

parser = argparse.ArgumentParser(description="Device profiler test options")
parser.add_argument("--simulate", action="store_true", help="use simulated metrics")
ARGS, _ = parser.parse_known_args()


def _apply_simulation(monkeypatch):
    class Mem:
        total = 1024 ** 3
        available = total // 2
        used = total // 2
        percent = 50.0

    monkeypatch.setattr(psutil, "virtual_memory", lambda: Mem)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 4)
    monkeypatch.setattr(psutil, "cpu_freq", lambda: type("cf", (), {"current": 1000})())
    monkeypatch.setattr(psutil, "disk_usage", lambda path: type("du", (), {"total": 1000, "used": 500, "free": 500})())
    monkeypatch.setattr(psutil, "net_io_counters", lambda: type("nio", (), {"bytes_sent": 0, "bytes_recv": 0})())
    monkeypatch.setattr(psutil, "net_connections", lambda: [])
    monkeypatch.setattr(psutil, "pids", lambda: [1])
    monkeypatch.setattr(psutil, "sensors_battery", lambda: type("bat", (), {"percent": 100, "power_plugged": True})())
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {"coretemp": [type("t", (), {"current": 50})()]})


def test_take_snapshot(monkeypatch):
    if ARGS.simulate:
        _apply_simulation(monkeypatch)
    profiler = DeviceProfiler(monitoring_interval=0.1, enable_background_monitoring=False)
    snapshot = profiler.take_snapshot()
    assert snapshot.memory_total > 0
    assert snapshot.cpu_percent >= 0


def test_resource_snapshot_flags():
    snapshot = ResourceSnapshot(
        timestamp=0,
        memory_total=1000,
        memory_available=100,
        memory_used=900,
        memory_percent=90.0,
        cpu_percent=95.0,
        cpu_cores=4,
        cpu_freq_current=None,
        cpu_freq_max=None,
        cpu_temp=90.0,
        storage_total=1000,
        storage_used=900,
        storage_free=100,
        storage_percent=90.0,
        battery_percent=4.0,
        power_plugged=False,
        power_state=PowerState.BATTERY_CRITICAL,
        thermal_state=ThermalState.CRITICAL,
        network_sent=0,
        network_received=0,
        network_connections=0,
        process_count=0,
        gpu_memory_used=None,
        gpu_memory_total=None,
        gpu_utilization=None,
    )
    assert snapshot.is_resource_constrained
