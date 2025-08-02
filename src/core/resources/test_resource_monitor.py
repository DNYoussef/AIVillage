import argparse
import asyncio
from .device_profiler import DeviceProfiler, ResourceSnapshot, PowerState, ThermalState
from .resource_monitor import ResourceMonitor, MonitoringMode

parser = argparse.ArgumentParser(description="Resource monitor test options")
parser.add_argument("--simulate", action="store_true", help="use simulated snapshots")
ARGS, _ = parser.parse_known_args()


def _make_snapshot(mem_percent: float) -> ResourceSnapshot:
    mem_total = 1000
    mem_used = mem_total * mem_percent / 100
    return ResourceSnapshot(
        timestamp=0,
        memory_total=mem_total,
        memory_available=mem_total - mem_used,
        memory_used=mem_used,
        memory_percent=mem_percent,
        cpu_percent=10.0,
        cpu_cores=4,
        cpu_freq_current=None,
        cpu_freq_max=None,
        cpu_temp=None,
        storage_total=0,
        storage_used=0,
        storage_free=0,
        storage_percent=0,
        battery_percent=None,
        power_plugged=None,
        power_state=PowerState.UNKNOWN,
        thermal_state=ThermalState.UNKNOWN,
        network_sent=0,
        network_received=0,
        network_connections=0,
        process_count=0,
        gpu_memory_used=None,
        gpu_memory_total=None,
        gpu_utilization=None,
    )


def test_mode_interval_adjustment():
    profiler = DeviceProfiler(monitoring_interval=1.0, enable_background_monitoring=False)
    monitor = ResourceMonitor(profiler)
    monitor.set_monitoring_mode(MonitoringMode.EVOLUTION)
    expected = monitor.base_interval * monitor.interval_adjustments[MonitoringMode.EVOLUTION]
    assert monitor.current_interval == expected


def test_trend_simulation():
    profiler = DeviceProfiler(monitoring_interval=1.0, enable_background_monitoring=False)
    snapshots = [_make_snapshot(10), _make_snapshot(11), _make_snapshot(12)]
    profiler.snapshots = snapshots
    monitor = ResourceMonitor(profiler)
    asyncio.run(monitor._update_trends(snapshots[-1]))
    assert "memory_percent" in monitor.trends
