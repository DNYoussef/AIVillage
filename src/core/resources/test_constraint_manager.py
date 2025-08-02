import argparse
import asyncio
from .device_profiler import DeviceProfiler, ResourceSnapshot, PowerState, ThermalState
from .constraint_manager import ConstraintManager, ResourceConstraints, ConstraintType

parser = argparse.ArgumentParser(description="Constraint manager test options")
parser.add_argument("--simulate", action="store_true", help="use simulated data")
ARGS, _ = parser.parse_known_args()


def _sample_snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=0,
        memory_total=200 * 1024 * 1024,
        memory_available=50 * 1024 * 1024,
        memory_used=150 * 1024 * 1024,
        memory_percent=75.0,
        cpu_percent=60.0,
        cpu_cores=4,
        cpu_freq_current=None,
        cpu_freq_max=None,
        cpu_temp=80.0,
        storage_total=10 * 1024 * 1024 * 1024,
        storage_used=9 * 1024 * 1024 * 1024,
        storage_free=0.5 * 1024 * 1024 * 1024,
        storage_percent=90.0,
        battery_percent=10.0,
        power_plugged=False,
        power_state=PowerState.BATTERY_LOW,
        thermal_state=ThermalState.HOT,
        network_sent=0,
        network_received=0,
        network_connections=0,
        process_count=1,
        gpu_memory_used=None,
        gpu_memory_total=None,
        gpu_utilization=None,
    )


def test_detects_violations():
    profiler = DeviceProfiler(monitoring_interval=0.1, enable_background_monitoring=False)
    cm = ConstraintManager(profiler)
    constraints = ResourceConstraints(
        max_memory_mb=100,
        memory_warning_mb=80,
        memory_critical_mb=90,
        max_cpu_percent=50.0,
        cpu_warning_percent=40.0,
        cpu_critical_percent=45.0,
        min_battery_percent=20.0,
        battery_warning_percent=30.0,
        max_temperature_celsius=70.0,
        temperature_warning_celsius=60.0,
        min_free_storage_gb=1.0,
        storage_warning_gb=2.0,
    )
    snapshot = _sample_snapshot()
    violations = asyncio.run(cm._check_task_constraints("task", constraints, snapshot))
    types = {v.constraint_type for v in violations}
    assert ConstraintType.MEMORY in types
    assert ConstraintType.CPU in types
