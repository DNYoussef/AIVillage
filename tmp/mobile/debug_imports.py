#!/usr/bin/env python3
"""Debug import issues with DeviceProfile"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Try to import DeviceProfile and check what we get
try:
    import inspect

    from src.production.monitoring.mobile.device_profiler import DeviceProfile

    print("DeviceProfile imported successfully")
    print("Constructor signature:")
    print(inspect.signature(DeviceProfile.__init__))
    print("\nDeviceProfile fields:")
    print([f.name for f in DeviceProfile.__dataclass_fields__.values()])

    # Try to create instance with the correct signature
    try:
        profile = DeviceProfile(
            timestamp=1234567890.0,
            cpu_percent=50.0,
            cpu_freq_mhz=2400.0,
            cpu_temp_celsius=35.0,
            cpu_cores=4,
            ram_used_mb=2048,
            ram_available_mb=2048,
            ram_total_mb=4096,
            battery_percent=80,
            battery_charging=True,
            battery_temp_celsius=None,
            battery_health=None,
            network_type="wifi",
            network_bandwidth_mbps=None,
            network_latency_ms=50.0,
            storage_available_gb=50.0,
            storage_total_gb=100.0,
            gpu_available=False,
            gpu_memory_mb=None,
            thermal_state="normal",
            power_mode="balanced",
            screen_brightness=None,
            device_type="laptop",
        )
        print("✓ DeviceProfile instance created successfully")
        print(
            f"Profile: battery={profile.battery_percent}%, temp={profile.cpu_temp_celsius}°C"
        )
    except Exception as e:
        print(f"✗ Failed to create DeviceProfile: {e}")

except Exception as e:
    print(f"Failed to import DeviceProfile: {e}")

print("\n" + "=" * 50)

# Try importing the resource management
try:
    from src.production.monitoring.mobile.resource_management import (
        BatteryThermalResourceManager,
    )

    print("BatteryThermalResourceManager imported successfully")

    # Try to create a manager and test env mode
    os.environ["BATTERY"] = "50"
    manager = BatteryThermalResourceManager()
    print(f"Environment simulation mode: {manager.env_simulation_mode}")

    if manager.env_simulation_mode:
        try:
            import asyncio

            async def test():
                state = await manager.evaluate_and_adapt()
                print(f"✓ Evaluation successful: power={state.power_mode.value}")
                return state

            state = asyncio.run(test())

        except Exception as e:
            print(f"✗ Evaluation failed: {e}")

except Exception as e:
    print(f"Failed to import BatteryThermalResourceManager: {e}")
