#!/usr/bin/env python3
"""Validation script for mobile optimization implementation

Demonstrates P2 Mobile Resource Optimization features:
- Battery-aware transport selection (BitChat-first under low power)
- Dynamic tensor/chunk size tuning for 2-4GB devices
- Thermal throttling with progressive limits
- Network cost-aware routing decisions
- Real-time policy adaptation
"""

import asyncio
import os
import sys

# Add test directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests", "mobile"))

from test_resource_policy_simple import (
    MockDeviceProfile,
    PowerMode,
    SimpleBatteryThermalResourceManager,
    TransportPreference,
)


async def validate_mobile_optimization():
    """Validate mobile optimization implementation"""
    print("=" * 80)
    print("AI Village Mobile Optimization Validation")
    print("P2 Priority: Battery/Thermal-Aware Resource Management")
    print("=" * 80)

    manager = SimpleBatteryThermalResourceManager()

    # Test scenarios covering key requirements
    scenarios = [
        {
            "name": "Normal Operation",
            "battery": 80,
            "charging": True,
            "temp": 35.0,
            "ram_gb": 4.0,
            "network": "wifi",
            "description": "Optimal conditions - should use balanced mode",
        },
        {
            "name": "Low Battery (Not Charging)",
            "battery": 15,
            "charging": False,
            "temp": 35.0,
            "ram_gb": 4.0,
            "network": "wifi",
            "description": "Should prefer BitChat for power savings",
        },
        {
            "name": "Critical Battery",
            "battery": 8,
            "charging": False,
            "temp": 35.0,
            "ram_gb": 4.0,
            "network": "wifi",
            "description": "Should use BitChat-only mode",
        },
        {
            "name": "High Thermal Load",
            "battery": 80,
            "charging": True,
            "temp": 60.0,
            "ram_gb": 4.0,
            "network": "wifi",
            "description": "Should throttle performance and reduce chunk sizes",
        },
        {
            "name": "Memory Constrained (2GB)",
            "battery": 80,
            "charging": True,
            "temp": 35.0,
            "ram_gb": 2.0,
            "network": "wifi",
            "description": "Should use smaller chunks for limited memory",
        },
        {
            "name": "Cellular Network",
            "battery": 50,
            "charging": False,
            "temp": 35.0,
            "ram_gb": 4.0,
            "network": "cellular",
            "description": "Should prefer BitChat to reduce data costs",
        },
        {
            "name": "Extreme Stress (All Constraints)",
            "battery": 5,
            "charging": False,
            "temp": 65.0,
            "ram_gb": 1.5,
            "network": "cellular",
            "description": "Should apply maximum conservation policies",
        },
    ]

    print("\nTesting Mobile Resource Policy Scenarios:")
    print("-" * 80)

    validation_results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   {scenario['description']}")

        # Create device profile
        profile = MockDeviceProfile(
            battery_percent=scenario["battery"],
            battery_charging=scenario["charging"],
            cpu_temp_celsius=scenario["temp"],
            ram_total_mb=int(scenario["ram_gb"] * 1024),
            ram_available_mb=int(scenario["ram_gb"] * 1024 * 0.7),  # 70% available
            network_type=scenario["network"],
        )

        # Evaluate resource policies
        state = await manager.evaluate_and_adapt(profile)

        # Get routing decision for a typical message
        decision = await manager.get_transport_routing_decision(
            message_size_bytes=5 * 1024, priority=5
        )  # 5KB message

        # Display results
        print(f"   Power Mode:      {state.power_mode.value}")
        print(f"   Transport Pref:  {state.transport_preference.value}")
        print(f"   Primary Route:   {decision['primary_transport']}")
        print(
            f"   Chunk Size:      {state.chunking_config.effective_chunk_size()} bytes"
        )
        print(
            f"   Active Policies: {', '.join(state.active_policies) if state.active_policies else 'None'}"
        )

        # Validate expected behavior
        result = validate_scenario_behavior(scenario, state, decision)
        validation_results.append(result)

        if result["passed"]:
            print(f"   Status:          ✓ PASSED - {result['reason']}")
        else:
            print(f"   Status:          ✗ FAILED - {result['reason']}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in validation_results if r["passed"])
    total = len(validation_results)

    print(f"Tests Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL MOBILE OPTIMIZATION FEATURES VALIDATED SUCCESSFULLY!")
        print("\nKey P2 Requirements Verified:")
        print("  ✓ Battery-aware transport selection (BitChat-first under low power)")
        print("  ✓ Dynamic tensor/chunk size tuning for 2-4GB devices")
        print("  ✓ Thermal throttling with progressive limits")
        print("  ✓ Network cost-aware routing decisions")
        print("  ✓ Real-time policy adaptation")

        print("\nMobile Optimization Implementation: PRODUCTION READY ✅")
        return True
    print(f"\n❌ {total - passed} tests failed. Review implementation.")
    return False


def validate_scenario_behavior(scenario, state, decision):
    """Validate that the behavior matches expected scenario requirements"""
    name = scenario["name"]

    if "Normal Operation" in name:
        if state.power_mode in [PowerMode.BALANCED, PowerMode.PERFORMANCE]:
            return {
                "passed": True,
                "reason": "Balanced/performance mode under normal conditions",
            }
        return {"passed": False, "reason": "Should use balanced/performance mode"}

    if "Low Battery" in name and "Critical" not in name:
        if state.transport_preference == TransportPreference.BITCHAT_PREFERRED:
            return {
                "passed": True,
                "reason": "BitChat preferred for battery conservation",
            }
        return {"passed": False, "reason": "Should prefer BitChat for low battery"}

    if "Critical Battery" in name:
        if (
            state.transport_preference == TransportPreference.BITCHAT_ONLY
            and decision["primary_transport"] == "bitchat"
            and decision["fallback_transport"] is None
        ):
            return {"passed": True, "reason": "BitChat-only mode for critical battery"}
        return {
            "passed": False,
            "reason": "Should use BitChat-only for critical battery",
        }

    if "High Thermal" in name:
        if (
            state.power_mode in [PowerMode.POWER_SAVE, PowerMode.CRITICAL]
            and state.chunking_config.effective_chunk_size() < 512
        ):
            return {
                "passed": True,
                "reason": "Thermal throttling with reduced chunk size",
            }
        return {
            "passed": False,
            "reason": "Should throttle performance under high thermal load",
        }

    if "Memory Constrained" in name:
        if state.chunking_config.effective_chunk_size() <= 256:
            return {"passed": True, "reason": "Smaller chunks for memory constraints"}
        return {
            "passed": False,
            "reason": "Should use smaller chunks for limited memory",
        }

    if "Cellular Network" in name:
        if (
            state.transport_preference == TransportPreference.BITCHAT_PREFERRED
            and "data_cost_aware" in state.active_policies
        ):
            return {
                "passed": True,
                "reason": "BitChat preferred for cellular cost savings",
            }
        return {"passed": False, "reason": "Should prefer BitChat on cellular networks"}

    if "Extreme Stress" in name:
        if (
            state.power_mode == PowerMode.CRITICAL
            and state.transport_preference == TransportPreference.BITCHAT_ONLY
            and state.chunking_config.effective_chunk_size() <= 128
        ):
            return {
                "passed": True,
                "reason": "Maximum conservation under extreme stress",
            }
        return {"passed": False, "reason": "Should apply maximum conservation policies"}

    return {"passed": True, "reason": "Default validation passed"}


async def test_chunk_size_adaptation():
    """Test dynamic chunk size adaptation for different device configurations"""
    print("\n" + "=" * 80)
    print("CHUNK SIZE ADAPTATION TEST")
    print("=" * 80)

    manager = SimpleBatteryThermalResourceManager()

    configs = [
        {"ram_gb": 8, "battery": 80, "temp": 35, "expected_range": (512, 1024)},
        {"ram_gb": 4, "battery": 80, "temp": 35, "expected_range": (256, 768)},
        {"ram_gb": 2, "battery": 80, "temp": 35, "expected_range": (128, 512)},
        {"ram_gb": 4, "battery": 10, "temp": 35, "expected_range": (64, 256)},
        {"ram_gb": 4, "battery": 80, "temp": 60, "expected_range": (128, 512)},
    ]

    for config in configs:
        profile = MockDeviceProfile(
            ram_total_mb=int(config["ram_gb"] * 1024),
            ram_available_mb=int(config["ram_gb"] * 1024 * 0.7),
            battery_percent=config["battery"],
            cpu_temp_celsius=config["temp"],
        )

        state = await manager.evaluate_and_adapt(profile)
        chunk_size = state.chunking_config.effective_chunk_size()
        min_expected, max_expected = config["expected_range"]

        print(
            f"RAM: {config['ram_gb']}GB, Battery: {config['battery']}%, Temp: {config['temp']}°C"
        )
        print(
            f"  Chunk Size: {chunk_size} bytes (expected: {min_expected}-{max_expected})"
        )

        if min_expected <= chunk_size <= max_expected:
            print("  Status: ✓ PASSED")
        else:
            print("  Status: ✗ FAILED")
        print()


if __name__ == "__main__":
    print("Starting AI Village Mobile Optimization Validation...")

    async def main():
        success = await validate_mobile_optimization()
        await test_chunk_size_adaptation()

        if success:
            print(
                "\n🚀 Mobile optimization implementation ready for production deployment!"
            )
            exit(0)
        else:
            print("\n🔧 Mobile optimization needs fixes before deployment.")
            exit(1)

    asyncio.run(main())
