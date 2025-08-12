#!/usr/bin/env python3
"""Comprehensive mobile policy tests with environment-driven simulation.

Tests the P2 Mobile Resource Optimization features:
- Battery-aware transport selection (BitChat-first under low power)
- Dynamic tensor/chunk size tuning for 2-4GB devices
- Thermal throttling with progressive limits
- Network cost-aware routing decisions
- Real-time policy adaptation
"""

import asyncio
import os
import sys
import unittest

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from test_mobile_policy_simple import (
    PowerMode,
    SimpleBatteryThermalResourceManager,
    TransportPreference,
)


class TestMobilePolicyWithEnv(unittest.TestCase):
    """Test mobile resource policies using environment-driven simulation"""

    def setUp(self):
        """Set up test environment"""
        # Clear any existing environment variables
        self.original_env = {}
        env_vars = [
            "AIV_MOBILE_PROFILE",
            "BATTERY",
            "THERMAL",
            "MEMORY_GB",
            "NETWORK_TYPE",
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore original environment"""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    async def test_low_ram_profile(self):
        """Test AIV_MOBILE_PROFILE=low_ram scenario"""
        os.environ["AIV_MOBILE_PROFILE"] = "low_ram"
        os.environ["BATTERY"] = "15"
        os.environ["THERMAL"] = "hot"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # Verify smaller tensor chunks for limited memory
        self.assertLessEqual(state.chunking_config.effective_chunk_size(), 512)

        # Verify BitChat-first routing
        self.assertEqual(
            state.transport_preference, TransportPreference.BITCHAT_PREFERRED
        )

        # Verify power conservation
        self.assertIn(state.power_mode, [PowerMode.POWER_SAVE, PowerMode.CRITICAL])

        print(
            f"+ Low RAM profile: chunk_size={state.chunking_config.effective_chunk_size()}, "
            f"transport={state.transport_preference.value}, power={state.power_mode.value}"
        )

    async def test_critical_battery_scenario(self):
        """Test BATTERY=5 critical battery scenario"""
        os.environ["BATTERY"] = "5"
        os.environ["THERMAL"] = "normal"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # Critical battery should force BitChat-only
        self.assertEqual(state.transport_preference, TransportPreference.BITCHAT_ONLY)
        self.assertEqual(state.power_mode, PowerMode.CRITICAL)

        # Should use minimal chunk sizes
        self.assertLessEqual(state.chunking_config.effective_chunk_size(), 256)

        # Verify routing decision
        decision = await manager.get_transport_routing_decision(1024, priority=5)
        self.assertEqual(decision["primary_transport"], "bitchat")
        self.assertIsNone(decision["fallback_transport"])

        print(
            f"+ Critical battery: transport_only={decision['primary_transport']}, "
            f"chunk_size={state.chunking_config.effective_chunk_size()}"
        )

    async def test_thermal_throttling(self):
        """Test THERMAL=hot thermal throttling scenario"""
        os.environ["THERMAL"] = "65"  # 65°C
        os.environ["BATTERY"] = "80"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # Thermal throttling should reduce chunk sizes
        self.assertLessEqual(state.chunking_config.effective_chunk_size(), 400)

        # Should prefer power save mode
        self.assertIn(state.power_mode, [PowerMode.POWER_SAVE, PowerMode.CRITICAL])

        # Should have thermal mitigation active
        thermal_policies = [p for p in state.active_policies if "thermal" in p.lower()]
        self.assertGreater(len(thermal_policies), 0)

        print(
            f"+ Thermal throttling: temp=65°C, power={state.power_mode.value}, "
            f"chunk_size={state.chunking_config.effective_chunk_size()}"
        )

    async def test_cellular_network_optimization(self):
        """Test NETWORK_TYPE=cellular data cost awareness"""
        os.environ["NETWORK_TYPE"] = "cellular"
        os.environ["BATTERY"] = "50"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # Should prefer BitChat for data cost savings
        self.assertEqual(
            state.transport_preference, TransportPreference.BITCHAT_PREFERRED
        )

        # Should have data cost awareness active
        self.assertIn("data_cost_aware", state.active_policies)

        # Routing should prefer BitChat
        decision = await manager.get_transport_routing_decision(5120, priority=5)
        self.assertEqual(decision["primary_transport"], "bitchat")

        print(
            f"+ Cellular network: transport={decision['primary_transport']}, "
            f"policies={state.active_policies}"
        )

    async def test_memory_constrained_device(self):
        """Test MEMORY_GB=2 constrained device scenario"""
        os.environ["MEMORY_GB"] = "2"
        os.environ["BATTERY"] = "70"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # 2GB device should use smaller chunks
        self.assertLessEqual(state.chunking_config.effective_chunk_size(), 512)

        # Should have memory constraint policy active
        self.assertIn("memory_constrained", state.active_policies)

        # Get chunking recommendations
        tensor_rec = manager.get_chunking_recommendations("tensor")
        self.assertLessEqual(tensor_rec["chunk_size"], 512)
        self.assertGreaterEqual(tensor_rec["batch_size"], 1)

        print(
            f"+ Memory constrained: chunk_size={tensor_rec['chunk_size']}, "
            f"batch_size={tensor_rec['batch_size']}"
        )

    async def test_performance_mode(self):
        """Test AIV_MOBILE_PROFILE=performance scenario"""
        os.environ["AIV_MOBILE_PROFILE"] = "performance"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # Performance mode should allow larger chunks
        self.assertGreaterEqual(state.chunking_config.effective_chunk_size(), 500)

        # Should use balanced transport
        self.assertEqual(state.transport_preference, TransportPreference.BALANCED)

        # Should be in performance or balanced power mode
        self.assertIn(state.power_mode, [PowerMode.PERFORMANCE, PowerMode.BALANCED])

        print(
            f"+ Performance mode: chunk_size={state.chunking_config.effective_chunk_size()}, "
            f"power={state.power_mode.value}"
        )

    async def test_extreme_stress_scenario(self):
        """Test extreme stress: low battery + high thermal + limited memory"""
        os.environ["BATTERY"] = "5"
        os.environ["THERMAL"] = "critical"
        os.environ["MEMORY_GB"] = "1.5"
        os.environ["NETWORK_TYPE"] = "cellular"

        manager = SimpleBatteryThermalResourceManager()
        state = await manager.evaluate_and_adapt()

        # Should apply maximum conservation
        self.assertEqual(state.power_mode, PowerMode.CRITICAL)
        self.assertEqual(state.transport_preference, TransportPreference.BITCHAT_ONLY)
        self.assertLessEqual(state.chunking_config.effective_chunk_size(), 128)

        # Multiple policies should be active
        self.assertGreaterEqual(len(state.active_policies), 3)

        # Routing should be BitChat-only
        decision = await manager.get_transport_routing_decision(1024, priority=10)
        self.assertEqual(decision["primary_transport"], "bitchat")
        self.assertIsNone(decision["fallback_transport"])

        print(
            f"+ Extreme stress: chunk_size={state.chunking_config.effective_chunk_size()}, "
            f"policies={len(state.active_policies)}, transport_only={decision['primary_transport']}"
        )

    async def test_chunk_size_adaptation_thresholds(self):
        """Test chunk size adaptation at different memory/battery/thermal thresholds"""
        scenarios = [
            {"memory_gb": "8", "battery": "90", "thermal": "30", "expected_min": 500},
            {"memory_gb": "4", "battery": "80", "thermal": "35", "expected_min": 200},
            {"memory_gb": "2", "battery": "60", "thermal": "40", "expected_min": 100},
            {"memory_gb": "2", "battery": "30", "thermal": "50", "expected_max": 300},
            {"memory_gb": "1.5", "battery": "10", "thermal": "60", "expected_max": 150},
        ]

        results = []
        for scenario in scenarios:
            # Clear and set environment
            for var in ["MEMORY_GB", "BATTERY", "THERMAL"]:
                if var in os.environ:
                    del os.environ[var]

            os.environ["MEMORY_GB"] = scenario["memory_gb"]
            os.environ["BATTERY"] = scenario["battery"]
            os.environ["THERMAL"] = scenario["thermal"]

            manager = SimpleBatteryThermalResourceManager()
            state = await manager.evaluate_and_adapt()
            chunk_size = state.chunking_config.effective_chunk_size()

            results.append(
                {
                    "config": f"RAM:{scenario['memory_gb']}GB B:{scenario['battery']}% T:{scenario['thermal']}°C",
                    "chunk_size": chunk_size,
                }
            )

            # Verify expectations
            if "expected_min" in scenario:
                self.assertGreaterEqual(chunk_size, scenario["expected_min"])
            if "expected_max" in scenario:
                self.assertLessEqual(chunk_size, scenario["expected_max"])

        print("+ Chunk size adaptation test results:")
        for result in results:
            print(f"   {result['config']} -> {result['chunk_size']} bytes")

    async def test_transport_selection_logic(self):
        """Test transport selection based on different constraints"""
        scenarios = [
            {
                "battery": "90",
                "network": "wifi",
                "expected": TransportPreference.BALANCED,
            },
            {
                "battery": "50",
                "network": "wifi",
                "expected": TransportPreference.BALANCED,
            },  # Above conservative threshold
            {
                "battery": "15",
                "network": "wifi",
                "expected": TransportPreference.BITCHAT_PREFERRED,
            },
            {
                "battery": "8",
                "network": "wifi",
                "expected": TransportPreference.BITCHAT_ONLY,
            },
            {
                "battery": "60",
                "network": "cellular",
                "expected": TransportPreference.BITCHAT_PREFERRED,
            },
            {
                "battery": "90",
                "network": "3g",
                "expected": TransportPreference.BITCHAT_PREFERRED,
            },
        ]

        for scenario in scenarios:
            # Clear and set environment
            for var in ["BATTERY", "NETWORK_TYPE"]:
                if var in os.environ:
                    del os.environ[var]

            os.environ["BATTERY"] = scenario["battery"]
            os.environ["NETWORK_TYPE"] = scenario["network"]

            manager = SimpleBatteryThermalResourceManager()
            state = await manager.evaluate_and_adapt()

            self.assertEqual(state.transport_preference, scenario["expected"])

            print(
                f"+ Transport: B:{scenario['battery']}% {scenario['network']} -> {state.transport_preference.value}"
            )


async def run_all_tests():
    """Run all mobile policy tests"""
    print("=" * 80)
    print("Mobile Resource Policy Tests - Environment-Driven Simulation")
    print("=" * 80)

    test_instance = TestMobilePolicyWithEnv()

    tests = [
        test_instance.test_low_ram_profile,
        test_instance.test_critical_battery_scenario,
        test_instance.test_thermal_throttling,
        test_instance.test_cellular_network_optimization,
        test_instance.test_memory_constrained_device,
        test_instance.test_performance_mode,
        test_instance.test_extreme_stress_scenario,
        test_instance.test_chunk_size_adaptation_thresholds,
        test_instance.test_transport_selection_logic,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test_instance.setUp()
            await test()
            test_instance.tearDown()
            passed += 1
        except Exception as e:
            print(f"X {test.__name__} FAILED: {e}")
            failed += 1
            test_instance.tearDown()

    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
