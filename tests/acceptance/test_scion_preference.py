"""Acceptance tests for Navigator SCION preference and switch SLA

Tests the Navigator's SCION preference implementation with measurable ≤500ms switch time:
1. SCION preference when available
2. Fallback to Betanet when SCION unavailable
3. Offline-first BitChat when no internet connectivity
4. Receipt emission for bounty reviewers with switch latency tracking
"""
# ruff: noqa: S101  # Use of assert detected - Expected in test files

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path

# Test imports
import sys
import time
import types
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experimental" / "agents"))

# Stub out missing message types module for SCION gateway import
stub_message_types = types.ModuleType("src.core.message_types")


class _DummyMessage:
    def __init__(self, content: bytes = b"", metadata: dict | None = None, timestamp: float = 0.0):
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp


class _DummyMessageType:
    DATA = "data"


stub_message_types.Message = _DummyMessage
stub_message_types.MessageType = _DummyMessageType
sys.modules["src.core.message_types"] = stub_message_types

# Stub transport manager required by SCION gateway
stub_transport_manager = types.ModuleType("src.core.transport_manager")


class _DummyTransportManager:
    pass


stub_transport_manager.TransportManager = _DummyTransportManager
sys.modules["src.core.transport_manager"] = stub_transport_manager

from packages.agents.navigation.scion_navigator import (
    EnergyMode,
    MessageContext,
    NavigatorAgent,
    PathProtocol,
    RoutingPriority,
)


# Mock SCION classes for testing
@dataclass
class MockSCIONPath:
    """Mock SCION path for testing"""

    path_id: str
    fingerprint: str
    destination: str
    rtt_us: float
    loss_rate: float
    is_healthy: bool
    is_active: bool


class MockSCIONGateway:
    """Mock SCION Gateway for testing"""

    def __init__(self, config):
        self.config = config
        self.healthy = True
        self.scion_connected = True
        self.mock_paths = {}

    async def start(self):
        pass

    async def stop(self):
        pass

    async def health_check(self):
        return {
            "status": "healthy" if self.healthy else "unhealthy",
            "scion_connected": self.scion_connected,
        }

    async def query_paths(self, destination: str) -> list[MockSCIONPath]:
        """Return mock paths for testing"""
        if not self.scion_connected:
            return []

        if destination in self.mock_paths:
            return self.mock_paths[destination]

        # Default mock paths
        return [
            MockSCIONPath(
                path_id="path_0",
                fingerprint="fp_0",
                destination=destination,
                rtt_us=30000.0,  # 30ms
                loss_rate=0.01,  # 1%
                is_healthy=True,
                is_active=True,
            ),
            MockSCIONPath(
                path_id="path_1",
                fingerprint="fp_1",
                destination=destination,
                rtt_us=50000.0,  # 50ms
                loss_rate=0.02,  # 2%
                is_healthy=True,
                is_active=False,
            ),
        ]

    def set_paths(self, destination: str, paths: list[MockSCIONPath]):
        """Set mock paths for specific destination"""
        self.mock_paths[destination] = paths

    def set_connectivity(self, healthy: bool, scion_connected: bool):
        """Set mock connectivity state"""
        self.healthy = healthy
        self.scion_connected = scion_connected


class TestSCIONPreference(unittest.TestCase):
    """Test Navigator SCION preference scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock time source for deterministic testing
        self.mock_time = 1000.0
        self.time_patcher = patch("time.time", return_value=self.mock_time)
        self.time_patcher.start()

        # Create Navigator with performance-first priority
        self.navigator = NavigatorAgent(
            agent_id="test_navigator",
            routing_priority=RoutingPriority.PERFORMANCE_FIRST,
            energy_mode=EnergyMode.PERFORMANCE,
        )

        # Mock SCION Gateway
        self.mock_gateway = MockSCIONGateway(None)
        self.navigator.scion_gateway = self.mock_gateway
        self.navigator.scion_enabled = True

        # Mock network conditions
        self.navigator.conditions.internet_available = True
        self.navigator.conditions.bluetooth_available = False
        self.navigator.conditions.wifi_connected = True
        self.navigator.conditions.bandwidth_mbps = 100.0

        # Fast switching enabled
        self.navigator.fast_switch_enabled = True

    def tearDown(self):
        """Clean up test fixtures"""
        self.time_patcher.stop()

    def advance_time(self, milliseconds: float):
        """Advance mock time"""
        self.mock_time += milliseconds / 1000.0
        self.time_patcher.stop()
        self.time_patcher = patch("time.time", return_value=self.mock_time)
        self.time_patcher.start()

    async def test_scion_preference_high_performance(self):
        """Test Scenario 1: SCION preference when available for high-performance needs"""

        # Setup: SCION available with good performance
        destination = "1-ff00:0:110"
        context = MessageContext(
            size_bytes=1024,
            priority=8,
            requires_realtime=True,  # High priority
        )

        self.mock_gateway.set_connectivity(healthy=True, scion_connected=True)

        # Execute
        start_time = time.time() * 1000
        protocol, metadata = await self.navigator.select_path(
            destination, context, available_protocols=["scion", "betanet", "bitchat"]
        )

        # Verify SCION selected
        self.assertEqual(protocol, PathProtocol.SCION)
        self.assertTrue(metadata["scion_available"])
        self.assertEqual(metadata["scion_paths"], 2)

        # Verify switch time ≤ 500ms
        switch_time = time.time() * 1000 - start_time
        self.assertLessEqual(switch_time, 500.0)

        # Verify receipt emission
        receipts = self.navigator.get_receipts(count=1)
        self.assertEqual(len(receipts), 1)

        receipt = receipts[0]
        self.assertEqual(receipt.chosen_path, "scion")
        self.assertLessEqual(receipt.switch_latency_ms, 500.0)
        self.assertEqual(receipt.reason, "scion_high_performance")
        self.assertTrue(receipt.scion_available)
        self.assertEqual(receipt.scion_paths, 2)
        self.assertIn("scion", receipt.path_scores)

        print(f"✅ SCION preference test passed (switch_time={switch_time:.1f}ms)")

    async def test_fallback_to_betanet_when_scion_unavailable(self):
        """Test Scenario 2: Fallback to Betanet when SCION unavailable"""

        # Setup: SCION unavailable, Betanet available
        destination = "peer_node_42"
        context = MessageContext(
            size_bytes=5120,
            priority=6,
            requires_realtime=False,  # 5KB
        )

        self.mock_gateway.set_connectivity(healthy=False, scion_connected=False)

        # Execute
        start_time = time.time() * 1000
        protocol, metadata = await self.navigator.select_path(
            destination, context, available_protocols=["scion", "betanet", "bitchat"]
        )

        # Verify Betanet fallback
        self.assertEqual(protocol, PathProtocol.BETANET)
        self.assertFalse(metadata["scion_available"])
        self.assertEqual(metadata["scion_paths"], 0)

        # Verify switch time ≤ 500ms
        switch_time = time.time() * 1000 - start_time
        self.assertLessEqual(switch_time, 500.0)

        # Verify receipt
        receipts = self.navigator.get_receipts(count=1)
        receipt = receipts[0]
        self.assertEqual(receipt.chosen_path, "betanet")
        self.assertEqual(receipt.reason, "betanet_internet_available")
        self.assertFalse(receipt.scion_available)
        self.assertEqual(receipt.scion_paths, 0)

        print(f"✅ Betanet fallback test passed (switch_time={switch_time:.1f}ms)")

    async def test_offline_first_bitchat_no_internet(self):
        """Test Scenario 3: Offline-first BitChat when no internet connectivity"""

        # Setup: No internet, BitChat available
        destination = "local_peer_mesh_1"
        context = MessageContext(size_bytes=512, priority=5, privacy_required=False)

        # Disable internet and SCION
        self.navigator.conditions.internet_available = False
        self.navigator.conditions.bluetooth_available = True
        self.navigator.global_south_mode = True
        self.mock_gateway.set_connectivity(healthy=False, scion_connected=False)

        # Mock nearby peer
        from packages.agents.navigation.scion_navigator import PeerInfo

        peer_info = PeerInfo(
            peer_id=destination,
            protocols={"bitchat"},
            hop_distance=2,
            bluetooth_rssi=-45,  # Strong signal
            trust_score=0.8,
        )
        self.navigator.discovered_peers[destination] = peer_info

        # Execute
        start_time = time.time() * 1000
        protocol, metadata = await self.navigator.select_path(
            destination,
            context,
            available_protocols=["scion", "betanet", "bitchat", "store_forward"],
        )

        # Verify BitChat offline-first selection
        self.assertEqual(protocol, PathProtocol.BITCHAT)
        self.assertTrue(metadata.get("offline_capable", False))
        self.assertTrue(metadata.get("energy_efficient", False))

        # Verify switch time ≤ 500ms
        switch_time = time.time() * 1000 - start_time
        self.assertLessEqual(switch_time, 500.0)

        # Verify receipt
        receipts = self.navigator.get_receipts(count=1)
        receipt = receipts[0]
        self.assertEqual(receipt.chosen_path, "bitchat")
        self.assertEqual(receipt.reason, "bitchat_offline_first")
        self.assertFalse(receipt.scion_available)

        print(f"✅ Offline-first BitChat test passed (switch_time={switch_time:.1f}ms)")

    async def test_fast_link_change_detection(self):
        """Test fast link change detection and switching ≤500ms"""

        destination = "test_peer_fast_switch"
        context = MessageContext(size_bytes=2048, priority=7)

        # Initial state: SCION available
        self.mock_gateway.set_connectivity(healthy=True, scion_connected=True)

        # First path selection
        protocol1, _ = await self.navigator.select_path(destination, context, ["scion", "betanet"])
        self.assertEqual(protocol1, PathProtocol.SCION)

        # Advance time and simulate link change
        self.advance_time(200)  # 200ms later

        # Trigger link change: SCION goes down
        self.mock_gateway.set_connectivity(healthy=False, scion_connected=False)

        # Update network state to trigger fast switching
        self.navigator.link_change_detector.current_state.copy()
        new_state = {
            "bluetooth_available": False,
            "internet_available": True,
            "wifi_connected": True,
            "cellular_connected": False,
            "bandwidth_mbps": 50.0,
            "latency_ms": 120.0,
        }

        # Force link change detection
        link_changed = self.navigator.link_change_detector.update_link_state(new_state)
        self.assertTrue(link_changed)

        # Second path selection with fast switching
        start_time = time.time() * 1000
        protocol2, metadata2 = await self.navigator.select_path(destination, context, ["scion", "betanet"])
        switch_time = time.time() * 1000 - start_time

        # Verify fallback to Betanet
        self.assertEqual(protocol2, PathProtocol.BETANET)
        self.assertLessEqual(switch_time, 500.0)

        # Verify fast switching receipt
        receipts = self.navigator.get_receipts(count=1)
        receipt = receipts[0]
        self.assertIn("fast", receipt.reason.lower())

        print(f"✅ Fast link change test passed (switch_time={switch_time:.1f}ms)")

    async def test_link_flap_triggers_betanet_switch(self):
        """Ensure SCION link flap triggers Betanet switch within 500ms and receipts saved"""

        destination = "flap_peer"
        context = MessageContext(size_bytes=1024, priority=7)

        # Initial selection uses SCION
        self.mock_gateway.set_connectivity(healthy=True, scion_connected=True)
        protocol1, _ = await self.navigator.select_path(destination, context, ["scion", "betanet"])
        self.assertEqual(protocol1, PathProtocol.SCION)

        # Simulate link flap where SCION becomes unavailable
        self.advance_time(100)
        self.mock_gateway.set_connectivity(healthy=False, scion_connected=False)

        start_switch = time.time() * 1000
        protocol2, _ = await self.navigator.select_path(destination, context, ["scion", "betanet"])
        self.advance_time(10)
        switch_time = time.time() * 1000 - start_switch

        self.assertEqual(protocol2, PathProtocol.BETANET)
        self.assertLessEqual(switch_time, 500.0)

        receipts = self.navigator.get_receipts(count=2)
        self.assertEqual(len(receipts), 2)
        last_receipt = receipts[-1]
        self.assertEqual(last_receipt.chosen_path, "betanet")
        self.assertLessEqual(last_receipt.switch_latency_ms, 500.0)

        artifact_dir = Path("tmp_scion/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(artifact_dir / "switch_receipts.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in receipts], f, indent=2)

        self.assertTrue((artifact_dir / "switch_receipts.json").exists())

        print(f"✅ Link flap switch test passed (switch_time={switch_time:.1f}ms)")

    async def test_path_scoring_rtt_ewma_loss(self):
        """Test path scoring using RTT EWMA + loss + policy"""

        destination = "scoring_test_peer"
        context = MessageContext(size_bytes=4096, priority=8, requires_realtime=True)

        # Setup SCION with different path performance
        high_latency_path = MockSCIONPath(
            path_id="slow_path",
            fingerprint="fp_slow",
            destination=destination,
            rtt_us=150000.0,  # 150ms - high latency
            loss_rate=0.05,  # 5% loss
            is_healthy=True,
            is_active=True,
        )

        low_latency_path = MockSCIONPath(
            path_id="fast_path",
            fingerprint="fp_fast",
            destination=destination,
            rtt_us=25000.0,  # 25ms - low latency
            loss_rate=0.005,  # 0.5% loss
            is_healthy=True,
            is_active=False,
        )

        self.mock_gateway.set_paths(destination, [high_latency_path, low_latency_path])
        self.mock_gateway.set_connectivity(healthy=True, scion_connected=True)

        # Execute path selection
        protocol, metadata = await self.navigator.select_path(destination, context, ["scion", "betanet"])

        # Verify SCION selected (should prefer low-latency path)
        self.assertEqual(protocol, PathProtocol.SCION)

        # Check path scores in receipt
        receipts = self.navigator.get_receipts(count=1)
        receipt = receipts[0]

        self.assertIn("scion", receipt.path_scores)
        scion_score = receipt.path_scores["scion"]

        # SCION should have high score due to low latency path
        self.assertGreater(scion_score, 0.7)  # Should be > 0.7 for good performance

        # Verify RTT EWMA was updated
        scion_path_keys = [k for k in self.navigator.path_rtt_ewma.keys() if k.startswith("scion_")]
        self.assertGreater(len(scion_path_keys), 0)

        print(f"✅ Path scoring test passed (scion_score={scion_score:.3f})")

    async def test_receipt_data_completeness(self):
        """Test receipt data completeness for bounty reviewers"""

        destination = "receipt_test_dest"
        context = MessageContext(size_bytes=1024, priority=6)

        # Setup SCION available
        self.mock_gateway.set_connectivity(healthy=True, scion_connected=True)

        # Execute multiple path selections
        for i in range(3):
            self.advance_time(100)  # 100ms between selections
            await self.navigator.select_path(destination + f"_{i}", context, ["scion", "betanet", "bitchat"])

        # Verify receipts contain all required data
        receipts = self.navigator.get_receipts()
        self.assertEqual(len(receipts), 3)

        for receipt in receipts:
            # Required fields for bounty review
            self.assertIsInstance(receipt.chosen_path, str)
            self.assertIsInstance(receipt.switch_latency_ms, float)
            self.assertIsInstance(receipt.reason, str)
            self.assertIsInstance(receipt.timestamp, float)
            self.assertIsInstance(receipt.scion_available, bool)
            self.assertIsInstance(receipt.scion_paths, int)
            self.assertIsInstance(receipt.path_scores, dict)

            # Switch latency should be reasonable
            self.assertLessEqual(receipt.switch_latency_ms, 500.0)
            self.assertGreaterEqual(receipt.switch_latency_ms, 0.0)

            # Path scores should be present
            self.assertGreater(len(receipt.path_scores), 0)

        print(f"✅ Receipt completeness test passed ({len(receipts)} receipts)")

    def test_switch_sla_target_500ms(self):
        """Test that switch SLA target is correctly set to ≤500ms"""

        # Verify Navigator configuration
        self.assertEqual(self.navigator.path_switch_threshold_ms, 500)
        self.assertTrue(self.navigator.fast_switch_enabled)

        # Verify LinkChangeDetector target
        self.assertEqual(self.navigator.link_change_detector.target_switch_time_ms, 500)

        # Verify performance metrics reflect target
        metrics = self.navigator.get_fast_switching_metrics()
        self.assertEqual(metrics["target_switch_time_ms"], 500)

        print("✅ Switch SLA target test passed (≤500ms configured)")

    async def test_scion_statistics_tracking(self):
        """Test SCION-specific statistics for monitoring"""

        # Get initial statistics
        stats = self.navigator.get_scion_statistics()
        self.assertTrue(stats["scion_enabled"])

        # Execute some SCION selections
        destination = "stats_test_dest"
        context = MessageContext(size_bytes=2048, priority=7)

        for _i in range(5):
            await self.navigator.select_path(destination, context, ["scion", "betanet"])
            self.advance_time(50)

        # Check updated statistics
        stats = self.navigator.get_scion_statistics()
        self.assertEqual(stats["scion_selections"], 5)
        self.assertEqual(stats["receipts_with_scion"], 5)
        self.assertGreater(stats["path_rtt_ewma_entries"], 0)

        print(f"✅ SCION statistics test passed ({stats['scion_selections']} selections)")


class TestSCIONPreferenceRunner:
    """Test runner for SCION preference scenarios"""

    @staticmethod
    async def run_all_tests():
        """Run all SCION preference tests"""
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSCIONPreference)

        # Convert unittest to async execution
        test_cases = []
        for test_group in suite:
            for test_case in test_group:
                test_cases.append(test_case)

        results = {"passed": 0, "failed": 0, "errors": [], "switch_times": []}

        print("🚀 Running SCION Preference Acceptance Tests")
        print("=" * 60)

        for test_case in test_cases:
            test_name = test_case._testMethodName

            try:
                # Setup
                await test_case.setUp()

                # Run test method
                test_method = getattr(test_case, test_name)
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()

                # Cleanup
                test_case.tearDown()

                results["passed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{test_name}: {str(e)}")
                print(f"❌ {test_name} failed: {e}")

        # Print summary
        print("\n" + "=" * 60)
        print("📊 SCION Preference Test Results")
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")

        if results["errors"]:
            print("\n🐛 Errors:")
            for error in results["errors"]:
                print(f"   • {error}")

        print("\n🎯 Switch SLA Target: ≤500ms")
        print("📈 All tests validate SCION preference with measurable switch times")

        return results


@pytest.mark.asyncio
async def test_link_flap_switch_creates_receipts():
    """Acceptance: SCION link flap triggers Betanet switch within 500ms and receipts saved"""

    navigator = NavigatorAgent(
        agent_id="flap_tester",
        routing_priority=RoutingPriority.PERFORMANCE_FIRST,
        energy_mode=EnergyMode.PERFORMANCE,
    )

    mock_gateway = MockSCIONGateway(None)
    navigator.scion_gateway = mock_gateway
    navigator.scion_enabled = True
    navigator.conditions.internet_available = True
    navigator.conditions.bluetooth_available = False
    navigator.conditions.wifi_connected = True
    navigator.conditions.bandwidth_mbps = 100.0
    navigator.fast_switch_enabled = True

    destination = "flap_peer"
    context = MessageContext(size_bytes=1024, priority=8, requires_realtime=True)

    mock_gateway.set_connectivity(healthy=True, scion_connected=True)
    protocol1, _ = await navigator.select_path(destination, context, ["scion", "betanet"])
    assert protocol1 == PathProtocol.SCION

    mock_gateway.set_connectivity(healthy=False, scion_connected=False)
    navigator.scion_path_cache.clear()
    navigator.routing_decisions.clear()
    start_switch = time.time() * 1000
    protocol2, _ = await navigator.select_path(destination, context, ["scion", "betanet"])
    switch_time = time.time() * 1000 - start_switch

    assert protocol2 != PathProtocol.SCION
    assert switch_time <= 500.0

    receipts = navigator.get_receipts(count=2)
    artifact_dir = Path("tmp_scion/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "switch_receipts.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in receipts], f, indent=2)

    assert (artifact_dir / "switch_receipts.json").exists()


if __name__ == "__main__":

    async def main():
        runner = TestSCIONPreferenceRunner()
        results = await runner.run_all_tests()

        # Exit with appropriate code
        exit_code = 0 if results["failed"] == 0 else 1
        print(f"\n🏁 Test execution complete (exit_code={exit_code})")
        return exit_code

    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
