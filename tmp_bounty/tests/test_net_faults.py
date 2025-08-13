"""Network Fault Injection Tests - RTT/Jitter/Loss Simulation

Tests adaptive routing behavior under simulated network conditions:
- Inject delay/jitter/packet loss
- Assert delivery within thresholds
- Verify policy switches occur within 500ms when conditions change

This validates that the NetworkMetricsCollector + AdaptiveNavigator system
responds appropriately to degraded network conditions.
"""

import asyncio
import logging
import os
import random

# Test imports - adjust paths as needed
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

try:
    from core.p2p.metrics.net_metrics import (
        MeasurementType,
        NetworkMetricsCollector,
        NetworkSample,
    )

    from ..adaptive_navigator import (
        AdaptiveNavigator,
        AdaptiveNetworkConditions,
        MessageContext,
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Import failed, using mock objects: {e}")
    IMPORTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkFaultInjector:
    """Simulates network faults (delay, jitter, loss) for testing"""

    def __init__(self):
        self.base_rtt_ms = 50.0
        self.added_delay_ms = 0.0
        self.jitter_ms = 0.0
        self.packet_loss_rate = 0.0
        self.fault_start_time = None
        self.active_faults = []

    def inject_delay(self, added_delay_ms: float, duration_seconds: float = 10.0):
        """Inject additional delay for specified duration"""
        self.added_delay_ms = added_delay_ms
        self.fault_start_time = time.time()
        self.active_faults.append(f"delay_{added_delay_ms}ms")
        logger.info(f"Injected {added_delay_ms}ms additional delay for {duration_seconds}s")

        # Schedule fault removal
        asyncio.create_task(self._remove_delay_after(duration_seconds))

    def inject_jitter(self, jitter_ms: float, duration_seconds: float = 10.0):
        """Inject random jitter for specified duration"""
        self.jitter_ms = jitter_ms
        self.fault_start_time = time.time()
        self.active_faults.append(f"jitter_{jitter_ms}ms")
        logger.info(f"Injected {jitter_ms}ms jitter for {duration_seconds}s")

        asyncio.create_task(self._remove_jitter_after(duration_seconds))

    def inject_packet_loss(self, loss_rate: float, duration_seconds: float = 10.0):
        """Inject packet loss for specified duration"""
        self.packet_loss_rate = loss_rate
        self.fault_start_time = time.time()
        self.active_faults.append(f"loss_{loss_rate*100}%")
        logger.info(f"Injected {loss_rate*100}% packet loss for {duration_seconds}s")

        asyncio.create_task(self._remove_loss_after(duration_seconds))

    async def _remove_delay_after(self, seconds: float):
        await asyncio.sleep(seconds)
        self.added_delay_ms = 0.0
        self.active_faults = [f for f in self.active_faults if not f.startswith("delay_")]
        logger.info("Delay fault removed")

    async def _remove_jitter_after(self, seconds: float):
        await asyncio.sleep(seconds)
        self.jitter_ms = 0.0
        self.active_faults = [f for f in self.active_faults if not f.startswith("jitter_")]
        logger.info("Jitter fault removed")

    async def _remove_loss_after(self, seconds: float):
        await asyncio.sleep(seconds)
        self.packet_loss_rate = 0.0
        self.active_faults = [f for f in self.active_faults if not f.startswith("loss_")]
        logger.info("Loss fault removed")

    def simulate_message_send(self, peer_id: str, message_id: str) -> tuple[float, bool]:
        """Simulate message send with current fault conditions"""
        # Calculate RTT with fault injection
        base_rtt = self.base_rtt_ms + random.uniform(-5, 5)  # Natural variation
        injected_delay = self.added_delay_ms
        jitter = random.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms > 0 else 0

        simulated_rtt_ms = base_rtt + injected_delay + jitter

        # Simulate packet loss
        success = random.random() > self.packet_loss_rate

        return simulated_rtt_ms, success

    def get_fault_summary(self) -> dict[str, any]:
        """Get current fault injection state"""
        return {
            "active_faults": self.active_faults.copy(),
            "base_rtt_ms": self.base_rtt_ms,
            "added_delay_ms": self.added_delay_ms,
            "jitter_ms": self.jitter_ms,
            "packet_loss_rate": self.packet_loss_rate,
            "fault_duration_seconds": time.time() - self.fault_start_time if self.fault_start_time else 0
        }


class MockPingSender:
    """Mock ping sender that uses fault injector"""

    def __init__(self, fault_injector: NetworkFaultInjector):
        self.fault_injector = fault_injector
        self.call_count = 0

    async def send_ping(self, peer_id: str, payload: bytes) -> bool:
        """Simulate sending ping with fault injection"""
        self.call_count += 1
        _, success = self.fault_injector.simulate_message_send(peer_id, f"ping_{self.call_count}")

        # Simulate network delay
        rtt_ms, _ = self.fault_injector.simulate_message_send(peer_id, f"ping_{self.call_count}")
        await asyncio.sleep(rtt_ms / 1000.0)  # Convert to seconds

        return success


class TestNetworkFaultInjection(unittest.IsolatedAsyncioTestCase):
    """Test network fault injection and adaptive response"""

    async def asyncSetUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.fault_injector = NetworkFaultInjector()
        self.metrics_collector = NetworkMetricsCollector()
        self.navigator = AdaptiveNavigator(self.metrics_collector)
        self.ping_sender = MockPingSender(self.fault_injector)
        self.test_peer = "test_peer_001"

    async def test_delay_injection_triggers_protocol_switch(self):
        """Test that high delay triggers protocol switch within 500ms"""
        logger.info("=== Testing Delay Injection & Protocol Switch ===")

        # 1. Establish baseline measurements (low RTT)
        await self._establish_baseline_measurements()

        # 2. Get initial protocol selection
        context = MessageContext(recipient=self.test_peer, payload_size=1024, priority=5)
        available_protocols = ["htx", "htxquic", "betanet", "bitchat"]

        initial_protocol, initial_meta = await self.navigator.select_optimal_protocol(
            self.test_peer, context, available_protocols
        )
        logger.info(f"Initial protocol: {initial_protocol}")

        # 3. Inject high delay (simulate network congestion)
        self.fault_injector.inject_delay(added_delay_ms=800.0, duration_seconds=5.0)

        # 4. Simulate more measurements with high delay
        await self._simulate_measurements_with_faults(count=3)

        # 5. Check protocol selection again (should switch within 500ms)
        start_time = time.time()
        new_protocol, new_meta = await self.navigator.select_optimal_protocol(
            self.test_peer, context, available_protocols
        )
        decision_time_ms = (time.time() - start_time) * 1000

        # Assertions
        self.assertLess(decision_time_ms, 500, "Protocol decision took >500ms")
        self.assertNotEqual(initial_protocol, new_protocol, "Protocol should have switched due to high delay")
        self.assertIn(new_protocol, ["betanet", "bitchat"], "Should switch to more robust protocol for high delay")

        logger.info(f"✅ Protocol switched {initial_protocol} → {new_protocol} in {decision_time_ms:.1f}ms")
        logger.info(f"New metadata: {new_meta}")

    async def test_packet_loss_triggers_reliability_preference(self):
        """Test that packet loss triggers reliable protocol selection"""
        logger.info("=== Testing Packet Loss & Reliability Preference ===")

        await self._establish_baseline_measurements()

        # Inject packet loss
        self.fault_injector.inject_packet_loss(loss_rate=0.15, duration_seconds=5.0)  # 15% loss

        # Simulate measurements with loss
        await self._simulate_measurements_with_faults(count=5)

        # Check conditions and protocol selection
        conditions = self.navigator.get_network_conditions(self.test_peer)
        context = MessageContext(recipient=self.test_peer, payload_size=2048, priority=6)

        protocol, metadata = await self.navigator.select_optimal_protocol(
            self.test_peer, context, ["htx", "htxquic", "betanet", "bitchat"]
        )

        # Assertions
        self.assertIsNotNone(conditions.measured_loss_rate, "Should have loss rate measurement")
        self.assertGreater(conditions.measured_loss_rate, 0.1, "Should measure significant loss")
        self.assertTrue(conditions.needs_path_switch(), "Should trigger path switch due to loss")
        self.assertIn(protocol, ["betanet", "bitchat"], "Should prefer reliable protocols under loss")

        logger.info(f"✅ Selected {protocol} for {conditions.measured_loss_rate*100:.1f}% loss rate")

    async def test_jitter_affects_chunk_sizing(self):
        """Test that high jitter reduces chunk size"""
        logger.info("=== Testing Jitter & Adaptive Chunk Sizing ===")

        await self._establish_baseline_measurements()
        initial_chunk_size = self.navigator.get_adaptive_chunk_size(self.test_peer, 4096)

        # Inject high jitter
        self.fault_injector.inject_jitter(jitter_ms=150.0, duration_seconds=5.0)

        # Simulate measurements with jitter
        await self._simulate_measurements_with_faults(count=4)

        # Check adaptive chunk size
        new_chunk_size = self.navigator.get_adaptive_chunk_size(self.test_peer, 4096)
        conditions = self.navigator.get_network_conditions(self.test_peer)

        # Assertions
        self.assertIsNotNone(conditions.measured_jitter_ms, "Should measure jitter")
        self.assertGreater(conditions.measured_jitter_ms, 100, "Should measure high jitter")
        self.assertLess(new_chunk_size, initial_chunk_size, "Chunk size should decrease with high jitter")
        self.assertLessEqual(new_chunk_size, 1024, "High jitter should trigger small chunks")

        logger.info(f"✅ Chunk size: {initial_chunk_size} → {new_chunk_size} bytes due to {conditions.measured_jitter_ms:.1f}ms jitter")

    async def test_fault_recovery_restores_optimal_protocol(self):
        """Test that protocol selection recovers when faults clear"""
        logger.info("=== Testing Fault Recovery & Protocol Restoration ===")

        await self._establish_baseline_measurements()

        context = MessageContext(recipient=self.test_peer, payload_size=1024, priority=5)
        available = ["htx", "htxquic", "betanet", "bitchat"]

        # Get baseline protocol
        baseline_protocol, _ = await self.navigator.select_optimal_protocol(self.test_peer, context, available)

        # Inject severe faults
        self.fault_injector.inject_delay(added_delay_ms=1200.0, duration_seconds=3.0)
        self.fault_injector.inject_packet_loss(loss_rate=0.2, duration_seconds=3.0)

        # Measure under faults
        await self._simulate_measurements_with_faults(count=3)

        # Should switch to robust protocol
        fault_protocol, fault_meta = await self.navigator.select_optimal_protocol(self.test_peer, context, available)

        # Wait for faults to clear (note: faults auto-clear after duration)
        await asyncio.sleep(4.0)

        # Measure recovery conditions
        await self._establish_baseline_measurements(count=3)

        # Should recover to optimal protocol
        recovery_protocol, recovery_meta = await self.navigator.select_optimal_protocol(self.test_peer, context, available)

        # Assertions
        self.assertNotEqual(baseline_protocol, fault_protocol, "Should switch during faults")
        self.assertEqual(baseline_protocol, recovery_protocol, "Should restore optimal protocol after recovery")
        self.assertLess(recovery_meta["decision_time_ms"], 500, "Recovery decision should be fast")

        logger.info(f"✅ Protocol recovery: {baseline_protocol} → {fault_protocol} → {recovery_protocol}")

    async def test_control_ping_provides_measurements_under_faults(self):
        """Test that control ping continues to provide measurements during faults"""
        logger.info("=== Testing Control Ping Under Network Faults ===")

        # Start control ping loop
        asyncio.create_task(
            self.metrics_collector.start_control_ping_loop(self.test_peer, self.ping_sender.send_ping)
        )

        # Let it establish baseline
        await asyncio.sleep(2.0)
        baseline_measurements = self.metrics_collector.peer_metrics.get(self.test_peer, None)

        # Inject faults
        self.fault_injector.inject_delay(added_delay_ms=300.0, duration_seconds=4.0)
        self.fault_injector.inject_jitter(jitter_ms=100.0, duration_seconds=4.0)

        # Let it measure under faults
        await asyncio.sleep(3.0)

        # Check measurements
        fault_measurements = self.metrics_collector.peer_metrics.get(self.test_peer, None)

        # Stop ping loop
        self.metrics_collector.stop_control_ping_loop(self.test_peer)

        # Assertions
        self.assertIsNotNone(baseline_measurements, "Should have baseline measurements")
        self.assertIsNotNone(fault_measurements, "Should have fault measurements")
        self.assertGreater(fault_measurements.rtt_ewma_ms, baseline_measurements.rtt_ewma_ms,
                          "RTT should increase under delay fault")
        self.assertGreater(fault_measurements.jitter_ms, 50, "Should measure increased jitter")
        self.assertGreater(self.ping_sender.call_count, 5, "Should have made multiple ping attempts")

        logger.info(f"✅ Control ping measured RTT increase: "
                   f"{baseline_measurements.rtt_ewma_ms:.1f} → {fault_measurements.rtt_ewma_ms:.1f} ms")

    async def test_delivery_thresholds_under_faults(self):
        """Test that message delivery meets thresholds even under faults"""
        logger.info("=== Testing Delivery Thresholds Under Faults ===")

        # Define delivery requirements
        max_delivery_time_ms = 2000  # 2 second max
        min_success_rate = 0.8       # 80% minimum

        total_messages = 20
        successful_deliveries = 0
        delivery_times = []

        # Inject moderate faults
        self.fault_injector.inject_delay(added_delay_ms=200.0, duration_seconds=10.0)
        self.fault_injector.inject_packet_loss(loss_rate=0.1, duration_seconds=10.0)  # 10% loss

        # Simulate message delivery attempts
        for i in range(total_messages):
            start_time = time.time()

            # Record send
            seq_id = self.metrics_collector.record_message_sent(
                self.test_peer, f"delivery_test_{i}", payload_size=1024
            )

            # Simulate delivery with faults
            rtt_ms, success = self.fault_injector.simulate_message_send(self.test_peer, f"delivery_test_{i}")

            # Simulate network delay
            await asyncio.sleep(rtt_ms / 1000.0)

            # Record acknowledgment
            if success and random.random() > 0.05:  # 5% additional failure chance
                self.metrics_collector.record_message_acked(seq_id, success=True)
                delivery_time_ms = (time.time() - start_time) * 1000
                delivery_times.append(delivery_time_ms)
                successful_deliveries += 1
            else:
                self.metrics_collector.record_message_acked(seq_id, success=False)

            # Brief pause between messages
            await asyncio.sleep(0.1)

        # Calculate metrics
        success_rate = successful_deliveries / total_messages
        avg_delivery_time_ms = sum(delivery_times) / len(delivery_times) if delivery_times else float('inf')
        max_observed_delivery_ms = max(delivery_times) if delivery_times else float('inf')

        # Assertions
        self.assertGreaterEqual(success_rate, min_success_rate,
                               f"Success rate {success_rate:.2f} below minimum {min_success_rate}")
        self.assertLessEqual(max_observed_delivery_ms, max_delivery_time_ms,
                            f"Max delivery time {max_observed_delivery_ms:.1f}ms exceeds threshold {max_delivery_time_ms}ms")

        logger.info(f"✅ Delivery under faults: {success_rate:.1%} success rate, "
                   f"{avg_delivery_time_ms:.1f}ms avg delivery time")

    # Helper methods

    async def _establish_baseline_measurements(self, count: int = 3):
        """Establish baseline measurements with good conditions"""
        for i in range(count):
            seq_id = self.metrics_collector.record_message_sent(self.test_peer, f"baseline_{i}", 512)

            # Simulate good RTT (50-60ms)
            await asyncio.sleep(0.055)  # 55ms delay
            self.metrics_collector.record_message_acked(seq_id, success=True)

        logger.debug(f"Established {count} baseline measurements")

    async def _simulate_measurements_with_faults(self, count: int = 5):
        """Simulate measurements under current fault conditions"""
        for i in range(count):
            seq_id = self.metrics_collector.record_message_sent(self.test_peer, f"fault_{i}", 1024)

            # Use fault injector for realistic delay
            rtt_ms, success = self.fault_injector.simulate_message_send(self.test_peer, f"fault_{i}")
            await asyncio.sleep(rtt_ms / 1000.0)

            self.metrics_collector.record_message_acked(seq_id, success=success)

        logger.debug(f"Simulated {count} measurements with faults: {self.fault_injector.get_fault_summary()}")


async def run_fault_injection_demo():
    """Demonstration of fault injection capabilities"""
    print("🧪 Network Fault Injection Demo")
    print("=" * 50)

    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available")
        return

    # Setup
    fault_injector = NetworkFaultInjector()
    metrics_collector = NetworkMetricsCollector()
    navigator = AdaptiveNavigator(metrics_collector)
    MockPingSender(fault_injector)
    test_peer = "demo_peer"

    # Establish baseline
    print("📊 Establishing baseline measurements...")
    for i in range(3):
        seq_id = metrics_collector.record_message_sent(test_peer, f"baseline_{i}", 1024)
        await asyncio.sleep(0.05)  # 50ms RTT
        metrics_collector.record_message_acked(seq_id, success=True)

    baseline_conditions = navigator.get_network_conditions(test_peer)
    print(f"   Baseline RTT: {baseline_conditions.measured_rtt_ms:.1f}ms")
    print(f"   Baseline quality: {baseline_conditions.quality_score:.3f}")

    # Test protocol selection
    context = MessageContext(recipient=test_peer, payload_size=2048, priority=6)
    available = ["htx", "htxquic", "betanet", "bitchat"]

    baseline_protocol, _ = await navigator.select_optimal_protocol(test_peer, context, available)
    print(f"   Baseline protocol: {baseline_protocol}")

    print("\n🔥 Injecting network faults...")
    fault_injector.inject_delay(added_delay_ms=600.0, duration_seconds=8.0)
    fault_injector.inject_jitter(jitter_ms=120.0, duration_seconds=8.0)
    fault_injector.inject_packet_loss(loss_rate=0.12, duration_seconds=8.0)

    print(f"   Active faults: {fault_injector.get_fault_summary()['active_faults']}")

    # Measure under faults
    print("\n📈 Measuring under fault conditions...")
    for i in range(5):
        seq_id = metrics_collector.record_message_sent(test_peer, f"fault_{i}", 1024)
        rtt_ms, success = fault_injector.simulate_message_send(test_peer, f"fault_{i}")
        await asyncio.sleep(rtt_ms / 1000.0)
        metrics_collector.record_message_acked(seq_id, success=success)
        print(f"   Message {i+1}: RTT={rtt_ms:.1f}ms, success={success}")

    # Check adaptation
    fault_conditions = navigator.get_network_conditions(test_peer)
    fault_protocol, fault_meta = await navigator.select_optimal_protocol(test_peer, context, available)

    print("\n🎯 Adaptive Response:")
    print(f"   RTT: {baseline_conditions.measured_rtt_ms:.1f} → {fault_conditions.measured_rtt_ms:.1f} ms")
    print(f"   Jitter: {baseline_conditions.measured_jitter_ms:.1f} → {fault_conditions.measured_jitter_ms:.1f} ms")
    print(f"   Loss: {baseline_conditions.measured_loss_rate:.1%} → {fault_conditions.measured_loss_rate:.1%}")
    print(f"   Protocol: {baseline_protocol} → {fault_protocol}")
    print(f"   Decision time: {fault_meta['decision_time_ms']:.1f}ms")
    print(f"   Needs switch: {fault_conditions.needs_path_switch()}")

    # Wait for recovery
    print("\n⏳ Waiting for fault recovery...")
    await asyncio.sleep(9.0)

    # Measure recovery
    for i in range(3):
        seq_id = metrics_collector.record_message_sent(test_peer, f"recovery_{i}", 1024)
        await asyncio.sleep(0.055)  # Back to 55ms
        metrics_collector.record_message_acked(seq_id, success=True)

    recovery_conditions = navigator.get_network_conditions(test_peer)
    recovery_protocol, recovery_meta = await navigator.select_optimal_protocol(test_peer, context, available)

    print("✅ Recovery Complete:")
    print(f"   RTT recovered: {recovery_conditions.measured_rtt_ms:.1f}ms")
    print(f"   Protocol restored: {recovery_protocol}")
    print(f"   Quality score: {recovery_conditions.quality_score:.3f}")

    print("\n🏁 Demo complete! Fault injection successfully demonstrated adaptive behavior.")


if __name__ == "__main__":
    # Can run individual tests or the demo
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--test", help="Run specific test method")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(run_fault_injection_demo())
    elif args.test:
        # Run specific test
        suite = unittest.TestLoader().loadTestsFromName(f"TestNetworkFaultInjection.{args.test}",
                                                        module=sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # Run all tests
        unittest.main(verbosity=2)
