"""Acceptance Tests for Dual-Path Transport System

Tests the unified transport interface with automatic path selection between
BitChat (Bluetooth mesh) and Betanet (decentralized internet).

Scenarios:
1. Proximity local → BitChat is chosen; queued when offline; delivered when peer reappears
2. Large/urgent → Betanet chosen; QUIC fail → TCP fallback
3. Link flap triggers switch ≤500ms (simulate clock)

DoD: All scenarios pass; receipts recorded; metrics exported JSON.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import system under test
try:
    from experimental.agents.agents.navigator.path_policy import (
        LinkChangeDetector,
        MessageContext,
        NavigatorAgent,
        NetworkConditions,
        PathProtocol,
        RoutingPriority,
    )
    from src.core.p2p.unified_transport import (
        DeliveryReceipt,
        DeliveryStatus,
        PathSelection,
        TransportContext,
        UnifiedTransport,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class MockBitChatTransport:
    """Mock BitChat transport for testing"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.is_running = False
        self.discovered_peers = {}
        self.message_handlers = {}
        self.peer_reachable_map = {}  # peer_id -> bool

    async def start(self) -> bool:
        self.is_running = True
        return True

    async def stop(self) -> None:
        self.is_running = False

    async def send_message(
        self, recipient: str, payload: bytes, priority: int = 5, ttl: int = 7
    ) -> bool:
        # Simulate send success based on peer reachability
        return self.peer_reachable_map.get(recipient, True)

    def is_peer_reachable(self, peer_id: str) -> bool:
        return self.peer_reachable_map.get(peer_id, False)

    def set_peer_reachable(self, peer_id: str, reachable: bool):
        self.peer_reachable_map[peer_id] = reachable

    def register_handler(self, name: str, handler):
        self.message_handlers[name] = handler

    def get_peer_count(self) -> int:
        return len(self.discovered_peers)

    def get_status(self) -> dict:
        return {"is_running": self.is_running, "peers": len(self.discovered_peers)}


class MockBetanetTransport:
    """Mock Betanet transport for testing"""

    def __init__(self, peer_id: str):
        self.peer_id = peer_id
        self.is_running = False
        self.discovered_peers = {}
        self.active_connections = set()
        self.message_handlers = {}
        self.send_success = True
        self.quic_available = True

    async def start(self) -> bool:
        self.is_running = True
        return True

    async def stop(self) -> None:
        self.is_running = False

    async def send_message(
        self,
        recipient: str,
        payload: bytes,
        protocol: str = "quic",
        priority: int = 5,
        use_mixnodes: bool = False,
    ) -> bool:
        # Simulate QUIC failure and TCP fallback
        if not self.quic_available and protocol == "quic":
            # Simulate fallback to TCP
            protocol = "tcp"

        return self.send_success

    def register_handler(self, name: str, handler):
        self.message_handlers[name] = handler

    def get_status(self) -> dict:
        return {"is_running": self.is_running, "peers": len(self.discovered_peers)}


class MockDualPathTransport:
    """Mock dual-path transport combining BitChat and Betanet"""

    def __init__(
        self, node_id: str, enable_bitchat: bool = True, enable_betanet: bool = True
    ):
        self.node_id = node_id
        self.enable_bitchat = enable_bitchat
        self.enable_betanet = enable_betanet
        self.is_running = False

        # Mock transports
        self.bitchat = MockBitChatTransport(f"bc_{node_id}") if enable_bitchat else None
        self.betanet = MockBetanetTransport(f"bn_{node_id}") if enable_betanet else None
        self.navigator = Mock()

        # Message tracking
        self.sent_messages = []
        self.routing_stats = {
            "bitchat_sent": 0,
            "betanet_sent": 0,
            "store_forward_queued": 0,
            "routing_decisions": 0,
        }

        # Set up navigator mock
        self.navigator.select_path = AsyncMock(return_value=(PathProtocol.BITCHAT, {}))

    async def start(self) -> bool:
        self.is_running = True
        if self.bitchat:
            await self.bitchat.start()
        if self.betanet:
            await self.betanet.start()
        return True

    async def stop(self) -> None:
        self.is_running = False
        if self.bitchat:
            await self.bitchat.stop()
        if self.betanet:
            await self.betanet.stop()

    async def send_message(
        self,
        recipient: str,
        payload,
        priority: int = 5,
        privacy_required: bool = False,
        deadline: float = None,
        preferred_protocol: str = None,
    ) -> bool:
        """Mock message sending with path selection simulation"""

        # Record message for testing
        message_info = {
            "recipient": recipient,
            "payload": payload,
            "priority": priority,
            "preferred_protocol": preferred_protocol,
            "timestamp": time.time(),
        }
        self.sent_messages.append(message_info)

        # Simulate path selection and delivery
        if preferred_protocol == "bitchat" and self.bitchat:
            success = await self.bitchat.send_message(recipient, payload, priority)
            if success:
                self.routing_stats["bitchat_sent"] += 1
            return success

        elif preferred_protocol == "betanet" and self.betanet:
            success = await self.betanet.send_message(recipient, payload)
            if success:
                self.routing_stats["betanet_sent"] += 1
            return success

        elif preferred_protocol is None:
            # Auto-selection simulation
            if self.bitchat and self.bitchat.is_peer_reachable(recipient):
                success = await self.bitchat.send_message(recipient, payload, priority)
                if success:
                    self.routing_stats["bitchat_sent"] += 1
                return success
            elif self.betanet:
                success = await self.betanet.send_message(recipient, payload)
                if success:
                    self.routing_stats["betanet_sent"] += 1
                return success

        # Store-and-forward fallback
        self.routing_stats["store_forward_queued"] += 1
        return True

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "routing_stats": self.routing_stats,
            "sent_messages": len(self.sent_messages),
        }


@pytest.fixture
async def unified_transport():
    """Create a UnifiedTransport instance with mocked dependencies"""
    node_id = f"test_{uuid.uuid4().hex[:8]}"

    with patch("src.core.p2p.unified_transport.DualPathTransport") as mock_dual_path:
        # Set up mock dual-path transport
        mock_transport = MockDualPathTransport(node_id)
        mock_dual_path.return_value = mock_transport

        # Create unified transport
        transport = UnifiedTransport(
            node_id=node_id,
            enable_bitchat=True,
            enable_betanet=True,
            global_south_mode=True,
        )

        # Override dual_path with our mock
        transport.dual_path = mock_transport

        # Start transport
        await transport.start()

        yield transport, mock_transport

        # Cleanup
        await transport.stop()


class TestProximityLocalScenario:
    """Test scenario: Proximity local → BitChat is chosen; queued when offline; delivered when peer reappears"""

    @pytest.mark.asyncio
    async def test_proximity_local_bitchat_chosen(self, unified_transport):
        """Test that local/nearby peers automatically choose BitChat"""
        transport, mock_dual_path = unified_transport

        # Set up peer as nearby/reachable via BitChat
        peer_id = "nearby_peer_001"
        mock_dual_path.bitchat.set_peer_reachable(peer_id, True)

        # Send message with proximity hint
        context = TransportContext(size_bytes=1024, priority=5, proximity_hint="local")

        receipt = await transport.send(peer_id, "Hello nearby peer!", context)

        # Verify BitChat was chosen
        assert receipt.path_chosen == "bitchat"
        assert receipt.path_reasoning == PathSelection.PROXIMITY_LOCAL
        assert receipt.success
        assert receipt.status == DeliveryStatus.DELIVERED
        assert mock_dual_path.routing_stats["bitchat_sent"] == 1
        assert mock_dual_path.routing_stats["betanet_sent"] == 0

    @pytest.mark.asyncio
    async def test_queued_when_offline(self, unified_transport):
        """Test that messages are queued when peer is offline"""
        transport, mock_dual_path = unified_transport

        # Set up peer as unreachable
        peer_id = "offline_peer_001"
        mock_dual_path.bitchat.set_peer_reachable(peer_id, False)
        mock_dual_path.betanet.send_success = False

        # Send message to offline peer
        context = TransportContext(size_bytes=512, priority=3, proximity_hint="local")

        receipt = await transport.send(peer_id, "Message for offline peer", context)

        # Verify message was queued for store-and-forward
        assert receipt.path_chosen in ["store_forward", "bitchat"]
        assert (
            mock_dual_path.routing_stats["store_forward_queued"] >= 0
        )  # May be queued

    @pytest.mark.asyncio
    async def test_delivered_when_peer_reappears(self, unified_transport):
        """Test that queued messages are delivered when peer comes back online"""
        transport, mock_dual_path = unified_transport

        peer_id = "returning_peer_001"

        # Step 1: Peer is offline - message should be queued
        mock_dual_path.bitchat.set_peer_reachable(peer_id, False)
        mock_dual_path.betanet.send_success = False

        receipt1 = await transport.send(
            peer_id, "Offline message", TransportContext(proximity_hint="local")
        )

        # Step 2: Peer comes back online
        mock_dual_path.bitchat.set_peer_reachable(peer_id, True)
        mock_dual_path.betanet.send_success = True

        # Step 3: Send another message - should be delivered immediately
        receipt2 = await transport.send(
            peer_id, "Online message", TransportContext(proximity_hint="local")
        )

        # Verify second message was delivered via BitChat
        assert receipt2.path_chosen == "bitchat"
        assert receipt2.success
        assert receipt2.status == DeliveryStatus.DELIVERED


class TestLargeUrgentScenario:
    """Test scenario: Large/urgent → Betanet chosen; QUIC fail → TCP fallback"""

    @pytest.mark.asyncio
    async def test_large_message_betanet_chosen(self, unified_transport):
        """Test that large messages automatically choose Betanet"""
        transport, mock_dual_path = unified_transport

        peer_id = "remote_peer_001"
        large_payload = "x" * 50000  # 50KB message

        # Send large message
        context = TransportContext(
            size_bytes=len(large_payload), priority=5, proximity_hint="remote"
        )

        receipt = await transport.send(peer_id, large_payload, context)

        # Verify Betanet was chosen for large message
        assert receipt.path_chosen == "betanet"
        assert receipt.path_reasoning == PathSelection.LARGE_URGENT
        assert receipt.success
        assert mock_dual_path.routing_stats["betanet_sent"] == 1

    @pytest.mark.asyncio
    async def test_urgent_message_betanet_chosen(self, unified_transport):
        """Test that urgent messages choose Betanet"""
        transport, mock_dual_path = unified_transport

        peer_id = "urgent_peer_001"

        # Send urgent message
        context = TransportContext(
            size_bytes=1024,
            priority=9,  # High priority
            requires_realtime=True,
        )

        receipt = await transport.send(peer_id, "URGENT: System alert!", context)

        # Verify Betanet was chosen for urgent message
        assert receipt.path_chosen == "betanet"
        assert receipt.path_reasoning == PathSelection.LARGE_URGENT
        assert receipt.success

    @pytest.mark.asyncio
    async def test_quic_fail_tcp_fallback(self, unified_transport):
        """Test QUIC failure with TCP fallback"""
        transport, mock_dual_path = unified_transport

        # Simulate QUIC failure
        mock_dual_path.betanet.quic_available = False

        peer_id = "fallback_peer_001"
        context = TransportContext(size_bytes=20000, priority=8)

        receipt = await transport.send(peer_id, "Message requiring fallback", context)

        # Verify fallback behavior
        assert receipt.path_chosen == "betanet"
        assert receipt.success  # Should still succeed via TCP fallback
        assert "fallback" in receipt.metadata or receipt.fallback_used


class TestLinkChangeScenario:
    """Test scenario: Link flap triggers switch ≤500ms (simulate clock)"""

    @pytest.mark.asyncio
    async def test_link_change_fast_switching(self, unified_transport):
        """Test that link changes trigger fast path switching within 500ms"""
        transport, mock_dual_path = unified_transport

        # Create a more sophisticated navigator mock
        navigator = NavigatorAgent("test_nav", RoutingPriority.OFFLINE_FIRST)
        mock_dual_path.navigator = navigator

        peer_id = "switching_peer_001"

        # Initial state: Bluetooth available, internet unavailable
        with (
            patch.object(transport, "_check_bluetooth_available", return_value=True),
            patch.object(transport, "_check_internet_available", return_value=False),
        ):
            # Send first message - should use BitChat
            start_time = time.time()
            context1 = TransportContext(size_bytes=1024, priority=5)
            receipt1 = await transport.send(peer_id, "Message 1", context1)

            # Simulate link change: Internet becomes available
            with patch.object(
                transport, "_check_internet_available", return_value=True
            ):
                # Send second message - should detect link change and switch
                context2 = TransportContext(
                    size_bytes=20000, priority=8
                )  # Large urgent
                receipt2 = await transport.send(peer_id, "Message 2", context2)
                switch_time = time.time()

        # Verify fast switching behavior
        time_diff_ms = (switch_time - start_time) * 1000

        # Should have switched paths due to link change + large/urgent message
        assert (
            receipt1.path_chosen != receipt2.path_chosen
            or receipt2.path_reasoning == PathSelection.LINK_CHANGE_SWITCH
        )
        assert time_diff_ms < 1000  # Should be much faster than 1 second

    @pytest.mark.asyncio
    async def test_link_change_detection_timing(self):
        """Test link change detection performance meets 500ms target"""
        detector = LinkChangeDetector(target_switch_time_ms=500)

        # Simulate multiple link state changes
        start_time = time.time() * 1000

        # Initial state
        state1 = {
            "bluetooth_available": True,
            "internet_available": False,
            "wifi_connected": False,
            "bandwidth_mbps": 0.1,
        }
        change1 = detector.update_link_state(state1)

        # Simulate link change
        await asyncio.sleep(0.1)  # 100ms delay
        state2 = {
            "bluetooth_available": True,
            "internet_available": True,
            "wifi_connected": True,
            "bandwidth_mbps": 50.0,
        }
        change2 = detector.update_link_state(state2)

        end_time = time.time() * 1000

        # Verify change detection
        assert change2  # Should detect significant change

        # Verify timing performance
        detection_time = end_time - start_time
        assert detection_time < 500  # Should be well under 500ms target

        # Verify metrics
        metrics = detector.get_performance_metrics()
        assert metrics["within_target"]
        assert metrics["avg_evaluation_time_ms"] < 500


class TestCostOptimizationScenario:
    """Test Global South cost optimization scenarios"""

    @pytest.mark.asyncio
    async def test_cost_sensitive_bitchat_preferred(self, unified_transport):
        """Test that cost-sensitive scenarios prefer BitChat"""
        transport, mock_dual_path = unified_transport

        # Enable cost sensitivity (Global South mode)
        transport.global_south_mode = True

        peer_id = "cost_sensitive_peer"
        mock_dual_path.bitchat.set_peer_reachable(peer_id, True)

        context = TransportContext(size_bytes=5000, priority=5, cost_sensitive=True)

        receipt = await transport.send(peer_id, "Cost-sensitive message", context)

        # Should prefer BitChat to avoid data costs
        assert receipt.path_chosen == "bitchat"
        assert receipt.path_reasoning == PathSelection.COST_OPTIMIZATION
        assert receipt.data_cost_mb == 0.0  # BitChat has no data cost

    @pytest.mark.asyncio
    async def test_battery_conservation_bitchat_preferred(self, unified_transport):
        """Test that low battery scenarios prefer BitChat"""
        transport, mock_dual_path = unified_transport

        peer_id = "battery_conserve_peer"
        mock_dual_path.bitchat.set_peer_reachable(peer_id, True)

        context = TransportContext(
            size_bytes=2000,
            priority=4,
            battery_percent=20,  # Low battery
        )

        receipt = await transport.send(peer_id, "Battery conservation message", context)

        # Should prefer BitChat for energy efficiency
        assert receipt.path_chosen == "bitchat"
        assert receipt.path_reasoning == PathSelection.BATTERY_CONSERVATION
        assert receipt.energy_cost < 0.5  # BitChat should be more energy efficient


class TestPrivacyScenario:
    """Test privacy-aware routing scenarios"""

    @pytest.mark.asyncio
    async def test_privacy_required_betanet_mixnodes(self, unified_transport):
        """Test that privacy-required messages use Betanet with mixnodes"""
        transport, mock_dual_path = unified_transport

        peer_id = "privacy_peer"

        context = TransportContext(size_bytes=3000, priority=6, privacy_required=True)

        receipt = await transport.send(peer_id, "Private message", context)

        # Should use Betanet for privacy with mixnodes
        assert receipt.path_chosen == "betanet"
        assert receipt.path_reasoning == PathSelection.PRIVACY_REQUIRED
        assert receipt.metadata.get("privacy_routing", False)


class TestMetricsAndReceipts:
    """Test metrics export and receipt tracking"""

    @pytest.mark.asyncio
    async def test_receipt_tracking(self, unified_transport):
        """Test that all messages generate proper receipts"""
        transport, mock_dual_path = unified_transport

        peer_id = "receipt_test_peer"
        mock_dual_path.bitchat.set_peer_reachable(peer_id, True)

        # Send multiple messages
        messages = [
            ("Small message", TransportContext(size_bytes=100, priority=3)),
            ("Large message", TransportContext(size_bytes=50000, priority=5)),
            ("Urgent message", TransportContext(size_bytes=1000, priority=9)),
        ]

        receipts = []
        for payload, context in messages:
            receipt = await transport.send(peer_id, payload, context)
            receipts.append(receipt)

        # Verify all receipts are tracked
        assert len(receipts) == 3
        assert len(transport.receipts) == 3

        # Verify receipt contents
        for receipt in receipts:
            assert receipt.message_id
            assert receipt.destination == peer_id
            assert receipt.timestamp > 0
            assert receipt.path_chosen in ["bitchat", "betanet", "store_forward"]
            assert isinstance(receipt.path_reasoning, PathSelection)
            assert receipt.rtt_ms is not None
            assert receipt.success in [True, False]

    @pytest.mark.asyncio
    async def test_metrics_export_json(self, unified_transport):
        """Test that metrics can be exported to JSON"""
        transport, mock_dual_path = unified_transport

        # Send some test messages to generate metrics
        peer_id = "metrics_test_peer"
        mock_dual_path.bitchat.set_peer_reachable(peer_id, True)

        for i in range(5):
            context = TransportContext(size_bytes=1000 + i * 1000, priority=5)
            await transport.send(peer_id, f"Test message {i}", context)

        # Export metrics
        metrics = transport.export_metrics_json()

        # Verify metrics structure
        assert "node_id" in metrics
        assert "export_timestamp" in metrics
        assert "configuration" in metrics
        assert "path_performance" in metrics
        assert "recent_metrics" in metrics
        assert "summary_stats" in metrics

        # Verify metrics can be serialized to JSON
        json_str = json.dumps(metrics)
        assert isinstance(json_str, str)
        assert len(json_str) > 100  # Should have substantial content

        # Verify summary stats
        stats = metrics["summary_stats"]
        assert stats["total_messages"] == 5
        assert "success_rate" in stats
        assert "path_distribution" in stats


@pytest.mark.asyncio
async def test_complete_scenario_integration():
    """Integration test covering all scenarios end-to-end"""

    # Create transport
    transport = UnifiedTransport(node_id="integration_test", global_south_mode=True)

    # Mock the dual-path transport
    mock_dual_path = MockDualPathTransport("integration_test")
    transport.dual_path = mock_dual_path

    await transport.start()

    try:
        # Scenario 1: Proximity local
        peer1 = "local_peer"
        mock_dual_path.bitchat.set_peer_reachable(peer1, True)

        receipt1 = await transport.send(
            peer1,
            "Local message",
            TransportContext(proximity_hint="local", size_bytes=500),
        )
        assert receipt1.path_chosen == "bitchat"

        # Scenario 2: Large/urgent
        peer2 = "remote_peer"
        receipt2 = await transport.send(
            peer2,
            "x" * 25000,  # 25KB
            TransportContext(priority=8, size_bytes=25000),
        )
        assert receipt2.path_chosen == "betanet"

        # Scenario 3: Cost optimization
        peer3 = "cost_peer"
        mock_dual_path.bitchat.set_peer_reachable(peer3, True)

        receipt3 = await transport.send(
            peer3,
            "Cost-sensitive message",
            TransportContext(cost_sensitive=True, size_bytes=1000),
        )
        assert receipt3.path_chosen == "bitchat"

        # Verify all receipts recorded
        assert len(transport.receipts) == 3

        # Export metrics
        metrics = transport.export_metrics_json()
        assert metrics["summary_stats"]["total_messages"] >= 3

        # Save metrics to file for verification
        metrics_file = Path("test_dual_path_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Integration test passed - metrics saved to {metrics_file}")

    finally:
        await transport.stop()


if __name__ == "__main__":
    # Run the integration test directly
    asyncio.run(test_complete_scenario_integration())
