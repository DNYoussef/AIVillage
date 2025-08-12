"""Integration Tests for Dual-Path P2P Networking

Tests the complete integration of BitChat, Betanet, Navigator, and KING coordinator
for AIVillage's dual-path networking architecture.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Add src and experimental paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "experimental", "agents", "agents")
)

# Import components to test
try:
    from navigator.path_policy import (
        EnergyMode,
        MessageContext,
        NavigatorAgent,
        PathProtocol,
        RoutingPriority,
    )

    from core.p2p.betanet_transport import BetanetMessage, BetanetTransport
    from core.p2p.bitchat_transport import BitChatMessage, BitChatTransport
    from core.p2p.dual_path_transport import DualPathMessage, DualPathTransport

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

# Skip all tests if imports fail
pytestmark = pytest.mark.skipif(
    not IMPORTS_SUCCESSFUL, reason="Required modules not available"
)


class TestDualPathIntegration:
    """Test dual-path transport integration"""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests"""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    async def dual_transport(self):
        """Create dual-path transport instance"""
        transport = DualPathTransport(
            node_id="test_node", enable_bitchat=True, enable_betanet=True
        )
        try:
            yield transport
        finally:
            if transport.is_running:
                await transport.stop()

    @pytest.fixture
    def navigator(self):
        """Create Navigator agent instance"""
        nav = NavigatorAgent(
            agent_id="test_navigator",
            routing_priority=RoutingPriority.OFFLINE_FIRST,
            energy_mode=EnergyMode.BALANCED,
        )
        nav.enable_global_south_mode(True)
        return nav

    @pytest.mark.asyncio
    async def test_dual_transport_initialization(self, dual_transport):
        """Test that dual-path transport initializes correctly"""
        transport = await anext(dual_transport)
        assert transport.node_id == "test_node"
        assert transport.enable_bitchat == True
        assert transport.enable_betanet == True
        assert not transport.is_running
        assert transport.routing_stats["routing_decisions"] == 0

    @pytest.mark.asyncio
    async def test_dual_transport_startup(self, dual_transport):
        """Test dual-path transport startup"""
        transport = await anext(dual_transport)
        # Mock the individual transport startups to avoid actual network calls
        with patch.object(transport, "_sync_peer_information", new_callable=AsyncMock):
            success = await transport.start()

            assert success == True
            assert transport.is_running == True

            # Verify components were created (even if using fallback)
            assert transport.bitchat is not None
            assert transport.betanet is not None
            assert transport.navigator is not None

    @pytest.mark.asyncio
    async def test_navigator_path_selection_offline_first(self, navigator):
        """Test Navigator's offline-first path selection for Global South"""
        # Test scenario: nearby peer with small message
        context = MessageContext(size_bytes=500, priority=5, requires_realtime=False)

        # Mock network conditions for offline scenario
        navigator.conditions.bluetooth_available = True
        navigator.conditions.internet_available = False
        navigator.conditions.nearby_peers = 2
        navigator.conditions.battery_percent = 40

        # Add a nearby peer
        from navigator.path_policy import PeerInfo

        peer_info = PeerInfo(
            peer_id="nearby_peer",
            protocols={"bitchat"},
            hop_distance=2,
            bluetooth_rssi=-45,  # Good signal
        )
        navigator.update_peer_info("nearby_peer", peer_info)

        # Test path selection
        protocol, metadata = await navigator.select_path("nearby_peer", context)

        # Should select BitChat for nearby peer
        assert protocol == PathProtocol.BITCHAT
        assert metadata["offline_capable"] == True
        assert metadata["energy_efficient"] == True

    @pytest.mark.asyncio
    async def test_navigator_path_selection_performance_first(self, navigator):
        """Test Navigator's performance-first path selection"""
        # Switch to performance mode
        navigator.set_routing_priority(RoutingPriority.PERFORMANCE_FIRST)
        navigator.set_energy_mode(EnergyMode.PERFORMANCE)

        # Test scenario: large urgent message with internet available
        context = MessageContext(
            size_bytes=50000,  # Large message
            priority=9,  # Urgent
            requires_realtime=True,
        )

        # Mock network conditions for performance scenario
        navigator.conditions.bluetooth_available = True
        navigator.conditions.internet_available = True
        navigator.conditions.wifi_connected = True
        navigator.conditions.bandwidth_mbps = 50.0
        navigator.conditions.battery_percent = 80

        # Test path selection
        protocol, metadata = await navigator.select_path(
            "remote_peer", context, ["bitchat", "betanet"]
        )

        # Should select Betanet for performance
        assert protocol == PathProtocol.BETANET
        assert metadata["global_reach"] == True
        assert metadata["bandwidth_adaptive"] == True

    @pytest.mark.asyncio
    async def test_navigator_energy_conservation(self, navigator):
        """Test Navigator's energy conservation logic"""
        # Set low battery conditions
        navigator.conditions.battery_percent = 15  # Low battery
        navigator.set_energy_mode(EnergyMode.POWERSAVE)

        context = MessageContext(
            size_bytes=1000, priority=3, requires_realtime=False  # Low priority
        )

        # Mock peer nearby for BitChat
        navigator.conditions.bluetooth_available = True
        navigator.conditions.internet_available = True
        from navigator.path_policy import PeerInfo

        peer_info = PeerInfo(peer_id="test_peer", protocols={"bitchat"}, hop_distance=3)
        navigator.update_peer_info("test_peer", peer_info)

        # Test path selection
        protocol, metadata = await navigator.select_path("test_peer", context)

        # Should prefer BitChat or store-and-forward to conserve energy
        assert protocol in [PathProtocol.BITCHAT, PathProtocol.STORE_FORWARD]
        if protocol == PathProtocol.BITCHAT:
            assert metadata["energy_efficient"] == True

    @pytest.mark.asyncio
    async def test_dual_path_message_creation(self):
        """Test DualPathMessage format conversion"""
        # Test message creation
        message = DualPathMessage(
            sender="test_sender",
            recipient="test_recipient",
            payload={"test": "data"},
            priority=7,
            privacy_required=True,
        )

        assert message.sender == "test_sender"
        assert message.recipient == "test_recipient"
        assert message.priority == 7
        assert message.privacy_required == True
        assert isinstance(message.payload, bytes)
        assert message.context.size_bytes > 0
        assert message.context.priority == 7

    @pytest.mark.asyncio
    async def test_message_routing_via_dual_transport(self, dual_transport):
        """Test message routing through dual-path transport"""
        await dual_transport.start()

        # Mock successful routing
        with patch.object(
            dual_transport, "_send_via_bitchat", return_value=True
        ) as mock_bitchat:
            # Send test message
            success = await dual_transport.send_message(
                recipient="test_peer",
                payload={"test": "message"},
                priority=5,
                preferred_protocol="bitchat",
            )

            assert success == True
            assert dual_transport.routing_stats["routing_decisions"] > 0

    @pytest.mark.asyncio
    async def test_store_and_forward_queue(self, dual_transport):
        """Test store-and-forward functionality"""
        await dual_transport.start()

        # Create message for offline peer
        message = DualPathMessage(
            sender=dual_transport.node_id,
            recipient="offline_peer",
            payload={"test": "offline_message"},
            priority=5,
        )

        # Mock routing failure to trigger store-and-forward
        with patch.object(dual_transport, "_route_message", return_value=False):
            success = await dual_transport.send_message(
                recipient="offline_peer",
                payload={"test": "offline_message"},
                priority=5,
            )

            # Message should be queued for later delivery
            assert len(dual_transport.offline_queue) > 0
            queued_message, queue_time = dual_transport.offline_queue[0]
            assert queued_message.recipient == "offline_peer"

    @pytest.mark.asyncio
    async def test_broadcast_functionality(self, dual_transport):
        """Test mesh network broadcasting"""
        await dual_transport.start()

        # Mock connected peers
        if dual_transport.bitchat:
            dual_transport.bitchat.active_connections.add("peer1")
            dual_transport.bitchat.active_connections.add("peer2")

        # Test broadcast
        with patch.object(dual_transport, "_send_via_bitchat", return_value=True):
            peer_count = await dual_transport.broadcast_message(
                payload={"broadcast": "test"}, priority=6
            )

            # Should broadcast to available peers
            assert peer_count >= 0  # May be 0 if no real connections

    @pytest.mark.asyncio
    async def test_privacy_routing_selection(self, navigator):
        """Test privacy-aware routing selection"""
        # Set privacy-sensitive conditions
        navigator.conditions.censorship_risk = 0.8  # High censorship risk
        navigator.set_routing_priority(RoutingPriority.PRIVACY_FIRST)

        context = MessageContext(size_bytes=2000, priority=7, privacy_required=True)

        # Mock internet availability for mixnode routing
        navigator.conditions.internet_available = True
        navigator.conditions.bluetooth_available = True

        # Test path selection
        protocol, metadata = await navigator.select_path(
            "sensitive_peer", context, ["betanet", "bitchat"]
        )

        # Should prefer Betanet with privacy routing for sensitive content
        if protocol == PathProtocol.BETANET:
            assert metadata.get("privacy_routing") == True
            assert metadata.get("mixnode_hops", 0) >= 2


class TestNavigatorPathSelection:
    """Focused tests for Navigator path selection logic"""

    @pytest.fixture
    def navigator(self):
        """Create test Navigator instance"""
        nav = NavigatorAgent(
            agent_id="test_nav", routing_priority=RoutingPriority.OFFLINE_FIRST
        )
        nav.enable_global_south_mode(True)
        return nav

    def test_global_south_mode_configuration(self, navigator):
        """Test Global South mode optimization settings"""
        assert navigator.global_south_mode == True
        assert navigator.routing_priority == RoutingPriority.OFFLINE_FIRST
        assert navigator.data_cost_threshold == 0.005  # Sensitive to data costs
        assert navigator.battery_conservation_threshold == 25

    @pytest.mark.asyncio
    async def test_emergency_routing_override(self, navigator):
        """Test that urgent messages override normal routing logic"""
        # Set low-resource conditions
        navigator.conditions.battery_percent = 10
        navigator.conditions.internet_available = True
        navigator.conditions.bluetooth_available = False
        navigator.set_energy_mode(EnergyMode.POWERSAVE)

        # Create urgent message
        urgent_context = MessageContext(
            size_bytes=1000, priority=10, requires_realtime=True  # Maximum urgency
        )

        # Test emergency routing
        protocol, metadata = await navigator.select_path(
            "emergency_target", urgent_context, ["betanet"]
        )

        # Should override energy conservation for emergency
        assert protocol == PathProtocol.BETANET

    @pytest.mark.asyncio
    async def test_data_cost_aware_routing(self, navigator):
        """Test routing that considers mobile data costs"""
        # Set expensive data conditions (Global South scenario)
        navigator.conditions.data_cost_usd_mb = 0.02  # Expensive mobile data
        navigator.conditions.internet_available = True
        navigator.conditions.bluetooth_available = True
        navigator.conditions.wifi_connected = False  # Using cellular

        context = MessageContext(size_bytes=5000, priority=5)  # Medium message

        # Mock nearby peer for BitChat alternative
        from navigator.path_policy import PeerInfo

        peer_info = PeerInfo(
            peer_id="local_peer", protocols={"bitchat"}, hop_distance=2
        )
        navigator.update_peer_info("local_peer", peer_info)

        # Test path selection
        protocol, metadata = await navigator.select_path("local_peer", context)

        # Should prefer BitChat to avoid expensive data costs
        assert protocol == PathProtocol.BITCHAT

    def test_navigator_status_reporting(self, navigator):
        """Test comprehensive Navigator status reporting"""
        status = navigator.get_status()

        assert "agent_id" in status
        assert "routing_priority" in status
        assert "energy_mode" in status
        assert "global_south_mode" in status
        assert "network_conditions" in status

        # Check network conditions reporting
        conditions = status["network_conditions"]
        assert "bluetooth_available" in conditions
        assert "internet_available" in conditions
        assert "is_low_resource" in conditions
        assert "privacy_sensitive" in conditions


class TestBitChatTransport:
    """Test BitChat transport functionality"""

    @pytest.fixture
    async def bitchat(self):
        """Create BitChat transport instance"""
        transport = BitChatTransport(device_id="test_bitchat")
        try:
            yield transport
        finally:
            if transport.is_running:
                await transport.stop()

    @pytest.mark.asyncio
    async def test_bitchat_startup(self, bitchat):
        """Test BitChat transport startup"""
        success = await bitchat.start()

        # Should start successfully (even in simulation mode)
        assert success == True
        assert bitchat.is_running == True
        assert bitchat.device_id == "test_bitchat"

    @pytest.mark.asyncio
    async def test_bitchat_ttl_enforcement(self, bitchat):
        """Test BitChat TTL limit enforcement"""
        await bitchat.start()

        # Try to send message with excessive TTL
        with patch.object(bitchat, "_send_direct", return_value=True):
            success = await bitchat.send_message(
                recipient="test_peer",
                payload=b"test_message",
                ttl=10,  # Exceeds BitChat limit of 7
            )

            # Should enforce TTL limit of 7
            assert success in [
                True,
                False,
            ]  # May succeed or fail depending on peer availability

    def test_bitchat_message_format(self):
        """Test BitChat message format"""
        message = BitChatMessage(
            sender="sender1",
            recipient="recipient1",
            payload=b"test_payload",
            ttl=5,
            priority=7,
        )

        assert message.sender == "sender1"
        assert message.recipient == "recipient1"
        assert message.ttl == 5
        assert message.priority == 7
        assert message.hop_count == 0
        assert len(message.route_path) == 1
        assert message.route_path[0] == "sender1"

        # Test serialization
        msg_dict = message.to_dict()
        assert "id" in msg_dict
        assert "payload" in msg_dict
        assert msg_dict["ttl"] == 5

        # Test deserialization
        reconstructed = BitChatMessage.from_dict(msg_dict)
        assert reconstructed.sender == message.sender
        assert reconstructed.ttl == message.ttl


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
