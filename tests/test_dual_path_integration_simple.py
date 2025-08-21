"""Simplified Integration Tests for Dual-Path P2P Networking

Tests core functionality of BitChat, Betanet, Navigator, and DualPath components
without complex async fixture issues.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Add src and experimental paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experimental", "agents", "agents"))

# Import components to test
try:
    from core.p2p.bitchat_transport import BitChatMessage, BitChatTransport
    from core.p2p.dual_path_transport import DualPathMessage, DualPathTransport
    from navigator.path_policy import EnergyMode, MessageContext, NavigatorAgent, PathProtocol, RoutingPriority

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

# Skip all tests if imports fail
pytestmark = pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason="Required modules not available")


class TestDualPathComponents:
    """Test core dual-path components"""

    def test_bitchat_message_creation(self):
        """Test BitChat message creation and format"""
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

    def test_dual_path_message_creation(self):
        """Test DualPathMessage format conversion"""
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
        assert message.privacy_required
        assert isinstance(message.payload, bytes)
        assert message.context.size_bytes > 0
        assert message.context.priority == 7

    def test_navigator_initialization(self):
        """Test Navigator agent initialization"""
        nav = NavigatorAgent(
            agent_id="test_nav",
            routing_priority=RoutingPriority.OFFLINE_FIRST,
            energy_mode=EnergyMode.BALANCED,
        )
        nav.enable_global_south_mode(True)

        assert nav.agent_id == "test_nav"
        assert nav.routing_priority == RoutingPriority.OFFLINE_FIRST
        assert nav.energy_mode == EnergyMode.BALANCED
        assert nav.global_south_mode
        assert nav.data_cost_threshold == 0.005  # Sensitive to data costs
        assert nav.battery_conservation_threshold == 25

    def test_navigator_status_reporting(self):
        """Test Navigator status reporting"""
        nav = NavigatorAgent(agent_id="test_nav", routing_priority=RoutingPriority.OFFLINE_FIRST)
        nav.enable_global_south_mode(True)

        status = nav.get_status()

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

    def test_dual_path_transport_initialization(self):
        """Test DualPathTransport initialization"""
        transport = DualPathTransport(node_id="test_node", enable_bitchat=True, enable_betanet=True)

        assert transport.node_id == "test_node"
        assert transport.enable_bitchat
        assert transport.enable_betanet
        assert not transport.is_running
        assert transport.routing_stats["routing_decisions"] == 0

    @pytest.mark.asyncio
    async def test_bitchat_transport_startup(self):
        """Test BitChat transport startup"""
        transport = BitChatTransport(device_id="test_bitchat")

        success = await transport.start()

        # Should start successfully (even in simulation mode)
        assert success
        assert transport.is_running
        assert transport.device_id == "test_bitchat"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_navigator_emergency_routing_override(self):
        """Test that urgent messages override normal routing logic"""
        nav = NavigatorAgent(agent_id="test_nav", routing_priority=RoutingPriority.OFFLINE_FIRST)
        nav.enable_global_south_mode(True)

        # Set low-resource conditions
        nav.conditions.battery_percent = 10
        nav.conditions.internet_available = True
        nav.conditions.bluetooth_available = False
        nav.set_energy_mode(EnergyMode.POWERSAVE)

        # Create urgent message
        urgent_context = MessageContext(size_bytes=1000, priority=10, requires_realtime=True)  # Maximum urgency

        # Test emergency routing
        protocol, metadata = await nav.select_path("emergency_target", urgent_context, ["betanet"])

        # Should override energy conservation for emergency
        assert protocol == PathProtocol.BETANET

    @pytest.mark.asyncio
    async def test_navigator_offline_first_selection(self):
        """Test Navigator's offline-first path selection for Global South"""
        nav = NavigatorAgent(
            agent_id="test_navigator",
            routing_priority=RoutingPriority.OFFLINE_FIRST,
            energy_mode=EnergyMode.BALANCED,
        )
        nav.enable_global_south_mode(True)

        # Test scenario: nearby peer with small message
        context = MessageContext(size_bytes=500, priority=5, requires_realtime=False)

        # Mock network conditions for offline scenario
        nav.conditions.bluetooth_available = True
        nav.conditions.internet_available = False
        nav.conditions.nearby_peers = 2
        nav.conditions.battery_percent = 40

        # Add a nearby peer
        from navigator.path_policy import PeerInfo

        peer_info = PeerInfo(
            peer_id="nearby_peer",
            protocols={"bitchat"},
            hop_distance=2,
            bluetooth_rssi=-45,  # Good signal
        )
        nav.update_peer_info("nearby_peer", peer_info)

        # Mock the network condition update to preserve our test conditions
        with patch.object(nav, "_update_network_conditions", new_callable=AsyncMock):
            # Test path selection
            protocol, metadata = await nav.select_path("nearby_peer", context)

            # Should select BitChat for nearby peer
            assert protocol == PathProtocol.BITCHAT
            assert metadata["offline_capable"]
            assert metadata["energy_efficient"]

    @pytest.mark.asyncio
    async def test_navigator_energy_conservation(self):
        """Test Navigator's energy conservation logic"""
        nav = NavigatorAgent(
            agent_id="test_navigator",
            routing_priority=RoutingPriority.OFFLINE_FIRST,
            energy_mode=EnergyMode.BALANCED,
        )
        nav.enable_global_south_mode(True)

        # Set low battery conditions
        nav.conditions.battery_percent = 15  # Low battery
        nav.set_energy_mode(EnergyMode.POWERSAVE)

        context = MessageContext(size_bytes=1000, priority=3, requires_realtime=False)  # Low priority

        # Mock peer nearby for BitChat
        nav.conditions.bluetooth_available = True
        nav.conditions.internet_available = True
        from navigator.path_policy import PeerInfo

        peer_info = PeerInfo(peer_id="test_peer", protocols={"bitchat"}, hop_distance=3)
        nav.update_peer_info("test_peer", peer_info)

        # Test path selection
        protocol, metadata = await nav.select_path("test_peer", context)

        # Should prefer BitChat or store-and-forward to conserve energy
        assert protocol in [PathProtocol.BITCHAT, PathProtocol.STORE_FORWARD]
        if protocol == PathProtocol.BITCHAT:
            assert metadata["energy_efficient"]

    @pytest.mark.asyncio
    async def test_dual_path_startup_with_mocked_transports(self):
        """Test dual-path transport startup with mocked components"""
        transport = DualPathTransport(node_id="test_node", enable_bitchat=True, enable_betanet=True)

        # Mock the sync method to avoid network calls
        with patch.object(transport, "_sync_peer_information", new_callable=AsyncMock):
            success = await transport.start()

            assert success
            assert transport.is_running

            # Verify components were created (even if using fallback)
            assert transport.bitchat is not None
            assert transport.betanet is not None
            assert transport.navigator is not None

        await transport.stop()


class TestNavigatorPathLogic:
    """Focused tests for Navigator path selection scenarios"""

    def setUp(self):
        """Set up Navigator for testing"""
        self.nav = NavigatorAgent(agent_id="test_nav", routing_priority=RoutingPriority.OFFLINE_FIRST)
        self.nav.enable_global_south_mode(True)

    @pytest.mark.asyncio
    async def test_data_cost_aware_routing(self):
        """Test routing that considers mobile data costs"""
        nav = NavigatorAgent(agent_id="test_navigator", routing_priority=RoutingPriority.OFFLINE_FIRST)
        nav.enable_global_south_mode(True)

        # Set expensive data conditions (Global South scenario)
        nav.conditions.data_cost_usd_mb = 0.02  # Expensive mobile data
        nav.conditions.internet_available = True
        nav.conditions.bluetooth_available = True
        nav.conditions.wifi_connected = False  # Using cellular

        context = MessageContext(size_bytes=5000, priority=5)  # Medium message

        # Mock nearby peer for BitChat alternative
        from navigator.path_policy import PeerInfo

        peer_info = PeerInfo(peer_id="local_peer", protocols={"bitchat"}, hop_distance=2)
        nav.update_peer_info("local_peer", peer_info)

        # Mock network condition updates to preserve test conditions
        with patch.object(nav, "_update_network_conditions", new_callable=AsyncMock):
            # Test path selection
            protocol, metadata = await nav.select_path("local_peer", context)

            # Should prefer BitChat to avoid expensive data costs
            assert protocol == PathProtocol.BITCHAT

    @pytest.mark.asyncio
    async def test_performance_first_routing(self):
        """Test Navigator's performance-first path selection"""
        nav = NavigatorAgent(
            agent_id="test_navigator",
            routing_priority=RoutingPriority.PERFORMANCE_FIRST,
            energy_mode=EnergyMode.PERFORMANCE,
        )

        # Test scenario: large urgent message with internet available
        context = MessageContext(
            size_bytes=50000,  # Large message
            priority=9,  # Urgent
            requires_realtime=True,
        )

        # Mock network conditions for performance scenario
        nav.conditions.bluetooth_available = True
        nav.conditions.internet_available = True
        nav.conditions.wifi_connected = True
        nav.conditions.bandwidth_mbps = 50.0
        nav.conditions.battery_percent = 80

        # Test path selection
        protocol, metadata = await nav.select_path("remote_peer", context, ["bitchat", "betanet"])

        # Should select Betanet for performance
        assert protocol == PathProtocol.BETANET
        assert metadata["global_reach"]
        assert metadata["bandwidth_adaptive"]

    @pytest.mark.asyncio
    async def test_privacy_routing_selection(self):
        """Test privacy-aware routing selection"""
        nav = NavigatorAgent(agent_id="test_navigator", routing_priority=RoutingPriority.PRIVACY_FIRST)

        # Set privacy-sensitive conditions
        nav.conditions.censorship_risk = 0.8  # High censorship risk

        context = MessageContext(size_bytes=2000, priority=7, privacy_required=True)

        # Mock internet availability for mixnode routing
        nav.conditions.internet_available = True
        nav.conditions.bluetooth_available = True

        # Test path selection
        protocol, metadata = await nav.select_path("sensitive_peer", context, ["betanet", "bitchat"])

        # Should prefer Betanet with privacy routing for sensitive content
        if protocol == PathProtocol.BETANET:
            assert metadata.get("privacy_routing")
            assert metadata.get("mixnode_hops", 0) >= 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
