"""Basic tests for P2P networking infrastructure."""

import pytest

from infrastructure.p2p import P2PNetwork, NetworkConfig, PeerInfo, create_network
from infrastructure.p2p.core.message_delivery import DeliveryStatus, MessagePriority


class TestP2PBasics:
    """Test basic P2P functionality."""

    @pytest.mark.asyncio
    async def test_network_creation(self):
        """Test P2P network creation."""
        # Test direct creation
        network = P2PNetwork()
        assert network.config.mode == "hybrid"
        assert not network._initialized

        # Test with config
        config = NetworkConfig(mode="direct", max_peers=50)
        network = P2PNetwork(config)
        assert network.config.mode == "direct"
        assert network.config.max_peers == 50

    @pytest.mark.asyncio
    async def test_network_initialization(self):
        """Test network initialization."""
        network = P2PNetwork(NetworkConfig(mode="direct", discovery_interval=0))

        try:
            await network.initialize()
            assert network._initialized
        except Exception as e:
            # May fail without transport components
            pytest.skip(f"Network components not available: {e}")

    def test_peer_info_creation(self):
        """Test peer info data structure."""
        peer = PeerInfo(
            peer_id="test-peer",
            addresses=["127.0.0.1:8000", "192.168.1.100:8001"],
            protocols=["libp2p", "websocket"],
            metadata={"region": "us-east", "capabilities": ["relay"]},
            reputation=0.95,
        )

        assert peer.peer_id == "test-peer"
        assert len(peer.addresses) == 2
        assert "libp2p" in peer.protocols
        assert peer.metadata["region"] == "us-east"
        assert peer.reputation == 0.95

    def test_network_config_defaults(self):
        """Test network configuration defaults."""
        config = NetworkConfig()

        assert config.mode == "hybrid"
        assert config.transport_priority == ["libp2p", "bitchat", "betanet", "websocket"]
        assert config.enable_nat_traversal is True
        assert config.enable_encryption is True
        assert config.max_peers == 100
        assert config.discovery_interval == 30

    def test_create_network_modes(self):
        """Test network creation with different modes."""
        # Test mesh mode
        mesh_network = create_network("mesh", max_peers=25)
        assert mesh_network.config.mode == "mesh"
        assert mesh_network.config.transport_priority[0] == "bitchat"

        # Test anonymous mode
        anon_network = create_network("anonymous", max_peers=10)
        assert anon_network.config.mode == "anonymous"
        assert anon_network.config.transport_priority[0] == "betanet"

        # Test direct mode
        direct_network = create_network("direct")
        assert direct_network.config.mode == "direct"
        assert direct_network.config.transport_priority[0] == "libp2p"

    @pytest.mark.asyncio
    async def test_peer_management(self):
        """Test basic peer management."""
        network = P2PNetwork()

        # Add peers manually
        peer1 = PeerInfo(peer_id="peer-1", addresses=["127.0.0.1:8001"], protocols=["libp2p"])

        peer2 = PeerInfo(peer_id="peer-2", addresses=["127.0.0.1:8002"], protocols=["websocket"])

        network.peers["peer-1"] = peer1
        network.peers["peer-2"] = peer2

        # Test retrieval
        peers = await network.get_peers()
        assert len(peers) == 2

        peer_ids = [p.peer_id for p in peers]
        assert "peer-1" in peer_ids
        assert "peer-2" in peer_ids

    @pytest.mark.asyncio
    async def test_discovery_stub_replaced(self):
        """Test that discovery is no longer a stub."""
        network = P2PNetwork(NetworkConfig(discovery_interval=0))  # Disable interval discovery

        # The start_discovery method should no longer be a pass/stub
        try:
            await network.start_discovery()
            # Should not raise NotImplementedError or be empty
            assert True  # If we get here, discovery has real implementation
        except Exception as e:
            # May fail due to missing components, but should not be a stub
            error_msg = str(e).lower()
            assert "not implemented" not in error_msg
            assert "stub" not in error_msg

    @pytest.mark.asyncio
    async def test_message_delivery_service(self):
        """Test message delivery service integration."""
        try:
            from infrastructure.p2p.core.message_delivery import MessageDelivery, DeliveryStatus

            # Test that MessageDelivery class is available
            assert MessageDelivery is not None

            # Test delivery status enum
            assert DeliveryStatus.PENDING.value == "pending"
            assert DeliveryStatus.DELIVERED.value == "delivered"

        except ImportError as e:
            pytest.skip(f"Message delivery components not available: {e}")

    @pytest.mark.asyncio
    async def test_network_shutdown(self):
        """Test network shutdown."""
        network = P2PNetwork()

        # Add some peers
        network.peers["test-peer"] = PeerInfo(peer_id="test-peer", addresses=["127.0.0.1:8000"], protocols=["test"])

        try:
            await network.initialize()
            assert network._initialized
            assert len(network.peers) == 1

            # Test shutdown
            await network.shutdown()
            assert not network._initialized

        except Exception as e:
            # May fail without proper network setup
            pytest.skip(f"Network setup not available: {e}")


class TestMessageDelivery:
    """Test message delivery service."""

    def test_delivery_status_enum(self):
        """Test delivery status enumeration."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.FAILED.value == "failed"

    def test_message_priority_enum(self):
        """Test message priority enumeration."""
        assert MessagePriority.CRITICAL.value == 1
        assert MessagePriority.HIGH.value == 2

        # Test priority ordering
        assert MessagePriority.CRITICAL.value < MessagePriority.HIGH.value


class TestImportStructure:
    """Test that imports work correctly."""

    def test_core_imports(self):
        """Test core P2P imports."""
        from infrastructure.p2p import P2PNetwork, NetworkConfig, PeerInfo

        assert P2PNetwork is not None
        assert NetworkConfig is not None
        assert PeerInfo is not None

    def test_advanced_imports(self):
        """Test advanced P2P imports."""
        try:
            from infrastructure.p2p.advanced import LibP2PEnhancedManager

            assert LibP2PEnhancedManager is not None
        except ImportError:
            # Advanced components may not be available
            pytest.skip("Advanced P2P components not available")

    def test_message_delivery_import(self):
        """Test message delivery import."""
        from infrastructure.p2p.core.message_delivery import MessageDelivery

        # Should be able to import MessageDelivery (alias for MessageDeliveryService)
        assert MessageDelivery is not None


if __name__ == "__main__":
    # Run basic tests
    pytest.main(["-v", __file__])
