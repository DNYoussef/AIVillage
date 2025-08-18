"""Tests for libp2p_mesh stub implementations."""

from unittest.mock import AsyncMock, patch

import pytest

from .libp2p_mesh import (
    LibP2PMeshNetwork,
    MeshConfiguration,
    MeshMessage,
    MeshMessageType,
)


class TestPlaceholderTypes:
    """Test placeholder types when LibP2P is not available."""

    @patch("src.core.p2p.libp2p_mesh.LIBP2P_AVAILABLE", False)
    def test_placeholder_inetstream(self):
        """Test INetStream placeholder functionality."""
        from .libp2p_mesh import INetStream

        stream = INetStream()
        assert stream is not None

    @pytest.mark.asyncio
    @patch("src.core.p2p.libp2p_mesh.LIBP2P_AVAILABLE", False)
    async def test_placeholder_inetstream_methods(self):
        """Test INetStream placeholder methods."""
        from .libp2p_mesh import INetStream

        stream = INetStream()

        # Test read method
        data = await stream.read(1024)
        assert data == b""

        # Test write method (should not raise)
        await stream.write(b"test data")

        # Test close method (should not raise)
        await stream.close()

    @patch("src.core.p2p.libp2p_mesh.LIBP2P_AVAILABLE", False)
    def test_placeholder_pubsub(self):
        """Test Pubsub placeholder functionality."""
        from .libp2p_mesh import Pubsub

        pubsub = Pubsub()
        assert pubsub is not None

    @pytest.mark.asyncio
    @patch("src.core.p2p.libp2p_mesh.LIBP2P_AVAILABLE", False)
    async def test_placeholder_pubsub_methods(self):
        """Test Pubsub placeholder methods."""
        from .libp2p_mesh import Pubsub

        pubsub = Pubsub()

        # Test subscribe method (should log warning)
        with patch("src.core.p2p.libp2p_mesh.logger") as mock_logger:
            await pubsub.subscribe("test_topic")
            mock_logger.warning.assert_called_once()

        # Test publish method (should log warning)
        with patch("src.core.p2p.libp2p_mesh.logger") as mock_logger:
            await pubsub.publish("test_topic", b"test data")
            mock_logger.warning.assert_called_once()

    @patch("src.core.p2p.libp2p_mesh.LIBP2P_AVAILABLE", False)
    def test_placeholder_kaddht(self):
        """Test KadDHT placeholder functionality."""
        from .libp2p_mesh import KadDHT

        dht = KadDHT()
        assert dht is not None

    @pytest.mark.asyncio
    @patch("src.core.p2p.libp2p_mesh.LIBP2P_AVAILABLE", False)
    async def test_placeholder_kaddht_methods(self):
        """Test KadDHT placeholder methods."""
        from .libp2p_mesh import KadDHT

        dht = KadDHT()

        # Test provide method (should log warning)
        with patch("src.core.p2p.libp2p_mesh.logger") as mock_logger:
            await dht.provide("test_key")
            mock_logger.warning.assert_called_once()

        # Test get_providers method (should log warning and return empty list)
        with patch("src.core.p2p.libp2p_mesh.logger") as mock_logger:
            providers = await dht.get_providers("test_key")
            mock_logger.warning.assert_called_once()
            assert providers == []


class TestLibP2PMeshNetworkFallback:
    """Test LibP2PMeshNetwork fallback discovery implementation."""

    def test_mesh_network_initialization(self):
        """Test mesh network initialization."""
        config = MeshConfiguration(
            node_id="test_node", listen_port=4001, mdns_enabled=True, dht_enabled=True
        )

        network = LibP2PMeshNetwork(config)
        assert network.config == config
        assert network.node_id == "test_node"

    @pytest.mark.asyncio
    async def test_fallback_discovery_mechanism(self):
        """Test fallback discovery when LibP2P is not available."""
        config = MeshConfiguration(
            node_id="test_node", listen_port=4001, mdns_enabled=True, dht_enabled=True
        )

        network = LibP2PMeshNetwork(config)

        # Mock fallback node
        mock_fallback = AsyncMock()
        mock_fallback.discover_peers = AsyncMock()
        mock_fallback.get_active_peers.return_value = ["peer1", "peer2", "peer3"]
        network.fallback_node = mock_fallback

        # Test the fallback discovery mechanism
        await network._dht_peer_discovery()

        # Verify fallback discovery was called
        mock_fallback.discover_peers.assert_called_once()
        mock_fallback.get_active_peers.assert_called_once()

        # Verify peers were added to connected_peers
        assert "peer1" in network.connected_peers
        assert "peer2" in network.connected_peers
        assert "peer3" in network.connected_peers

    @pytest.mark.asyncio
    async def test_fallback_discovery_error_handling(self):
        """Test error handling in fallback discovery."""
        config = MeshConfiguration(
            node_id="test_node", listen_port=4001, mdns_enabled=True, dht_enabled=True
        )

        network = LibP2PMeshNetwork(config)

        # Mock fallback node that raises exception
        mock_fallback = AsyncMock()
        mock_fallback.discover_peers.side_effect = Exception("Discovery failed")
        network.fallback_node = mock_fallback

        # Test fallback discovery with error (should not raise)
        with patch("src.core.p2p.libp2p_mesh.logger") as mock_logger:
            await network._dht_peer_discovery()
            mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_fallback_discovery_no_fallback_node(self):
        """Test fallback discovery when no fallback node is available."""
        config = MeshConfiguration(
            node_id="test_node", listen_port=4001, mdns_enabled=True, dht_enabled=True
        )

        network = LibP2PMeshNetwork(config)
        network.fallback_node = None

        # Test fallback discovery without fallback node (should not raise)
        await network._dht_peer_discovery()

        # No peers should be added
        assert len(network.connected_peers) == 0

    @pytest.mark.asyncio
    async def test_fallback_discovery_duplicate_peers(self):
        """Test fallback discovery handles duplicate peers correctly."""
        config = MeshConfiguration(
            node_id="test_node", listen_port=4001, mdns_enabled=True, dht_enabled=True
        )

        network = LibP2PMeshNetwork(config)

        # Pre-populate some connected peers
        network.connected_peers = {"peer1": 1234567890.0}

        # Mock fallback node with overlapping peers
        mock_fallback = AsyncMock()
        mock_fallback.discover_peers = AsyncMock()
        mock_fallback.get_active_peers.return_value = ["peer1", "peer2"]
        network.fallback_node = mock_fallback

        # Test fallback discovery
        await network._dht_peer_discovery()

        # Verify only new peer was added
        assert "peer1" in network.connected_peers  # Was already there
        assert "peer2" in network.connected_peers  # Newly added
        assert len(network.connected_peers) == 2

    def test_mesh_message_creation(self):
        """Test MeshMessage creation and serialization."""
        message = MeshMessage(
            type=MeshMessageType.AGENT_TASK,
            sender="test_node",
            recipient="target_node",
            payload=b"test payload",
            ttl=5,
        )

        assert message.type == MeshMessageType.AGENT_TASK
        assert message.sender == "test_node"
        assert message.recipient == "target_node"
        assert message.payload == b"test payload"
        assert message.ttl == 5

    def test_mesh_message_serialization(self):
        """Test MeshMessage to_dict serialization."""
        message = MeshMessage(
            type=MeshMessageType.DATA_MESSAGE,
            sender="sender",
            recipient="recipient",
            payload=b"binary data",
            ttl=3,
        )

        data = message.to_dict()

        assert data["type"] == "DATA_MESSAGE"
        assert data["sender"] == "sender"
        assert data["recipient"] == "recipient"
        assert data["payload"] == b"binary data".hex()
        assert data["ttl"] == 3

    def test_mesh_message_deserialization(self):
        """Test MeshMessage from_dict deserialization."""
        data = {
            "id": "test-id",
            "type": "PARAMETER_UPDATE",
            "sender": "sender",
            "recipient": "recipient",
            "payload": "48656c6c6f",  # "Hello" in hex
            "ttl": 2,
            "timestamp": 1234567890.0,
            "hop_count": 1,
        }

        message = MeshMessage.from_dict(data)

        assert message.id == "test-id"
        assert message.type == MeshMessageType.PARAMETER_UPDATE
        assert message.sender == "sender"
        assert message.recipient == "recipient"
        assert message.payload == b"Hello"
        assert message.ttl == 2
        assert message.timestamp == 1234567890.0
        assert message.hop_count == 1


class TestMeshConfiguration:
    """Test MeshConfiguration functionality."""

    def test_mesh_configuration_creation(self):
        """Test MeshConfiguration creation with defaults."""
        config = MeshConfiguration(node_id="test_node", listen_port=4001)

        assert config.node_id == "test_node"
        assert config.listen_port == 4001
        assert config.mdns_enabled is True  # Default
        assert config.dht_enabled is True  # Default

    def test_mesh_configuration_custom_values(self):
        """Test MeshConfiguration with custom values."""
        config = MeshConfiguration(
            node_id="custom_node",
            listen_port=8080,
            mdns_enabled=False,
            dht_enabled=False,
        )

        assert config.node_id == "custom_node"
        assert config.listen_port == 8080
        assert config.mdns_enabled is False
        assert config.dht_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
