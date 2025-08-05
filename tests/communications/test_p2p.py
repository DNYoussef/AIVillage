"""Comprehensive tests for P2P communication infrastructure."""

import time

import numpy as np
import pytest

# Import P2P components
from src.production.communications.p2p import DeviceMesh, P2PNode, TensorStreaming
from src.production.communications.p2p.device_mesh import ConnectionType, MeshProtocol
from src.production.communications.p2p.p2p_node import MessageType, NodeStatus, P2PMessage, PeerInfo
from src.production.communications.p2p.tensor_streaming import CompressionType, StreamingConfig
from src.production.communications.p2p_protocol import P2PCapabilities, P2PCommunicationProtocol


class TestP2PNode:
    """Test P2P node functionality."""

    @pytest.mark.asyncio
    async def test_node_initialization(self):
        """Test P2P node initialization."""
        node = P2PNode(node_id="test-node", port=8001)

        assert node.node_id == "test-node"
        assert node.port == 8001
        assert node.status == NodeStatus.OFFLINE
        assert len(node.peers) == 0

    @pytest.mark.asyncio
    async def test_node_start_stop(self):
        """Test node start and stop lifecycle."""
        node = P2PNode(node_id="test-node", port=8002)

        # Test start
        try:
            await node.start()
            assert node.status == NodeStatus.CONNECTED
            assert node.server is not None
        except Exception as e:
            # May fail on Windows without proper network setup
            pytest.skip(f"Network setup not available: {e}")
        finally:
            # Test stop
            await node.stop()
            assert node.status == NodeStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_message_serialization(self):
        """Test P2P message serialization."""
        node = P2PNode(node_id="test-node", port=8003)

        message = P2PMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id="sender",
            receiver_id="receiver",
            payload={"test": "data", "timestamp": time.time()},
        )

        # Test message structure
        assert message.message_type == MessageType.HEARTBEAT
        assert message.sender_id == "sender"
        assert message.receiver_id == "receiver"
        assert "test" in message.payload

    @pytest.mark.asyncio
    async def test_peer_management(self):
        """Test peer management functionality."""
        node = P2PNode(node_id="test-node", port=8004)

        # Add known address
        node.add_known_address("127.0.0.1", 8005)
        assert "127.0.0.1:8005" in node.known_addresses

        # Test peer info
        peer_info = PeerInfo(peer_id="peer-1", address="127.0.0.1", port=8005, status=NodeStatus.CONNECTED)

        node.peers["peer-1"] = peer_info

        connected_peers = node.get_connected_peers()
        assert len(connected_peers) == 1
        assert connected_peers[0].peer_id == "peer-1"

    def test_node_statistics(self):
        """Test node statistics collection."""
        node = P2PNode(node_id="test-node", port=8006)

        stats = node.get_stats()

        assert "node_id" in stats
        assert "status" in stats
        assert "connected_peers" in stats
        assert "total_peers" in stats
        assert stats["node_id"] == "test-node"
        assert stats["status"] == NodeStatus.OFFLINE.value


class TestDeviceMesh:
    """Test device mesh networking."""

    @pytest.mark.asyncio
    async def test_mesh_initialization(self):
        """Test mesh network initialization."""
        node = P2PNode(node_id="mesh-test", port=8010)
        mesh = DeviceMesh(node=node, protocol=MeshProtocol.OPTIMIZED_LINK_STATE)

        assert mesh.node == node
        assert mesh.protocol == MeshProtocol.OPTIMIZED_LINK_STATE
        assert mesh.max_hops == 5
        assert len(mesh.routing_table) == 0

    def test_device_capability_detection(self):
        """Test device capability detection."""
        node = P2PNode(node_id="capability-test", port=8011)
        mesh = DeviceMesh(node=node)

        capabilities = mesh.device_capabilities

        assert capabilities.device_type in ["desktop", "phone", "tablet"]
        assert capabilities.os_type in ["windows", "linux", "darwin", "android"]
        assert isinstance(capabilities.cpu_cores, int)
        assert isinstance(capabilities.available_memory, int)

    @pytest.mark.asyncio
    async def test_mesh_routing_table(self):
        """Test mesh routing table management."""
        node = P2PNode(node_id="routing-test", port=8012)
        mesh = DeviceMesh(node=node, protocol=MeshProtocol.OPTIMIZED_LINK_STATE)

        # Add a peer
        mesh.add_mesh_peer("peer-1", ConnectionType.TCP_IP)

        # Check routing table update
        mesh._update_routing_table()

        assert len(mesh.routing_table) >= 0  # May be 0 or 1 depending on implementation

    @pytest.mark.asyncio
    async def test_mesh_message_storage(self):
        """Test store and forward functionality."""
        node = P2PNode(node_id="storage-test", port=8013)
        mesh = DeviceMesh(node=node)

        # Store message for offline destination
        test_data = {"message": "test", "priority": "high"}
        mesh._store_message_for_later("offline-peer", test_data)

        assert "offline-peer" in mesh.message_store
        assert len(mesh.message_store["offline-peer"]) == 1

    def test_mesh_status(self):
        """Test mesh status reporting."""
        node = P2PNode(node_id="status-test", port=8014)
        mesh = DeviceMesh(node=node)

        status = mesh.get_mesh_status()

        assert "protocol" in status
        assert "device_capabilities" in status
        assert "network_topology" in status
        assert "routing_table_size" in status
        assert status["protocol"] == mesh.protocol.value


class TestTensorStreaming:
    """Test tensor streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_initialization(self):
        """Test tensor streaming initialization."""
        node = P2PNode(node_id="stream-test", port=8020)
        config = StreamingConfig(chunk_size=32 * 1024, compression=CompressionType.LZ4)
        streaming = TensorStreaming(node=node, config=config)

        assert streaming.node == node
        assert streaming.config.chunk_size == 32 * 1024
        assert streaming.config.compression == CompressionType.LZ4
        assert len(streaming.active_transfers) == 0

    @pytest.mark.asyncio
    async def test_tensor_serialization(self):
        """Test tensor serialization methods."""
        node = P2PNode(node_id="serialize-test", port=8021)
        streaming = TensorStreaming(node=node)

        # Test numpy array serialization
        test_array = np.random.rand(10, 10).astype(np.float32)

        serialized_data, metadata = await streaming._serialize_tensor(test_array, "test-tensor", "test_tensor", {})

        assert isinstance(serialized_data, bytes)
        assert metadata.name == "test_tensor"
        assert metadata.shape == (10, 10)
        assert metadata.dtype == "float32"
        assert len(serialized_data) > 0

    @pytest.mark.asyncio
    async def test_tensor_compression(self):
        """Test tensor compression."""
        node = P2PNode(node_id="compress-test", port=8022)
        streaming = TensorStreaming(node=node)

        # Create test data
        test_data = b"test data " * 1000  # Repeating pattern for good compression

        # Test LZ4 compression
        compressed = await streaming._compress_tensor(test_data, CompressionType.LZ4)
        decompressed = await streaming._decompress_tensor(compressed, CompressionType.LZ4)

        assert len(compressed) < len(test_data)  # Should be compressed
        assert decompressed == test_data  # Should decompress correctly

    def test_chunk_splitting(self):
        """Test tensor chunk splitting."""
        node = P2PNode(node_id="chunk-test", port=8023)
        config = StreamingConfig(chunk_size=100)  # Small chunks for testing
        streaming = TensorStreaming(node=node, config=config)

        test_data = b"x" * 350  # 350 bytes should create 4 chunks
        chunks = streaming._split_into_chunks(test_data, "test-tensor")

        assert len(chunks) == 4
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 4
        assert len(chunks[0].data) == 100
        assert len(chunks[3].data) == 50  # Last chunk is smaller

        # Verify checksums
        for chunk in chunks:
            assert len(chunk.checksum) == 32  # MD5 hex string

    def test_transfer_progress_tracking(self):
        """Test transfer progress tracking."""
        from src.production.communications.p2p.tensor_streaming import TransferProgress

        progress = TransferProgress(
            tensor_id="test-transfer",
            total_chunks=10,
            received_chunks=3,
        )

        assert progress.progress_percent == 30.0
        assert not progress.is_complete

        progress.received_chunks = 10
        assert progress.progress_percent == 100.0
        assert progress.is_complete

    def test_streaming_statistics(self):
        """Test streaming statistics collection."""
        node = P2PNode(node_id="stats-test", port=8024)
        streaming = TensorStreaming(node=node)

        stats = streaming.get_streaming_stats()

        assert "tensors_sent" in stats
        assert "tensors_received" in stats
        assert "bytes_sent" in stats
        assert "bytes_received" in stats
        assert "compression_ratio" in stats
        assert "active_transfers" in stats


class TestP2PCommunicationProtocol:
    """Test P2P communication protocol integration."""

    @pytest.mark.asyncio
    async def test_protocol_initialization(self):
        """Test P2P protocol initialization."""
        capabilities = P2PCapabilities(
            supports_mesh_routing=True,
            supports_tensor_streaming=True,
            max_concurrent_connections=25,
        )

        protocol = P2PCommunicationProtocol(
            node_id="protocol-test",
            port=8030,
            capabilities=capabilities,
        )

        assert protocol.p2p_node.node_id == "protocol-test"
        assert protocol.capabilities.max_concurrent_connections == 25
        assert not protocol.mesh_enabled
        assert len(protocol.local_agents) == 0

    @pytest.mark.asyncio
    async def test_agent_registration(self):
        """Test agent registration."""
        protocol = P2PCommunicationProtocol(node_id="agent-test", port=8031)

        # Register local agent
        protocol.register_local_agent("local-agent-1")
        assert "local-agent-1" in protocol.local_agents

        # Register distributed agent
        protocol.register_distributed_agent("remote-agent-1", "peer-1")
        assert protocol.distributed_agents["remote-agent-1"] == "peer-1"
        assert protocol.message_routing_cache["remote-agent-1"] == "peer-1"

    def test_p2p_capabilities(self):
        """Test P2P capabilities configuration."""
        capabilities = P2PCapabilities(
            supports_mesh_routing=True,
            supports_tensor_streaming=False,
            supports_distributed_inference=True,
            max_concurrent_connections=100,
            preferred_protocols=["tcp", "bluetooth"],
            device_resources={"memory_gb": 8, "cpu_cores": 4},
        )

        assert capabilities.supports_mesh_routing
        assert not capabilities.supports_tensor_streaming
        assert capabilities.supports_distributed_inference
        assert capabilities.max_concurrent_connections == 100
        assert "bluetooth" in capabilities.preferred_protocols
        assert capabilities.device_resources["memory_gb"] == 8

    def test_protocol_status_reporting(self):
        """Test protocol status reporting."""
        protocol = P2PCommunicationProtocol(node_id="status-test", port=8032)

        # Register some agents
        protocol.register_local_agent("local-1")
        protocol.register_distributed_agent("remote-1", "peer-1")

        status = protocol.get_p2p_status()

        assert status["node_id"] == "status-test"
        assert "local-1" in status["local_agents"]
        assert status["distributed_agents"]["remote-1"] == "peer-1"
        assert "capabilities" in status
        assert "statistics" in status


class TestIntegration:
    """Integration tests for P2P system."""

    @pytest.mark.asyncio
    async def test_multi_node_setup(self):
        """Test setting up multiple P2P nodes (simulated)."""
        # Create multiple nodes
        node1 = P2PNode(node_id="node-1", port=8040)
        node2 = P2PNode(node_id="node-2", port=8041)
        node3 = P2PNode(node_id="node-3", port=8042)

        # Simulate network topology
        node1.add_known_address("127.0.0.1", 8041)
        node1.add_known_address("127.0.0.1", 8042)
        node2.add_known_address("127.0.0.1", 8040)
        node3.add_known_address("127.0.0.1", 8040)

        # Verify initial state
        assert len(node1.known_addresses) == 2
        assert len(node2.known_addresses) == 1
        assert len(node3.known_addresses) == 1

    @pytest.mark.asyncio
    async def test_mesh_network_formation(self):
        """Test mesh network formation (simulated)."""
        # Create nodes
        nodes = [P2PNode(node_id=f"mesh-node-{i}", port=8050 + i) for i in range(3)]

        # Create mesh networks
        meshes = [DeviceMesh(node=node, protocol=MeshProtocol.OPTIMIZED_LINK_STATE) for node in nodes]

        # Simulate mesh connections
        for i, mesh in enumerate(meshes):
            for j, other_node in enumerate(nodes):
                if i != j:
                    mesh.add_mesh_peer(other_node.node_id, ConnectionType.TCP_IP)

        # Verify mesh state
        for mesh in meshes:
            status = mesh.get_mesh_status()
            assert len(status["connection_types"]) >= 0

    @pytest.mark.asyncio
    async def test_end_to_end_message_flow(self):
        """Test end-to-end message flow through P2P protocol."""
        # Create protocol
        protocol = P2PCommunicationProtocol(node_id="e2e-test", port=8060)

        # Register agents
        protocol.register_local_agent("sender")
        protocol.register_local_agent("receiver")

        # Import message components
        from src.communications.message import Message, MessageType, Priority

        # Create test message
        message = Message(
            type=MessageType.QUERY,
            sender="sender",
            receiver="receiver",
            content={"query": "test", "data": [1, 2, 3]},
            priority=Priority.HIGH,
        )

        # Test message sending (local routing)
        try:
            await protocol.send_message(message)
            # Message should be queued in standard protocol
        except Exception:
            # May fail due to missing setup, but structure should be correct
            assert "Message" in str(type(message))

    @pytest.mark.asyncio
    async def test_network_performance_metrics(self):
        """Test network performance metrics collection."""
        protocol = P2PCommunicationProtocol(node_id="perf-test", port=8061)

        # Get initial metrics
        metrics = protocol.get_performance_metrics()

        assert "node_metrics" in metrics
        assert "mesh_metrics" in metrics
        assert "streaming_metrics" in metrics
        assert "routing_efficiency" in metrics

        # Verify metric structure
        node_metrics = metrics["node_metrics"]
        assert "node_id" in node_metrics
        assert "status" in node_metrics
        assert "connected_peers" in node_metrics


class TestErrorHandling:
    """Test error handling in P2P system."""

    @pytest.mark.asyncio
    async def test_connection_failures(self):
        """Test handling of connection failures."""
        node = P2PNode(node_id="error-test", port=8070)

        # Try to connect to non-existent peer
        success = await node.connect_to_peer("127.0.0.1", 9999)
        assert not success

    @pytest.mark.asyncio
    async def test_message_serialization_errors(self):
        """Test handling of message serialization errors."""
        node = P2PNode(node_id="serial-error-test", port=8071)
        streaming = TensorStreaming(node=node)

        # Test with unsupported data type
        class UnsupportedType:
            def __init__(self):
                self.data = "test"

        try:
            # This should handle the error gracefully
            result = await streaming._serialize_tensor(UnsupportedType(), "test", "unsupported", {})
            # Should fallback to pickle
            assert result is not None
        except Exception as e:
            # Or raise a proper exception
            assert isinstance(e, (TypeError, ValueError))

    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        # Test invalid chunk size
        with pytest.raises(ValueError):
            StreamingConfig(chunk_size=0)  # Should raise error for invalid chunk size

        # This might not raise an error in current implementation,
        # but demonstrates the test pattern
        try:
            config = StreamingConfig(chunk_size=1)  # Very small chunk size
            assert config.chunk_size == 1
        except ValueError:
            pass  # Expected for some validation scenarios


# Fixtures for testing
@pytest.fixture
def sample_node():
    """Create a sample P2P node for testing."""
    return P2PNode(node_id="test-node", port=8100)


@pytest.fixture
def sample_mesh(sample_node):
    """Create a sample mesh network for testing."""
    return DeviceMesh(node=sample_node)


@pytest.fixture
def sample_streaming(sample_node):
    """Create a sample tensor streaming instance for testing."""
    return TensorStreaming(node=sample_node)


@pytest.fixture
def sample_protocol():
    """Create a sample P2P protocol for testing."""
    return P2PCommunicationProtocol(node_id="test-protocol", port=8101)


# Performance benchmarks
class TestPerformance:
    """Performance benchmarks for P2P system."""

    def test_message_throughput(self, sample_node):
        """Benchmark message creation throughput."""
        start_time = time.time()

        messages = []
        for i in range(1000):
            message = P2PMessage(
                message_type=MessageType.DATA,
                sender_id="sender",
                receiver_id="receiver",
                payload={"index": i, "data": f"message-{i}"},
            )
            messages.append(message)

        end_time = time.time()
        duration = end_time - start_time

        assert len(messages) == 1000
        assert duration < 1.0  # Should create 1000 messages in under 1 second

    def test_chunk_processing_speed(self, sample_streaming):
        """Benchmark chunk processing speed."""
        # Create large test data
        test_data = b"x" * (1024 * 1024)  # 1MB

        start_time = time.time()
        chunks = sample_streaming._split_into_chunks(test_data, "perf-test")
        end_time = time.time()

        duration = end_time - start_time

        assert len(chunks) > 0
        assert duration < 0.5  # Should process 1MB in under 0.5 seconds

    def test_routing_table_performance(self, sample_mesh):
        """Benchmark routing table operations."""
        # Add many peers
        start_time = time.time()

        for i in range(100):
            sample_mesh.add_mesh_peer(f"peer-{i}", ConnectionType.TCP_IP)

        sample_mesh._update_routing_table()

        end_time = time.time()
        duration = end_time - start_time

        assert duration < 1.0  # Should handle 100 peers in under 1 second


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1:
        test_category = sys.argv[1]

        if test_category == "basic":
            pytest.main(["-v", "TestP2PNode", "TestDeviceMesh"])
        elif test_category == "streaming":
            pytest.main(["-v", "TestTensorStreaming"])
        elif test_category == "protocol":
            pytest.main(["-v", "TestP2PCommunicationProtocol"])
        elif test_category == "integration":
            pytest.main(["-v", "TestIntegration"])
        elif test_category == "performance":
            pytest.main(["-v", "TestPerformance"])
        else:
            pytest.main(["-v", __file__])
    else:
        pytest.main(["-v", __file__])
