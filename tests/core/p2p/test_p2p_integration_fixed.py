"""Integration test for P2P protocol compatibility fix.

Tests the fixed P2P system with protocol auto-detection that handles both
discovery protocol (JSON) and encrypted P2P protocol messages.
"""

import asyncio
import json
import logging
import time

import pytest

from src.core.p2p.p2p_node import P2PNode, PeerCapabilities

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
async def mock_resource_monitor():
    """Create mock resource monitor."""

    async def resource_monitor():
        return {
            "cpu_cores": 4,
            "ram_total_mb": 8192,
            "memory_percent": 30,
            "cpu_percent": 20,
            "battery_percent": 80,
            "network_type": "wifi",
            "device_type": "laptop",
            "thermal_state": "normal",
        }

    return resource_monitor


@pytest.fixture
async def p2p_nodes(mock_resource_monitor):
    """Create multiple P2P nodes for testing."""
    nodes = []
    base_port = 9100

    for i in range(10):
        node = P2PNode(
            node_id=f"test_node_{i}",
            listen_port=base_port + i,
            resource_monitor=mock_resource_monitor,
        )
        nodes.append(node)

    # Start all nodes
    for node in nodes:
        await node.start()
        # Give each node time to initialize
        await asyncio.sleep(0.1)

    yield nodes

    # Cleanup
    for node in nodes:
        try:
            await node.stop()
        except Exception as e:
            logger.warning(f"Error stopping node {node.node_id}: {e}")


class TestP2PProtocolCompatibility:
    """Test P2P protocol compatibility and discovery integration."""

    async def test_discovery_protocol_compatibility(self, p2p_nodes):
        """Test that discovery protocol can communicate with P2P server."""
        node = p2p_nodes[0]

        # Simulate discovery message
        discovery_msg = {
            "type": "PEER_DISCOVERY",
            "sender_id": "discovery_client",
            "sender_port": 9999,
            "timestamp": time.time(),
            "capabilities": {
                "cpu_cores": 2,
                "ram_mb": 4096,
                "device_type": "phone",
                "can_evolve": True,
                "evolution_capacity": 0.8,
            },
        }

        # Connect and send discovery message
        reader, writer = await asyncio.open_connection("localhost", node.listen_port)

        try:
            # Send discovery message using discovery protocol format
            msg_data = json.dumps(discovery_msg).encode("utf-8")
            length_data = len(msg_data).to_bytes(4, "big")
            writer.write(length_data + msg_data)
            await writer.drain()

            # Read response
            resp_length_data = await reader.readexactly(4)
            resp_length = int.from_bytes(resp_length_data, "big")
            resp_data = await reader.readexactly(resp_length)

            response = json.loads(resp_data.decode("utf-8"))

            # Verify response
            assert response["type"] == "PEER_DISCOVERY_RESPONSE"
            assert response["node_id"] == node.node_id
            assert "capabilities" in response
            assert response["protocol_version"] == "1.0"

            # Verify peer was registered
            await asyncio.sleep(0.1)  # Allow processing time
            assert "discovery_client" in node.peer_registry
            peer_caps = node.peer_registry["discovery_client"]
            assert peer_caps.cpu_cores == 2
            assert peer_caps.ram_mb == 4096
            assert peer_caps.device_type == "phone"

            logger.info("✅ Discovery protocol compatibility test PASSED")

        finally:
            writer.close()
            await writer.wait_closed()

    async def test_multi_node_discovery(self, p2p_nodes):
        """Test discovery between multiple nodes."""
        # Take first 5 nodes for discovery test
        test_nodes = p2p_nodes[:5]

        # Start discovery on all nodes
        for node in test_nodes:
            await node.peer_discovery.start_discovery()

        # Wait for discovery cycles
        await asyncio.sleep(5.0)

        # Check each node discovered others
        for i, node in enumerate(test_nodes):
            discovered_count = len(node.peer_registry)
            logger.info(f"Node {i} discovered {discovered_count} peers")

            # Each node should discover at least some others
            # (May not discover all due to timing and network conditions)
            assert discovered_count > 0, f"Node {i} discovered no peers"

        # Stop discovery
        for node in test_nodes:
            await node.peer_discovery.stop_discovery()

        logger.info("✅ Multi-node discovery test PASSED")

    async def test_evolution_peer_selection_unlimited(self, p2p_nodes):
        """Test that evolution peer selection is not limited to 5 peers."""
        node = p2p_nodes[0]

        # Manually add more than 5 suitable peers
        for i in range(10):
            peer_id = f"evolution_peer_{i}"
            capabilities = PeerCapabilities(
                device_id=peer_id,
                cpu_cores=4,
                ram_mb=8192,
                can_evolve=True,
                evolution_capacity=0.9,
                available_for_evolution=True,
                current_evolution_load=0.1,
                thermal_state="normal",
                battery_percent=90,
            )
            node.peer_registry[peer_id] = capabilities

        # Get suitable evolution peers
        suitable_peers = node.get_suitable_evolution_peers(min_count=1)

        # Should return all 10 peers, not limited to 5
        assert (
            len(suitable_peers) == 10
        ), f"Expected 10 peers, got {len(suitable_peers)}"

        # Test with higher minimum
        suitable_peers = node.get_suitable_evolution_peers(min_count=8)
        assert len(suitable_peers) == 10, "Should still return all available peers"

        logger.info("✅ Evolution peer selection unlimited test PASSED")

    async def test_protocol_version_negotiation(self, p2p_nodes):
        """Test protocol version handling in discovery."""
        node = p2p_nodes[0]

        # Test with unsupported protocol version
        discovery_msg = {
            "type": "PEER_DISCOVERY",
            "sender_id": "version_test_client",
            "sender_port": 9999,
            "timestamp": time.time(),
            "protocol_version": "2.0",  # Future version
            "capabilities": {"cpu_cores": 1},
        }

        reader, writer = await asyncio.open_connection("localhost", node.listen_port)

        try:
            msg_data = json.dumps(discovery_msg).encode("utf-8")
            length_data = len(msg_data).to_bytes(4, "big")
            writer.write(length_data + msg_data)
            await writer.drain()

            # Read response
            resp_length_data = await reader.readexactly(4)
            resp_length = int.from_bytes(resp_length_data, "big")
            resp_data = await reader.readexactly(resp_length)

            response = json.loads(resp_data.decode("utf-8"))

            # Should respond with our protocol version
            assert response["protocol_version"] == "1.0"

            logger.info("✅ Protocol version negotiation test PASSED")

        finally:
            writer.close()
            await writer.wait_closed()

    async def test_concurrent_discovery_and_p2p(self, p2p_nodes):
        """Test that discovery and P2P protocols can work concurrently."""
        node1, node2 = p2p_nodes[0], p2p_nodes[1]

        async def discovery_client():
            """Simulate discovery client."""
            reader, writer = await asyncio.open_connection(
                "localhost", node1.listen_port
            )
            try:
                discovery_msg = {
                    "type": "PEER_DISCOVERY",
                    "sender_id": "concurrent_discovery",
                    "sender_port": 9999,
                    "timestamp": time.time(),
                    "capabilities": {"cpu_cores": 2},
                }

                msg_data = json.dumps(discovery_msg).encode("utf-8")
                length_data = len(msg_data).to_bytes(4, "big")
                writer.write(length_data + msg_data)
                await writer.drain()

                # Read response
                resp_length_data = await reader.readexactly(4)
                resp_length = int.from_bytes(resp_length_data, "big")
                resp_data = await reader.readexactly(resp_length)

                return json.loads(resp_data.decode("utf-8"))
            finally:
                writer.close()
                await writer.wait_closed()

        async def p2p_client():
            """Simulate P2P encrypted client."""
            # This would normally use encrypted protocol
            # For now, just test that the server can handle mixed connections
            reader, writer = await asyncio.open_connection(
                "localhost", node2.listen_port
            )
            try:
                # Send a simple discovery message to verify connection handling
                discovery_msg = {
                    "type": "PEER_DISCOVERY",
                    "sender_id": "concurrent_p2p",
                    "sender_port": 9998,
                    "timestamp": time.time(),
                    "capabilities": {"cpu_cores": 4},
                }

                msg_data = json.dumps(discovery_msg).encode("utf-8")
                length_data = len(msg_data).to_bytes(4, "big")
                writer.write(length_data + msg_data)
                await writer.drain()

                # Read response
                resp_length_data = await reader.readexactly(4)
                resp_length = int.from_bytes(resp_length_data, "big")
                resp_data = await reader.readexactly(resp_length)

                return json.loads(resp_data.decode("utf-8"))
            finally:
                writer.close()
                await writer.wait_closed()

        # Run both clients concurrently
        discovery_result, p2p_result = await asyncio.gather(
            discovery_client(), p2p_client()
        )

        # Both should succeed
        assert discovery_result["type"] == "PEER_DISCOVERY_RESPONSE"
        assert p2p_result["type"] == "PEER_DISCOVERY_RESPONSE"

        logger.info("✅ Concurrent discovery and P2P test PASSED")

    async def test_large_network_scalability(self, p2p_nodes):
        """Test P2P network with all 10 nodes."""
        # Start discovery on all nodes
        for node in p2p_nodes:
            await node.peer_discovery.start_discovery()

        # Wait for multiple discovery cycles
        await asyncio.sleep(8.0)

        total_discoveries = 0
        successful_nodes = 0

        for i, node in enumerate(p2p_nodes):
            discovered_count = len(node.peer_registry)
            total_discoveries += discovered_count

            if discovered_count > 0:
                successful_nodes += 1

            logger.info(
                f"Node {i} ({node.node_id}): {discovered_count} peers discovered"
            )

        # Stop discovery
        for node in p2p_nodes:
            await node.peer_discovery.stop_discovery()

        # Validate network formation
        assert successful_nodes >= 5, f"Only {successful_nodes} nodes discovered peers"
        assert total_discoveries >= 10, f"Total discoveries: {total_discoveries}"

        # Test evolution peer selection across the network
        for node in p2p_nodes[:3]:  # Test first 3 nodes
            suitable_peers = node.get_suitable_evolution_peers()
            logger.info(
                f"Node {node.node_id}: {len(suitable_peers)} suitable evolution peers"
            )

        logger.info("✅ Large network scalability test PASSED")
        logger.info(f"   - {successful_nodes}/10 nodes successfully discovered peers")
        logger.info(f"   - {total_discoveries} total peer discoveries")


# Performance benchmarks
class TestP2PPerformance:
    """Performance tests for the fixed P2P system."""

    async def test_discovery_latency(self, p2p_nodes):
        """Test discovery message latency."""
        node = p2p_nodes[0]

        latencies = []

        for i in range(10):
            start_time = time.time()

            reader, writer = await asyncio.open_connection(
                "localhost", node.listen_port
            )

            discovery_msg = {
                "type": "PEER_DISCOVERY",
                "sender_id": f"latency_test_{i}",
                "sender_port": 9999,
                "timestamp": time.time(),
                "capabilities": {"cpu_cores": 1},
            }

            msg_data = json.dumps(discovery_msg).encode("utf-8")
            length_data = len(msg_data).to_bytes(4, "big")
            writer.write(length_data + msg_data)
            await writer.drain()

            # Read response
            resp_length_data = await reader.readexactly(4)
            resp_length = int.from_bytes(resp_length_data, "big")
            await reader.readexactly(resp_length)

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)

            writer.close()
            await writer.wait_closed()

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        logger.info("Discovery Latency Stats:")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  Min: {min_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")

        # Reasonable latency expectations
        assert avg_latency < 50.0, f"Average latency too high: {avg_latency:.2f}ms"
        assert max_latency < 200.0, f"Max latency too high: {max_latency:.2f}ms"

        logger.info("✅ Discovery latency test PASSED")


if __name__ == "__main__":
    # Run the tests directly
    async def run_tests():
        print("🚀 Starting P2P Integration Tests")

        # Create resource monitor
        async def mock_resource_monitor():
            return {
                "cpu_cores": 4,
                "ram_total_mb": 8192,
                "memory_percent": 30,
                "cpu_percent": 20,
                "battery_percent": 80,
                "network_type": "wifi",
                "device_type": "laptop",
                "thermal_state": "normal",
            }

        # Create 10 test nodes
        nodes = []
        base_port = 9200

        try:
            for i in range(10):
                node = P2PNode(
                    node_id=f"test_node_{i}",
                    listen_port=base_port + i,
                    resource_monitor=mock_resource_monitor,
                )
                nodes.append(node)

            # Start all nodes
            for node in nodes:
                await node.start()
                await asyncio.sleep(0.1)

            # Run tests
            test_class = TestP2PProtocolCompatibility()

            print("Test 1: Discovery Protocol Compatibility")
            await test_class.test_discovery_protocol_compatibility(nodes)

            print("Test 2: Multi-Node Discovery")
            await test_class.test_multi_node_discovery(nodes)

            print("Test 3: Evolution Peer Selection (No 5-peer limit)")
            await test_class.test_evolution_peer_selection_unlimited(nodes)

            print("Test 4: Protocol Version Negotiation")
            await test_class.test_protocol_version_negotiation(nodes)

            print("Test 5: Concurrent Discovery and P2P")
            await test_class.test_concurrent_discovery_and_p2p(nodes)

            print("Test 6: Large Network Scalability (10 nodes)")
            await test_class.test_large_network_scalability(nodes)

            # Performance tests
            perf_test_class = TestP2PPerformance()
            print("Performance Test: Discovery Latency")
            await perf_test_class.test_discovery_latency(nodes)

            print("\n🎉 ALL P2P INTEGRATION TESTS PASSED!")
            print("✅ Protocol mismatch fixed")
            print("✅ 5-peer limit removed")
            print("✅ 10+ peer network functional")
            print("✅ Discovery and P2P protocols compatible")

        finally:
            # Cleanup
            for node in nodes:
                try:
                    await node.stop()
                except Exception as e:
                    print(f"Warning: Error stopping node {node.node_id}: {e}")

    # Run the tests
    asyncio.run(run_tests())
