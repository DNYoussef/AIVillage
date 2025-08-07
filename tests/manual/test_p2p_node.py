#!/usr/bin/env python3
"""Test P2P node functionality by starting actual nodes."""

import asyncio
import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the P2P components
from src.core.p2p import P2PNode


async def test_single_node():
    """Test starting a single P2P node."""
    logger.info("=== Testing Single P2P Node ===")

    # Mock resource monitor
    async def mock_resource_monitor():
        return {
            "cpu_cores": 4,
            "ram_total_mb": 8192,
            "ram_available_mb": 4096,
            "battery_percent": 75,
            "network_type": "wifi",
            "device_type": "desktop",
            "thermal_state": "normal",
            "memory_percent": 50,
            "cpu_percent": 20,
        }

    # Create and start node
    node = P2PNode(node_id="test-node-1", listen_port=0)

    try:
        await node.start(resource_monitor=mock_resource_monitor)

        logger.info(f"Node started successfully on port {node.listen_port}")

        # Check status
        status = node.get_network_status()
        logger.info(f"Network status: {json.dumps(status, indent=2)}")

        # Wait a bit to see if anything happens
        await asyncio.sleep(2)

        # Check stats
        encryption_stats = node.encryption.get_encryption_stats()
        logger.info(f"Encryption stats: {json.dumps(encryption_stats, indent=2)}")

        discovery_stats = node.peer_discovery.get_discovery_stats()
        logger.info(f"Discovery stats: {json.dumps(discovery_stats, indent=2)}")

    except Exception as e:
        logger.exception(f"Failed to start node: {e}")
        raise
    finally:
        await node.stop()
        logger.info("Node stopped")


async def test_two_nodes_discovery():
    """Test two nodes discovering each other."""
    logger.info("=== Testing Two Nodes Discovery ===")

    async def mock_resource_monitor():
        return {
            "cpu_cores": 4,
            "ram_total_mb": 8192,
            "ram_available_mb": 4096,
            "battery_percent": 75,
            "network_type": "wifi",
            "device_type": "desktop",
            "thermal_state": "normal",
            "memory_percent": 50,
            "cpu_percent": 20,
        }

    # Create two nodes
    node1 = P2PNode(node_id="node-1", listen_port=0)
    node2 = P2PNode(node_id="node-2", listen_port=0)

    try:
        # Start both nodes
        await node1.start(resource_monitor=mock_resource_monitor)
        await node2.start(resource_monitor=mock_resource_monitor)

        logger.info(f"Node 1 listening on port {node1.listen_port}")
        logger.info(f"Node 2 listening on port {node2.listen_port}")

        # Add each other as known peers
        node1.peer_discovery.add_known_peer("127.0.0.1", node2.listen_port)
        node2.peer_discovery.add_known_peer("127.0.0.1", node1.listen_port)

        # Wait for discovery
        logger.info("Waiting for peer discovery...")
        await asyncio.sleep(5)

        # Check if they found each other
        logger.info(f"Node 1 peers: {list(node1.peer_registry.keys())}")
        logger.info(f"Node 2 peers: {list(node2.peer_registry.keys())}")

        # Check connection status
        logger.info(f"Node 1 connections: {list(node1.connections.keys())}")
        logger.info(f"Node 2 connections: {list(node2.connections.keys())}")

        # Try sending a message from node 1 to node 2
        if node2.node_id in node1.connections:
            logger.info("Attempting to send message...")
            success = await node1.send_to_peer(
                node2.node_id,
                {
                    "type": "TEST_MESSAGE",
                    "data": {"message": "Hello from node 1!", "timestamp": time.time()},
                },
            )
            logger.info(f"Message send result: {success}")
        else:
            logger.warning("Nodes are not connected")

        # Wait a bit more to see messages
        await asyncio.sleep(2)

    except Exception as e:
        logger.exception(f"Two-node test failed: {e}")
        raise
    finally:
        await node1.stop()
        await node2.stop()
        logger.info("Both nodes stopped")


async def test_evolution_messaging():
    """Test evolution-specific messaging."""
    logger.info("=== Testing Evolution Messaging ===")

    async def mock_resource_monitor():
        return {
            "cpu_cores": 8,
            "ram_total_mb": 16384,
            "ram_available_mb": 12288,
            "battery_percent": 80,
            "network_type": "ethernet",
            "device_type": "desktop",
            "thermal_state": "normal",
            "memory_percent": 25,
            "cpu_percent": 15,
        }

    node = P2PNode(node_id="evolution-test-node", listen_port=0)

    try:
        await node.start(resource_monitor=mock_resource_monitor)

        logger.info(f"Evolution test node started on port {node.listen_port}")

        # Check evolution capabilities
        if node.local_capabilities:
            caps = node.local_capabilities
            logger.info(f"Evolution capacity: {caps.evolution_capacity}")
            logger.info(f"Suitable for evolution: {caps.is_suitable_for_evolution()}")
            logger.info(f"Evolution priority: {caps.get_evolution_priority()}")

        # Simulate evolution events
        await node.broadcast_evolution_event(
            "START", {"evolution_type": "test_evolution", "parameters": {"test": True}}
        )

        await asyncio.sleep(1)

        await node.broadcast_evolution_event("PROGRESS", {"progress": 0.5})

        await asyncio.sleep(1)

        await node.broadcast_evolution_event(
            "COMPLETE", {"success": True, "results": {"test_result": "passed"}}
        )

        # Check stats
        stats = node.get_network_status()
        logger.info(f"Final network stats: {json.dumps(stats, indent=2)}")

    except Exception as e:
        logger.exception(f"Evolution messaging test failed: {e}")
        raise
    finally:
        await node.stop()
        logger.info("Evolution test node stopped")


async def main():
    """Run all P2P tests."""
    logger.info("Starting P2P Node Testing...")

    try:
        # Test 1: Single node
        await test_single_node()
        await asyncio.sleep(1)

        # Test 2: Two nodes discovery
        await test_two_nodes_discovery()
        await asyncio.sleep(1)

        # Test 3: Evolution messaging
        await test_evolution_messaging()

        logger.info("All P2P tests completed successfully!")

    except Exception as e:
        logger.exception(f"P2P testing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
