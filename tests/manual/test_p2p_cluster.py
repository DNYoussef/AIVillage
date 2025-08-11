#!/usr/bin/env python3
"""Test P2P networking with multiple nodes to verify "5+ nodes" claim."""

import asyncio
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import both P2P implementations
from src.core.p2p import P2PNode as CoreP2PNode
from src.production.communications.p2p import P2PNode as ProductionP2PNode


async def test_core_p2p_cluster(num_nodes: int = 5):
    """Test the core P2P implementation with multiple nodes."""
    logger.info(f"=== Testing Core P2P with {num_nodes} nodes ===")

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

    nodes = []

    try:
        # Create and start nodes
        for i in range(num_nodes):
            node = CoreP2PNode(node_id=f"core-node-{i}", listen_port=0)
            await node.start(resource_monitor=mock_resource_monitor)
            nodes.append(node)
            logger.info(f"Started core node {i} on port {node.listen_port}")

        # Connect nodes in a mesh (each node knows about all others)
        for i, node in enumerate(nodes):
            for j, other_node in enumerate(nodes):
                if i != j:
                    node.peer_discovery.add_known_peer("127.0.0.1", other_node.listen_port)

        logger.info("Waiting for peer discovery and connections...")
        await asyncio.sleep(10)  # Give more time for discovery

        # Check connections
        total_connections = 0
        for i, node in enumerate(nodes):
            peer_count = len(node.peer_registry)
            connection_count = len(node.connections)
            total_connections += connection_count

            logger.info(f"Core Node {i}: {peer_count} peers discovered, {connection_count} active connections")
            logger.info(f"  Peers: {list(node.peer_registry.keys())}")
            logger.info(f"  Connections: {list(node.connections.keys())}")

        logger.info(f"Total connections across all nodes: {total_connections}")

        # Test message passing if nodes are connected
        if total_connections > 0:
            logger.info("Testing message broadcast...")
            test_message = {
                "type": "TEST_BROADCAST",
                "data": {"test": "Hello from cluster test!", "timestamp": time.time()},
            }

            success_count = await nodes[0].broadcast_to_peers("TEST_BROADCAST", test_message["data"])
            logger.info(f"Broadcast reached {success_count} peers")

        # Test evolution messaging
        logger.info("Testing evolution coordination...")
        await nodes[0].broadcast_evolution_event("TEST_START", {"evolution_type": "cluster_test", "nodes": num_nodes})

        await asyncio.sleep(2)

        # Get final stats
        for i, node in enumerate(nodes):
            stats = node.get_network_status()
            logger.info(
                f"Core Node {i} final stats: peers={stats['known_peers']}, connections={stats['connected_peers']}, evolution_messages={stats['network_stats']['evolution_messages']}"
            )

    except Exception as e:
        logger.exception(f"Core P2P cluster test failed: {e}")
        raise
    finally:
        # Clean up
        for i, node in enumerate(nodes):
            await node.stop()
            logger.info(f"Stopped core node {i}")


async def test_production_p2p_cluster(num_nodes: int = 5):
    """Test the production P2P implementation with multiple nodes."""
    logger.info(f"=== Testing Production P2P with {num_nodes} nodes ===")

    nodes = []

    try:
        # Create and start nodes
        for i in range(num_nodes):
            node = ProductionP2PNode(node_id=f"prod-node-{i}", port=8100 + i)
            nodes.append(node)
            logger.info(f"Created production node {i}")

        # Try to start nodes (may fail due to missing dependencies)
        started_nodes = []
        for i, node in enumerate(nodes):
            try:
                await node.start()
                started_nodes.append(node)
                logger.info(f"Started production node {i} on port {node.port}")
            except Exception as e:
                logger.warning(f"Failed to start production node {i}: {e}")

        if not started_nodes:
            logger.warning("No production nodes started successfully")
            return

        # Connect nodes
        for node in started_nodes:
            for other_node in started_nodes:
                if node != other_node:
                    node.add_known_address("127.0.0.1", other_node.port)

        logger.info("Waiting for connections...")
        await asyncio.sleep(5)

        # Check connections
        for i, node in enumerate(started_nodes):
            peer_count = len(node.peers)
            logger.info(f"Production Node {i}: {peer_count} peers")
            logger.info(f"  Status: {node.status}")

            stats = node.get_stats()
            logger.info(f"  Stats: {json.dumps(stats, indent=2)}")

    except Exception as e:
        logger.exception(f"Production P2P cluster test failed: {e}")
    finally:
        # Clean up
        for i, node in enumerate(started_nodes if "started_nodes" in locals() else []):
            try:
                await node.stop()
                logger.info(f"Stopped production node {i}")
            except Exception as e:
                logger.warning(f"Error stopping production node {i}: {e}")


async def test_mixed_network_simulation():
    """Test network simulation with various configurations."""
    logger.info("=== Testing Mixed Network Simulation ===")

    # Test different network sizes
    for size in [2, 3, 5, 7]:
        logger.info(f"\n--- Testing {size} nodes ---")

        try:
            await test_core_p2p_cluster(size)
            await asyncio.sleep(2)  # Cool down between tests
        except Exception as e:
            logger.exception(f"Failed with {size} nodes: {e}")

    logger.info("Mixed network simulation complete")


async def benchmark_p2p_performance():
    """Benchmark P2P performance with multiple nodes."""
    logger.info("=== P2P Performance Benchmark ===")

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

    nodes = []
    num_nodes = 5

    try:
        start_time = time.time()

        # Create nodes
        for i in range(num_nodes):
            node = CoreP2PNode(node_id=f"bench-node-{i}", listen_port=0)
            nodes.append(node)

        # Start all nodes
        await asyncio.gather(*[node.start(resource_monitor=mock_resource_monitor) for node in nodes])

        startup_time = time.time() - start_time
        logger.info(f"Started {num_nodes} nodes in {startup_time:.2f} seconds")

        # Set up mesh connectivity
        for node in nodes:
            for other_node in nodes:
                if node != other_node:
                    node.peer_discovery.add_known_peer("127.0.0.1", other_node.listen_port)

        # Wait for discovery
        discovery_start = time.time()
        await asyncio.sleep(8)
        discovery_time = time.time() - discovery_start

        # Measure final connectivity
        total_peers = sum(len(node.peer_registry) for node in nodes)
        total_connections = sum(len(node.connections) for node in nodes)

        logger.info(f"Discovery completed in {discovery_time:.2f} seconds")
        logger.info(f"Total peers discovered: {total_peers}")
        logger.info(f"Total active connections: {total_connections}")
        logger.info(f"Average peers per node: {total_peers / num_nodes:.1f}")
        logger.info(f"Average connections per node: {total_connections / num_nodes:.1f}")

        # Message throughput test
        if total_connections > 0:
            message_start = time.time()
            message_count = 0

            for i in range(10):  # Send 10 broadcasts
                success = await nodes[0].broadcast_to_peers(f"BENCHMARK_{i}", {"test": i, "timestamp": time.time()})
                message_count += success

            message_time = time.time() - message_start
            logger.info(f"Sent {message_count} messages in {message_time:.2f} seconds")
            logger.info(f"Message rate: {message_count / message_time:.1f} messages/second")

    except Exception as e:
        logger.exception(f"Performance benchmark failed: {e}")
    finally:
        # Shutdown all nodes
        await asyncio.gather(*[node.stop() for node in nodes], return_exceptions=True)


async def main():
    """Run all P2P cluster tests."""
    logger.info("Starting P2P Cluster Testing...")

    try:
        # Test 1: Core P2P with 5 nodes (claimed functionality)
        await test_core_p2p_cluster(5)
        await asyncio.sleep(2)

        # Test 2: Production P2P (if available)
        try:
            await test_production_p2p_cluster(3)
            await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"Production P2P test skipped: {e}")

        # Test 3: Mixed network sizes
        await test_mixed_network_simulation()
        await asyncio.sleep(2)

        # Test 4: Performance benchmark
        await benchmark_p2p_performance()

        logger.info("All P2P cluster tests completed!")

    except Exception as e:
        logger.exception(f"P2P cluster testing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
