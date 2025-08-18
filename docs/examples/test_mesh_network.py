#!/usr/bin/env python3
"""LibP2P Mesh Network Test Script.

Tests the LibP2P mesh network implementation with multiple nodes.
Verifies peer discovery, message routing, and network resilience.

Usage:
    python examples/test_mesh_network.py --nodes 10
    python examples/test_mesh_network.py --nodes 5 --test-routing
    python examples/test_mesh_network.py --nodes 3 --test-dht
    python examples/test_mesh_network.py --android-bridge
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.android.jni.libp2p_mesh_bridge import LibP2PMeshBridge
from src.core.p2p.libp2p_mesh import LibP2PMeshNetwork, MeshConfiguration, MeshMessage, MeshMessageType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MeshNetworkTester:
    """Test harness for LibP2P mesh network."""

    def __init__(self, num_nodes: int = 5) -> None:
        self.num_nodes = num_nodes
        self.nodes: list[LibP2PMeshNetwork] = []
        self.test_results: dict[str, bool] = {}
        self.message_stats = {"sent": 0, "received": 0, "routed": 0, "failed": 0}

    async def setup_nodes(self) -> None:
        """Create and start test nodes."""
        logger.info(f"Setting up {self.num_nodes} mesh nodes")

        base_port = 4001
        for i in range(self.num_nodes):
            config = MeshConfiguration(
                node_id=f"test-node-{i:02d}",
                listen_port=base_port + i,
                discovery_interval=10.0,  # Faster discovery for testing
                heartbeat_interval=5.0,  # Faster heartbeat for testing
                max_peers=self.num_nodes - 1,
                mdns_enabled=True,
                dht_enabled=True,
            )

            node = LibP2PMeshNetwork(config)

            # Register message handlers
            await self.register_test_handlers(node, i)

            self.nodes.append(node)

        # Start all nodes
        for i, node in enumerate(self.nodes):
            try:
                await node.start()
                logger.info(f"Started node {i}: {node.node_id}")
                await asyncio.sleep(1)  # Stagger startup
            except Exception as e:
                logger.exception(f"Failed to start node {i}: {e}")

    async def register_test_handlers(self, node: LibP2PMeshNetwork, node_index: int) -> None:
        """Register message handlers for testing."""

        async def handle_test_message(message: MeshMessage) -> None:
            self.message_stats["received"] += 1
            logger.debug(f"Node {node_index} received {message.type.value} from {message.sender}")

            # Echo back for ping-pong tests
            if message.type == MeshMessageType.DATA_MESSAGE:
                try:
                    payload = json.loads(message.payload.decode())
                    if payload.get("test_type") == "ping":
                        # Send pong back
                        pong_message = MeshMessage(
                            type=MeshMessageType.DATA_MESSAGE,
                            sender=node.node_id,
                            recipient=message.sender,
                            payload=json.dumps(
                                {
                                    "test_type": "pong",
                                    "original_id": payload.get("message_id"),
                                    "hop_count": message.hop_count,
                                }
                            ).encode(),
                        )
                        await node.send_message(pong_message)

                except Exception as e:
                    logger.debug(f"Error handling ping message: {e}")

        async def handle_agent_task(message: MeshMessage) -> None:
            self.message_stats["received"] += 1
            logger.debug(f"Node {node_index} received agent task from {message.sender}")

            # Simulate task processing
            await asyncio.sleep(0.1)

            # Send task result back
            if message.sender != node.node_id:
                result_message = MeshMessage(
                    type=MeshMessageType.DATA_MESSAGE,
                    sender=node.node_id,
                    recipient=message.sender,
                    payload=json.dumps(
                        {
                            "task_result": "completed",
                            "processing_node": node.node_id,
                            "original_task": message.id,
                        }
                    ).encode(),
                )
                await node.send_message(result_message)

        async def handle_parameter_update(message: MeshMessage) -> None:
            self.message_stats["received"] += 1
            logger.debug(f"Node {node_index} received parameter update from {message.sender}")

            # Simulate parameter processing
            try:
                params = json.loads(message.payload.decode())
                logger.debug(f"Node {node_index} updated parameters: {list(params.keys())}")
            except Exception as e:
                logger.debug(f"Error processing parameters: {e}")

        async def handle_gradient_sharing(message: MeshMessage) -> None:
            self.message_stats["received"] += 1
            logger.debug(f"Node {node_index} received gradients from {message.sender}")

            # Simulate gradient aggregation
            await asyncio.sleep(0.05)

        # Register handlers
        node.register_message_handler(MeshMessageType.DATA_MESSAGE, handle_test_message)
        node.register_message_handler(MeshMessageType.AGENT_TASK, handle_agent_task)
        node.register_message_handler(MeshMessageType.PARAMETER_UPDATE, handle_parameter_update)
        node.register_message_handler(MeshMessageType.GRADIENT_SHARING, handle_gradient_sharing)

    async def wait_for_network_convergence(self, timeout: float = 60.0) -> bool:
        """Wait for network to converge (all nodes connected)."""
        logger.info("Waiting for network convergence...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_connected = True

            for node in self.nodes:
                if len(node.connected_peers) < min(3, self.num_nodes - 1):
                    all_connected = False
                    break

            if all_connected:
                logger.info(f"Network converged in {time.time() - start_time:.2f} seconds")
                return True

            await asyncio.sleep(2)

        logger.warning("Network failed to converge within timeout")
        return False

    async def test_basic_connectivity(self) -> bool:
        """Test basic peer connectivity."""
        logger.info("Testing basic connectivity...")

        try:
            # Check if all nodes have peers
            for i, node in enumerate(self.nodes):
                peer_count = len(node.connected_peers)
                logger.info(f"Node {i} has {peer_count} peers: {list(node.connected_peers.keys())}")

                if peer_count == 0:
                    logger.error(f"Node {i} has no peers")
                    return False

            logger.info("✅ Basic connectivity test passed")
            return True

        except Exception as e:
            logger.exception(f"❌ Basic connectivity test failed: {e}")
            return False

    async def test_message_routing(self) -> bool:
        """Test message routing between nodes."""
        logger.info("Testing message routing...")

        try:
            # Send messages from first node to all others
            sender_node = self.nodes[0]
            success_count = 0

            for i in range(1, len(self.nodes)):
                target_node = self.nodes[i]

                message = MeshMessage(
                    type=MeshMessageType.DATA_MESSAGE,
                    sender=sender_node.node_id,
                    recipient=target_node.node_id,
                    payload=json.dumps(
                        {
                            "test_type": "ping",
                            "message_id": f"test-{i}",
                            "timestamp": time.time(),
                        }
                    ).encode(),
                )

                success = await sender_node.send_message(message)
                if success:
                    success_count += 1
                    self.message_stats["sent"] += 1

            # Wait for responses
            await asyncio.sleep(5)

            success_rate = success_count / (len(self.nodes) - 1) if len(self.nodes) > 1 else 0
            logger.info(f"Message routing success rate: {success_rate:.2f} ({success_count}/{len(self.nodes) - 1})")

            if success_rate >= 0.8:  # 80% success rate
                logger.info("✅ Message routing test passed")
                return True
            logger.error("❌ Message routing test failed")
            return False

        except Exception as e:
            logger.exception(f"❌ Message routing test failed: {e}")
            return False

    async def test_broadcast_messaging(self) -> bool:
        """Test broadcast messaging."""
        logger.info("Testing broadcast messaging...")

        try:
            sender_node = self.nodes[0]

            # Send broadcast message
            broadcast_message = MeshMessage(
                type=MeshMessageType.PARAMETER_UPDATE,
                sender=sender_node.node_id,
                recipient=None,  # Broadcast
                payload=json.dumps(
                    {
                        "parameters": {"learning_rate": 0.01, "batch_size": 32},
                        "epoch": 1,
                        "timestamp": time.time(),
                    }
                ).encode(),
            )

            success = await sender_node.send_message(broadcast_message)
            self.message_stats["sent"] += 1

            # Wait for propagation
            await asyncio.sleep(3)

            if success:
                logger.info("✅ Broadcast messaging test passed")
                return True
            logger.error("❌ Broadcast messaging test failed")
            return False

        except Exception as e:
            logger.exception(f"❌ Broadcast messaging test failed: {e}")
            return False

    async def test_dht_functionality(self) -> bool:
        """Test DHT store and retrieve."""
        logger.info("Testing DHT functionality...")

        try:
            if not any(node.dht for node in self.nodes):
                logger.warning("No DHT available, skipping DHT test")
                return True

            # Find a node with DHT
            dht_node = None
            for node in self.nodes:
                if node.dht:
                    dht_node = node
                    break

            if not dht_node:
                logger.warning("No DHT node found")
                return True

            # Test store
            test_key = "test-key"
            test_value = json.dumps({"test_data": "hello world", "timestamp": time.time()}).encode()

            store_success = await dht_node.dht_store(test_key, test_value)
            if not store_success:
                logger.error("❌ DHT store failed")
                return False

            # Wait for propagation
            await asyncio.sleep(2)

            # Test retrieve from different node
            retrieve_node = None
            for node in self.nodes:
                if node != dht_node and node.dht:
                    retrieve_node = node
                    break

            if retrieve_node:
                retrieved_value = await retrieve_node.dht_get(test_key)
                if retrieved_value and retrieved_value == test_value:
                    logger.info("✅ DHT functionality test passed")
                    return True
                logger.error(f"❌ DHT retrieve failed: expected {test_value}, got {retrieved_value}")
                return False
            logger.warning("Only one DHT node available, cannot test retrieve")
            return True

        except Exception as e:
            logger.exception(f"❌ DHT functionality test failed: {e}")
            return False

    async def test_network_resilience(self) -> bool:
        """Test network resilience by removing nodes."""
        logger.info("Testing network resilience...")

        try:
            if len(self.nodes) < 3:
                logger.warning("Need at least 3 nodes for resilience test")
                return True

            # Remove a random node
            node_to_remove = random.choice(self.nodes[1:])  # Don't remove the first node
            logger.info(f"Removing node: {node_to_remove.node_id}")

            await node_to_remove.stop()
            self.nodes.remove(node_to_remove)

            # Wait for network to adapt
            await asyncio.sleep(10)

            # Test if remaining nodes can still communicate
            return await self.test_message_routing()

        except Exception as e:
            logger.exception(f"❌ Network resilience test failed: {e}")
            return False

    async def test_high_load_messaging(self) -> bool:
        """Test high-load messaging scenario."""
        logger.info("Testing high-load messaging...")

        try:
            # Send many messages concurrently
            tasks = []
            message_count = 50

            for i in range(message_count):
                sender = random.choice(self.nodes)
                recipient = random.choice([n for n in self.nodes if n != sender])

                message = MeshMessage(
                    type=MeshMessageType.AGENT_TASK,
                    sender=sender.node_id,
                    recipient=recipient.node_id,
                    payload=json.dumps(
                        {
                            "task_id": i,
                            "task_type": "computation",
                            "data": f"task_data_{i}",
                        }
                    ).encode(),
                )

                tasks.append(sender.send_message(message))

            # Execute all sends concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for result in results if result is True)
            success_rate = success_count / message_count

            self.message_stats["sent"] += message_count

            logger.info(f"High-load messaging success rate: {success_rate:.2f} ({success_count}/{message_count})")

            # Wait for message processing
            await asyncio.sleep(10)

            if success_rate >= 0.9:  # 90% success rate for high load
                logger.info("✅ High-load messaging test passed")
                return True
            logger.error("❌ High-load messaging test failed")
            return False

        except Exception as e:
            logger.exception(f"❌ High-load messaging test failed: {e}")
            return False

    async def run_all_tests(self) -> dict[str, bool]:
        """Run all tests and return results."""
        logger.info("Starting comprehensive mesh network tests")

        # Setup
        await self.setup_nodes()

        # Wait for network to stabilize
        if not await self.wait_for_network_convergence():
            logger.error("Network failed to converge, aborting tests")
            return {"setup": False}

        # Run tests
        tests = [
            ("basic_connectivity", self.test_basic_connectivity),
            ("message_routing", self.test_message_routing),
            ("broadcast_messaging", self.test_broadcast_messaging),
            ("dht_functionality", self.test_dht_functionality),
            ("high_load_messaging", self.test_high_load_messaging),
            ("network_resilience", self.test_network_resilience),
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"\n--- Running {test_name} test ---")
                result = await test_func()
                self.test_results[test_name] = result
                await asyncio.sleep(2)  # Brief pause between tests
            except Exception as e:
                logger.exception(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = False

        return self.test_results

    async def cleanup(self) -> None:
        """Clean up test resources."""
        logger.info("Cleaning up test nodes...")

        for node in self.nodes:
            try:
                await node.stop()
            except Exception as e:
                logger.debug(f"Error stopping node: {e}")

        self.nodes.clear()

    def print_results(self) -> None:
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("MESH NETWORK TEST RESULTS")
        print("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<25} {status}")

        print("-" * 60)
        print(f"Overall: {passed}/{total} tests passed")
        print(f"Success Rate: {(passed / total * 100):.1f}%" if total > 0 else "No tests run")

        print("\nMessage Statistics:")
        print(f"  Sent: {self.message_stats['sent']}")
        print(f"  Received: {self.message_stats['received']}")
        print(f"  Routed: {self.message_stats['routed']}")
        print(f"  Failed: {self.message_stats['failed']}")

        print("\nNetwork Status:")
        for i, node in enumerate(self.nodes):
            if hasattr(node, "get_mesh_status"):
                status = node.get_mesh_status()
                peer_count = status.get("connected_peers", 0)
                print(f"  Node {i}: {peer_count} peers, status: {status.get('status', 'unknown')}")

        print("=" * 60)


async def test_android_bridge() -> None:
    """Test the Android bridge functionality."""
    logger.info("Testing Android bridge...")

    try:
        # Create and start bridge
        bridge = LibP2PMeshBridge(port=8080)
        await bridge.start_bridge()

        logger.info("Bridge started, testing basic functionality...")

        # Let it run for a moment to test
        await asyncio.sleep(5)

        # Check bridge info
        info = bridge.get_bridge_info()
        logger.info(f"Bridge info: {info}")

        logger.info("✅ Android bridge test completed")

        # Cleanup
        bridge.stop_bridge()

    except Exception as e:
        logger.exception(f"❌ Android bridge test failed: {e}")


def main() -> None:
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test LibP2P Mesh Network")
    parser.add_argument("--nodes", type=int, default=5, help="Number of test nodes")
    parser.add_argument("--test-routing", action="store_true", help="Focus on routing tests")
    parser.add_argument("--test-dht", action="store_true", help="Focus on DHT tests")
    parser.add_argument("--android-bridge", action="store_true", help="Test Android bridge")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    async def run_tests() -> None:
        if args.android_bridge:
            await test_android_bridge()
            return

        tester = MeshNetworkTester(args.nodes)

        try:
            if args.test_routing:
                # Focus on routing tests
                await tester.setup_nodes()
                await tester.wait_for_network_convergence()
                tester.test_results["message_routing"] = await tester.test_message_routing()
                tester.test_results["network_resilience"] = await tester.test_network_resilience()
            elif args.test_dht:
                # Focus on DHT tests
                await tester.setup_nodes()
                await tester.wait_for_network_convergence()
                tester.test_results["dht_functionality"] = await tester.test_dht_functionality()
            else:
                # Run all tests
                await tester.run_all_tests()

            tester.print_results()

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.exception(f"Test failed: {e}")
        finally:
            await tester.cleanup()

    # Run the tests
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.exception(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
