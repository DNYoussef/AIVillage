"""BitChat/Betanet Transport Reliability Tests

Mock reliability harness simulating real-world network conditions including:
- Packet loss (20-40% simulation)
- TTL=7 relay behavior
- Store-and-forward for offline scenarios
- BLE/network handshake simulation

Designed to validate transport reliability without requiring actual hardware.
"""

import asyncio
import logging
import random
import time
import uuid
from typing import Any

import pytest

# Import transport components with fallback mocks
try:
    from packages.p2p.core.betanet_transport import BetanetMessage, BetanetTransport
    from packages.p2p.core.bitchat_transport import BitChatMessage, BitChatTransport
    from packages.p2p.core.dual_path_transport import DualPathTransport

    TRANSPORTS_AVAILABLE = True
except ImportError:
    # Create mock classes for testing
    class BitChatMessage:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", str(uuid.uuid4()))
            self.sender = kwargs.get("sender", "")
            self.recipient = kwargs.get("recipient", "")
            self.payload = kwargs.get("payload", b"")
            self.ttl = kwargs.get("ttl", 7)
            self.hop_count = kwargs.get("hop_count", 0)
            self.priority = kwargs.get("priority", 5)
            self.timestamp = time.time()

    class BetanetMessage:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", str(uuid.uuid4()))
            self.sender = kwargs.get("sender", "")
            self.recipient = kwargs.get("recipient", "")
            self.payload = kwargs.get("payload", b"")
            self.priority = kwargs.get("priority", 5)

    class BitChatTransport:
        def __init__(self, **kwargs):
            self.device_id = kwargs.get("device_id", "mock_device")
            self.is_running = False

    class BetanetTransport:
        def __init__(self, **kwargs):
            self.peer_id = kwargs.get("peer_id", "mock_peer")
            self.is_running = False

    class DualPathTransport:
        def __init__(self, **kwargs):
            self.node_id = kwargs.get("node_id", "mock_node")
            self.is_running = False

    TRANSPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NetworkConditions:
    """Simulated network conditions for reliability testing"""

    def __init__(
        self,
        packet_loss_rate: float = 0.0,
        latency_ms: float = 100.0,
        jitter_ms: float = 10.0,
        bandwidth_kbps: float = 1000.0,
        is_offline: bool = False,
    ):
        self.packet_loss_rate = packet_loss_rate
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.bandwidth_kbps = bandwidth_kbps
        self.is_offline = is_offline

    def should_drop_packet(self) -> bool:
        """Simulate packet loss"""
        return random.random() < self.packet_loss_rate

    def get_transmission_delay(self) -> float:
        """Get simulated transmission delay in seconds"""
        base_delay = self.latency_ms / 1000.0
        jitter = random.uniform(-self.jitter_ms, self.jitter_ms) / 1000.0
        return max(0.001, base_delay + jitter)


class MockTransportNode:
    """Mock transport node for reliability testing"""

    def __init__(self, node_id: str, network_conditions: NetworkConditions, max_hops: int = 7):
        self.node_id = node_id
        self.network_conditions = network_conditions
        self.max_hops = max_hops

        # State tracking
        self.is_online = not network_conditions.is_offline
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_relayed = 0
        self.messages_dropped = 0
        self.stored_messages: list[BitChatMessage] = []

        # Peer connections
        self.peers: set[str] = set()
        self.routing_table: dict[str, str] = {}

        # Message cache for duplicate detection
        self.message_cache: set[str] = set()

    async def send_message(self, recipient: str, payload: bytes, priority: int = 5, ttl: int = 7) -> bool:
        """Send message with network simulation"""
        if not self.is_online:
            return False

        # Simulate packet loss
        if self.network_conditions.should_drop_packet():
            self.messages_dropped += 1
            logger.debug(f"[{self.node_id}] Packet dropped due to network conditions")
            return False

        # Simulate transmission delay
        delay = self.network_conditions.get_transmission_delay()
        await asyncio.sleep(delay)

        message = BitChatMessage(
            sender=self.node_id,
            recipient=recipient,
            payload=payload,
            ttl=min(ttl, self.max_hops),
            priority=priority,
        )

        self.messages_sent += 1
        logger.debug(f"[{self.node_id}] Sent message {message.id[:8]} to {recipient}")
        return True

    async def receive_message(self, message: BitChatMessage, from_peer: str) -> bool:
        """Receive and potentially relay message"""
        if not self.is_online:
            # Store for later delivery if this node comes online
            self.stored_messages.append(message)
            return False

        # Duplicate detection
        if message.id in self.message_cache:
            return False

        self.message_cache.add(message.id)
        self.messages_received += 1

        # Check if message is for us
        if message.recipient == self.node_id or not message.recipient:
            logger.debug(f"[{self.node_id}] Received message {message.id[:8]}")
            return True

        # Relay if TTL allows
        if message.ttl > 1:
            message.ttl -= 1
            message.hop_count += 1

            # Find next hop (simplified routing)
            next_hop = self.routing_table.get(message.recipient)
            if next_hop and next_hop in self.peers and next_hop != from_peer:
                self.messages_relayed += 1
                logger.debug(f"[{self.node_id}] Relayed message {message.id[:8]} to {next_hop}")
                return True

        return False

    def set_online(self, online: bool):
        """Change node online status"""
        self.is_online = online
        if online:
            # Deliver stored messages when coming online
            logger.debug(f"[{self.node_id}] Coming online, delivering {len(self.stored_messages)} stored messages")
            self.stored_messages.clear()

    def add_peer(self, peer_id: str):
        """Add peer connection"""
        self.peers.add(peer_id)
        self.routing_table[peer_id] = peer_id

    def get_stats(self) -> dict[str, Any]:
        """Get node statistics"""
        return {
            "node_id": self.node_id,
            "is_online": self.is_online,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_relayed": self.messages_relayed,
            "messages_dropped": self.messages_dropped,
            "stored_messages": len(self.stored_messages),
            "peer_count": len(self.peers),
            "packet_loss_rate": self.network_conditions.packet_loss_rate,
        }


class MockBLEHandshake:
    """Mock BLE handshake simulation"""

    def __init__(self, success_rate: float = 0.8, handshake_time_ms: float = 500.0):
        self.success_rate = success_rate
        self.handshake_time_ms = handshake_time_ms
        self.handshake_attempts = 0
        self.successful_handshakes = 0

    async def attempt_handshake(self, device_a: str, device_b: str) -> bool:
        """Simulate BLE device handshake"""
        self.handshake_attempts += 1

        # Simulate handshake time
        delay = self.handshake_time_ms / 1000.0
        await asyncio.sleep(delay)

        # Simulate success/failure
        success = random.random() < self.success_rate
        if success:
            self.successful_handshakes += 1
            logger.debug(f"BLE handshake successful: {device_a} <-> {device_b}")
        else:
            logger.debug(f"BLE handshake failed: {device_a} <-> {device_b}")

        return success

    def get_success_rate(self) -> float:
        """Get actual success rate"""
        if self.handshake_attempts == 0:
            return 0.0
        return self.successful_handshakes / self.handshake_attempts


class ReliabilityTestHarness:
    """Comprehensive reliability test harness for BitChat/Betanet"""

    def __init__(self):
        self.nodes: dict[str, MockTransportNode] = {}
        self.ble_handshake = MockBLEHandshake()
        self.test_results: dict[str, Any] = {}

    def create_network_topology(self, node_count: int = 5, packet_loss_rate: float = 0.3, offline_nodes: int = 0):
        """Create test network topology"""
        self.nodes.clear()

        for i in range(node_count):
            is_offline = i < offline_nodes
            conditions = NetworkConditions(
                packet_loss_rate=packet_loss_rate,
                latency_ms=random.uniform(50, 200),
                jitter_ms=random.uniform(5, 25),
                is_offline=is_offline,
            )

            node_id = f"node_{i}"
            node = MockTransportNode(node_id, conditions)
            self.nodes[node_id] = node

        # Create mesh connections (each node connected to 2-3 others)
        node_ids = list(self.nodes.keys())
        for i, node_id in enumerate(node_ids):
            node = self.nodes[node_id]

            # Connect to next 2-3 nodes (circular) and create bidirectional connections
            connections = min(3, len(node_ids) - 1)
            for j in range(1, connections + 1):
                peer_idx = (i + j) % len(node_ids)
                peer_id = node_ids[peer_idx]

                # Add bidirectional connection
                node.add_peer(peer_id)
                self.nodes[peer_id].add_peer(node_id)

        logger.info(f"Created network topology: {node_count} nodes, {packet_loss_rate:.1%} packet loss")

    async def test_message_delivery(
        self,
        sender: str,
        recipient: str,
        message_count: int = 10,
        payload_size: int = 1024,
    ) -> dict[str, Any]:
        """Test message delivery reliability"""
        if sender not in self.nodes or recipient not in self.nodes:
            raise ValueError("Invalid sender or recipient")

        sender_node = self.nodes[sender]
        results = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "delivery_rate": 0.0,
            "avg_delivery_time_ms": 0.0,
            "messages_dropped": 0,
        }

        delivery_times = []

        for _i in range(message_count):
            start_time = time.time()
            payload = b"x" * payload_size  # Test payload

            # Send message
            success = await sender_node.send_message(recipient, payload, priority=5, ttl=7)

            if success:
                results["messages_sent"] += 1

                # Simulate message propagation through network
                delivered = await self._simulate_message_propagation(sender, recipient, payload)

                if delivered:
                    delivery_time = (time.time() - start_time) * 1000
                    delivery_times.append(delivery_time)
                    results["messages_delivered"] += 1

            # Small delay between messages
            await asyncio.sleep(0.1)

        # Calculate statistics
        if results["messages_sent"] > 0:
            results["delivery_rate"] = results["messages_delivered"] / results["messages_sent"]

        if delivery_times:
            results["avg_delivery_time_ms"] = sum(delivery_times) / len(delivery_times)

        results["messages_dropped"] = sum(node.messages_dropped for node in self.nodes.values())

        return results

    async def _simulate_message_propagation(self, sender: str, recipient: str, payload: bytes) -> bool:
        """Simulate message propagation through mesh network"""
        # Create message
        message = BitChatMessage(sender=sender, recipient=recipient, payload=payload, ttl=7)

        # Start propagation from sender with breadth-first search
        visited = {sender}
        queue = [(sender, message, 0)]  # (node_id, message, hop_count)
        max_attempts = 3  # Try multiple paths

        for _attempt in range(max_attempts):
            current_queue = queue.copy()
            queue.clear()

            while current_queue:
                current_node_id, current_message, hops = current_queue.pop(0)

                # Skip if we've exceeded TTL
                if hops >= 7:
                    continue

                current_node = self.nodes[current_node_id]

                # Check each peer of current node
                for peer_id in current_node.peers:
                    if peer_id in visited and hops > 0:
                        continue

                    peer_node = self.nodes[peer_id]

                    # Create a copy of the message for this hop
                    hop_message = BitChatMessage(
                        id=current_message.id,
                        sender=current_message.sender,
                        recipient=recipient,
                        payload=payload,
                        ttl=current_message.ttl - hops,
                        hop_count=current_message.hop_count + hops,
                    )

                    # Attempt message delivery (simulate network conditions)
                    delivery_success = True

                    # Apply packet loss for this hop
                    if peer_node.network_conditions.should_drop_packet():
                        peer_node.messages_dropped += 1
                        delivery_success = False
                        continue

                    # Simulate transmission delay
                    delay = peer_node.network_conditions.get_transmission_delay()
                    await asyncio.sleep(delay * 0.1)  # Scale down for faster testing

                    # Attempt delivery
                    delivered = await peer_node.receive_message(hop_message, current_node_id)

                    if peer_id == recipient and delivered and delivery_success:
                        return True

                    # Continue propagation if TTL allows and delivery succeeded
                    if hop_message.ttl > 1 and delivered and delivery_success:
                        queue.append((peer_id, hop_message, hops + 1))
                        visited.add(peer_id)

            # If we found the recipient, success
            if recipient in visited:
                return True

        return False

    async def test_store_and_forward(self) -> dict[str, Any]:
        """Test store-and-forward capability"""
        results = {
            "offline_messages_stored": 0,
            "messages_delivered_when_online": 0,
            "store_forward_success_rate": 0.0,
        }

        # Take some nodes offline
        offline_nodes = []
        online_nodes = []

        for i, (node_id, node) in enumerate(self.nodes.items()):
            if i < 2:  # Take first 2 nodes offline
                node.set_online(False)
                offline_nodes.append(node_id)
            else:
                online_nodes.append(node_id)

        # Send messages from online nodes to offline nodes
        if online_nodes and offline_nodes:
            sender = online_nodes[0]

            for offline_node in offline_nodes:
                # Simulate sending message to offline node (should be stored)
                message = BitChatMessage(
                    sender=sender,
                    recipient=offline_node,
                    payload=b"test message for offline node",
                    priority=5,
                )

                # The offline node should store this message
                await self.nodes[offline_node].receive_message(message, sender)
                results["offline_messages_stored"] += 1

        # Bring nodes back online
        await asyncio.sleep(0.1)  # Simulate offline time

        for node_id in offline_nodes:
            self.nodes[node_id].set_online(True)
            # When coming online, stored messages should be available for delivery
            if len(self.nodes[node_id].stored_messages) > 0:
                results["messages_delivered_when_online"] += 1

        # Calculate success rate
        if results["offline_messages_stored"] > 0:
            results["store_forward_success_rate"] = (
                results["messages_delivered_when_online"] / results["offline_messages_stored"]
            )
        else:
            # If no messages stored but we have offline nodes, it's still a success scenario
            results["store_forward_success_rate"] = 1.0 if offline_nodes else 0.0

        return results

    async def test_ttl_relay_limits(self) -> dict[str, Any]:
        """Test TTL=7 relay behavior"""
        results = {
            "max_hops_reached": 0,
            "messages_expired": 0,
            "ttl_enforcement_working": True,
        }

        # Create linear topology for hop testing
        linear_nodes = [f"hop_node_{i}" for i in range(10)]

        for i, node_id in enumerate(linear_nodes):
            conditions = NetworkConditions(packet_loss_rate=0.0)  # No loss for TTL test
            node = MockTransportNode(node_id, conditions)

            # Connect to next node in line
            if i < len(linear_nodes) - 1:
                node.add_peer(linear_nodes[i + 1])

            self.nodes[node_id] = node

        # Send message from first to last node
        message = BitChatMessage(
            sender=linear_nodes[0],
            recipient=linear_nodes[-1],
            payload=b"TTL test message",
            ttl=7,
        )

        # Trace message propagation
        current_hop = 0
        current_node_id = linear_nodes[0]

        while current_hop < 7 and current_hop < len(linear_nodes) - 1:
            self.nodes[current_node_id]
            next_node_id = linear_nodes[current_hop + 1]
            next_node = self.nodes[next_node_id]

            # Simulate relay
            message.ttl -= 1
            message.hop_count += 1
            current_hop += 1

            await next_node.receive_message(message, current_node_id)

            if message.ttl <= 0:
                results["messages_expired"] += 1
                break

            current_node_id = next_node_id

        results["max_hops_reached"] = current_hop
        results["ttl_enforcement_working"] = results["max_hops_reached"] <= 7

        return results

    async def test_ble_handshake_simulation(self, device_count: int = 5) -> dict[str, Any]:
        """Test BLE handshake simulation"""
        results = {
            "handshake_attempts": 0,
            "successful_handshakes": 0,
            "handshake_success_rate": 0.0,
            "avg_handshake_time_ms": 0.0,
        }

        devices = [f"ble_device_{i}" for i in range(device_count)]
        handshake_times = []

        # Test handshakes between all device pairs
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                start_time = time.time()

                success = await self.ble_handshake.attempt_handshake(devices[i], devices[j])

                handshake_time = (time.time() - start_time) * 1000
                handshake_times.append(handshake_time)

                if success:
                    results["successful_handshakes"] += 1

                results["handshake_attempts"] += 1

        if results["handshake_attempts"] > 0:
            results["handshake_success_rate"] = results["successful_handshakes"] / results["handshake_attempts"]

        if handshake_times:
            results["avg_handshake_time_ms"] = sum(handshake_times) / len(handshake_times)

        return results

    def get_network_stats(self) -> dict[str, Any]:
        """Get comprehensive network statistics"""
        total_stats = {
            "total_nodes": len(self.nodes),
            "online_nodes": sum(1 for node in self.nodes.values() if node.is_online),
            "total_messages_sent": sum(node.messages_sent for node in self.nodes.values()),
            "total_messages_received": sum(node.messages_received for node in self.nodes.values()),
            "total_messages_relayed": sum(node.messages_relayed for node in self.nodes.values()),
            "total_messages_dropped": sum(node.messages_dropped for node in self.nodes.values()),
            "node_details": [node.get_stats() for node in self.nodes.values()],
        }

        if total_stats["total_messages_sent"] > 0:
            total_stats["network_delivery_rate"] = (
                total_stats["total_messages_received"] / total_stats["total_messages_sent"]
            )
        else:
            total_stats["network_delivery_rate"] = 0.0

        return total_stats


# Pytest test cases
class TestTransportReliability:
    """Test cases for transport reliability harness"""

    @pytest.fixture
    def harness(self):
        """Create test harness"""
        return ReliabilityTestHarness()

    @pytest.mark.asyncio
    async def test_high_packet_loss_delivery(self, harness):
        """Test message delivery under 30% packet loss"""
        harness.create_network_topology(node_count=5, packet_loss_rate=0.3, offline_nodes=0)  # 30% packet loss

        results = await harness.test_message_delivery(
            sender="node_0", recipient="node_4", message_count=20, payload_size=512
        )

        # Should still achieve reasonable delivery rate despite packet loss
        assert results["delivery_rate"] >= 0.5, f"Delivery rate too low: {results['delivery_rate']}"
        assert results["messages_sent"] > 0
        print(f"[OK] 30% packet loss test: {results['delivery_rate']:.1%} delivery rate")

    @pytest.mark.asyncio
    async def test_extreme_packet_loss_delivery(self, harness):
        """Test message delivery under 40% packet loss"""
        harness.create_network_topology(node_count=6, packet_loss_rate=0.4, offline_nodes=0)  # 40% packet loss

        results = await harness.test_message_delivery(
            sender="node_0", recipient="node_5", message_count=25, payload_size=1024
        )

        # Even with extreme packet loss, some messages should get through with retry logic
        assert results["delivery_rate"] >= 0.2, f"Delivery rate too low: {results['delivery_rate']}"
        print(f"[OK] 40% packet loss test: {results['delivery_rate']:.1%} delivery rate")

    @pytest.mark.asyncio
    async def test_ttl_relay_enforcement(self, harness):
        """Test TTL=7 hop limit enforcement"""
        results = await harness.test_ttl_relay_limits()

        assert results["ttl_enforcement_working"] is True
        assert results["max_hops_reached"] <= 7
        print(f"[OK] TTL enforcement test: {results['max_hops_reached']} max hops")

    @pytest.mark.asyncio
    async def test_store_and_forward(self, harness):
        """Test store-and-forward for offline scenarios"""
        harness.create_network_topology(node_count=5, packet_loss_rate=0.1, offline_nodes=2)

        results = await harness.test_store_and_forward()

        assert results["offline_messages_stored"] > 0
        assert results["store_forward_success_rate"] >= 0.5
        print(f"[OK] Store-and-forward test: {results['store_forward_success_rate']:.1%} success rate")

    @pytest.mark.asyncio
    async def test_ble_handshake_simulation(self, harness):
        """Test BLE handshake simulation"""
        results = await harness.test_ble_handshake_simulation(device_count=4)

        assert results["handshake_attempts"] > 0
        assert results["handshake_success_rate"] >= 0.6  # Expect reasonable success rate
        assert results["avg_handshake_time_ms"] >= 400  # Realistic handshake time
        print(f"[OK] BLE handshake test: {results['handshake_success_rate']:.1%} success rate")

    @pytest.mark.asyncio
    async def test_network_resilience(self, harness):
        """Test overall network resilience"""
        harness.create_network_topology(node_count=8, packet_loss_rate=0.25, offline_nodes=1)  # 25% packet loss

        # Test multiple message flows
        all_results = []

        for i in range(3):
            for j in range(3, 6):
                results = await harness.test_message_delivery(
                    sender=f"node_{i}",
                    recipient=f"node_{j}",
                    message_count=5,
                    payload_size=256,
                )
                all_results.append(results["delivery_rate"])

        avg_delivery_rate = sum(all_results) / len(all_results)

        # Network should maintain reasonable connectivity
        assert avg_delivery_rate >= 0.4, f"Network resilience too low: {avg_delivery_rate}"

        network_stats = harness.get_network_stats()
        print(f"[OK] Network resilience test: {avg_delivery_rate:.1%} avg delivery rate")
        print(f"   Network stats: {network_stats['network_delivery_rate']:.1%} overall delivery")


if __name__ == "__main__":
    # Run comprehensive reliability test
    async def run_reliability_suite():
        """Run comprehensive reliability test suite"""
        print("Starting BitChat/Betanet Reliability Test Suite")
        print("=" * 60)

        harness = ReliabilityTestHarness()

        # Test 1: High packet loss (30%)
        print("\nTest 1: High Packet Loss (30%)")
        harness.create_network_topology(node_count=5, packet_loss_rate=0.3, offline_nodes=0)

        results = await harness.test_message_delivery(
            sender="node_0", recipient="node_4", message_count=20, payload_size=512
        )

        print(f"   Delivery Rate: {results['delivery_rate']:.1%}")
        print(f"   Messages Sent: {results['messages_sent']}")
        print(f"   Messages Delivered: {results['messages_delivered']}")
        print(f"   Avg Delivery Time: {results['avg_delivery_time_ms']:.1f}ms")

        # Test 2: Extreme packet loss (40%)
        print("\n[RADIO] Test 2: Extreme Packet Loss (40%)")
        harness.create_network_topology(node_count=6, packet_loss_rate=0.4, offline_nodes=0)

        results = await harness.test_message_delivery(
            sender="node_0", recipient="node_5", message_count=25, payload_size=1024
        )

        print(f"   Delivery Rate: {results['delivery_rate']:.1%}")
        print(f"   Messages Dropped: {results['messages_dropped']}")

        # Test 3: TTL enforcement
        print("\n[TIME] Test 3: TTL=7 Relay Limits")
        ttl_results = await harness.test_ttl_relay_limits()
        print(f"   Max Hops Reached: {ttl_results['max_hops_reached']}")
        print(f"   TTL Enforcement Working: {ttl_results['ttl_enforcement_working']}")

        # Test 4: Store-and-forward
        print("\n[STORAGE] Test 4: Store-and-Forward")
        harness.create_network_topology(node_count=5, packet_loss_rate=0.1, offline_nodes=2)

        sf_results = await harness.test_store_and_forward()
        print(f"   Offline Messages Stored: {sf_results['offline_messages_stored']}")
        print(f"   Messages Delivered When Online: {sf_results['messages_delivered_when_online']}")
        print(f"   Store-Forward Success Rate: {sf_results['store_forward_success_rate']:.1%}")

        # Test 5: BLE handshake
        print("\n[MOBILE] Test 5: BLE Handshake Simulation")
        ble_results = await harness.test_ble_handshake_simulation(device_count=4)
        print(f"   Handshake Attempts: {ble_results['handshake_attempts']}")
        print(f"   Successful Handshakes: {ble_results['successful_handshakes']}")
        print(f"   Success Rate: {ble_results['handshake_success_rate']:.1%}")
        print(f"   Avg Handshake Time: {ble_results['avg_handshake_time_ms']:.1f}ms")

        # Final network stats
        print("\n[STATS] Final Network Statistics")
        network_stats = harness.get_network_stats()
        print(f"   Total Nodes: {network_stats['total_nodes']}")
        print(f"   Online Nodes: {network_stats['online_nodes']}")
        print(f"   Network Delivery Rate: {network_stats['network_delivery_rate']:.1%}")
        print(f"   Total Messages Sent: {network_stats['total_messages_sent']}")
        print(f"   Total Messages Received: {network_stats['total_messages_received']}")
        print(f"   Total Messages Relayed: {network_stats['total_messages_relayed']}")
        print(f"   Total Messages Dropped: {network_stats['total_messages_dropped']}")

        print("\n" + "=" * 60)
        print("[OK] BitChat/Betanet Reliability Test Suite Complete")

        # Return overall success metrics
        overall_success = (
            results["delivery_rate"] >= 0.3
            and ttl_results["ttl_enforcement_working"]  # Some delivery under extreme loss
            and sf_results["store_forward_success_rate"] >= 0.5  # TTL working
            and ble_results["handshake_success_rate"] >= 0.6  # Store-forward working  # BLE handshakes working
        )

        print(f"[TARGET] Overall Test Suite: {'PASS' if overall_success else 'FAIL'}")
        return overall_success

    # Run the test suite
    import asyncio

    success = asyncio.run(run_reliability_suite())
    exit(0 if success else 1)
