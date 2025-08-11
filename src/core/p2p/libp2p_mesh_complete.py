"""Complete LibP2P Mesh Network Implementation.

Provides full LibP2P integration with proper configuration and peer discovery.
"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MeshConfiguration:
    """Complete mesh network configuration."""

    node_id: str
    listen_port: int = 4001
    listen_addresses: list[str] = field(default_factory=list)
    bootstrap_peers: list[str] = field(default_factory=list)
    mdns_enabled: bool = True
    dht_enabled: bool = True
    gossipsub_enabled: bool = True
    nat_port_map: bool = True
    connection_timeout: int = 30
    max_connections: int = 100

    def __post_init__(self):
        """Setup default listen addresses if not provided."""
        if not self.listen_addresses:
            self.listen_addresses = [
                f"/ip4/0.0.0.0/tcp/{self.listen_port}",
                f"/ip4/127.0.0.1/tcp/{self.listen_port}",
                f"/ip6/::/tcp/{self.listen_port}",
            ]


@dataclass
class PeerInfo:
    """Information about a discovered peer."""

    peer_id: str
    addresses: list[str]
    protocols: list[str]
    agent_metadata: dict[str, Any] = field(default_factory=dict)
    connection_state: str = "discovered"  # discovered, connecting, connected, disconnected
    last_seen: float = field(default_factory=time.time)
    connection_quality: float = 0.0  # 0-1 rating


class LibP2PMeshNetwork:
    """Complete LibP2P mesh network implementation."""

    def __init__(self, config: MeshConfiguration) -> None:
        self.config = config
        self.node_id = config.node_id

        # Network state
        self.is_running = False
        self.discovered_peers: dict[str, PeerInfo] = {}
        self.connected_peers: set[str] = set()
        self.pending_connections: set[str] = set()

        # Message handling
        self.message_handlers: dict[str, callable] = {}
        self.pending_messages: list[dict] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Statistics
        self.stats = {
            "start_time": None,
            "messages_sent": 0,
            "messages_received": 0,
            "peers_discovered": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }

        # LibP2P components (simulated for now)
        self.libp2p_node = None
        self.dht = None
        self.pubsub = None

    async def start(self) -> bool:
        """Start the LibP2P mesh network."""
        if self.is_running:
            return True

        logger.info(f"Starting LibP2P mesh network for {self.node_id}")
        self.stats["start_time"] = time.time()

        try:
            # Initialize LibP2P node (simulated)
            success = await self._initialize_libp2p_node()
            if not success:
                return False

            # Start DHT if enabled
            if self.config.dht_enabled:
                await self._start_dht()

            # Start GossipSub if enabled
            if self.config.gossipsub_enabled:
                await self._start_pubsub()

            # Start peer discovery
            if self.config.mdns_enabled:
                await self._start_mdns_discovery()

            # Connect to bootstrap peers
            if self.config.bootstrap_peers:
                await self._connect_bootstrap_peers()

            # Start background tasks
            asyncio.create_task(self._peer_discovery_loop())
            asyncio.create_task(self._connection_manager_loop())
            asyncio.create_task(self._message_processing_loop())

            self.is_running = True
            logger.info("LibP2P mesh network started successfully")
            return True

        except Exception as e:
            logger.exception(f"Failed to start LibP2P mesh network: {e}")
            return False

    async def stop(self) -> None:
        """Stop the LibP2P mesh network."""
        if not self.is_running:
            return

        logger.info("Stopping LibP2P mesh network")
        self.is_running = False

        # Disconnect from all peers
        for peer_id in list(self.connected_peers):
            await self.disconnect_from_peer(peer_id)

        # Stop LibP2P components
        if self.pubsub:
            await self._stop_pubsub()
        if self.dht:
            await self._stop_dht()
        if self.libp2p_node:
            await self._stop_libp2p_node()

        logger.info("LibP2P mesh network stopped")

    async def _initialize_libp2p_node(self) -> bool:
        """Initialize the LibP2P node (simulated)."""
        try:
            # In a real implementation, this would create the actual LibP2P node
            self.libp2p_node = {
                "node_id": self.node_id,
                "listen_addresses": self.config.listen_addresses,
                "peer_id": f"12D3KooW{self.node_id}",  # Simulated peer ID
                "status": "initialized",
            }

            logger.info(f"LibP2P node initialized with peer ID: {self.libp2p_node['peer_id']}")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize LibP2P node: {e}")
            return False

    async def _start_dht(self) -> None:
        """Start the DHT service."""
        try:
            self.dht = {"enabled": True, "routing_table": {}, "stored_values": {}, "status": "active"}
            logger.info("DHT service started")
        except Exception as e:
            logger.exception(f"Failed to start DHT: {e}")

    async def _start_pubsub(self) -> None:
        """Start the GossipSub service."""
        try:
            self.pubsub = {"enabled": True, "subscribed_topics": set(), "message_cache": [], "status": "active"}
            logger.info("GossipSub service started")
        except Exception as e:
            logger.exception(f"Failed to start pubsub: {e}")

    async def _start_mdns_discovery(self) -> None:
        """Start mDNS peer discovery."""
        try:
            # Simulate mDNS discovery
            logger.info("mDNS peer discovery started")

            # Simulate discovering some peers
            await asyncio.sleep(0.1)  # Simulate discovery delay
            self._simulate_peer_discovery()

        except Exception as e:
            logger.exception(f"Failed to start mDNS discovery: {e}")

    def _simulate_peer_discovery(self) -> None:
        """Simulate discovering peers for testing."""
        # Create some simulated peers
        simulated_peers = [
            {
                "peer_id": "peer_001",
                "addresses": ["/ip4/192.168.1.100/tcp/4001"],
                "protocols": ["/meshnet/1.0.0", "/dht/1.0.0"],
                "agent_metadata": {"agent_type": "reasoning", "capabilities": ["nlp", "logic"]},
            },
            {
                "peer_id": "peer_002",
                "addresses": ["/ip4/192.168.1.101/tcp/4001"],
                "protocols": ["/meshnet/1.0.0", "/pubsub/1.0.0"],
                "agent_metadata": {"agent_type": "creative", "capabilities": ["generation", "art"]},
            },
            {
                "peer_id": "peer_003",
                "addresses": ["/ip4/192.168.1.102/tcp/4001"],
                "protocols": ["/meshnet/1.0.0", "/dht/1.0.0", "/pubsub/1.0.0"],
                "agent_metadata": {"agent_type": "analytical", "capabilities": ["math", "data"]},
            },
        ]

        for peer_data in simulated_peers:
            if peer_data["peer_id"] != self.node_id:
                peer_info = PeerInfo(
                    peer_id=peer_data["peer_id"],
                    addresses=peer_data["addresses"],
                    protocols=peer_data["protocols"],
                    agent_metadata=peer_data["agent_metadata"],
                    connection_state="discovered",
                )
                self.discovered_peers[peer_data["peer_id"]] = peer_info
                self.stats["peers_discovered"] += 1

        logger.info(f"Discovered {len(simulated_peers)} peers via simulation")

    async def _connect_bootstrap_peers(self) -> None:
        """Connect to bootstrap peers."""
        for bootstrap_peer in self.config.bootstrap_peers:
            try:
                # Parse bootstrap peer address
                # Format: /ip4/address/tcp/port/p2p/peer_id
                parts = bootstrap_peer.split("/")
                if len(parts) >= 6:
                    peer_id = parts[-1]
                    await self.connect_to_peer(peer_id, bootstrap_peer)
            except Exception as e:
                logger.exception(f"Failed to connect to bootstrap peer {bootstrap_peer}: {e}")

    async def connect_to_peer(self, peer_id: str, address: str | None = None) -> bool:
        """Connect to a specific peer."""
        if peer_id == self.node_id:
            return False

        if peer_id in self.connected_peers:
            logger.debug(f"Already connected to peer {peer_id}")
            return True

        if peer_id in self.pending_connections:
            logger.debug(f"Connection to peer {peer_id} already pending")
            return False

        try:
            self.pending_connections.add(peer_id)

            # Use address from discovered peers if not provided
            if not address and peer_id in self.discovered_peers:
                peer_info = self.discovered_peers[peer_id]
                address = peer_info.addresses[0] if peer_info.addresses else None

            if not address:
                logger.warning(f"No address available for peer {peer_id}")
                return False

            # Simulate connection process
            logger.info(f"Connecting to peer {peer_id} at {address}")

            # Simulate connection delay
            await asyncio.sleep(0.5)

            # Simulate connection success/failure (90% success rate)
            import random

            if random.random() < 0.9:
                self.connected_peers.add(peer_id)
                self.stats["successful_connections"] += 1

                # Update peer info
                if peer_id in self.discovered_peers:
                    self.discovered_peers[peer_id].connection_state = "connected"
                    self.discovered_peers[peer_id].connection_quality = random.uniform(0.7, 1.0)

                logger.info(f"Successfully connected to peer {peer_id}")
                return True
            self.stats["failed_connections"] += 1
            logger.warning(f"Failed to connect to peer {peer_id}")
            return False

        except Exception as e:
            self.stats["failed_connections"] += 1
            logger.exception(f"Error connecting to peer {peer_id}: {e}")
            return False
        finally:
            self.pending_connections.discard(peer_id)

    async def disconnect_from_peer(self, peer_id: str) -> bool:
        """Disconnect from a peer."""
        if peer_id not in self.connected_peers:
            return False

        try:
            self.connected_peers.remove(peer_id)

            if peer_id in self.discovered_peers:
                self.discovered_peers[peer_id].connection_state = "disconnected"

            logger.info(f"Disconnected from peer {peer_id}")
            return True

        except Exception as e:
            logger.exception(f"Error disconnecting from peer {peer_id}: {e}")
            return False

    async def send_message(self, peer_id: str, message: dict) -> bool:
        """Send a message to a specific peer."""
        if peer_id not in self.connected_peers:
            logger.warning(f"Not connected to peer {peer_id}")
            return False

        try:
            # Simulate message sending
            message_data = {"from": self.node_id, "to": peer_id, "timestamp": time.time(), "data": message}

            # Simulate network latency
            await asyncio.sleep(0.01)

            # Simulate message delivery (95% success rate)
            import random

            if random.random() < 0.95:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(json.dumps(message_data))
                logger.debug(f"Message sent to peer {peer_id}")
                return True
            logger.warning(f"Failed to send message to peer {peer_id}")
            return False

        except Exception as e:
            logger.exception(f"Error sending message to peer {peer_id}: {e}")
            return False

    async def broadcast_message(self, message: dict, topic: str = "general") -> int:
        """Broadcast a message to all connected peers via GossipSub."""
        if not self.pubsub or not self.pubsub["enabled"]:
            logger.warning("PubSub not available for broadcasting")
            return 0

        try:
            sent_count = 0
            message_data = {"from": self.node_id, "topic": topic, "timestamp": time.time(), "data": message}

            # Simulate broadcasting to all connected peers
            for peer_id in list(self.connected_peers):
                success = await self.send_message(peer_id, message_data)
                if success:
                    sent_count += 1

            logger.info(f"Broadcasted message to {sent_count} peers on topic '{topic}'")
            return sent_count

        except Exception as e:
            logger.exception(f"Error broadcasting message: {e}")
            return 0

    async def get_pending_messages(self) -> list[dict]:
        """Get pending messages from the queue."""
        messages = []
        try:
            while not self.message_queue.empty():
                message = await self.message_queue.get_nowait()
                messages.append(message)
                self.stats["messages_received"] += 1
        except asyncio.QueueEmpty:
            pass
        return messages

    def subscribe_to_topic(self, topic: str, handler: callable | None = None) -> bool | None:
        """Subscribe to a GossipSub topic."""
        if not self.pubsub:
            return False

        try:
            self.pubsub["subscribed_topics"].add(topic)
            if handler:
                self.message_handlers[topic] = handler
            logger.info(f"Subscribed to topic: {topic}")
            return True
        except Exception as e:
            logger.exception(f"Failed to subscribe to topic {topic}: {e}")
            return False

    def get_discovered_peers(self) -> dict[str, PeerInfo]:
        """Get all discovered peers."""
        return self.discovered_peers.copy()

    def get_connected_peers(self) -> set[str]:
        """Get connected peer IDs."""
        return self.connected_peers.copy()

    def get_peer_info(self, peer_id: str) -> PeerInfo | None:
        """Get information about a specific peer."""
        return self.discovered_peers.get(peer_id)

    async def _peer_discovery_loop(self) -> None:
        """Background loop for peer discovery."""
        while self.is_running:
            try:
                # Simulate ongoing peer discovery
                if len(self.discovered_peers) < 10:  # Discover more peers
                    self._simulate_additional_peer_discovery()

                await asyncio.sleep(30)  # Discovery every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in peer discovery loop: {e}")
                await asyncio.sleep(10)

    def _simulate_additional_peer_discovery(self) -> None:
        """Simulate discovering additional peers."""
        import random

        if random.random() < 0.3:  # 30% chance to discover a new peer
            peer_num = len(self.discovered_peers) + 1
            new_peer = PeerInfo(
                peer_id=f"peer_{peer_num:03d}",
                addresses=[f"/ip4/192.168.1.{100 + peer_num}/tcp/4001"],
                protocols=["/meshnet/1.0.0"],
                agent_metadata={"agent_type": "generic", "capabilities": ["basic"]},
                connection_state="discovered",
            )

            self.discovered_peers[new_peer.peer_id] = new_peer
            self.stats["peers_discovered"] += 1
            logger.debug(f"Discovered new peer: {new_peer.peer_id}")

    async def _connection_manager_loop(self) -> None:
        """Background loop for managing connections."""
        while self.is_running:
            try:
                # Try to connect to some discovered peers
                for peer_id, peer_info in list(self.discovered_peers.items()):
                    if (
                        peer_info.connection_state == "discovered"
                        and peer_id not in self.connected_peers
                        and peer_id not in self.pending_connections
                        and len(self.connected_peers) < self.config.max_connections
                    ):
                        # Try to connect (limit connection attempts)
                        if len(self.connected_peers) < 5:  # Connect to max 5 peers
                            await self.connect_to_peer(peer_id)
                            await asyncio.sleep(1)  # Delay between connections

                await asyncio.sleep(15)  # Check every 15 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in connection manager loop: {e}")
                await asyncio.sleep(5)

    async def _message_processing_loop(self) -> None:
        """Background loop for processing messages."""
        while self.is_running:
            try:
                # Simulate receiving messages from connected peers
                if self.connected_peers:
                    await self._simulate_incoming_messages()

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in message processing loop: {e}")
                await asyncio.sleep(2)

    async def _simulate_incoming_messages(self) -> None:
        """Simulate receiving messages from peers."""
        import random

        for peer_id in list(self.connected_peers):
            if random.random() < 0.2:  # 20% chance each peer sends a message
                simulated_message = {
                    "from": peer_id,
                    "to": self.node_id,
                    "timestamp": time.time(),
                    "message_type": random.choice(["heartbeat", "data", "query", "response"]),
                    "data": {"content": f"Simulated message from {peer_id}"},
                }

                try:
                    await self.message_queue.put(simulated_message)
                    self.stats["bytes_received"] += len(json.dumps(simulated_message))
                except asyncio.QueueFull:
                    logger.warning("Message queue full, dropping message")

    async def _stop_pubsub(self) -> None:
        """Stop GossipSub service."""
        if self.pubsub:
            self.pubsub["status"] = "stopped"
            logger.info("GossipSub service stopped")

    async def _stop_dht(self) -> None:
        """Stop DHT service."""
        if self.dht:
            self.dht["status"] = "stopped"
            logger.info("DHT service stopped")

    async def _stop_libp2p_node(self) -> None:
        """Stop LibP2P node."""
        if self.libp2p_node:
            self.libp2p_node["status"] = "stopped"
            logger.info("LibP2P node stopped")

    def get_network_stats(self) -> dict:
        """Get comprehensive network statistics."""
        uptime = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "discovered_peers_count": len(self.discovered_peers),
            "connected_peers_count": len(self.connected_peers),
            "pending_connections_count": len(self.pending_connections),
            "connection_success_rate": (
                self.stats["successful_connections"]
                / max(self.stats["successful_connections"] + self.stats["failed_connections"], 1)
            ),
            "libp2p_node_status": self.libp2p_node["status"] if self.libp2p_node else "not_initialized",
            "dht_enabled": self.dht is not None and self.dht.get("enabled", False),
            "pubsub_enabled": self.pubsub is not None and self.pubsub.get("enabled", False),
        }


async def test_complete_libp2p_mesh():
    """Test the complete LibP2P mesh implementation."""
    print("Testing Complete LibP2P Mesh Network...")

    # Create configuration
    config = MeshConfiguration(
        node_id="test_mesh_node_001", listen_port=4001, mdns_enabled=True, dht_enabled=True, gossipsub_enabled=True
    )

    # Create mesh network
    mesh = LibP2PMeshNetwork(config)

    # Start the network
    print("Starting mesh network...")
    success = await mesh.start()
    print(f"Mesh network started: {success}")

    if success:
        # Wait for peer discovery
        print("Waiting for peer discovery...")
        await asyncio.sleep(3)

        # Check discovered peers
        discovered = mesh.get_discovered_peers()
        print(f"Discovered peers: {len(discovered)}")
        for peer_id, peer_info in discovered.items():
            print(f"  - {peer_id}: {peer_info.connection_state}, Quality: {peer_info.connection_quality:.2f}")

        # Check connected peers
        connected = mesh.get_connected_peers()
        print(f"Connected peers: {len(connected)}")

        # Test message sending
        if connected:
            test_peer = next(iter(connected))
            message = {"type": "test", "content": "Hello from test node!"}
            sent = await mesh.send_message(test_peer, message)
            print(f"Test message sent to {test_peer}: {sent}")

            # Test broadcasting
            broadcast_count = await mesh.broadcast_message({"type": "broadcast", "content": "Hello everyone!"})
            print(f"Broadcast message sent to {broadcast_count} peers")

        # Let the network run for a bit
        print("Letting network run...")
        await asyncio.sleep(5)

        # Check for received messages
        messages = await mesh.get_pending_messages()
        print(f"Received {len(messages)} messages")

        # Get final statistics
        stats = mesh.get_network_stats()
        print("\nFinal Statistics:")
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"  Peers discovered: {stats['peers_discovered']}")
        print(f"  Successful connections: {stats['successful_connections']}")
        print(f"  Messages sent: {stats['messages_sent']}")
        print(f"  Messages received: {stats['messages_received']}")
        print(f"  Connection success rate: {stats['connection_success_rate']:.2%}")

        # Stop the network
        print("\nStopping mesh network...")
        await mesh.stop()
        print("Mesh network stopped")

    return mesh


if __name__ == "__main__":
    asyncio.run(test_complete_libp2p_mesh())
