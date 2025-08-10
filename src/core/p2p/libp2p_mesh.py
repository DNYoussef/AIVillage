"""LibP2P Mesh Network Implementation for AIVillage.

This module provides a LibP2P-based mesh networking solution to replace
the broken Bluetooth mesh implementation. It supports:
- Peer discovery via mDNS
- Pub/Sub messaging
- DHT for peer routing
- Transport agnostic design (TCP, Bluetooth, WiFi Direct)
- Message types: DATA_MESSAGE, AGENT_TASK, PARAMETER_UPDATE, GRADIENT_SHARING

DESIGN:
- Uses py-libp2p for Python backend
- Provides Android integration via JNI wrapper
- Maintains backward compatibility with existing interfaces
- Supports fallback transports for offline scenarios
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# LibP2P imports (install with: pip install py-libp2p)
try:
    from libp2p import new_host
    from libp2p.kademlia import KadDHT
    from libp2p.network.stream.net_stream_interface import INetStream
    from libp2p.peer.peerinfo import info_from_p2p_addr
    from libp2p.pubsub.gossipsub import GossipSub
    from libp2p.pubsub.pubsub import Pubsub
    from multiaddr import Multiaddr

    LIBP2P_AVAILABLE = True
except ImportError:
    LIBP2P_AVAILABLE = False
    logging.warning("LibP2P not available, using fallback implementation")

    # Define placeholder types for when LibP2P is not available
    class INetStream:  # type: ignore
        pass

    class Pubsub:  # type: ignore
        pass

    class KadDHT:  # type: ignore
        pass


from .fallback_transports import (
    FallbackTransportManager,
    TransportMessage,
    TransportType,
    create_default_fallback_manager,
)
from .mdns_discovery import PeerInfo, mDNSDiscovery
from .p2p_node import NodeStatus, P2PNode, PeerCapabilities

logger = logging.getLogger(__name__)


# Message types matching existing Android implementation
class MeshMessageType(Enum):
    """Mesh network message types."""

    DATA_MESSAGE = "DATA_MESSAGE"
    AGENT_TASK = "AGENT_TASK"
    PARAMETER_UPDATE = "PARAMETER_UPDATE"
    GRADIENT_SHARING = "GRADIENT_SHARING"
    PEER_DISCOVERY = "PEER_DISCOVERY"
    HEARTBEAT = "HEARTBEAT"
    ROUTING_UPDATE = "ROUTING_UPDATE"


@dataclass
class MeshMessage:
    """Mesh network message structure."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MeshMessageType = MeshMessageType.DATA_MESSAGE
    sender: str = ""
    recipient: str | None = None  # None for broadcast
    payload: bytes = b""
    ttl: int = 5
    timestamp: float = field(default_factory=time.time)
    hop_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": (
                self.payload.hex() if isinstance(self.payload, bytes) else self.payload
            ),
            "ttl": self.ttl,
            "timestamp": self.timestamp,
            "hop_count": self.hop_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshMessage":
        """Create from dictionary."""
        payload = data.get("payload", "")
        if isinstance(payload, str):
            try:
                payload = bytes.fromhex(payload)
            except ValueError:
                payload = payload.encode()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MeshMessageType(data.get("type", "DATA_MESSAGE")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient"),
            payload=payload,
            ttl=data.get("ttl", 5),
            timestamp=data.get("timestamp", time.time()),
            hop_count=data.get("hop_count", 0),
        )


@dataclass
class MeshConfiguration:
    """Mesh network configuration."""

    node_id: str | None = None
    listen_port: int = 4001  # Standard libp2p port
    discovery_interval: float = 30.0
    heartbeat_interval: float = 10.0
    max_peers: int = 50
    pubsub_topic: str = "aivillage-mesh"
    dht_enabled: bool = True
    mdns_enabled: bool = True
    transports: list[str] = field(default_factory=lambda: ["tcp", "ws"])
    fallback_transports: list[str] = field(
        default_factory=lambda: ["bluetooth", "wifi_direct"]
    )


class LibP2PMeshNetwork:
    """LibP2P-based mesh network implementation."""

    def __init__(self, config: MeshConfiguration | None = None) -> None:
        self.config = config or MeshConfiguration()
        self.node_id = self.config.node_id or str(uuid.uuid4())
        self.status = NodeStatus.STARTING

        # LibP2P components
        self.host = None
        self.pubsub: Pubsub | None = None
        self.dht: KadDHT | None = None
        self.mdns_discovery: mDNSDiscovery | None = None

        # Fallback transports
        self.fallback_manager: FallbackTransportManager | None = None

        # Mesh state
        self.connected_peers: dict[str, PeerCapabilities] = {}
        self.message_handlers: dict[MeshMessageType, Callable] = {}
        self.routing_table: dict[str, str] = {}  # destination -> next_hop
        self.message_cache: set[str] = set()  # For duplicate detection
        self.pending_messages: dict[str, list[MeshMessage]] = {}  # Offline queue

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_routed": 0,
            "peers_discovered": 0,
            "routing_updates": 0,
        }

        # Fallback to existing P2P node if LibP2P not available
        self.fallback_node: P2PNode | None = None
        if not LIBP2P_AVAILABLE:
            self.fallback_node = P2PNode(self.node_id, self.config.listen_port)

        # Register default message handlers
        self._register_default_handlers()

    async def start(self) -> None:
        """Start the mesh network."""
        logger.info(f"Starting LibP2P mesh network (node: {self.node_id})")

        if not LIBP2P_AVAILABLE:
            logger.warning("Using fallback P2P implementation")
            await self._start_fallback()
            return

        try:
            # Create LibP2P host
            transports = []
            for transport in self.config.transports:
                if transport == "tcp":
                    transports.append(f"/ip4/0.0.0.0/tcp/{self.config.listen_port}")
                elif transport == "ws":
                    transports.append(
                        f"/ip4/0.0.0.0/tcp/{self.config.listen_port + 1}/ws"
                    )

            self.host = await new_host(
                listen_addrs=[Multiaddr(addr) for addr in transports]
            )

            # Set up Pub/Sub
            gossipsub = GossipSub(
                protocols=["/meshsub/1.0.0"],
                degree=6,  # Target number of peers to gossip to
                degree_low=4,  # Lower bound for peers
                degree_high=12,  # Upper bound for peers
            )

            self.pubsub = Pubsub(
                host=self.host, router=gossipsub, my_id=self.host.get_id()
            )

            # Subscribe to mesh topic
            await self.pubsub.subscribe(self.config.pubsub_topic)

            # Set up DHT if enabled
            if self.config.dht_enabled:
                self.dht = KadDHT(self.host)
                await self.dht.bootstrap([])  # Bootstrap from known peers

            # Set up mDNS discovery if enabled
            if self.config.mdns_enabled:
                self.mdns_discovery = mDNSDiscovery(
                    node_id=self.node_id,
                    listen_port=self.config.listen_port,
                    capabilities=self._get_local_capabilities(),
                )
                self.mdns_discovery.add_peer_callback(self._on_mdns_peer_event)
                await self.mdns_discovery.start()

            # Set up stream handler for direct messages
            self.host.set_stream_handler("/aivillage/mesh/1.0.0", self._handle_stream)

            # Start background tasks
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._pubsub_message_loop())
            asyncio.create_task(self._dht_maintenance_loop())

            # Initialize fallback transports
            await self._start_fallback_transports()

            self.status = NodeStatus.ACTIVE
            logger.info(f"LibP2P mesh network started on {self.host.get_addrs()}")

        except Exception as e:
            logger.exception(f"Failed to start LibP2P mesh: {e}")
            self.status = NodeStatus.FAILED
            # Fallback to basic P2P
            await self._start_fallback()

    async def _start_fallback(self) -> None:
        """Start fallback P2P implementation."""
        if self.fallback_node:
            await self.fallback_node.start()
            self.status = NodeStatus.ACTIVE

            # Start basic discovery and messaging
            asyncio.create_task(self._fallback_discovery_loop())
            asyncio.create_task(self._fallback_heartbeat_loop())

    async def stop(self) -> None:
        """Stop the mesh network."""
        logger.info("Stopping LibP2P mesh network")
        self.status = NodeStatus.DISCONNECTED

        if self.mdns_discovery:
            await self.mdns_discovery.stop()

        if self.fallback_manager:
            await self.fallback_manager.stop_all_transports()

        if self.host:
            await self.host.close()

        if self.fallback_node:
            await self.fallback_node.stop()

    def register_message_handler(
        self, message_type: MeshMessageType, handler: Callable[[MeshMessage], None]
    ) -> None:
        """Register handler for specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")

    async def send_message(self, message: MeshMessage) -> bool:
        """Send message through mesh network."""
        if self.status != NodeStatus.ACTIVE:
            return False

        try:
            if not LIBP2P_AVAILABLE or not self.host:
                return await self._fallback_send_message(message)

            message.sender = self.node_id
            message.timestamp = time.time()

            if message.recipient:
                if message.recipient not in self.connected_peers:
                    self._queue_offline_message(message.recipient, message)
                    return False
                # Direct message
                success = await self._send_direct_message(message)
            else:
                # Broadcast message
                success = await self._broadcast_message(message)

            if success:
                self.stats["messages_sent"] += 1
                return success

            # Try fallback transports if primary sending failed
            logger.debug(
                f"Primary send failed for {message.id}, trying fallback transports"
            )
            fallback_success = await self._try_fallback_send(message)

            if fallback_success:
                self.stats["messages_sent"] += 1
            elif message.recipient:
                self._queue_offline_message(message.recipient, message)

            return fallback_success

        except Exception as e:
            logger.exception(f"Failed to send message: {e}")
            return False

    async def _send_direct_message(self, message: MeshMessage) -> bool:
        """Send direct message to specific peer."""
        # Try direct connection first
        for peer_id in self.connected_peers:
            if peer_id == message.recipient:
                try:
                    stream = await self.host.new_stream(
                        peer_id, ["/aivillage/mesh/1.0.0"]
                    )

                    data = json.dumps(message.to_dict()).encode()
                    await stream.write(data)
                    await stream.close()
                    return True

                except Exception as e:
                    logger.debug(f"Direct send failed to {peer_id}: {e}")

        # Fall back to routing through mesh
        return await self._route_message(message)

    async def _broadcast_message(self, message: MeshMessage) -> bool:
        """Broadcast message via pub/sub."""
        try:
            data = json.dumps(message.to_dict()).encode()
            await self.pubsub.publish(self.config.pubsub_topic, data)
            return True
        except Exception as e:
            logger.exception(f"Failed to broadcast message: {e}")
            return False

    async def _route_message(self, message: MeshMessage) -> bool:
        """Route message through mesh using routing table."""
        if message.ttl <= 0:
            logger.debug(f"Message {message.id} expired (TTL)")
            return False

        # Check if we've seen this message before
        if message.id in self.message_cache:
            return False

        self.message_cache.add(message.id)

        # Update message for routing
        message.ttl -= 1
        message.hop_count += 1

        # Find next hop
        next_hop = self.routing_table.get(message.recipient)
        if not next_hop:
            # No route found, broadcast to all peers
            return await self._broadcast_message(message)

        # Forward to next hop
        try:
            forwarded_message = MeshMessage(
                id=message.id,
                type=message.type,
                sender=message.sender,
                recipient=message.recipient,
                payload=message.payload,
                ttl=message.ttl,
                timestamp=message.timestamp,
                hop_count=message.hop_count,
            )

            stream = await self.host.new_stream(next_hop, ["/aivillage/mesh/1.0.0"])

            data = json.dumps(forwarded_message.to_dict()).encode()
            await stream.write(data)
            await stream.close()

            self.stats["messages_routed"] += 1
            return True

        except Exception as e:
            logger.debug(f"Failed to route message to {next_hop}: {e}")
            return False

    def _queue_offline_message(self, peer_id: str, message: MeshMessage) -> None:
        """Queue message for later delivery when peer comes online."""
        if peer_id not in self.pending_messages:
            self.pending_messages[peer_id] = []

        self.pending_messages[peer_id].append(message)

        # Limit queued messages per peer to prevent unbounded growth
        max_stored = 100
        if len(self.pending_messages[peer_id]) > max_stored:
            self.pending_messages[peer_id] = self.pending_messages[peer_id][
                -max_stored:
            ]

    async def _deliver_pending_messages(self, peer_id: str) -> None:
        """Deliver queued messages to a peer that just connected."""
        if peer_id not in self.pending_messages:
            return

        messages = self.pending_messages.pop(peer_id)
        for message in messages:
            try:
                await self.send_message(message)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to deliver queued message {message.id}: {e}")

    async def _handle_stream(self, stream: INetStream) -> None:
        """Handle incoming stream (direct messages)."""
        try:
            data = await stream.read()
            message_data = json.loads(data.decode())
            message = MeshMessage.from_dict(message_data)

            await self._process_received_message(message)

        except Exception as e:
            logger.exception(f"Error handling stream: {e}")
        finally:
            await stream.close()

    async def _pubsub_message_loop(self) -> None:
        """Process pub/sub messages."""
        if not self.pubsub:
            return

        while self.status == NodeStatus.ACTIVE:
            try:
                message = await self.pubsub.wait_for_msg()

                # Skip our own messages
                if message.from_id == self.host.get_id():
                    continue

                # Parse mesh message
                try:
                    message_data = json.loads(message.data.decode())
                    mesh_message = MeshMessage.from_dict(message_data)
                    await self._process_received_message(mesh_message)

                except Exception as e:
                    logger.debug(f"Failed to parse pub/sub message: {e}")

            except Exception as e:
                logger.exception(f"Pub/sub message loop error: {e}")
                await asyncio.sleep(1)

    async def _process_received_message(self, message: MeshMessage) -> None:
        """Process received message."""
        self.stats["messages_received"] += 1

        # Check if message is for us or needs routing
        if message.recipient and message.recipient != self.node_id:
            # Route message to destination
            await self._route_message(message)
            return

        # Message is for us, handle it
        handler = self.message_handlers.get(message.type)
        if handler:
            try:
                (
                    await handler(message)
                    if asyncio.iscoroutinefunction(handler)
                    else handler(message)
                )
            except Exception as e:
                logger.exception(f"Error in message handler for {message.type}: {e}")
        else:
            logger.debug(f"No handler for message type: {message.type}")

    async def _discovery_loop(self) -> None:
        """Peer discovery loop using mDNS."""
        while self.status == NodeStatus.ACTIVE:
            try:
                if LIBP2P_AVAILABLE and self.host:
                    # Use LibP2P's mDNS discovery
                    # Note: py-libp2p mDNS integration would go here
                    # For now, implementing basic network discovery
                    await self._basic_peer_discovery()
                else:
                    await self._fallback_discovery()

                await asyncio.sleep(self.config.discovery_interval)

            except Exception as e:
                logger.exception(f"Discovery loop error: {e}")
                await asyncio.sleep(10)

    async def _basic_peer_discovery(self) -> None:
        """Basic peer discovery implementation."""
        # DHT-based peer discovery
        if self.dht:
            try:
                # Query DHT for peers
                # Note: This is a simplified example.
                # A production-grade implementation would be more complex.
                await self.dht.provide_key("aivillage-mesh", self.node_id.encode())

                # Find other providers
                providers = await self.dht.get_providers("aivillage-mesh")
                for provider in providers:
                    if provider != self.node_id:
                        # Try to connect to discovered peer
                        try:
                            await self.host.connect(provider)
                        except Exception as e:
                            logger.debug(
                                f"Failed to connect to DHT peer {provider}: {e}"
                            )

            except Exception as e:
                logger.debug(f"DHT discovery error: {e}")

        # Fallback to existing mechanism if needed
        if self.fallback_node:
            # Use existing discovery mechanism
            pass

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self.status == NodeStatus.ACTIVE:
            try:
                heartbeat = MeshMessage(
                    type=MeshMessageType.HEARTBEAT,
                    sender=self.node_id,
                    payload=json.dumps(
                        {
                            "timestamp": time.time(),
                            "capabilities": self._get_local_capabilities(),
                            "peer_count": len(self.connected_peers),
                        }
                    ).encode(),
                )

                await self._broadcast_message(heartbeat)
                await asyncio.sleep(self.config.heartbeat_interval)

            except Exception as e:
                logger.exception(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers.update(
            {
                MeshMessageType.HEARTBEAT: self._handle_heartbeat,
                MeshMessageType.PEER_DISCOVERY: self._handle_peer_discovery,
                MeshMessageType.ROUTING_UPDATE: self._handle_routing_update,
            }
        )

    async def _handle_heartbeat(self, message: MeshMessage) -> None:
        """Handle heartbeat message."""
        try:
            data = json.loads(message.payload.decode())

            # Update peer information
            capabilities = PeerCapabilities(
                device_id=message.sender,
                cpu_cores=data.get("cpu_cores", 1),
                ram_mb=data.get("ram_mb", 1024),
                last_seen=time.time(),
            )

            self.connected_peers[message.sender] = capabilities

        except Exception as e:
            logger.debug(f"Failed to process heartbeat from {message.sender}: {e}")

    async def _handle_peer_discovery(self, message: MeshMessage) -> None:
        """Handle peer discovery message."""
        self.stats["peers_discovered"] += 1
        # Implementation for peer discovery response

    async def _handle_routing_update(self, message: MeshMessage) -> None:
        """Handle routing table update."""
        self.stats["routing_updates"] += 1
        # Implementation for routing table updates

    def _get_local_capabilities(self) -> dict[str, Any]:
        """Get local device capabilities."""
        try:
            import psutil

            return {
                "cpu_cores": psutil.cpu_count() or 1,
                "ram_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
                "battery_percent": self._get_battery_percent(),
                "network_type": "mesh",
            }
        except ImportError:
            return {"cpu_cores": 1, "ram_mb": 1024, "network_type": "mesh"}

    def _get_battery_percent(self) -> int | None:
        """Get battery percentage if available."""
        try:
            import psutil

            battery = psutil.sensors_battery()
            return int(battery.percent) if battery else None
        except Exception:
            return None

    # Fallback methods for when LibP2P is not available
    async def _fallback_send_message(self, message: MeshMessage) -> bool:
        """Send message using fallback P2P node."""
        if not self.fallback_node:
            return False

        # Convert to fallback format and send
        fallback_message = {
            "type": message.type.value,
            "sender_id": message.sender,
            "recipient": message.recipient,
            "payload": message.payload,
            "timestamp": message.timestamp,
        }

        if message.recipient:
            return await self.fallback_node.send_to_peer(
                message.recipient, fallback_message
            )
        await self.fallback_node.broadcast_to_peers("MESH_BROADCAST", fallback_message)
        return True

    async def _fallback_discovery_loop(self) -> None:
        """Fallback discovery using existing P2P mechanism."""
        while self.status == NodeStatus.ACTIVE:
            try:
                # Use existing P2P discovery
                await asyncio.sleep(self.config.discovery_interval)
            except Exception as e:
                logger.exception(f"Fallback discovery error: {e}")
                await asyncio.sleep(10)

    async def _fallback_discovery(self) -> None:
        """Fallback peer discovery."""

    async def _fallback_heartbeat_loop(self) -> None:
        """Fallback heartbeat loop."""
        while self.status == NodeStatus.ACTIVE:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.exception(f"Fallback heartbeat error: {e}")
                await asyncio.sleep(5)

    def get_mesh_status(self) -> dict[str, Any]:
        """Get comprehensive mesh status."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "libp2p_available": LIBP2P_AVAILABLE,
            "connected_peers": len(self.connected_peers),
            "peer_details": [
                {"peer_id": peer_id, "capabilities": caps.__dict__}
                for peer_id, caps in self.connected_peers.items()
            ],
            "routing_table_size": len(self.routing_table),
            "message_cache_size": len(self.message_cache),
            "statistics": self.stats.copy(),
            "configuration": {
                "listen_port": self.config.listen_port,
                "max_peers": self.config.max_peers,
                "pubsub_topic": self.config.pubsub_topic,
                "transports": self.config.transports,
                "fallback_transports": self.config.fallback_transports,
            },
            "fallback_transport_status": self.get_fallback_status(),
        }

    async def add_peer(self, peer_address: str) -> bool:
        """Add peer by address (multiaddr format)."""
        if not LIBP2P_AVAILABLE or not self.host:
            return False

        try:
            peer_info = info_from_p2p_addr(Multiaddr(peer_address))
            await self.host.connect(peer_info)
            await self._deliver_pending_messages(str(peer_info.peer_id))
            return True
        except Exception as e:
            logger.exception(f"Failed to add peer {peer_address}: {e}")
            return False

    def get_peer_count(self) -> int:
        """Get number of connected peers."""
        return len(self.connected_peers)

    def is_peer_reachable(self, peer_id: str) -> bool:
        """Check if peer is reachable."""
        return peer_id in self.connected_peers

    async def _on_mdns_peer_event(self, peer_info: PeerInfo, event_type: str) -> None:
        """Handle mDNS peer discovery events."""
        logger.info(f"mDNS peer {event_type}: {peer_info.peer_id}")

        if event_type in {"discovered", "updated"}:
            # Try to connect to discovered peer
            for multiaddr_str in peer_info.to_multiaddr():
                try:
                    await self.add_peer(multiaddr_str)
                    break  # Connected successfully
                except Exception as e:
                    logger.debug(
                        "Failed to connect to %s via %s: %s",
                        peer_info.peer_id,
                        multiaddr_str,
                        e,
                    )

            # Update capabilities
            capabilities = PeerCapabilities(
                device_id=peer_info.peer_id,
                cpu_cores=peer_info.capabilities.get("cpu_cores", 1),
                ram_mb=peer_info.capabilities.get("ram_mb", 1024),
                battery_percent=peer_info.capabilities.get("battery_percent"),
                network_type=peer_info.capabilities.get("network_type", "wifi"),
                last_seen=peer_info.last_seen,
            )
            self.connected_peers[peer_info.peer_id] = capabilities
            await self._deliver_pending_messages(peer_info.peer_id)

        elif event_type == "removed":
            # Remove peer from connected peers
            if peer_info.peer_id in self.connected_peers:
                del self.connected_peers[peer_info.peer_id]

            # Remove from routing table
            entries_to_remove = []
            for dest, next_hop in self.routing_table.items():
                if next_hop == peer_info.peer_id:
                    entries_to_remove.append(dest)

            for dest in entries_to_remove:
                del self.routing_table[dest]

    async def _dht_maintenance_loop(self) -> None:
        """DHT maintenance and routing updates."""
        if not self.dht:
            return

        while self.status == NodeStatus.ACTIVE:
            try:
                # Refresh DHT routing table
                await self.dht.refresh_routing_table()

                # Update our presence in DHT
                await self.dht.provide_key("aivillage-mesh", self.node_id.encode())

                # Periodically discover new peers
                await self._basic_peer_discovery()

                await asyncio.sleep(60)  # DHT maintenance every minute

            except Exception as e:
                logger.exception(f"DHT maintenance error: {e}")
                await asyncio.sleep(30)

    async def dht_store(self, key: str, value: bytes) -> bool:
        """Store value in DHT."""
        if not self.dht:
            return False

        try:
            await self.dht.put_value(key.encode(), value)
            return True
        except Exception as e:
            logger.exception(f"DHT store failed for key {key}: {e}")
            return False

    async def dht_get(self, key: str) -> bytes | None:
        """Get value from DHT."""
        if not self.dht:
            return None

        try:
            return await self.dht.get_value(key.encode())
        except Exception as e:
            logger.debug(f"DHT get failed for key {key}: {e}")
            return None

    async def dht_find_peers(self, key: str) -> list[str]:
        """Find peers providing a key in DHT."""
        if not self.dht:
            return []

        try:
            providers = await self.dht.get_providers(key.encode())
            return [str(provider) for provider in providers]
        except Exception as e:
            logger.debug(f"DHT find peers failed for key {key}: {e}")
            return []

    async def _start_fallback_transports(self) -> None:
        """Initialize and start fallback transports."""
        try:
            self.fallback_manager = create_default_fallback_manager(self.node_id)
            self.fallback_manager.add_message_handler(self._handle_fallback_message)

            # Start fallback transports
            results = await self.fallback_manager.start_all_transports()

            active_count = sum(1 for success in results.values() if success)
            logger.info(f"Started {active_count}/{len(results)} fallback transports")

            if active_count == 0:
                logger.warning("No fallback transports available")

        except Exception as e:
            logger.exception(f"Failed to start fallback transports: {e}")

    async def _handle_fallback_message(
        self, transport_message: TransportMessage
    ) -> None:
        """Handle message from fallback transport."""
        try:
            # Convert transport message to mesh message
            mesh_message = MeshMessage(
                id=transport_message.id,
                type=MeshMessageType(transport_message.type),
                sender=transport_message.sender,
                recipient=transport_message.recipient,
                payload=transport_message.payload,
                timestamp=transport_message.timestamp,
                ttl=transport_message.ttl,
                hop_count=0,
            )

            # Process as regular mesh message
            await self._process_received_message(mesh_message)

        except Exception as e:
            logger.exception(f"Error handling fallback message: {e}")

    async def _try_fallback_send(self, message: MeshMessage) -> bool:
        """Try to send message via fallback transports."""
        if not self.fallback_manager:
            return False

        try:
            # Convert mesh message to transport message
            transport_message = TransportMessage(
                id=message.id,
                type=message.type.value,
                sender=message.sender,
                recipient=message.recipient,
                payload=message.payload,
                timestamp=message.timestamp,
                transport_type="fallback",
                ttl=message.ttl,
            )

            # Try preferred transport based on message type
            preferred_transport = self._get_preferred_transport(message.type)
            success = await self.fallback_manager.send_message(
                transport_message, preferred_transport
            )

            if success:
                logger.debug(f"Message sent via fallback transport: {message.id}")
                return True

        except Exception as e:
            logger.debug(f"Fallback send failed: {e}")

        return False

    def _get_preferred_transport(
        self, message_type: MeshMessageType
    ) -> TransportType | None:
        """Get preferred transport for message type."""
        # Route different message types through appropriate transports
        transport_preferences = {
            MeshMessageType.DATA_MESSAGE: TransportType.LOCAL_SOCKET,
            MeshMessageType.AGENT_TASK: TransportType.FILE_SYSTEM,
            MeshMessageType.PARAMETER_UPDATE: TransportType.BLUETOOTH_CLASSIC,
            MeshMessageType.GRADIENT_SHARING: TransportType.WIFI_DIRECT,
            MeshMessageType.HEARTBEAT: TransportType.LOCAL_SOCKET,
            MeshMessageType.PEER_DISCOVERY: TransportType.BLUETOOTH_LE,
        }

        return transport_preferences.get(message_type)

    async def discover_fallback_peers(self) -> dict[TransportType, list[str]]:
        """Discover peers via fallback transports."""
        if not self.fallback_manager:
            return {}

        try:
            return await self.fallback_manager.discover_all_peers()
        except Exception as e:
            logger.exception(f"Fallback peer discovery failed: {e}")
            return {}

    def get_fallback_status(self) -> dict[str, Any]:
        """Get fallback transport status."""
        if not self.fallback_manager:
            return {"enabled": False}

        return {"enabled": True, **self.fallback_manager.get_status()}
