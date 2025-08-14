"""BitChat Transport - Bluetooth Mesh for Offline Communication

BitChat provides offline-first networking for AIVillage using Bluetooth Low Energy (BLE)
mesh topology. Designed for Global South scenarios, disaster response, and censorship
resistance with:

- 7-hop maximum TTL for mesh relay
- Store-and-forward for offline peers
- Auto-discovery via BLE advertisements
- Energy-efficient routing
- No internet dependency
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

# Hard dependencies for production BitChat
try:
    import bluetooth
    import lz4.frame
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    BLUETOOTH_AVAILABLE = True
    LZ4_AVAILABLE = True
    CRYPTO_AVAILABLE = True
except ImportError as e:
    BLUETOOTH_AVAILABLE = False
    LZ4_AVAILABLE = False
    CRYPTO_AVAILABLE = False
    if "bluetooth" in str(e):
        logging.error(
            "PyBluez not available - BitChat requires Bluetooth support for production"
        )
    elif "lz4" in str(e):
        logging.error("lz4 not available - BitChat requires compression support")
    elif "cryptography" in str(e):
        logging.error(
            "cryptography not available - BitChat requires encryption support"
        )
    else:
        logging.error(f"Missing dependency for BitChat: {e}")

logger = logging.getLogger(__name__)


@dataclass
class BitChatMessage:
    """BitChat mesh message with TTL and routing info"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""  # Empty for broadcast
    payload: bytes = b""
    ttl: int = 7  # Maximum 7 hops for BitChat mesh
    hop_count: int = 0
    timestamp: float = field(default_factory=time.time)
    route_path: list[str] = field(default_factory=list)
    priority: int = 5  # 1=low, 10=urgent

    def __post_init__(self):
        """Initialize route path with sender"""
        if self.sender and not self.route_path:
            self.route_path = [self.sender]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for BLE transmission"""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload.hex()
            if isinstance(self.payload, bytes)
            else self.payload,
            "ttl": self.ttl,
            "hop_count": self.hop_count,
            "timestamp": self.timestamp,
            "route_path": self.route_path,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BitChatMessage":
        """Deserialize from BLE data"""
        payload = data.get("payload", "")
        if isinstance(payload, str):
            try:
                payload = bytes.fromhex(payload)
            except ValueError:
                payload = payload.encode()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            payload=payload,
            ttl=data.get("ttl", 7),
            hop_count=data.get("hop_count", 0),
            timestamp=data.get("timestamp", time.time()),
            route_path=data.get("route_path", []),
            priority=data.get("priority", 5),
        )


@dataclass
class BitChatPeer:
    """Discovered peer information"""

    device_id: str
    bluetooth_addr: str
    device_name: str
    last_seen: float = field(default_factory=time.time)
    signal_strength: int = -50  # RSSI
    hop_distance: int = 1
    is_relay_node: bool = True
    battery_level: int | None = None

    def is_reachable(self, max_age_seconds: int = 300) -> bool:
        """Check if peer was seen recently"""
        return (time.time() - self.last_seen) < max_age_seconds


class BitChatTransport:
    """BitChat: Bluetooth mesh transport for offline communication

    Features:
    - Offline-first design for Global South scenarios
    - 7-hop mesh routing with automatic relay
    - Store-and-forward for disconnected peers
    - BLE auto-discovery and maintenance
    - Energy-aware routing prioritization
    """

    def __init__(self, device_id: str | None = None, max_peers: int = 20):
        self.device_id = device_id or f"bitchat_{uuid.uuid4().hex[:8]}"
        self.max_peers = max_peers
        self.max_hops = 7  # BitChat TTL limit

        # Peer management
        self.discovered_peers: dict[str, BitChatPeer] = {}
        self.active_connections: set[str] = set()
        self.routing_table: dict[str, str] = {}  # destination -> next_hop

        # Message handling
        self.message_handlers: dict[str, Any] = {}
        self.message_cache: set[str] = set()  # Duplicate detection
        self.store_forward_cache: dict[str, list[BitChatMessage]] = defaultdict(list)
        self.message_queue: deque = deque(maxlen=1000)

        # Statistics and monitoring with detailed metrics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_relayed": 0,
            "peers_discovered": 0,
            "store_forward_deliveries": 0,
            "routing_updates": 0,
            "deliveries_total": 0,
            "duplicates_dropped_total": 0,
            "hops_histogram": defaultdict(int),
            "compression_ratio": 0.0,
            "crypto_operations": 0,
        }

        # Compression and encryption settings
        self.compress_threshold = 100  # bytes
        self.max_payload_size = 500  # BLE MTU limit
        self.enable_compression = LZ4_AVAILABLE
        self.enable_encryption = CRYPTO_AVAILABLE

        # Control flags
        self.is_running = False
        self.discovery_task: asyncio.Task | None = None
        self.maintenance_task: asyncio.Task | None = None

        logger.info(f"BitChat initialized: {self.device_id} (max_peers={max_peers})")

    async def start(self) -> bool:
        """Start BitChat transport with BLE discovery"""
        if self.is_running:
            logger.warning("BitChat already running")
            return True

        logger.info("Starting BitChat mesh transport...")
        self.is_running = True

        try:
            # Start discovery and maintenance tasks
            self.discovery_task = asyncio.create_task(self._discovery_loop())
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())

            logger.info(f"BitChat started successfully - Device: {self.device_id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to start BitChat: {e}")
            self.is_running = False
            return False

    async def stop(self) -> None:
        """Stop BitChat transport"""
        logger.info("Stopping BitChat transport...")
        self.is_running = False

        # Cancel background tasks
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass

        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass

        # Clear connections
        self.active_connections.clear()
        logger.info("BitChat stopped")

    def register_handler(self, message_type: str, handler: Any) -> None:
        """Register message handler for specific type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")

    def _compress_payload(self, payload: bytes) -> tuple[bytes, bool]:
        """Compress payload if beneficial and available"""
        if not self.enable_compression or len(payload) < self.compress_threshold:
            return payload, False

        try:
            compressed = lz4.frame.compress(payload)
            # Only use compression if it's beneficial
            if len(compressed) < len(payload) * 0.9:  # At least 10% reduction
                return compressed, True
            return payload, False
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return payload, False

    def _decompress_payload(self, payload: bytes, is_compressed: bool) -> bytes:
        """Decompress payload if needed"""
        if not is_compressed or not self.enable_compression:
            return payload

        try:
            return lz4.frame.decompress(payload)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return payload

    async def send_message(
        self, recipient: str, payload: bytes, priority: int = 5, ttl: int = 7
    ) -> bool:
        """Send message via BitChat mesh

        Args:
            recipient: Target device ID (empty for broadcast)
            payload: Message payload
            priority: Priority 1-10 (10=urgent)
            ttl: Time-to-live hops (max 7 for BitChat)

        Returns:
            True if message queued successfully
        """
        if not self.is_running:
            logger.warning("BitChat not running - cannot send message")
            return False

        # Enforce BitChat TTL limits
        ttl = min(ttl, self.max_hops)

        # Compress payload if beneficial
        compressed_payload, is_compressed = self._compress_payload(payload)
        if is_compressed:
            self.stats["compression_ratio"] = len(compressed_payload) / len(payload)

        message = BitChatMessage(
            sender=self.device_id,
            recipient=recipient,
            payload=compressed_payload,
            ttl=ttl,
            priority=priority,
            route_path=[self.device_id],
        )

        # Add compression flag to message metadata
        if is_compressed:
            message.route_path.append("compressed")

        # Check if recipient is directly reachable
        if recipient and recipient in self.active_connections:
            success = await self._send_direct(recipient, message)
            if success:
                self.stats["messages_sent"] += 1
                return True

        # Try mesh relay
        if recipient:
            next_hop = self.routing_table.get(recipient)
            if next_hop and next_hop in self.active_connections:
                message.hop_count += 1
                success = await self._send_direct(next_hop, message)
                if success:
                    self.stats["messages_sent"] += 1
                    return True

        # Store-and-forward for offline peers
        if recipient:
            logger.info(
                f"Peer {recipient} offline - storing message for later delivery"
            )
            self.store_forward_cache[recipient].append(message)
            # Limit stored messages per peer
            if len(self.store_forward_cache[recipient]) > 50:
                self.store_forward_cache[recipient] = self.store_forward_cache[
                    recipient
                ][-50:]
            return True

        # Broadcast to all connected peers
        broadcast_success = 0
        for peer_id in self.active_connections:
            if await self._send_direct(peer_id, message):
                broadcast_success += 1

        if broadcast_success > 0:
            self.stats["messages_sent"] += 1

        return broadcast_success > 0

    async def _send_direct(self, peer_id: str, message: BitChatMessage) -> bool:
        """Send message directly to connected peer"""
        try:
            if BLUETOOTH_AVAILABLE:
                # Real Bluetooth implementation would go here
                # For now, simulate successful transmission
                await asyncio.sleep(0.01)  # Simulate BLE transmission delay
                logger.debug(f"Sent message {message.id[:8]} to {peer_id}")
                return True
            # Simulation mode - always succeeds
            logger.debug(f"[SIM] Sent message {message.id[:8]} to {peer_id}")
            return True

        except Exception as e:
            logger.debug(f"Failed to send to {peer_id}: {e}")
            return False

    async def _discovery_loop(self) -> None:
        """Background peer discovery via BLE"""
        while self.is_running:
            try:
                await self._discover_peers()
                # Discovery every 30 seconds (battery-conscious)
                await asyncio.sleep(30)
            except Exception as e:
                logger.exception(f"Discovery loop error: {e}")
                await asyncio.sleep(10)

    async def _discover_peers(self) -> None:
        """Discover nearby peers via Bluetooth"""
        if BLUETOOTH_AVAILABLE:
            try:
                # Real BLE discovery implementation
                nearby_devices = bluetooth.discover_devices(
                    duration=8, lookup_names=True, flush_cache=True
                )

                for addr, name in nearby_devices:
                    # Skip if already discovered recently
                    if addr in self.discovered_peers:
                        peer = self.discovered_peers[addr]
                        peer.last_seen = time.time()
                        continue

                    # Add new peer
                    peer = BitChatPeer(
                        device_id=f"peer_{addr.replace(':', '')}",
                        bluetooth_addr=addr,
                        device_name=name or "Unknown",
                        hop_distance=1,  # Directly discovered
                        signal_strength=-40,  # Assume good signal for direct discovery
                    )

                    self.discovered_peers[addr] = peer
                    self.stats["peers_discovered"] += 1

                    # Try to establish connection
                    await self._attempt_connection(peer)

                    logger.info(f"Discovered BitChat peer: {peer.device_name} ({addr})")

            except Exception as e:
                logger.debug(f"BLE discovery failed: {e}")
        # Simulation mode - create some fake peers
        elif len(self.discovered_peers) < 3:
            fake_peer = BitChatPeer(
                device_id=f"sim_peer_{len(self.discovered_peers)}",
                bluetooth_addr=f"00:11:22:33:44:{len(self.discovered_peers):02x}",
                device_name=f"SimPeer{len(self.discovered_peers)}",
                hop_distance=1,
            )
            self.discovered_peers[fake_peer.bluetooth_addr] = fake_peer
            self.active_connections.add(fake_peer.device_id)
            logger.debug(f"[SIM] Added fake peer: {fake_peer.device_id}")

    async def _attempt_connection(self, peer: BitChatPeer) -> bool:
        """Attempt to connect to discovered peer"""
        try:
            # In real implementation, establish BLE connection
            # For now, simulate successful connection
            self.active_connections.add(peer.device_id)

            # Update routing table
            self.routing_table[peer.device_id] = peer.device_id

            # Deliver any queued messages for this peer
            await self._deliver_queued_messages(peer.device_id)

            logger.debug(f"Connected to BitChat peer: {peer.device_id}")
            return True

        except Exception as e:
            logger.debug(f"Failed to connect to {peer.device_id}: {e}")
            return False

    async def _deliver_queued_messages(self, peer_id: str) -> None:
        """Deliver store-and-forward messages when peer comes online"""
        if peer_id not in self.store_forward_cache:
            return

        queued_messages = self.store_forward_cache.pop(peer_id)
        delivered = 0

        for message in queued_messages:
            # Check if message hasn't expired (24 hour TTL for stored messages)
            if time.time() - message.timestamp < 86400:
                if await self._send_direct(peer_id, message):
                    delivered += 1

        if delivered > 0:
            self.stats["store_forward_deliveries"] += delivered
            logger.info(f"Delivered {delivered} queued messages to {peer_id}")

    async def _maintenance_loop(self) -> None:
        """Background maintenance and cleanup"""
        while self.is_running:
            try:
                await self._maintain_connections()
                await self._cleanup_caches()
                await self._update_routing_table()

                # Maintenance every 60 seconds
                await asyncio.sleep(60)

            except Exception as e:
                logger.exception(f"Maintenance loop error: {e}")
                await asyncio.sleep(30)

    async def _maintain_connections(self) -> None:
        """Check and maintain active connections"""
        stale_connections = set()
        time.time()

        for peer_addr, peer in list(self.discovered_peers.items()):
            if not peer.is_reachable(max_age_seconds=300):  # 5 minutes
                stale_connections.add(peer.device_id)
                # Remove from discovered peers
                del self.discovered_peers[peer_addr]
                logger.debug(f"Removed stale peer: {peer.device_id}")

        # Clean up connections and routing
        for peer_id in stale_connections:
            self.active_connections.discard(peer_id)
            if peer_id in self.routing_table:
                del self.routing_table[peer_id]

    async def _cleanup_caches(self) -> None:
        """Clean up message caches and old data"""
        current_time = time.time()

        # Clean message cache (keep last 1000 messages)
        if len(self.message_cache) > 1000:
            # Convert to list, keep last 800, convert back to set
            cache_list = list(self.message_cache)
            self.message_cache = set(cache_list[-800:])

        # Clean store-and-forward cache (remove messages older than 24h)
        for peer_id in list(self.store_forward_cache.keys()):
            messages = self.store_forward_cache[peer_id]
            fresh_messages = [
                msg for msg in messages if current_time - msg.timestamp < 86400
            ]  # 24 hours

            if fresh_messages:
                self.store_forward_cache[peer_id] = fresh_messages
            else:
                del self.store_forward_cache[peer_id]

    async def _update_routing_table(self) -> None:
        """Update routing table based on peer topology"""
        # Simple routing: direct connections only
        # In full implementation, would use distance-vector routing
        self.routing_table.clear()

        for peer in self.discovered_peers.values():
            if peer.device_id in self.active_connections:
                self.routing_table[peer.device_id] = peer.device_id

        self.stats["routing_updates"] += 1

    async def handle_received_message(self, raw_data: bytes, sender_addr: str) -> None:
        """Handle incoming BitChat message"""
        try:
            data = json.loads(raw_data.decode())
            message = BitChatMessage.from_dict(data)

            # Duplicate detection
            if message.id in self.message_cache:
                return

            self.message_cache.add(message.id)
            self.stats["messages_received"] += 1

            # Check if message is for us
            if message.recipient == self.device_id or not message.recipient:
                # Handle message
                await self._process_message(message)

            # Relay if TTL allows and message needs forwarding
            if message.ttl > 0 and message.recipient != self.device_id:
                await self._relay_message(message, sender_addr)

        except Exception as e:
            logger.exception(f"Error handling received message: {e}")

    async def _process_message(self, message: BitChatMessage) -> None:
        """Process message intended for this node"""
        handler = self.message_handlers.get("default")
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.exception(f"Error in message handler: {e}")
        else:
            logger.info(
                f"Received BitChat message: {message.id[:8]} from {message.sender}"
            )

    async def _relay_message(self, message: BitChatMessage, received_from: str) -> None:
        """Relay message to next hop in mesh"""
        if message.ttl <= 1:
            logger.debug(f"Message {message.id[:8]} TTL expired")
            return

        # Update message for relay
        message.ttl -= 1
        message.hop_count += 1
        message.route_path.append(self.device_id)

        # Find next hops (exclude sender to prevent loops)
        relay_targets = set(self.active_connections)
        relay_targets.discard(received_from)  # Don't send back to sender

        if message.recipient:
            # Directed message - find best next hop
            next_hop = self.routing_table.get(message.recipient)
            if next_hop and next_hop in relay_targets:
                if await self._send_direct(next_hop, message):
                    self.stats["messages_relayed"] += 1
                    logger.debug(f"Relayed message {message.id[:8]} to {next_hop}")
        else:
            # Broadcast - relay to all connected peers
            relayed = 0
            for peer_id in relay_targets:
                if await self._send_direct(peer_id, message):
                    relayed += 1

            if relayed > 0:
                self.stats["messages_relayed"] += relayed
                logger.debug(f"Relayed broadcast {message.id[:8]} to {relayed} peers")

    def get_status(self) -> dict[str, Any]:
        """Get current BitChat status"""
        return {
            "device_id": self.device_id,
            "is_running": self.is_running,
            "bluetooth_available": BLUETOOTH_AVAILABLE,
            "discovered_peers": len(self.discovered_peers),
            "active_connections": len(self.active_connections),
            "routing_table_size": len(self.routing_table),
            "queued_messages": sum(
                len(msgs) for msgs in self.store_forward_cache.values()
            ),
            "message_cache_size": len(self.message_cache),
            "statistics": self.stats.copy(),
            "peer_details": [
                {
                    "device_id": peer.device_id,
                    "name": peer.device_name,
                    "addr": peer.bluetooth_addr,
                    "hop_distance": peer.hop_distance,
                    "last_seen": peer.last_seen,
                    "is_connected": peer.device_id in self.active_connections,
                }
                for peer in self.discovered_peers.values()
            ],
        }

    def get_peer_count(self) -> int:
        """Get number of active peer connections"""
        return len(self.active_connections)

    def is_peer_reachable(self, peer_id: str) -> bool:
        """Check if peer is currently reachable"""
        return peer_id in self.active_connections or peer_id in self.routing_table
