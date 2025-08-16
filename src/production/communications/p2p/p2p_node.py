"""P2P Node Implementation with Enhanced Features.

This is the canonical P2P node implementation for AIVillage, consolidating
features from multiple implementations:
- Production-ready Windows compatibility
- Evolution-aware peer coordination
- Enhanced discovery and routing capabilities
- Comprehensive message validation
- Mobile optimization support

Replaces:
- src/core/p2p/p2p_node.py (evolution features merged)
- src/infrastructure/p2p/p2p_node.py (infrastructure features merged)
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any

# Use local encryption instead of libp2p for Windows compatibility
from cryptography.fernet import Fernet
from pydantic import (
    BaseModel,
    ConfigDict,
    FieldValidationInfo,
    ValidationError,
    field_validator,
)

# Optional system monitoring for capability reporting
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node connection status."""

    OFFLINE = "offline"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class PeerCapabilities(Flag):
    """Peer capability flags for protocol support."""

    SUPPORTS_HTX = auto()
    SUPPORTS_BITCHAT = auto()
    SUPPORTS_QUIC = auto()
    SUPPORTS_EVOLUTION = auto()
    SUPPORTS_ENCRYPTION = auto()
    SUPPORTS_COMPRESSION = auto()

    @property
    def supports_htx(self) -> bool:
        return bool(self & PeerCapabilities.SUPPORTS_HTX)

    @property
    def supports_bitchat(self) -> bool:
        return bool(self & PeerCapabilities.SUPPORTS_BITCHAT)

    @property
    def supports_quic(self) -> bool:
        return bool(self & PeerCapabilities.SUPPORTS_QUIC)


@dataclass
class PeerCapabilityInfo:
    """Information about peer capabilities including ALPN."""

    flags: PeerCapabilities = PeerCapabilities.SUPPORTS_ENCRYPTION
    alpn: set[str] = field(default_factory=lambda: {"http/1.1", "h2"})


@dataclass
class PeerInfo:
    """Enhanced information about a peer node with evolution capabilities."""

    peer_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None

    # Evolution-specific attributes
    can_evolve: bool = True
    evolution_capacity: float = 1.0  # 0-1 scale of evolution participation ability
    available_for_evolution: bool = True
    current_evolution_load: float = 0.0  # Current evolution workload (0-1)

    # Resource tracking
    cpu_cores: int = 1
    ram_mb: int = 1024
    battery_percent: int | None = None
    network_type: str = "unknown"
    trust_score: float = 0.5
    bandwidth_kbps: int | None = None

    def is_suitable_for_evolution(self) -> bool:
        """Check if peer is suitable for evolution tasks."""
        return (
            self.can_evolve
            and self.available_for_evolution
            and self.status == NodeStatus.CONNECTED
            and self.current_evolution_load < 0.8
            and self.trust_score > 0.3
        )

    def get_evolution_priority(self) -> float:
        """Calculate evolution priority score for this peer."""
        base_score = self.evolution_capacity * 0.4

        # Adjust for resources
        if self.cpu_cores >= 4:
            base_score += 0.1
        if self.ram_mb >= 4096:
            base_score += 0.1
        if self.battery_percent and self.battery_percent > 50:
            base_score += 0.1

        # Adjust for current load
        base_score -= self.current_evolution_load * 0.3

        # Adjust for latency
        if self.latency_ms and self.latency_ms < 100:
            base_score += 0.1

        # Adjust for trust
        base_score += (self.trust_score - 0.5) * 0.2

        return max(0.0, min(1.0, base_score))


class MessageType(Enum):
    """Enhanced P2P message types with evolution support."""

    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    DATA = "data"
    TENSOR_CHUNK = "tensor_chunk"
    SYNC_REQUEST = "sync_request"
    DISCOVERY = "discovery"
    ERROR = "error"

    # Evolution-specific message types
    EVOLUTION_START = "evolution_start"
    EVOLUTION_PROGRESS = "evolution_progress"
    EVOLUTION_COMPLETE = "evolution_complete"
    EVOLUTION_REQUEST_HELP = "evolution_request_help"
    EVOLUTION_OFFER_HELP = "evolution_offer_help"
    EVOLUTION_CONSENSUS = "evolution_consensus"
    EVOLUTION_METRICS_SHARE = "evolution_metrics_share"

    # Mobile optimization
    CAPABILITY_UPDATE = "capability_update"
    PING = "ping"
    PONG = "pong"


class HandshakePayload(BaseModel):
    """Payload for handshake messages."""

    node_id: str
    capabilities: dict[str, Any]
    timestamp: float
    model_config = ConfigDict(extra="ignore")


class HeartbeatPayload(BaseModel):
    """Payload for heartbeat messages."""

    timestamp: float
    status: str
    model_config = ConfigDict(extra="ignore")


class TensorChunkPayload(BaseModel):
    """Payload for tensor chunk messages."""

    tensor_id: str
    chunk_index: int
    total_chunks: int
    data: str
    checksum: str
    timestamp: float
    is_compressed: bool
    compression_type: str | None = None
    model_config = ConfigDict(extra="ignore")


class DiscoveryPeer(BaseModel):
    """Information about a discovered peer."""

    address: str
    port: int
    model_config = ConfigDict(extra="ignore")


class DiscoveryPayload(BaseModel):
    """Payload for discovery messages."""

    peers: list[DiscoveryPeer]
    model_config = ConfigDict(extra="ignore")


class SyncRequestPayload(BaseModel):
    """Payload for sync request messages."""

    timestamp: float | None = None
    model_config = ConfigDict(extra="ignore")


class ErrorPayload(BaseModel):
    """Payload for error messages."""

    message: str
    code: int | None = None
    model_config = ConfigDict(extra="ignore")


class DataPayload(BaseModel):
    """Payload for generic data messages."""

    model_config = ConfigDict(extra="allow")


PAYLOAD_MODELS: dict[MessageType, type[BaseModel]] = {
    MessageType.HANDSHAKE: HandshakePayload,
    MessageType.HEARTBEAT: HeartbeatPayload,
    MessageType.DATA: DataPayload,
    MessageType.TENSOR_CHUNK: TensorChunkPayload,
    MessageType.SYNC_REQUEST: SyncRequestPayload,
    MessageType.DISCOVERY: DiscoveryPayload,
    MessageType.ERROR: ErrorPayload,
}


class MessageModel(BaseModel):
    """Pydantic model for validating P2P messages."""

    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: dict[str, Any]
    timestamp: float
    message_id: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("payload")
    @classmethod
    def validate_payload(
        cls, v: dict[str, Any], info: FieldValidationInfo
    ) -> dict[str, Any]:
        message_type: MessageType | None = info.data.get("message_type")
        if message_type:
            payload_model = PAYLOAD_MODELS.get(message_type)
            if payload_model:
                payload_model(**v)
        return v


@dataclass
class P2PMessage:
    """P2P message structure."""

    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class P2PNode:
    """P2P Node for decentralized communication with Windows fallback."""

    def __init__(
        self,
        node_id: str | None = None,
        port: int = 8000,
        encryption_key: bytes | None = None,
    ) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.status = NodeStatus.OFFLINE

        # Initialize encryption
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

        # Peer management
        self.peers: dict[str, PeerInfo] = {}
        self.known_addresses: set[str] = set()

        # Message handling
        self.message_handlers: dict[MessageType, Callable] = {}
        self.pending_responses: dict[str, asyncio.Future] = {}

        # Evolution coordination
        self.evolution_handlers: dict[str, Callable] = {}
        self.evolution_results: dict[str, Any] = {}

        # Network components
        self.server: asyncio.Server | None = None
        self.discovery_task: asyncio.Task | None = None
        self.heartbeat_task: asyncio.Task | None = None

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connections_established": 0,
            "discovery_rounds": 0,
            "evolution_messages": 0,
            "evolution_tasks_completed": 0,
        }

        # Configuration
        self.config = {
            "heartbeat_interval": 30.0,  # seconds
            "discovery_interval": 60.0,  # seconds
            "connection_timeout": 10.0,  # seconds
            "max_message_size": 1024 * 1024,  # 1MB
            "max_peers": 100,
        }

        # Register default handlers
        self._register_default_handlers()
        self._register_evolution_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers[MessageType.HANDSHAKE] = self._handle_handshake
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.DISCOVERY] = self._handle_discovery
        self.message_handlers[MessageType.PING] = self._handle_ping
        self.message_handlers[MessageType.PONG] = self._handle_pong
        self.message_handlers[MessageType.CAPABILITY_UPDATE] = (
            self._handle_capability_update
        )

    def _register_evolution_handlers(self) -> None:
        """Register evolution-specific message handlers."""
        self.evolution_handlers.update(
            {
                "EVOLUTION_START": self._handle_evolution_start,
                "EVOLUTION_PROGRESS": self._handle_evolution_progress,
                "EVOLUTION_COMPLETE": self._handle_evolution_complete,
                "EVOLUTION_REQUEST_HELP": self._handle_evolution_request_help,
                "EVOLUTION_OFFER_HELP": self._handle_evolution_offer_help,
                "EVOLUTION_CONSENSUS": self._handle_evolution_consensus,
                "EVOLUTION_METRICS_SHARE": self._handle_evolution_metrics_share,
            }
        )

    async def start(self) -> None:
        """Start the P2P node."""
        try:
            logger.info(f"Starting P2P node {self.node_id} on port {self.port}")

            # Start TCP server
            self.server = await asyncio.start_server(
                self._handle_connection, "0.0.0.0", self.port
            )

            # Start background tasks
            self.discovery_task = asyncio.create_task(self._discovery_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self.status = NodeStatus.CONNECTED
            logger.info(f"P2P node {self.node_id} started successfully")

        except Exception as e:
            logger.exception(f"Failed to start P2P node: {e}")
            self.status = NodeStatus.ERROR
            raise

    async def stop(self) -> None:
        """Stop the P2P node."""
        logger.info(f"Stopping P2P node {self.node_id}")

        self.status = NodeStatus.OFFLINE

        # Cancel background tasks
        if self.discovery_task:
            self.discovery_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Clear peers
        self.peers.clear()

        logger.info(f"P2P node {self.node_id} stopped")

    async def connect_to_peer(self, address: str, port: int) -> bool:
        """Connect to a peer node."""
        try:
            logger.debug(f"Connecting to peer at {address}:{port}")

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=self.config["connection_timeout"],
            )

            # Send handshake
            handshake_msg = P2PMessage(
                message_type=MessageType.HANDSHAKE,
                sender_id=self.node_id,
                receiver_id="",
                payload={
                    "node_id": self.node_id,
                    "capabilities": self._get_capabilities(),
                    "timestamp": time.time(),
                },
            )

            await self._send_message_to_writer(writer, handshake_msg)

            # Wait for response
            response = await self._receive_message_from_reader(reader)

            if response and response.message_type == MessageType.HANDSHAKE:
                peer_id = response.payload.get("node_id")
                if peer_id:
                    peer_info = PeerInfo(
                        peer_id=peer_id,
                        address=address,
                        port=port,
                        status=NodeStatus.CONNECTED,
                        capabilities=response.payload.get("capabilities", {}),
                    )
                    self.peers[peer_id] = peer_info
                    self.stats["connections_established"] += 1

                    logger.info(f"Successfully connected to peer {peer_id}")
                    writer.close()
                    await writer.wait_closed()
                    return True

            writer.close()
            await writer.wait_closed()
            return False

        except Exception as e:
            logger.exception(f"Failed to connect to peer {address}:{port}: {e}")
            return False

    async def send_message(
        self, peer_id: str, message_type: MessageType, payload: dict[str, Any]
    ) -> bool:
        """Send a message to a specific peer."""
        if peer_id not in self.peers:
            logger.warning(f"Peer {peer_id} not found")
            return False

        peer = self.peers[peer_id]
        if peer.status != NodeStatus.CONNECTED:
            logger.warning(f"Peer {peer_id} not connected")
            return False

        try:
            message = P2PMessage(
                message_type=message_type,
                sender_id=self.node_id,
                receiver_id=peer_id,
                payload=payload,
            )

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.address, peer.port),
                timeout=self.config["connection_timeout"],
            )

            await self._send_message_to_writer(writer, message)
            self.stats["messages_sent"] += 1

            writer.close()
            await writer.wait_closed()
            return True

        except Exception as e:
            logger.exception(f"Failed to send message to peer {peer_id}: {e}")
            peer.status = NodeStatus.ERROR
            return False

    async def broadcast_message(
        self,
        message_type: MessageType,
        payload: dict[str, Any],
        exclude_peers: set[str] | None = None,
    ) -> int:
        """Broadcast a message to all connected peers."""
        exclude_peers = exclude_peers or set()
        successful_sends = 0

        for peer_id in self.peers:
            if peer_id not in exclude_peers:
                if await self.send_message(peer_id, message_type, payload):
                    successful_sends += 1

        return successful_sends

    async def query_peer(
        self,
        peer_id: str,
        query_type: str,
        query_data: dict[str, Any],
        timeout: float = 10.0,
    ) -> dict[str, Any] | None:
        """Send a query to a peer and wait for response."""
        message_id = str(uuid.uuid4())

        # Set up response future
        response_future = asyncio.Future()
        self.pending_responses[message_id] = response_future

        try:
            # Send query
            payload = {
                "query_type": query_type,
                "query_data": query_data,
                "message_id": message_id,
                "expect_response": True,
            }

            success = await self.send_message(peer_id, MessageType.DATA, payload)

            if not success:
                return None

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except TimeoutError:
            logger.warning(f"Query to peer {peer_id} timed out")
            return None
        except Exception as e:
            logger.exception(f"Query to peer {peer_id} failed: {e}")
            return None
        finally:
            self.pending_responses.pop(message_id, None)

    def add_known_address(self, address: str, port: int) -> None:
        """Add a known peer address for discovery."""
        self.known_addresses.add(f"{address}:{port}")

    def register_handler(
        self, message_type: MessageType, handler: Callable[[P2PMessage], Any]
    ) -> None:
        """Register a custom message handler."""
        self.message_handlers[message_type] = handler

    def get_peer_info(self, peer_id: str) -> PeerInfo | None:
        """Get information about a specific peer."""
        return self.peers.get(peer_id)

    def get_connected_peers(self) -> list[PeerInfo]:
        """Get list of all connected peers."""
        return [
            peer for peer in self.peers.values() if peer.status == NodeStatus.CONNECTED
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get enhanced node statistics including evolution metrics."""
        return {
            **self.stats,
            "node_id": self.node_id,
            "status": self.status.value,
            "connected_peers": len(self.get_connected_peers()),
            "total_peers": len(self.peers),
            "uptime": time.time() - self.stats.get("start_time", time.time()),
            "suitable_evolution_peers": len(self.get_suitable_evolution_peers()),
            "evolution_results": len(self.evolution_results),
            "average_peer_latency": sum(p.latency_ms or 0 for p in self.peers.values())
            / len(self.peers)
            if self.peers
            else 0,
        }

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming connection."""
        peer_address = writer.get_extra_info("peername")
        logger.debug(f"Incoming connection from {peer_address}")

        try:
            message = await self._receive_message_from_reader(reader)

            if message:
                await self._process_message(message, writer)

        except Exception as e:
            logger.exception(f"Error handling connection from {peer_address}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _send_message_to_writer(
        self, writer: asyncio.StreamWriter, message: P2PMessage
    ) -> None:
        """Send a message through a writer."""
        # Serialize and encrypt message
        message_data = json.dumps(
            {
                "message_type": message.message_type.value,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "payload": message.payload,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
            }
        ).encode()

        encrypted_data = self.cipher.encrypt(message_data)

        # Send length prefix + encrypted data
        length = len(encrypted_data)
        writer.write(length.to_bytes(4, byteorder="big"))
        writer.write(encrypted_data)
        await writer.drain()

        self.stats["bytes_sent"] += 4 + length

    def _validate_message_dict(self, message_dict: dict[str, Any]) -> P2PMessage | None:
        """Validate and construct a ``P2PMessage`` from a dictionary."""
        try:
            model = MessageModel(**message_dict)
        except ValidationError as e:
            logger.warning("Invalid message received: %s", e)
            return None
        return P2PMessage(
            message_type=model.message_type,
            sender_id=model.sender_id,
            receiver_id=model.receiver_id,
            payload=model.payload,
            timestamp=model.timestamp,
            message_id=model.message_id,
        )

    async def _receive_message_from_reader(
        self, reader: asyncio.StreamReader
    ) -> P2PMessage | None:
        """Receive a message from a reader."""
        try:
            # Read length prefix
            length_data = await reader.readexactly(4)
            length = int.from_bytes(length_data, byteorder="big")

            if length > self.config["max_message_size"]:
                logger.warning(f"Message too large: {length} bytes")
                return None

            # Read encrypted message
            encrypted_data = await reader.readexactly(length)

            # Decrypt and deserialize
            message_data = self.cipher.decrypt(encrypted_data)
            message_dict = json.loads(message_data.decode())

            required_keys = {
                "message_type",
                "sender_id",
                "receiver_id",
                "payload",
                "timestamp",
                "message_id",
            }
            message_keys = set(message_dict)
            missing = required_keys - message_keys
            unexpected = message_keys - required_keys
            if missing or unexpected:
                logger.warning(
                    "Invalid message schema. Missing: %s, unexpected: %s",
                    missing,
                    unexpected,
                )
                return None

            self.stats["bytes_received"] += 4 + length
            self.stats["messages_received"] += 1

            return self._validate_message_dict(message_dict)

        except Exception as e:
            logger.exception(f"Failed to receive message: {e}")
            return None

    async def _process_message(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Process an incoming message."""
        logger.debug(
            f"Processing {message.message_type.value} from {message.sender_id}"
        )

        # Handle response to pending query
        if message.payload.get("response_to"):
            response_id = message.payload["response_to"]
            if response_id in self.pending_responses:
                self.pending_responses[response_id].set_result(message.payload)
                return

        # Handle evolution messages
        msg_type = message.payload.get("type")
        if msg_type in self.evolution_handlers:
            try:
                await self.evolution_handlers[msg_type](message.payload, writer)
                self.stats["evolution_messages"] += 1
                return
            except Exception as e:
                logger.exception(f"Error in evolution handler for {msg_type}: {e}")

        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message, writer)
            except Exception as e:
                logger.exception(
                    f"Error in handler for {message.message_type.value}: {e}"
                )
        else:
            logger.warning(f"No handler for message type {message.message_type.value}")

    async def _handle_handshake(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle handshake message."""
        sender_id = message.payload.get("node_id")

        if sender_id and sender_id != self.node_id:
            # Create peer info
            peer = PeerInfo(
                peer_id=sender_id,
                address="",  # Will be updated
                port=0,
                status=NodeStatus.CONNECTED,
                capabilities=message.payload.get("capabilities", {}),
            )
            self.peers[sender_id] = peer

            # Send handshake response
            if writer:
                response = P2PMessage(
                    message_type=MessageType.HANDSHAKE,
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    payload={
                        "node_id": self.node_id,
                        "capabilities": self._get_capabilities(),
                        "timestamp": time.time(),
                    },
                )
                await self._send_message_to_writer(writer, response)

            logger.info(f"Handshake completed with peer {sender_id}")

    async def _handle_heartbeat(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle heartbeat message."""
        sender_id = message.sender_id

        if sender_id in self.peers:
            self.peers[sender_id].last_seen = time.time()
            self.peers[sender_id].status = NodeStatus.CONNECTED

    async def _handle_discovery(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle peer discovery message."""
        peer_list = message.payload.get("peers", [])

        for peer_info in peer_list:
            address = peer_info.get("address")
            port = peer_info.get("port")

            if address and port:
                self.add_known_address(address, port)

    async def _handle_ping(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle ping message with pong response."""
        if writer:
            response = P2PMessage(
                message_type=MessageType.PONG,
                sender_id=self.node_id,
                receiver_id=message.sender_id,
                payload={
                    "timestamp": time.time(),
                    "original_timestamp": message.payload.get("timestamp"),
                },
            )
            await self._send_message_to_writer(writer, response)

    async def _handle_pong(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle pong message and update latency."""
        sender_id = message.sender_id
        if sender_id in self.peers:
            original_timestamp = message.payload.get("original_timestamp")
            if original_timestamp:
                latency = (time.time() - original_timestamp) * 1000  # Convert to ms
                self.peers[sender_id].latency_ms = latency

    async def _handle_capability_update(
        self, message: P2PMessage, writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle capability update from peer."""
        sender_id = message.sender_id
        if sender_id in self.peers:
            capabilities = message.payload.get("capabilities", {})
            self.peers[sender_id].capabilities.update(capabilities)

            # Update evolution-specific attributes
            peer = self.peers[sender_id]
            peer.can_evolve = capabilities.get("can_evolve", peer.can_evolve)
            peer.evolution_capacity = capabilities.get(
                "evolution_capacity", peer.evolution_capacity
            )
            peer.available_for_evolution = capabilities.get(
                "available_for_evolution", peer.available_for_evolution
            )
            peer.cpu_cores = capabilities.get("cpu_cores", peer.cpu_cores)
            peer.ram_mb = capabilities.get("ram_mb", peer.ram_mb)
            peer.battery_percent = capabilities.get(
                "battery_percent", peer.battery_percent
            )
            peer.network_type = capabilities.get("network_type", peer.network_type)

    # Evolution handler methods
    async def _handle_evolution_start(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle evolution start notification."""
        sender_id = message.get("sender_id")
        evolution_type = message.get("evolution_type", "unknown")

        logger.info(f"Peer {sender_id} started {evolution_type} evolution")

        # Update peer evolution status
        if sender_id and sender_id in self.peers:
            self.peers[sender_id].current_evolution_load = 1.0

        self.stats["evolution_messages"] += 1

    async def _handle_evolution_progress(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle evolution progress update."""
        sender_id = message.get("sender_id")
        progress = message.get("progress", 0.0)

        if sender_id and sender_id in self.peers:
            self.peers[sender_id].current_evolution_load = progress

        self.stats["evolution_messages"] += 1

    async def _handle_evolution_complete(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle evolution completion notification."""
        sender_id = message.get("sender_id")

        logger.info(f"Peer {sender_id} completed evolution")

        # Reset peer evolution status
        if sender_id and sender_id in self.peers:
            self.peers[sender_id].current_evolution_load = 0.0

        # Store results
        self.evolution_results[sender_id] = {
            "completed_at": time.time(),
            "type": message.get("evolution_type"),
            "results": message.get("results", {}),
        }

        self.stats["evolution_tasks_completed"] += 1

    async def _handle_evolution_request_help(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle request for evolution assistance."""
        sender_id = message.get("sender_id")

        # Check if we can help
        if self.status == NodeStatus.CONNECTED:
            suitable_peers = self.get_suitable_evolution_peers(
                exclude=[sender_id] if sender_id else []
            )

            if suitable_peers:
                # Offer help if we have capacity
                response_data = {
                    "type": "EVOLUTION_OFFER_HELP",
                    "sender_id": self.node_id,
                    "available_capacity": 1.0
                    - sum(p.current_evolution_load for p in self.peers.values())
                    / len(self.peers)
                    if self.peers
                    else 1.0,
                    "capabilities": self._get_capabilities(),
                }

                if sender_id in self.peers and writer:
                    response = P2PMessage(
                        message_type=MessageType.EVOLUTION_OFFER_HELP,
                        sender_id=self.node_id,
                        receiver_id=sender_id,
                        payload=response_data,
                    )
                    await self._send_message_to_writer(writer, response)

    async def _handle_evolution_offer_help(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle offer of evolution assistance."""
        sender_id = message.get("sender_id")
        capacity = message.get("available_capacity", 0.0)

        logger.info(
            f"Peer {sender_id} offered evolution help with {capacity:.2f} capacity"
        )

    async def _handle_evolution_consensus(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle evolution consensus message."""
        sender_id = message.get("sender_id")
        message.get("consensus", {})

        logger.info(f"Received evolution consensus from {sender_id}")

    async def _handle_evolution_metrics_share(
        self, message: dict[str, Any], writer: asyncio.StreamWriter | None = None
    ) -> None:
        """Handle shared evolution metrics."""
        sender_id = message.get("sender_id")
        metrics = message.get("metrics", {})

        logger.debug(f"Received evolution metrics from {sender_id}: {metrics}")

    async def _discovery_loop(self) -> None:
        """Background task for peer discovery."""
        while self.status == NodeStatus.CONNECTED:
            try:
                # Try to connect to known addresses
                for addr_port in list(self.known_addresses):
                    if len(self.peers) >= self.config["max_peers"]:
                        break

                    address, port = addr_port.split(":")
                    port = int(port)

                    # Skip if already connected
                    if any(
                        p.address == address and p.port == port
                        for p in self.peers.values()
                    ):
                        continue

                    await self.connect_to_peer(address, port)

                self.stats["discovery_rounds"] += 1
                await asyncio.sleep(self.config["discovery_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in discovery loop: {e}")
                await asyncio.sleep(10)  # Back off on error

    async def _heartbeat_loop(self) -> None:
        """Background task for sending heartbeats."""
        while self.status == NodeStatus.CONNECTED:
            try:
                # Send heartbeat to all connected peers
                heartbeat_payload = {
                    "timestamp": time.time(),
                    "status": self.status.value,
                }

                await self.broadcast_message(MessageType.HEARTBEAT, heartbeat_payload)

                # Check for stale peers
                current_time = time.time()
                stale_peers = []

                for peer_id, peer in self.peers.items():
                    if (
                        current_time - peer.last_seen
                        > self.config["heartbeat_interval"] * 3
                    ):
                        stale_peers.append(peer_id)

                # Remove stale peers
                for peer_id in stale_peers:
                    del self.peers[peer_id]
                    logger.info(f"Removed stale peer {peer_id}")

                await asyncio.sleep(self.config["heartbeat_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)

    def _get_capabilities(self) -> dict[str, Any]:
        """Get enhanced node capabilities including evolution features."""
        if PSUTIL_AVAILABLE:
            try:
                # Get system info for capability reporting
                cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory()
                battery = psutil.sensors_battery()
            except Exception:
                # Fallback values if psutil fails
                cpu_count = 1
                memory = type("Memory", (), {"total": 1024 * 1024 * 1024})()
                battery = None
        else:
            # Fallback values when psutil not available
            cpu_count = 1
            memory = type("Memory", (), {"total": 1024 * 1024 * 1024})()
            battery = None

        return {
            "version": "2.0.0",  # Updated version for consolidated implementation
            "protocols": ["tensor_streaming", "mesh_routing", "evolution_coordination"],
            "max_message_size": self.config["max_message_size"],
            "encryption": "fernet",
            # Evolution capabilities
            "can_evolve": True,
            "evolution_capacity": 1.0,
            "available_for_evolution": True,
            # Resource information
            "cpu_cores": cpu_count or 1,
            "ram_mb": int(
                (memory.total if memory else 1024 * 1024 * 1024) / (1024 * 1024)
            ),
            "battery_percent": battery.percent if battery else None,
            "network_type": "ethernet",  # Default assumption
            "trust_score": 0.8,  # Start with good trust for own node
        }

    def get_suitable_evolution_peers(
        self, exclude: list[str] | None = None
    ) -> list[PeerInfo]:
        """Get peers suitable for evolution tasks."""
        exclude = exclude or []
        suitable_peers = []

        for peer_id, peer in self.peers.items():
            if peer_id not in exclude and peer.is_suitable_for_evolution():
                suitable_peers.append(peer)

        # Sort by evolution priority
        suitable_peers.sort(key=lambda p: p.get_evolution_priority(), reverse=True)
        return suitable_peers

    async def broadcast_evolution_event(
        self, event_type: str, data: dict[str, Any]
    ) -> int:
        """Broadcast an evolution event to all suitable peers."""
        payload = {
            "type": event_type,
            "sender_id": self.node_id,
            "timestamp": time.time(),
            **data,
        }

        suitable_peers = self.get_suitable_evolution_peers()
        successful_sends = 0

        for peer in suitable_peers:
            if await self.send_message(peer.peer_id, MessageType.DATA, payload):
                successful_sends += 1

        self.stats["evolution_messages"] += successful_sends
        return successful_sends

    async def request_evolution_assistance(
        self, evolution_type: str, requirements: dict[str, Any]
    ) -> list[str]:
        """Request assistance from suitable peers for evolution task."""
        payload = {
            "evolution_type": evolution_type,
            "requirements": requirements,
            "timestamp": time.time(),
        }

        helpers = []
        suitable_peers = self.get_suitable_evolution_peers()

        for peer in suitable_peers[:5]:  # Limit to top 5 candidates
            if await self.send_message(
                peer.peer_id, MessageType.EVOLUTION_REQUEST_HELP, payload
            ):
                helpers.append(peer.peer_id)

        return helpers
