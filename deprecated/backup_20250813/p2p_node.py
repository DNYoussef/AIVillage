"""DEPRECATED: P2P node implementation with discovery and routing.

This module has been consolidated into src/production/communications/p2p/p2p_node.py

All infrastructure features from this module have been merged into the production version.

Please update your imports to use:
  from src.production.communications.p2p.p2p_node import P2PNode, PeerInfo, NodeStatus

This file will be removed in a future version.
"""

# Import from the consolidated production implementation
import warnings

from src.production.communications.p2p.p2p_node_deprecated import *

warnings.warn(
    "src.infrastructure.p2p.p2p_node is deprecated and has been consolidated. "
    "Use src.production.communications.p2p.p2p_node instead.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import socket
import time
from typing import Any
import uuid

# For now, we'll implement a basic P2P system without libp2p dependency
# This can be upgraded to full libp2p later when dependencies are resolved

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a P2P node."""

    STARTING = "starting"
    ACTIVE = "active"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class PeerCapabilities:
    """Track peer device capabilities."""

    device_id: str
    cpu_cores: int
    ram_mb: int
    battery_percent: int | None = None
    network_type: str = "unknown"  # "bluetooth", "wifi", "cellular", "ethernet"
    trust_score: float = 0.5
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    bandwidth_kbps: int | None = None


class P2PNode:
    """Enhanced P2P node with capability tracking."""

    def __init__(self, node_id: str | None = None, listen_port: int = 0) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.listen_port = listen_port
        self.status = NodeStatus.STARTING

        # Networking
        self.server: asyncio.Server | None = None
        self.connections: dict[str, asyncio.StreamWriter] = {}
        self.peer_registry: dict[str, PeerCapabilities] = {}

        # Message handling
        self.message_handlers: dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_futures: dict[str, asyncio.Future] = {}

        # Discovery
        self.discovery_interval = 30  # seconds
        self.heartbeat_interval = 10  # seconds
        self.known_peers: set[tuple[str, int]] = set()  # (host, port) pairs

        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

        # Register default handlers
        self._register_default_handlers()

    async def start(self, listen_addrs: list[str] | None = None) -> None:
        """Initialize node with discovery."""
        try:
            # Start TCP server
            self.server = await asyncio.start_server(self._handle_connection, "0.0.0.0", self.listen_port)

            # Get actual port if random was assigned
            self.listen_port = self.server.sockets[0].getsockname()[1]

            logger.info(f"P2P Node {self.node_id} listening on port {self.listen_port}")

            # Start background tasks
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._capability_broadcast_loop())

            self.status = NodeStatus.ACTIVE

        except Exception as e:
            logger.exception(f"Failed to start P2P node: {e}")
            self.status = NodeStatus.FAILED
            raise

    async def stop(self) -> None:
        """Shutdown node gracefully."""
        logger.info(f"Stopping P2P Node {self.node_id}")

        # Close all connections
        for writer in self.connections.values():
            writer.close()
            await writer.wait_closed()

        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self.status = NodeStatus.DISCONNECTED

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming peer connection."""
        peer_addr = writer.get_extra_info("peername")
        peer_id = None

        try:
            while True:
                # Read message length first (4 bytes)
                length_data = await reader.readexactly(4)
                if not length_data:
                    break

                message_length = int.from_bytes(length_data, "big")

                # Read actual message
                message_data = await reader.readexactly(message_length)
                self.bytes_received += len(message_data)

                # Parse message
                try:
                    message = json.loads(message_data.decode("utf-8"))
                    self.messages_received += 1

                    # Track peer
                    if "sender_id" in message:
                        peer_id = message["sender_id"]
                        self.connections[peer_id] = writer

                    # Handle message
                    await self._handle_message(message, writer)

                except json.JSONDecodeError as e:
                    logger.exception(f"Failed to parse message from {peer_addr}: {e}")

        except asyncio.IncompleteReadError:
            # Connection closed
            pass
        except Exception as e:
            logger.exception(f"Error handling connection from {peer_addr}: {e}")
        finally:
            if peer_id and peer_id in self.connections:
                del self.connections[peer_id]
            writer.close()
            await writer.wait_closed()

    async def _handle_message(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Process incoming message."""
        msg_type = message.get("type", "UNKNOWN")

        if msg_type in self.message_handlers:
            try:
                await self.message_handlers[msg_type](message, writer)
            except Exception as e:
                logger.exception(f"Error handling {msg_type} message: {e}")
        else:
            logger.warning(f"No handler for message type: {msg_type}")

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers.update(
            {
                "PING": self._handle_ping,
                "PONG": self._handle_pong,
                "CAPABILITY_UPDATE": self._handle_capability_update,
                "DISCOVERY_REQUEST": self._handle_discovery_request,
                "DISCOVERY_RESPONSE": self._handle_discovery_response,
                "TENSOR_CHUNK": self._handle_tensor_chunk,
            }
        )

    async def _handle_ping(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle ping message."""
        response = {
            "type": "PONG",
            "sender_id": self.node_id,
            "timestamp": time.time(),
            "in_response_to": message.get("message_id"),
        }
        await self._send_message(response, writer)

    async def _handle_pong(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle pong response."""
        msg_id = message.get("in_response_to")
        if msg_id in self.response_futures:
            self.response_futures[msg_id].set_result(message)
            del self.response_futures[msg_id]

    async def _handle_capability_update(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle peer capability update."""
        sender_id = message.get("sender_id")
        if sender_id and "capabilities" in message:
            capabilities_data = message["capabilities"]

            # Create PeerCapabilities object
            capabilities = PeerCapabilities(
                device_id=sender_id,
                cpu_cores=capabilities_data.get("cpu_cores", 1),
                ram_mb=capabilities_data.get("ram_mb", 1024),
                battery_percent=capabilities_data.get("battery_percent"),
                network_type=capabilities_data.get("network_type", "unknown"),
                trust_score=capabilities_data.get("trust_score", 0.5),
                last_seen=time.time(),
            )

            self.peer_registry[sender_id] = capabilities
            logger.debug(f"Updated capabilities for peer {sender_id}")

    async def _handle_discovery_request(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle peer discovery request."""
        response = {
            "type": "DISCOVERY_RESPONSE",
            "sender_id": self.node_id,
            "peers": [
                {"peer_id": peer_id, "capabilities": caps.__dict__} for peer_id, caps in self.peer_registry.items()
            ],
            "node_info": {
                "node_id": self.node_id,
                "listen_port": self.listen_port,
                "capabilities": self._get_local_capabilities().__dict__,
            },
        }
        await self._send_message(response, writer)

    async def _handle_discovery_response(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle discovery response."""
        if "peers" in message:
            for peer_info in message["peers"]:
                peer_id = peer_info.get("peer_id")
                if peer_id and peer_id != self.node_id:
                    caps_data = peer_info.get("capabilities", {})
                    capabilities = PeerCapabilities(
                        device_id=peer_id,
                        cpu_cores=caps_data.get("cpu_cores", 1),
                        ram_mb=caps_data.get("ram_mb", 1024),
                        battery_percent=caps_data.get("battery_percent"),
                        network_type=caps_data.get("network_type", "unknown"),
                        trust_score=caps_data.get("trust_score", 0.5),
                        last_seen=time.time(),
                    )
                    self.peer_registry[peer_id] = capabilities

    async def _handle_tensor_chunk(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Handle incoming tensor chunk."""
        # This will be implemented when we add tensor streaming
        chunk_id = message.get("chunk_id", 0)
        total_chunks = message.get("total_chunks", 1)

        logger.debug(f"Received tensor chunk {chunk_id}/{total_chunks}")

        # For now, just acknowledge receipt
        response = {
            "type": "CHUNK_ACK",
            "sender_id": self.node_id,
            "chunk_id": chunk_id,
            "status": "received",
        }
        await self._send_message(response, writer)

    async def _send_message(self, message: dict, writer: asyncio.StreamWriter) -> None:
        """Send message to specific writer."""
        try:
            message_data = json.dumps(message).encode("utf-8")
            length_data = len(message_data).to_bytes(4, "big")

            writer.write(length_data)
            writer.write(message_data)
            await writer.drain()

            self.messages_sent += 1
            self.bytes_sent += len(message_data)

        except Exception as e:
            logger.exception(f"Failed to send message: {e}")

    async def send_to_peer(self, peer_id: str, message: dict) -> bool:
        """Send message to specific peer."""
        if peer_id not in self.connections:
            logger.warning(f"No connection to peer {peer_id}")
            return False

        message["sender_id"] = self.node_id
        message["message_id"] = str(uuid.uuid4())
        message["timestamp"] = time.time()

        await self._send_message(message, self.connections[peer_id])
        return True

    async def broadcast_to_peers(self, msg_type: str, data: dict) -> None:
        """Broadcast message to all connected peers."""
        message = {
            "type": msg_type,
            "sender_id": self.node_id,
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            **data,
        }

        for peer_id, writer in self.connections.items():
            try:
                await self._send_message(message, writer)
            except Exception as e:
                logger.exception(f"Failed to broadcast to {peer_id}: {e}")

    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a peer node."""
        try:
            reader, writer = await asyncio.open_connection(host, port)

            # Send introduction
            intro_message = {
                "type": "INTRODUCTION",
                "sender_id": self.node_id,
                "capabilities": self._get_local_capabilities().__dict__,
            }

            await self._send_message(intro_message, writer)

            # Start handling this connection
            asyncio.create_task(self._handle_peer_connection(reader, writer))

            self.known_peers.add((host, port))
            return True

        except Exception as e:
            logger.exception(f"Failed to connect to {host}:{port}: {e}")
            return False

    async def _handle_peer_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle outgoing peer connection."""
        await self._handle_connection(reader, writer)

    def _get_local_capabilities(self) -> PeerCapabilities:
        """Get current node capabilities."""
        import psutil

        return PeerCapabilities(
            device_id=self.node_id,
            cpu_cores=psutil.cpu_count() or 1,
            ram_mb=int(psutil.virtual_memory().total / (1024 * 1024)),
            battery_percent=self._get_battery_percent(),
            network_type="ethernet",  # Default assumption
            trust_score=1.0,  # We trust ourselves
            last_seen=time.time(),
        )

    def _get_battery_percent(self) -> int | None:
        """Get battery percentage if available."""
        try:
            import psutil

            battery = psutil.sensors_battery()
            return int(battery.percent) if battery else None
        except:
            return None

    async def _discovery_loop(self) -> None:
        """Periodic peer discovery."""
        while self.status == NodeStatus.ACTIVE:
            try:
                # Try to discover new peers on local network
                await self._discover_local_peers()
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                logger.exception(f"Discovery loop error: {e}")
                await asyncio.sleep(5)

    async def _discover_local_peers(self) -> None:
        """Discover peers on local network."""
        # Simple discovery: try common ports on local subnet
        import ipaddress

        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            # Get network
            network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)

            # Try to connect to a few IPs on common ports
            discovery_ports = [9000, 9001, 9002, 9003, 9004]

            tasks = []
            for ip in list(network.hosts())[:10]:  # Try first 10 IPs
                for port in discovery_ports:
                    if f"{ip}" != local_ip or port != self.listen_port:
                        tasks.append(self._try_discover_peer(str(ip), port))

            # Don't wait for all, just try
            if tasks:
                await asyncio.wait(tasks[:5], timeout=2.0)  # Quick discovery

        except Exception as e:
            logger.debug(f"Local discovery failed: {e}")

    async def _try_discover_peer(self, host: str, port: int) -> None:
        """Try to discover a single peer."""
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2.0)

            # Send discovery request
            discovery_msg = {"type": "DISCOVERY_REQUEST", "sender_id": self.node_id}

            await self._send_message(discovery_msg, writer)

            # Close discovery connection (peer will connect back if interested)
            writer.close()
            await writer.wait_closed()

        except:
            # Discovery failures are expected and normal
            pass

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to connected peers."""
        while self.status == NodeStatus.ACTIVE:
            try:
                ping_message = {
                    "type": "PING",
                    "sender_id": self.node_id,
                    "timestamp": time.time(),
                }

                for peer_id in list(self.connections.keys()):
                    try:
                        await self.send_to_peer(peer_id, ping_message.copy())
                    except Exception as e:
                        logger.warning(f"Heartbeat failed for {peer_id}: {e}")
                        # Remove failed connection
                        if peer_id in self.connections:
                            del self.connections[peer_id]
                        if peer_id in self.peer_registry:
                            del self.peer_registry[peer_id]

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.exception(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)

    async def _capability_broadcast_loop(self) -> None:
        """Periodically broadcast our capabilities."""
        while self.status == NodeStatus.ACTIVE:
            try:
                capabilities = self._get_local_capabilities()
                await self.broadcast_to_peers("CAPABILITY_UPDATE", {"capabilities": capabilities.__dict__})
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                logger.exception(f"Capability broadcast error: {e}")
                await asyncio.sleep(10)

    def register_handler(self, msg_type: str, handler: Callable) -> None:
        """Register message handler."""
        self.message_handlers[msg_type] = handler

    async def wait_for_message(self, msg_type: str, timeout: float = 10.0) -> dict | None:
        """Wait for specific message type."""
        msg_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.response_futures[msg_id] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            if msg_id in self.response_futures:
                del self.response_futures[msg_id]
            return None

    def get_peer_count(self) -> int:
        """Get number of connected peers."""
        return len(self.peer_registry)

    def get_network_stats(self) -> dict[str, Any]:
        """Get network statistics."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "listen_port": self.listen_port,
            "connected_peers": len(self.connections),
            "known_peers": len(self.peer_registry),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }
