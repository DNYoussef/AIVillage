"""Enhanced Transport Manager with WebSocket, TCP, and UDP support.

Provides comprehensive transport layer management with automatic failover.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import websockets

logger = logging.getLogger(__name__)


@dataclass
class TransportMessage:
    """Standardized transport message format."""

    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    payload: dict
    timestamp: float
    transport_metadata: dict = None

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "transport_metadata": self.transport_metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TransportMessage":
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=data["message_type"],
            payload=data["payload"],
            timestamp=data["timestamp"],
            transport_metadata=data.get("transport_metadata", {}),
        )


class BaseTransport(ABC):
    """Abstract base class for transport implementations."""

    def __init__(self, transport_id: str, config: dict) -> None:
        self.transport_id = transport_id
        self.config = config
        self.is_active = False
        self.connected_peers: set[str] = set()
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "errors": 0,
        }

    @abstractmethod
    async def start(self) -> bool:
        """Start the transport."""

    @abstractmethod
    async def stop(self):
        """Stop the transport."""

    @abstractmethod
    async def send_message(self, peer_id: str, message: TransportMessage) -> bool:
        """Send a message to a peer."""

    @abstractmethod
    async def receive_messages(self) -> list[TransportMessage]:
        """Receive pending messages."""

    @abstractmethod
    async def connect_to_peer(self, peer_id: str, address: str) -> bool:
        """Connect to a peer."""

    async def disconnect_from_peer(self, peer_id: str) -> bool:
        """Disconnect from a peer."""
        if peer_id in self.connected_peers:
            self.connected_peers.remove(peer_id)
            return True
        return False


class WebSocketTransport(BaseTransport):
    """WebSocket transport implementation."""

    def __init__(self, config: dict) -> None:
        super().__init__("websocket", config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8765)
        self.server = None
        self.client_connections: dict[str, websockets.WebSocketServerProtocol] = {}
        self.outbound_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> bool:
        """Start WebSocket server."""
        try:
            self.server = await websockets.serve(self._handle_client_connection, self.host, self.port)
            self.is_active = True
            logger.info(f"WebSocket transport started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.exception(f"Failed to start WebSocket transport: {e}")
            return False

    async def stop(self) -> None:
        """Stop WebSocket server and close connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all connections
        for conn in list(self.client_connections.values()):
            await conn.close()
        for conn in list(self.outbound_connections.values()):
            await conn.close()

        self.is_active = False
        logger.info("WebSocket transport stopped")

    async def _handle_client_connection(self, websocket, path) -> None:
        """Handle incoming WebSocket connection."""
        try:
            # Wait for peer identification
            peer_id = await self._authenticate_peer(websocket)
            if peer_id:
                self.client_connections[peer_id] = websocket
                self.connected_peers.add(peer_id)
                logger.info(f"WebSocket connection established with {peer_id}")

                # Listen for messages
                async for message_data in websocket:
                    await self._process_incoming_message(message_data, peer_id)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.exception(f"WebSocket connection error: {e}")
            self.stats["errors"] += 1

    async def _authenticate_peer(self, websocket) -> str | None:
        """Authenticate connecting peer."""
        try:
            # Simple authentication - in production, use proper auth
            auth_data = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_info = json.loads(auth_data)
            return auth_info.get("peer_id")
        except Exception as e:
            logger.exception(f"Peer authentication failed: {e}")
            return None

    async def _process_incoming_message(self, message_data: str, sender_id: str) -> None:
        """Process incoming message."""
        try:
            data = json.loads(message_data)
            message = TransportMessage.from_dict(data)
            await self.message_queue.put(message)

            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(message_data)
        except Exception as e:
            logger.exception(f"Failed to process message: {e}")
            self.stats["errors"] += 1

    async def send_message(self, peer_id: str, message: TransportMessage) -> bool:
        """Send message via WebSocket."""
        try:
            # Try outbound connection first
            if peer_id in self.outbound_connections:
                websocket = self.outbound_connections[peer_id]
            elif peer_id in self.client_connections:
                websocket = self.client_connections[peer_id]
            else:
                logger.warning(f"No WebSocket connection to {peer_id}")
                return False

            message_data = json.dumps(message.to_dict())
            await websocket.send(message_data)

            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message_data)
            return True

        except Exception as e:
            logger.exception(f"Failed to send WebSocket message: {e}")
            self.stats["errors"] += 1
            return False

    async def receive_messages(self) -> list[TransportMessage]:
        """Get pending messages."""
        messages = []
        try:
            while not self.message_queue.empty():
                message = await self.message_queue.get_nowait()
                messages.append(message)
        except asyncio.QueueEmpty:
            pass
        return messages

    async def connect_to_peer(self, peer_id: str, address: str) -> bool:
        """Connect to peer via WebSocket."""
        try:
            self.stats["connection_attempts"] += 1

            # Parse address (ws://host:port)
            if not address.startswith("ws://"):
                address = f"ws://{address}"

            websocket = await websockets.connect(address)

            # Send authentication
            auth_data = json.dumps({"peer_id": self.config.get("node_id", "unknown")})
            await websocket.send(auth_data)

            self.outbound_connections[peer_id] = websocket
            self.connected_peers.add(peer_id)
            self.stats["successful_connections"] += 1

            logger.info(f"Connected to {peer_id} at {address}")
            return True

        except Exception as e:
            logger.exception(f"Failed to connect to {peer_id}: {e}")
            self.stats["errors"] += 1
            return False


class TCPTransport(BaseTransport):
    """TCP transport implementation."""

    def __init__(self, config: dict) -> None:
        super().__init__("tcp", config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8766)
        self.server = None
        self.connections: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> bool:
        """Start TCP server."""
        try:
            self.server = await asyncio.start_server(self._handle_client_connection, self.host, self.port)
            self.is_active = True
            logger.info(f"TCP transport started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.exception(f"Failed to start TCP transport: {e}")
            return False

    async def stop(self) -> None:
        """Stop TCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all connections
        for _peer_id, (_reader, writer) in list(self.connections.items()):
            writer.close()
            await writer.wait_closed()

        self.is_active = False
        logger.info("TCP transport stopped")

    async def _handle_client_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming TCP connection."""
        peer_address = writer.get_extra_info("peername")
        logger.info(f"TCP connection from {peer_address}")

        try:
            # Read peer ID
            peer_id = await self._read_peer_id(reader)
            if peer_id:
                self.connections[peer_id] = (reader, writer)
                self.connected_peers.add(peer_id)

                # Listen for messages
                while True:
                    message_data = await self._read_message(reader)
                    if not message_data:
                        break
                    await self._process_incoming_message(message_data, peer_id)

        except Exception as e:
            logger.exception(f"TCP connection error: {e}")
            self.stats["errors"] += 1
        finally:
            writer.close()
            await writer.wait_closed()

    async def _read_peer_id(self, reader: asyncio.StreamReader) -> str | None:
        """Read peer ID from TCP connection."""
        try:
            length_data = await reader.read(4)
            if len(length_data) < 4:
                return None

            length = int.from_bytes(length_data, "big")
            peer_data = await reader.read(length)
            peer_info = json.loads(peer_data.decode("utf-8"))
            return peer_info.get("peer_id")
        except Exception as e:
            logger.exception(f"Failed to read peer ID: {e}")
            return None

    async def _read_message(self, reader: asyncio.StreamReader) -> str | None:
        """Read message from TCP connection."""
        try:
            length_data = await reader.read(4)
            if len(length_data) < 4:
                return None

            length = int.from_bytes(length_data, "big")
            message_data = await reader.read(length)
            return message_data.decode("utf-8")
        except Exception as e:
            logger.exception(f"Failed to read TCP message: {e}")
            return None

    async def _process_incoming_message(self, message_data: str, sender_id: str) -> None:
        """Process incoming TCP message."""
        try:
            data = json.loads(message_data)
            message = TransportMessage.from_dict(data)
            await self.message_queue.put(message)

            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(message_data)
        except Exception as e:
            logger.exception(f"Failed to process TCP message: {e}")
            self.stats["errors"] += 1

    async def send_message(self, peer_id: str, message: TransportMessage) -> bool:
        """Send message via TCP."""
        try:
            if peer_id not in self.connections:
                logger.warning(f"No TCP connection to {peer_id}")
                return False

            reader, writer = self.connections[peer_id]
            message_data = json.dumps(message.to_dict()).encode("utf-8")

            # Send length prefix + message
            length = len(message_data)
            writer.write(length.to_bytes(4, "big"))
            writer.write(message_data)
            await writer.drain()

            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message_data)
            return True

        except Exception as e:
            logger.exception(f"Failed to send TCP message: {e}")
            self.stats["errors"] += 1
            return False

    async def receive_messages(self) -> list[TransportMessage]:
        """Get pending TCP messages."""
        messages = []
        try:
            while not self.message_queue.empty():
                message = await self.message_queue.get_nowait()
                messages.append(message)
        except asyncio.QueueEmpty:
            pass
        return messages

    async def connect_to_peer(self, peer_id: str, address: str) -> bool:
        """Connect to peer via TCP."""
        try:
            self.stats["connection_attempts"] += 1

            # Parse address (host:port)
            if ":" in address:
                host, port = address.split(":")
                port = int(port)
            else:
                host = address
                port = self.port

            reader, writer = await asyncio.open_connection(host, port)

            # Send peer ID
            peer_data = json.dumps({"peer_id": self.config.get("node_id", "unknown")}).encode("utf-8")
            length = len(peer_data)
            writer.write(length.to_bytes(4, "big"))
            writer.write(peer_data)
            await writer.drain()

            self.connections[peer_id] = (reader, writer)
            self.connected_peers.add(peer_id)
            self.stats["successful_connections"] += 1

            logger.info(f"TCP connected to {peer_id} at {host}:{port}")
            return True

        except Exception as e:
            logger.exception(f"Failed to TCP connect to {peer_id}: {e}")
            self.stats["errors"] += 1
            return False


class UDPTransport(BaseTransport):
    """UDP transport implementation."""

    def __init__(self, config: dict) -> None:
        super().__init__("udp", config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8767)
        self.transport = None
        self.protocol = None
        self.peer_addresses: dict[str, tuple[str, int]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> bool:
        """Start UDP transport."""
        try:
            loop = asyncio.get_event_loop()
            self.transport, self.protocol = await loop.create_datagram_endpoint(
                lambda: UDPProtocol(self.message_queue, self.stats),
                local_addr=(self.host, self.port),
            )
            self.is_active = True
            logger.info(f"UDP transport started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.exception(f"Failed to start UDP transport: {e}")
            return False

    async def stop(self) -> None:
        """Stop UDP transport."""
        if self.transport:
            self.transport.close()
        self.is_active = False
        logger.info("UDP transport stopped")

    async def send_message(self, peer_id: str, message: TransportMessage) -> bool:
        """Send message via UDP."""
        try:
            if peer_id not in self.peer_addresses:
                logger.warning(f"No UDP address for {peer_id}")
                return False

            address = self.peer_addresses[peer_id]
            message_data = json.dumps(message.to_dict()).encode("utf-8")

            self.transport.sendto(message_data, address)

            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message_data)
            return True

        except Exception as e:
            logger.exception(f"Failed to send UDP message: {e}")
            self.stats["errors"] += 1
            return False

    async def receive_messages(self) -> list[TransportMessage]:
        """Get pending UDP messages."""
        messages = []
        try:
            while not self.message_queue.empty():
                message = await self.message_queue.get_nowait()
                messages.append(message)
        except asyncio.QueueEmpty:
            pass
        return messages

    async def connect_to_peer(self, peer_id: str, address: str) -> bool:
        """Register UDP peer address."""
        try:
            # Parse address (host:port)
            if ":" in address:
                host, port = address.split(":")
                port = int(port)
            else:
                host = address
                port = self.port

            self.peer_addresses[peer_id] = (host, port)
            self.connected_peers.add(peer_id)

            logger.info(f"UDP peer registered: {peer_id} at {host}:{port}")
            return True

        except Exception as e:
            logger.exception(f"Failed to register UDP peer {peer_id}: {e}")
            return False


class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler."""

    def __init__(self, message_queue: asyncio.Queue, stats: dict) -> None:
        self.message_queue = message_queue
        self.stats = stats

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle received UDP datagram."""
        try:
            message_data = data.decode("utf-8")
            data_dict = json.loads(message_data)
            message = TransportMessage.from_dict(data_dict)

            # Put message in queue (non-blocking)
            try:
                self.message_queue.put_nowait(message)
                self.stats["messages_received"] += 1
                self.stats["bytes_received"] += len(data)
            except asyncio.QueueFull:
                logger.warning("UDP message queue full, dropping message")
        except Exception as e:
            logger.exception(f"Failed to process UDP datagram: {e}")
            self.stats["errors"] += 1


class EnhancedTransportManager:
    """Enhanced transport manager with multiple transport support."""

    def __init__(self, node_id: str, config: dict | None = None) -> None:
        self.node_id = node_id
        self.config = config or {}

        # Initialize transports
        self.transports: dict[str, BaseTransport] = {}
        self.transport_preferences = ["websocket", "tcp", "udp"]  # Priority order

        # Message routing
        self.message_handlers: dict[str, callable] = {}
        self.routing_table: dict[str, str] = {}  # peer_id -> preferred_transport

        # Statistics
        self.stats = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "transport_failures": {},
            "active_transports": [],
            "start_time": None,
        }

        self.running = False

        # Initialize transports based on config
        self._initialize_transports()

    def _initialize_transports(self) -> None:
        """Initialize transport instances."""
        base_config = {"node_id": self.node_id}

        # WebSocket transport
        if self.config.get("enable_websocket", True):
            ws_config = {**base_config, **self.config.get("websocket", {})}
            self.transports["websocket"] = WebSocketTransport(ws_config)

        # TCP transport
        if self.config.get("enable_tcp", True):
            tcp_config = {**base_config, **self.config.get("tcp", {})}
            self.transports["tcp"] = TCPTransport(tcp_config)

        # UDP transport
        if self.config.get("enable_udp", True):
            udp_config = {**base_config, **self.config.get("udp", {})}
            self.transports["udp"] = UDPTransport(udp_config)

    async def start(self) -> bool:
        """Start all transports."""
        if self.running:
            return True

        logger.info(f"Starting enhanced transport manager for {self.node_id}")
        self.stats["start_time"] = time.time()

        # Start all transports
        for transport_name, transport in self.transports.items():
            try:
                success = await transport.start()
                if success:
                    self.stats["active_transports"].append(transport_name)
                    logger.info(f"✅ {transport_name} transport started")
                else:
                    logger.warning(f"❌ {transport_name} transport failed to start")
            except Exception as e:
                logger.exception(f"Error starting {transport_name} transport: {e}")

        if self.stats["active_transports"]:
            self.running = True
            # Start message processing loop
            asyncio.create_task(self._message_processing_loop())
            logger.info(f"Transport manager started with {len(self.stats['active_transports'])} active transports")
            return True
        logger.error("No transports could be started")
        return False

    async def stop(self) -> None:
        """Stop all transports."""
        if not self.running:
            return

        logger.info("Stopping transport manager")
        self.running = False

        for transport in self.transports.values():
            try:
                await transport.stop()
            except Exception as e:
                logger.exception(f"Error stopping transport: {e}")

        self.stats["active_transports"].clear()
        logger.info("Transport manager stopped")

    async def send_message(
        self,
        peer_id: str,
        message_type: str,
        payload: dict,
        preferred_transport: str | None = None,
    ) -> bool:
        """Send message with transport selection logic."""
        # Create transport message
        message = TransportMessage(
            message_id=f"{self.node_id}_{int(time.time() * 1000000)}",
            sender_id=self.node_id,
            recipient_id=peer_id,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
        )

        # Determine transport to use
        transports_to_try = self._get_transport_priority(peer_id, preferred_transport)

        for transport_name in transports_to_try:
            if transport_name in self.transports and transport_name in self.stats["active_transports"]:
                transport = self.transports[transport_name]

                try:
                    success = await transport.send_message(peer_id, message)
                    if success:
                        self.stats["total_messages_sent"] += 1
                        self.routing_table[peer_id] = transport_name  # Remember successful transport
                        logger.debug(f"Message sent to {peer_id} via {transport_name}")
                        return True
                except Exception as e:
                    logger.warning(f"Failed to send via {transport_name}: {e}")
                    self._record_transport_failure(transport_name)

        logger.error(f"Failed to send message to {peer_id} via any transport")
        return False

    def _get_transport_priority(self, peer_id: str, preferred_transport: str | None = None) -> list[str]:
        """Get ordered list of transports to try for a peer."""
        # Start with preferred transport
        if preferred_transport and preferred_transport in self.transports:
            priority_list = [preferred_transport]
        else:
            priority_list = []

        # Add transport from routing table (previously successful)
        if peer_id in self.routing_table:
            preferred = self.routing_table[peer_id]
            if preferred not in priority_list:
                priority_list.append(preferred)

        # Add remaining transports by preference
        for transport_name in self.transport_preferences:
            if transport_name not in priority_list and transport_name in self.transports:
                priority_list.append(transport_name)

        return priority_list

    def _record_transport_failure(self, transport_name: str) -> None:
        """Record transport failure for statistics."""
        if transport_name not in self.stats["transport_failures"]:
            self.stats["transport_failures"][transport_name] = 0
        self.stats["transport_failures"][transport_name] += 1

    async def connect_to_peer(self, peer_id: str, addresses: dict[str, str]) -> bool:
        """Connect to peer using available transports."""
        success = False

        for transport_name, address in addresses.items():
            if transport_name in self.transports and transport_name in self.stats["active_transports"]:
                try:
                    transport = self.transports[transport_name]
                    connected = await transport.connect_to_peer(peer_id, address)
                    if connected:
                        success = True
                        logger.info(f"Connected to {peer_id} via {transport_name}")
                except Exception as e:
                    logger.warning(f"Failed to connect to {peer_id} via {transport_name}: {e}")

        return success

    def register_message_handler(self, message_type: str, handler: callable) -> None:
        """Register handler for specific message types."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    async def _message_processing_loop(self) -> None:
        """Background loop for processing incoming messages."""
        while self.running:
            try:
                # Collect messages from all active transports
                for transport_name, transport in self.transports.items():
                    if transport_name in self.stats["active_transports"]:
                        messages = await transport.receive_messages()
                        for message in messages:
                            await self._handle_message(message)
                            self.stats["total_messages_received"] += 1

                await asyncio.sleep(0.1)  # Prevent busy waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)

    async def _handle_message(self, message: TransportMessage) -> None:
        """Handle incoming message."""
        try:
            # Look for registered handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                logger.debug(f"No handler for message type: {message.message_type}")
        except Exception as e:
            logger.exception(f"Error handling message: {e}")

    def get_transport_status(self) -> dict:
        """Get status of all transports."""
        status = {
            "active_transports": self.stats["active_transports"].copy(),
            "total_transports": len(self.transports),
            "transport_stats": {},
        }

        for transport_name, transport in self.transports.items():
            status["transport_stats"][transport_name] = {
                "is_active": transport.is_active,
                "connected_peers": len(transport.connected_peers),
                "stats": transport.stats.copy(),
            }

        return status

    def get_connected_peers(self) -> dict[str, list[str]]:
        """Get connected peers by transport."""
        peers = {}
        for transport_name, transport in self.transports.items():
            peers[transport_name] = list(transport.connected_peers)
        return peers


async def test_enhanced_transport_manager():
    """Test function for enhanced transport manager."""
    print("Testing Enhanced Transport Manager...")

    config = {"websocket": {"port": 8765}, "tcp": {"port": 8766}, "udp": {"port": 8767}}

    manager = EnhancedTransportManager("test_node_001", config)

    # Start transport manager
    success = await manager.start()
    print(f"Transport manager started: {success}")

    # Check status
    status = manager.get_transport_status()
    print(f"Active transports: {status['active_transports']}")

    # Test message handler
    async def test_handler(message) -> None:
        print(f"Received test message: {message.payload}")

    manager.register_message_handler("test", test_handler)

    # Let it run for a bit
    await asyncio.sleep(2)

    # Get final status
    final_status = manager.get_transport_status()
    print(f"Final status: {final_status}")

    # Stop
    await manager.stop()
    print("Transport manager stopped")

    return manager


if __name__ == "__main__":
    asyncio.run(test_enhanced_transport_manager())
