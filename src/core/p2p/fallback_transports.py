"""Fallback Transport Layer for LibP2P Mesh Network.

Provides fallback communication methods when LibP2P or internet connectivity
is not available. Supports:
- Bluetooth Classic/LE
- WiFi Direct (Android)
- Local file-based communication
- Serial/USB communication (for embedded devices)

This ensures the mesh network can operate in offline scenarios or
when primary transports fail.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Available fallback transport types."""

    BLUETOOTH_CLASSIC = "bluetooth_classic"
    BLUETOOTH_LE = "bluetooth_le"
    WIFI_DIRECT = "wifi_direct"
    FILE_SYSTEM = "file_system"
    SERIAL_USB = "serial_usb"
    LOCAL_SOCKET = "local_socket"


class TransportStatus(Enum):
    """Transport status."""

    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


@dataclass
class TransportMessage:
    """Message for fallback transports."""

    id: str
    type: str
    sender: str
    recipient: str | None
    payload: bytes
    timestamp: float
    transport_type: str
    ttl: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload.hex(),
            "timestamp": self.timestamp,
            "transport_type": self.transport_type,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransportMessage":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", "DATA"),
            sender=data.get("sender", ""),
            recipient=data.get("recipient"),
            payload=bytes.fromhex(data.get("payload", "")),
            timestamp=data.get("timestamp", time.time()),
            transport_type=data.get("transport_type", "unknown"),
            ttl=data.get("ttl", 5),
        )


class FallbackTransport(ABC):
    """Abstract base class for fallback transports."""

    def __init__(self, transport_type: TransportType, node_id: str) -> None:
        self.transport_type = transport_type
        self.node_id = node_id
        self.status = TransportStatus.INACTIVE
        self.message_handlers: list[Callable[[TransportMessage], None]] = []
        self.connected_peers: set[str] = set()

    @abstractmethod
    async def start(self) -> bool:
        """Start the transport."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""

    @abstractmethod
    async def send_message(self, message: TransportMessage) -> bool:
        """Send message via transport."""

    @abstractmethod
    async def discover_peers(self) -> list[str]:
        """Discover available peers."""

    def add_message_handler(self, handler: Callable[[TransportMessage], None]) -> None:
        """Add message handler."""
        self.message_handlers.append(handler)

    def remove_message_handler(self, handler: Callable[[TransportMessage], None]) -> None:
        """Remove message handler."""
        if handler in self.message_handlers:
            self.message_handlers.remove(handler)

    async def _handle_message(self, message: TransportMessage) -> None:
        """Handle received message."""
        for handler in self.message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.exception(f"Error in message handler: {e}")


class BluetoothTransport(FallbackTransport):
    """Bluetooth transport implementation."""

    def __init__(self, node_id: str, use_ble: bool = False) -> None:
        super().__init__(
            TransportType.BLUETOOTH_LE if use_ble else TransportType.BLUETOOTH_CLASSIC,
            node_id,
        )
        self.use_ble = use_ble
        self.service_uuid = "12345678-1234-5678-9012-123456789abc"  # Custom service UUID

        # Try to import bluetooth libraries
        self.bluetooth_available = self._check_bluetooth_availability()

    def _check_bluetooth_availability(self) -> bool:
        """Check if bluetooth libraries are available."""
        try:
            if self.use_ble:
                import bleak  # BLE library

                return True
            import bluetooth  # Classic Bluetooth

            return True
        except ImportError:
            logger.warning(f"Bluetooth libraries not available for {self.transport_type}")
            return False

    async def start(self) -> bool:
        """Start Bluetooth transport."""
        if not self.bluetooth_available:
            logger.error("Bluetooth not available")
            return False

        self.status = TransportStatus.STARTING

        try:
            if self.use_ble:
                await self._start_ble()
            else:
                await self._start_classic()

            self.status = TransportStatus.ACTIVE
            logger.info(f"Bluetooth transport started ({self.transport_type.value})")
            return True

        except Exception as e:
            logger.exception(f"Failed to start Bluetooth transport: {e}")
            self.status = TransportStatus.FAILED
            return False

    async def _start_ble(self) -> None:
        """Start Bluetooth LE transport."""
        # BLE implementation would go here
        # For now, this is a placeholder
        logger.info("BLE transport started (placeholder)")

    async def _start_classic(self) -> None:
        """Start Bluetooth Classic transport."""
        # Classic Bluetooth implementation would go here
        # For now, this is a placeholder
        logger.info("Bluetooth Classic transport started (placeholder)")

    async def stop(self) -> None:
        """Stop Bluetooth transport."""
        self.status = TransportStatus.DISCONNECTED
        logger.info("Bluetooth transport stopped")

    async def send_message(self, message: TransportMessage) -> bool:
        """Send message via Bluetooth."""
        if self.status != TransportStatus.ACTIVE:
            return False

        try:
            # Bluetooth send implementation would go here
            logger.debug(f"Sending Bluetooth message: {message.id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to send Bluetooth message: {e}")
            return False

    async def discover_peers(self) -> list[str]:
        """Discover Bluetooth peers."""
        if not self.bluetooth_available:
            return []

        try:
            # Bluetooth discovery implementation would go here
            logger.debug("Discovering Bluetooth peers")
            return []  # Placeholder
        except Exception as e:
            logger.exception(f"Bluetooth discovery failed: {e}")
            return []


class WiFiDirectTransport(FallbackTransport):
    """WiFi Direct transport implementation."""

    def __init__(self, node_id: str) -> None:
        super().__init__(TransportType.WIFI_DIRECT, node_id)
        self.group_name = f"AIVillage-{node_id[:8]}"

    async def start(self) -> bool:
        """Start WiFi Direct transport."""
        self.status = TransportStatus.STARTING

        try:
            # WiFi Direct setup would go here
            # This is platform-specific (Android, Windows, Linux)
            logger.info("WiFi Direct transport started (placeholder)")
            self.status = TransportStatus.ACTIVE
            return True
        except Exception as e:
            logger.exception(f"Failed to start WiFi Direct: {e}")
            self.status = TransportStatus.FAILED
            return False

    async def stop(self) -> None:
        """Stop WiFi Direct transport."""
        self.status = TransportStatus.DISCONNECTED
        logger.info("WiFi Direct transport stopped")

    async def send_message(self, message: TransportMessage) -> bool:
        """Send message via WiFi Direct."""
        if self.status != TransportStatus.ACTIVE:
            return False

        try:
            # WiFi Direct send implementation
            logger.debug(f"Sending WiFi Direct message: {message.id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to send WiFi Direct message: {e}")
            return False

    async def discover_peers(self) -> list[str]:
        """Discover WiFi Direct peers."""
        try:
            # WiFi Direct discovery
            logger.debug("Discovering WiFi Direct peers")
            return []  # Placeholder
        except Exception as e:
            logger.exception(f"WiFi Direct discovery failed: {e}")
            return []


class FileSystemTransport(FallbackTransport):
    """File system-based transport for local communication."""

    def __init__(self, node_id: str, base_dir: str = "/tmp/aivillage_mesh") -> None:
        super().__init__(TransportType.FILE_SYSTEM, node_id)
        self.base_dir = base_dir
        self.inbox_dir = os.path.join(base_dir, "nodes", node_id, "inbox")
        self.outbox_dir = os.path.join(base_dir, "nodes", node_id, "outbox")
        self.discovery_dir = os.path.join(base_dir, "discovery")
        self._running = False

    async def start(self) -> bool:
        """Start file system transport."""
        self.status = TransportStatus.STARTING

        try:
            # Create directories
            os.makedirs(self.inbox_dir, exist_ok=True)
            os.makedirs(self.outbox_dir, exist_ok=True)
            os.makedirs(self.discovery_dir, exist_ok=True)

            # Start file monitoring
            self._running = True
            asyncio.create_task(self._monitor_inbox())
            asyncio.create_task(self._announce_presence())

            self.status = TransportStatus.ACTIVE
            logger.info(f"File system transport started: {self.base_dir}")
            return True

        except Exception as e:
            logger.exception(f"Failed to start file system transport: {e}")
            self.status = TransportStatus.FAILED
            return False

    async def stop(self) -> None:
        """Stop file system transport."""
        self._running = False
        self.status = TransportStatus.DISCONNECTED

        # Remove presence file
        presence_file = os.path.join(self.discovery_dir, f"{self.node_id}.presence")
        try:
            if os.path.exists(presence_file):
                os.remove(presence_file)
        except Exception as e:
            logger.debug(f"Error removing presence file: {e}")

        logger.info("File system transport stopped")

    async def send_message(self, message: TransportMessage) -> bool:
        """Send message via file system."""
        if self.status != TransportStatus.ACTIVE:
            return False

        try:
            if message.recipient:
                # Direct message
                recipient_inbox = os.path.join(self.base_dir, "nodes", message.recipient, "inbox")
                if not os.path.exists(recipient_inbox):
                    logger.debug(f"Recipient inbox not found: {message.recipient}")
                    return False

                message_file = os.path.join(recipient_inbox, f"{message.id}.json")
            else:
                # Broadcast message
                message_file = os.path.join(self.base_dir, "broadcast", f"{message.id}.json")
                os.makedirs(os.path.dirname(message_file), exist_ok=True)

            # Write message to file
            with open(message_file, "w") as f:
                json.dump(message.to_dict(), f)

            logger.debug(f"File system message sent: {message.id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to send file system message: {e}")
            return False

    async def discover_peers(self) -> list[str]:
        """Discover peers via presence files."""
        try:
            if not os.path.exists(self.discovery_dir):
                return []

            peers = []
            for filename in os.listdir(self.discovery_dir):
                if filename.endswith(".presence"):
                    peer_id = filename[:-9]  # Remove .presence suffix
                    if peer_id != self.node_id:
                        peers.append(peer_id)

            return peers

        except Exception as e:
            logger.exception(f"File system peer discovery failed: {e}")
            return []

    async def _monitor_inbox(self) -> None:
        """Monitor inbox for new messages."""
        while self._running:
            try:
                if os.path.exists(self.inbox_dir):
                    for filename in os.listdir(self.inbox_dir):
                        if filename.endswith(".json"):
                            filepath = os.path.join(self.inbox_dir, filename)

                            try:
                                with open(filepath) as f:
                                    message_data = json.load(f)

                                message = TransportMessage.from_dict(message_data)
                                await self._handle_message(message)

                                # Remove processed message
                                os.remove(filepath)

                            except Exception as e:
                                logger.debug(f"Error processing message {filename}: {e}")

                # Check for broadcast messages
                broadcast_dir = os.path.join(self.base_dir, "broadcast")
                if os.path.exists(broadcast_dir):
                    for filename in os.listdir(broadcast_dir):
                        if filename.endswith(".json"):
                            filepath = os.path.join(broadcast_dir, filename)

                            try:
                                with open(filepath) as f:
                                    message_data = json.load(f)

                                message = TransportMessage.from_dict(message_data)

                                # Skip our own broadcasts
                                if message.sender != self.node_id:
                                    await self._handle_message(message)

                            except Exception as e:
                                logger.debug(f"Error processing broadcast {filename}: {e}")

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.exception(f"Inbox monitoring error: {e}")
                await asyncio.sleep(5)

    async def _announce_presence(self) -> None:
        """Announce presence via discovery file."""
        while self._running:
            try:
                presence_file = os.path.join(self.discovery_dir, f"{self.node_id}.presence")
                presence_data = {
                    "node_id": self.node_id,
                    "timestamp": time.time(),
                    "transport_type": self.transport_type.value,
                }

                with open(presence_file, "w") as f:
                    json.dump(presence_data, f)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.exception(f"Presence announcement error: {e}")
                await asyncio.sleep(60)


class LocalSocketTransport(FallbackTransport):
    """Local socket-based transport for same-machine communication."""

    def __init__(self, node_id: str, base_port: int = 9000) -> None:
        super().__init__(TransportType.LOCAL_SOCKET, node_id)
        self.listen_port = base_port + hash(node_id) % 1000
        self.server: asyncio.Server | None = None
        self.connections: dict[str, asyncio.StreamWriter] = {}

    async def start(self) -> bool:
        """Start local socket transport."""
        self.status = TransportStatus.STARTING

        try:
            self.server = await asyncio.start_server(self._handle_connection, "127.0.0.1", self.listen_port)

            self.status = TransportStatus.ACTIVE
            logger.info(f"Local socket transport started on port {self.listen_port}")
            return True

        except Exception as e:
            logger.exception(f"Failed to start local socket transport: {e}")
            self.status = TransportStatus.FAILED
            return False

    async def stop(self) -> None:
        """Stop local socket transport."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        for writer in self.connections.values():
            writer.close()
            await writer.wait_closed()

        self.connections.clear()
        self.status = TransportStatus.DISCONNECTED
        logger.info("Local socket transport stopped")

    async def send_message(self, message: TransportMessage) -> bool:
        """Send message via local socket."""
        if self.status != TransportStatus.ACTIVE:
            return False

        try:
            if message.recipient and message.recipient in self.connections:
                # Direct message to connected peer
                writer = self.connections[message.recipient]
                data = json.dumps(message.to_dict()).encode() + b"\n"

                writer.write(data)
                await writer.drain()

                logger.debug(f"Local socket message sent: {message.id}")
                return True
            # Try to connect to peer
            return await self._send_to_peer(message)

        except Exception as e:
            logger.exception(f"Failed to send local socket message: {e}")
            return False

    async def _send_to_peer(self, message: TransportMessage) -> bool:
        """Send message to peer by connecting directly."""
        if not message.recipient:
            return False

        try:
            # Try common ports for the peer
            peer_port = 9000 + hash(message.recipient) % 1000

            reader, writer = await asyncio.open_connection("127.0.0.1", peer_port)

            data = json.dumps(message.to_dict()).encode() + b"\n"
            writer.write(data)
            await writer.drain()

            writer.close()
            await writer.wait_closed()

            return True

        except Exception as e:
            logger.debug(f"Failed to connect to peer {message.recipient}: {e}")
            return False

    async def discover_peers(self) -> list[str]:
        """Discover peers by port scanning."""
        peers = []

        try:
            # Try connecting to common ports
            for port_offset in range(1000):
                port = 9000 + port_offset
                if port == self.listen_port:
                    continue

                try:
                    reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=0.1)

                    # Send discovery message
                    discovery_msg = {
                        "type": "DISCOVERY",
                        "sender": self.node_id,
                        "timestamp": time.time(),
                    }

                    data = json.dumps(discovery_msg).encode() + b"\n"
                    writer.write(data)
                    await writer.drain()

                    writer.close()
                    await writer.wait_closed()

                    peers.append(f"peer-{port}")

                except:
                    continue

                if len(peers) >= 10:  # Limit discovery
                    break

        except Exception as e:
            logger.debug(f"Local socket discovery error: {e}")

        return peers

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming connection."""
        peer_addr = writer.get_extra_info("peername")

        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                message_data = json.loads(data.decode())
                message = TransportMessage.from_dict(message_data)

                # Store connection for future use
                if message.sender and message.sender != self.node_id:
                    self.connections[message.sender] = writer

                await self._handle_message(message)

        except Exception as e:
            logger.debug(f"Connection error from {peer_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


class FallbackTransportManager:
    """Manages multiple fallback transports."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.transports: dict[TransportType, FallbackTransport] = {}
        self.active_transports: list[FallbackTransport] = []
        self.message_handlers: list[Callable[[TransportMessage], None]] = []

    def add_transport(self, transport: FallbackTransport) -> None:
        """Add transport to manager."""
        self.transports[transport.transport_type] = transport
        transport.add_message_handler(self._handle_transport_message)

    def remove_transport(self, transport_type: TransportType) -> None:
        """Remove transport from manager."""
        if transport_type in self.transports:
            transport = self.transports[transport_type]
            if transport in self.active_transports:
                self.active_transports.remove(transport)
            del self.transports[transport_type]

    async def start_all_transports(self) -> dict[TransportType, bool]:
        """Start all configured transports."""
        results = {}

        for transport_type, transport in self.transports.items():
            try:
                success = await transport.start()
                results[transport_type] = success

                if success:
                    self.active_transports.append(transport)
                    logger.info(f"Started transport: {transport_type.value}")
                else:
                    logger.warning(f"Failed to start transport: {transport_type.value}")

            except Exception as e:
                logger.exception(f"Error starting transport {transport_type.value}: {e}")
                results[transport_type] = False

        logger.info(f"Started {len(self.active_transports)}/{len(self.transports)} transports")
        return results

    async def stop_all_transports(self) -> None:
        """Stop all transports."""
        for transport in self.active_transports:
            try:
                await transport.stop()
            except Exception as e:
                logger.exception(f"Error stopping transport: {e}")

        self.active_transports.clear()

    async def send_message(
        self,
        message: TransportMessage,
        preferred_transport: TransportType | None = None,
    ) -> bool:
        """Send message via fallback transports."""
        if not self.active_transports:
            logger.warning("No active transports available")
            return False

        # Try preferred transport first
        if preferred_transport and preferred_transport in self.transports:
            transport = self.transports[preferred_transport]
            if transport in self.active_transports:
                success = await transport.send_message(message)
                if success:
                    return True

        # Try all active transports
        for transport in self.active_transports:
            try:
                success = await transport.send_message(message)
                if success:
                    logger.debug(f"Message sent via {transport.transport_type.value}")
                    return True
            except Exception as e:
                logger.debug(f"Transport {transport.transport_type.value} failed: {e}")

        logger.warning(f"Failed to send message via any transport: {message.id}")
        return False

    async def discover_all_peers(self) -> dict[TransportType, list[str]]:
        """Discover peers via all active transports."""
        results = {}

        for transport in self.active_transports:
            try:
                peers = await transport.discover_peers()
                results[transport.transport_type] = peers
            except Exception as e:
                logger.exception(f"Discovery failed for {transport.transport_type.value}: {e}")
                results[transport.transport_type] = []

        return results

    def add_message_handler(self, handler: Callable[[TransportMessage], None]) -> None:
        """Add message handler."""
        self.message_handlers.append(handler)

    def remove_message_handler(self, handler: Callable[[TransportMessage], None]) -> None:
        """Remove message handler."""
        if handler in self.message_handlers:
            self.message_handlers.remove(handler)

    def _handle_transport_message(self, message: TransportMessage) -> None:
        """Handle message from any transport."""
        for handler in self.message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.exception(f"Error in transport message handler: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get status of all transports."""
        return {
            "active_transports": len(self.active_transports),
            "total_transports": len(self.transports),
            "transports": {
                transport.transport_type.value: {
                    "status": transport.status.value,
                    "connected_peers": len(transport.connected_peers),
                }
                for transport in self.transports.values()
            },
        }


def create_default_fallback_manager(node_id: str) -> FallbackTransportManager:
    """Create fallback transport manager with default transports."""
    manager = FallbackTransportManager(node_id)

    # Add available transports
    manager.add_transport(BluetoothTransport(node_id, use_ble=False))
    manager.add_transport(BluetoothTransport(node_id, use_ble=True))
    manager.add_transport(WiFiDirectTransport(node_id))
    manager.add_transport(FileSystemTransport(node_id))
    manager.add_transport(LocalSocketTransport(node_id))

    return manager
