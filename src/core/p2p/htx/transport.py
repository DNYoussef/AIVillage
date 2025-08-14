"""HTX Transport Coordinator - Betanet v1.1

Main transport implementation that coordinates all HTX components:
- uTLS cover transport with fingerprint calibration
- Noise XK inner protocol security
- Access ticket authentication
- Frame format handling
- Flow control and connection management

This module provides the high-level HTX transport interface while
delegating specialized concerns to focused sub-modules.
"""

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .access_tickets import AccessTicket, AccessTicketManager, TicketStatus
from .frame_format import (
    HTXFrameBuffer,
    HTXFrameCodec,
    HTXFrameType,
    create_data_frame,
)
from .noise_protocol import NoiseHandshakeState, NoiseXKProtocol
from .utls_fingerprinting import ClientHelloFingerprint, uTLSFingerprintCalibrator

logger = logging.getLogger(__name__)


class HTXConnectionState(Enum):
    """HTX connection state machine."""

    DISCONNECTED = "disconnected"
    TLS_HANDSHAKING = "tls_handshaking"
    NOISE_HANDSHAKING = "noise_handshaking"
    ACCESS_AUTHENTICATING = "access_authenticating"
    CONNECTED = "connected"
    ERROR = "error"
    CLOSING = "closing"


class HTXStreamState(Enum):
    """HTX stream states."""

    IDLE = "idle"
    OPEN = "open"
    HALF_CLOSED_LOCAL = "half_closed_local"
    HALF_CLOSED_REMOTE = "half_closed_remote"
    CLOSED = "closed"


@dataclass
class HTXStream:
    """HTX stream tracking."""

    stream_id: int
    state: HTXStreamState = HTXStreamState.IDLE
    flow_control_window: int = 65536  # Default 64KB
    data_received: bytes = b""
    data_to_send: bytes = b""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


@dataclass
class HTXConnection:
    """HTX connection context."""

    connection_id: str
    remote_address: str
    state: HTXConnectionState = HTXConnectionState.DISCONNECTED

    # Protocol components
    fingerprint: ClientHelloFingerprint | None = None
    noise_protocol: NoiseXKProtocol | None = None
    access_ticket: AccessTicket | None = None

    # Stream management
    streams: dict[int, HTXStream] = field(default_factory=dict)
    next_stream_id: int = 1

    # Flow control
    connection_window: int = 65536  # Connection-level window
    max_frame_size: int = 16777215  # 2^24 - 1

    # Frame handling
    frame_buffer: HTXFrameBuffer = field(default_factory=HTXFrameBuffer)

    # Statistics
    bytes_sent: int = 0
    bytes_received: int = 0
    frames_sent: int = 0
    frames_received: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class HTXTransport:
    """HTX Transport Implementation with Modular Architecture.

    Coordinates all HTX protocol components:
    1. uTLS fingerprinting for cover transport
    2. Noise XK for inner protocol security
    3. Access tickets for authentication
    4. Frame format for data transmission
    5. Flow control and connection management
    """

    def __init__(
        self,
        known_server_key: bytes = None,
        ticket_manager: AccessTicketManager = None,
        fingerprint_template: str = "chrome_120_windows",
    ):
        """Initialize HTX transport.

        Args:
            known_server_key: Server's static Noise key (for XK pattern)
            ticket_manager: Access ticket manager instance
            fingerprint_template: uTLS fingerprint template to use
        """
        # Core components
        self.fingerprint_calibrator = uTLSFingerprintCalibrator()
        self.ticket_manager = ticket_manager or AccessTicketManager()
        self.fingerprint_template = fingerprint_template
        self.known_server_key = known_server_key

        # Connection management
        self.connections: dict[str, HTXConnection] = {}
        self.server_connections: dict[str, HTXConnection] = {}  # Inbound connections

        # Event handlers
        self.message_handlers: dict[str, Callable] = {}
        self.connection_handlers: dict[str, Callable] = {}

        # Configuration
        self.max_connections = 100
        self.connection_timeout = 300.0  # 5 minutes
        self.keepalive_interval = 30.0  # 30 seconds

        # State
        self.is_running = False
        self.cleanup_task: asyncio.Task | None = None

        logger.info("HTX Transport initialized with modular architecture")

    async def start(self) -> bool:
        """Start HTX transport service."""
        if self.is_running:
            return True

        try:
            # Start periodic cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            self.is_running = True
            logger.info("HTX Transport started")
            return True

        except Exception as e:
            logger.exception(f"Failed to start HTX transport: {e}")
            return False

    async def stop(self) -> None:
        """Stop HTX transport service."""
        logger.info("Stopping HTX Transport...")
        self.is_running = False

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        close_tasks = []
        for connection in list(self.connections.values()):
            close_tasks.append(self.close_connection(connection.connection_id))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        logger.info("HTX Transport stopped")

    async def connect(self, target_address: str, access_ticket: AccessTicket = None) -> str | None:
        """Establish HTX connection to target.

        Args:
            target_address: Target server address (host:port)
            access_ticket: Optional access ticket for authentication

        Returns:
            Connection ID if successful, None if failed
        """
        connection_id = f"client_{secrets.token_hex(8)}"

        try:
            # Create connection context
            connection = HTXConnection(
                connection_id=connection_id,
                remote_address=target_address,
                access_ticket=access_ticket,
            )

            # 1. Generate uTLS fingerprint
            connection.fingerprint = self.fingerprint_calibrator.calibrate_fingerprint(
                self.fingerprint_template, randomize=True
            )

            # 2. Initialize Noise XK protocol (client side)
            connection.noise_protocol = NoiseXKProtocol(is_initiator=True, known_remote_static=self.known_server_key)
            connection.noise_protocol.initialize_handshake()

            # Store connection
            self.connections[connection_id] = connection

            # 3. Perform TLS handshake (simulated for now)
            await self._perform_tls_handshake(connection)

            # 4. Perform Noise handshake
            await self._perform_noise_handshake(connection)

            # 5. Authenticate with access ticket
            if access_ticket:
                await self._authenticate_connection(connection)

            # Connection established
            connection.state = HTXConnectionState.CONNECTED
            connection.last_activity = time.time()

            # Trigger connection event
            if "connection_established" in self.connection_handlers:
                await self.connection_handlers["connection_established"](connection_id)

            logger.info(f"HTX connection established: {connection_id} -> {target_address}")
            return connection_id

        except Exception as e:
            logger.exception(f"Failed to establish HTX connection: {e}")
            if connection_id in self.connections:
                await self.close_connection(connection_id)
            return None

    async def send_message(self, connection_id: str, data: bytes, stream_id: int = None) -> bool:
        """Send data over HTX connection.

        Args:
            connection_id: Target connection
            data: Data to send
            stream_id: Optional specific stream (creates new if None)

        Returns:
            True if sent successfully
        """
        connection = self.connections.get(connection_id)
        if not connection or connection.state != HTXConnectionState.CONNECTED:
            logger.error(f"Connection not ready: {connection_id}")
            return False

        try:
            # Get or create stream
            if stream_id is None:
                stream_id = connection.next_stream_id
                connection.next_stream_id += 2  # Client uses odd IDs

            stream = self._get_or_create_stream(connection, stream_id)

            # Check flow control
            if len(data) > stream.flow_control_window:
                logger.warning(f"Data size exceeds flow control window: {len(data)} > {stream.flow_control_window}")
                return False

            # Create DATA frame
            frame_data = create_data_frame(stream_id, data)

            # Encrypt with Noise protocol
            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(frame_data)
            else:
                encrypted_frame = frame_data

            # Send frame (simulated - would go over actual TLS socket)
            success = await self._send_frame_data(connection, encrypted_frame)

            if success:
                # Update stream state
                stream.flow_control_window -= len(data)
                stream.last_activity = time.time()
                connection.bytes_sent += len(encrypted_frame)
                connection.frames_sent += 1
                connection.last_activity = time.time()

                logger.debug(f"Sent {len(data)} bytes on stream {stream_id}")

            return success

        except Exception as e:
            logger.exception(f"Failed to send message: {e}")
            return False

    async def receive_message(self, connection_id: str) -> tuple[int, bytes] | None:
        """Receive message from HTX connection.

        Args:
            connection_id: Source connection

        Returns:
            Tuple of (stream_id, data) if message available, None otherwise
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return None

        try:
            # Process any incoming frames
            await self._process_incoming_frames(connection)

            # Check streams for received data
            for stream in connection.streams.values():
                if stream.data_received:
                    data = stream.data_received
                    stream.data_received = b""
                    stream.last_activity = time.time()
                    connection.last_activity = time.time()
                    return (stream.stream_id, data)

            return None

        except Exception as e:
            logger.exception(f"Error receiving message: {e}")
            return None

    async def close_connection(self, connection_id: str) -> bool:
        """Close HTX connection.

        Args:
            connection_id: Connection to close

        Returns:
            True if closed successfully
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return False

        try:
            connection.state = HTXConnectionState.CLOSING

            # Send any pending frames
            # Close streams
            for stream in connection.streams.values():
                stream.state = HTXStreamState.CLOSED

            # Clean up connection
            del self.connections[connection_id]

            # Trigger connection event
            if "connection_closed" in self.connection_handlers:
                await self.connection_handlers["connection_closed"](connection_id)

            logger.info(f"HTX connection closed: {connection_id}")
            return True

        except Exception as e:
            logger.exception(f"Error closing connection: {e}")
            return False

    def add_message_handler(self, message_type: str, handler: Callable) -> None:
        """Add message handler for specific message type."""
        self.message_handlers[message_type] = handler

    def add_connection_handler(self, event_type: str, handler: Callable) -> None:
        """Add connection event handler."""
        self.connection_handlers[event_type] = handler

    async def _perform_tls_handshake(self, connection: HTXConnection) -> None:
        """Perform TLS handshake with uTLS fingerprinting."""
        connection.state = HTXConnectionState.TLS_HANDSHAKING

        # Generate ClientHello matching fingerprint
        client_hello = self.fingerprint_calibrator.generate_client_hello(
            connection.fingerprint, server_name=connection.remote_address.split(":")[0]
        )

        # Simulate TLS handshake
        await asyncio.sleep(0.05)  # Simulate network RTT

        # Validate fingerprint match (simplified)
        if not self.fingerprint_calibrator.validate_fingerprint_match(client_hello, connection.fingerprint):
            raise RuntimeError("TLS fingerprint validation failed")

        logger.debug(f"TLS handshake completed with JA3: {connection.fingerprint.ja3_hash[:8]}...")

    async def _perform_noise_handshake(self, connection: HTXConnection) -> None:
        """Perform Noise XK handshake."""
        connection.state = HTXConnectionState.NOISE_HANDSHAKING

        noise = connection.noise_protocol

        # Message 1: e, es (client -> server)
        msg1 = noise.create_message_1()
        await self._send_raw_data(connection, msg1)

        # Receive Message 2: e, ee (server -> client)
        msg2 = await self._receive_raw_data(connection, expected_min_size=32)
        noise.process_message_2(msg2)

        # Message 3: s, se (client -> server)
        msg3 = noise.create_message_3()
        await self._send_raw_data(connection, msg3)

        if noise.state != NoiseHandshakeState.HANDSHAKE_COMPLETE:
            raise RuntimeError(f"Noise handshake failed: {noise.state}")

        logger.debug("Noise XK handshake completed")

    async def _authenticate_connection(self, connection: HTXConnection) -> None:
        """Authenticate connection with access ticket."""
        connection.state = HTXConnectionState.ACCESS_AUTHENTICATING

        ticket = connection.access_ticket
        if not ticket:
            return

        # Validate ticket locally first
        status = self.ticket_manager.validate_ticket(ticket)
        if status != TicketStatus.VALID:
            raise RuntimeError(f"Invalid access ticket: {status}")

        # Send ticket to server (create ACCESS_TICKET frame)
        ticket_data = ticket.serialize()
        frame_data = HTXFrameCodec.encode_frame(HTXFrameType.ACCESS_TICKET, 0, ticket_data)

        # Encrypt and send
        encrypted_frame = connection.noise_protocol.encrypt_transport_message(frame_data)
        await self._send_frame_data(connection, encrypted_frame)

        # Wait for response (simplified)
        await asyncio.sleep(0.02)

        logger.debug("Access ticket authentication completed")

    def _get_or_create_stream(self, connection: HTXConnection, stream_id: int) -> HTXStream:
        """Get existing stream or create new one."""
        if stream_id not in connection.streams:
            stream = HTXStream(stream_id=stream_id, state=HTXStreamState.OPEN)
            connection.streams[stream_id] = stream

        return connection.streams[stream_id]

    async def _process_incoming_frames(self, connection: HTXConnection) -> None:
        """Process incoming frames from connection."""
        # Simulate receiving data (would read from actual socket)
        # This is where frame parsing and processing would happen
        pass

    async def _send_frame_data(self, connection: HTXConnection, frame_data: bytes) -> bool:
        """Send frame data over connection."""
        # Simulate sending (would write to actual socket)
        await asyncio.sleep(0.001)  # Simulate write latency
        return True

    async def _send_raw_data(self, connection: HTXConnection, data: bytes) -> None:
        """Send raw data during handshake."""
        # Simulate sending
        await asyncio.sleep(0.005)

    async def _receive_raw_data(self, connection: HTXConnection, expected_min_size: int) -> bytes:
        """Receive raw data during handshake."""
        # Simulate receiving (would read from actual socket)
        await asyncio.sleep(0.005)
        return secrets.token_bytes(expected_min_size)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of inactive connections."""
        while self.is_running:
            try:
                now = time.time()

                # Find expired connections
                expired_connections = []
                for conn_id, connection in self.connections.items():
                    if (now - connection.last_activity) > self.connection_timeout:
                        expired_connections.append(conn_id)

                # Close expired connections
                for conn_id in expired_connections:
                    await self.close_connection(conn_id)

                # Clean up ticket manager
                self.ticket_manager.cleanup_expired()

                # Sleep until next cleanup
                await asyncio.sleep(60.0)  # Check every minute

            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")
                await asyncio.sleep(60.0)

    def get_connection_status(self, connection_id: str) -> dict | None:
        """Get detailed connection status."""
        connection = self.connections.get(connection_id)
        if not connection:
            return None

        return {
            "connection_id": connection_id,
            "state": connection.state.value,
            "remote_address": connection.remote_address,
            "stream_count": len(connection.streams),
            "bytes_sent": connection.bytes_sent,
            "bytes_received": connection.bytes_received,
            "frames_sent": connection.frames_sent,
            "frames_received": connection.frames_received,
            "uptime_seconds": time.time() - connection.created_at,
            "fingerprint": connection.fingerprint.ja3_hash[:8] + "..." if connection.fingerprint else None,
            "noise_ready": connection.noise_protocol.state == NoiseHandshakeState.HANDSHAKE_COMPLETE
            if connection.noise_protocol
            else False,
            "has_ticket": connection.access_ticket is not None,
        }

    def get_transport_statistics(self) -> dict:
        """Get comprehensive transport statistics."""
        active_connections = len(self.connections)
        total_streams = sum(len(conn.streams) for conn in self.connections.values())
        total_bytes_sent = sum(conn.bytes_sent for conn in self.connections.values())
        total_bytes_received = sum(conn.bytes_received for conn in self.connections.values())

        return {
            "transport_running": self.is_running,
            "active_connections": active_connections,
            "total_streams": total_streams,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "fingerprint_template": self.fingerprint_template,
            "max_connections": self.max_connections,
            "connection_timeout": self.connection_timeout,
            "ticket_manager_stats": self.ticket_manager.get_statistics(),
            "fingerprint_stats": self.fingerprint_calibrator.get_fingerprint_stats(),
        }
