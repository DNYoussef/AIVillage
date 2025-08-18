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
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .access_tickets import AccessTicket, AccessTicketManager, TicketStatus
from .frame_format import (
    HTXFrame,
    HTXFrameBuffer,
    HTXFrameCodec,
    HTXFrameType,
    create_data_frame,
    create_window_update_frame,
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
        self.ping_task: asyncio.Task | None = None
        self.idle_padding_task: asyncio.Task | None = None

        # H2/H3 behavior emulation (BN-5.5)
        self.ping_interval_base = 35.0  # Base interval for [10-60]s ±10%
        self.priority_frame_rate = 0.02  # 2% of connections send PRIORITY frames
        self.idle_timeout_range = (200, 1200)  # ms
        self.idle_padding_range = (0, 3072)  # 0-3 KiB

        logger.info("HTX Transport initialized with modular architecture")

    async def start(self) -> bool:
        """Start HTX transport service."""
        if self.is_running:
            return True

        try:
            # Start periodic cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Start H2/H3 behavior emulation tasks (BN-5.5)
            self.ping_task = asyncio.create_task(self._ping_loop())
            self.idle_padding_task = asyncio.create_task(self._idle_padding_loop())

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

        # Cancel all background tasks
        tasks_to_cancel = [self.cleanup_task, self.ping_task, self.idle_padding_task]
        for task in tasks_to_cancel:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all connections
        close_tasks = []
        for connection in list(self.connections.values()):
            close_tasks.append(self.close_connection(connection.connection_id))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        logger.info("HTX Transport stopped")

    async def connect(
        self, target_address: str, access_ticket: AccessTicket = None
    ) -> str | None:
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
            connection.noise_protocol = NoiseXKProtocol(
                is_initiator=True, known_remote_static=self.known_server_key
            )
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

            logger.info(
                f"HTX connection established: {connection_id} -> {target_address}"
            )
            return connection_id

        except Exception as e:
            logger.exception(f"Failed to establish HTX connection: {e}")
            if connection_id in self.connections:
                await self.close_connection(connection_id)
            return None

    async def send_message(
        self, connection_id: str, data: bytes, stream_id: int = None
    ) -> bool:
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
            # Get or create stream (BN-5.4: client odd, server even)
            if stream_id is None:
                stream_id = connection.next_stream_id
                # Client uses odd IDs (1, 3, 5...), Server uses even IDs (2, 4, 6...)
                connection.next_stream_id += 2

            stream = self._get_or_create_stream(connection, stream_id)

            # Check flow control
            if len(data) > stream.flow_control_window:
                logger.warning(
                    f"Data size exceeds flow control window: {len(data)} > {stream.flow_control_window}"
                )
                return False

            # Create DATA frame
            frame_data = create_data_frame(stream_id, data)

            # Encrypt with Noise protocol
            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    frame_data
                )
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
        if not self.fingerprint_calibrator.validate_fingerprint_match(
            client_hello, connection.fingerprint
        ):
            raise RuntimeError("TLS fingerprint validation failed")

        logger.debug(
            f"TLS handshake completed with JA3: {connection.fingerprint.ja3_hash[:8]}..."
        )

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
        frame_data = HTXFrameCodec.encode_frame(
            HTXFrameType.ACCESS_TICKET, 0, ticket_data
        )

        # Encrypt and send
        encrypted_frame = connection.noise_protocol.encrypt_transport_message(
            frame_data
        )
        await self._send_frame_data(connection, encrypted_frame)

        # Wait for response (simplified)
        await asyncio.sleep(0.02)

        logger.debug("Access ticket authentication completed")

    def _get_or_create_stream(
        self, connection: HTXConnection, stream_id: int
    ) -> HTXStream:
        """Get existing stream or create new one."""
        if stream_id not in connection.streams:
            stream = HTXStream(stream_id=stream_id, state=HTXStreamState.OPEN)
            connection.streams[stream_id] = stream

        return connection.streams[stream_id]

    async def _process_incoming_frames(self, connection: HTXConnection) -> None:
        """Process incoming frames from connection."""
        # Process buffered frames
        frames = connection.frame_buffer.parse_frames()

        for frame in frames:
            connection.frames_received += 1
            connection.last_activity = time.time()

            if frame.frame_type == HTXFrameType.DATA:
                await self._handle_data_frame(connection, frame)
            elif frame.frame_type == HTXFrameType.WINDOW_UPDATE:
                await self._handle_window_update_frame(connection, frame)
            elif frame.frame_type == HTXFrameType.PING:
                await self._handle_ping_frame(connection, frame)
            elif frame.frame_type == HTXFrameType.KEY_UPDATE:
                await self._handle_key_update_frame(connection, frame)
            else:
                logger.debug(f"Unhandled frame type: {frame.frame_type}")

    async def _handle_data_frame(
        self, connection: HTXConnection, frame: HTXFrame
    ) -> None:
        """Handle incoming DATA frame (BN-5.4)."""
        stream = self._get_or_create_stream(connection, frame.stream_id)

        # Add data to stream
        stream.data_received += frame.payload
        stream.last_activity = time.time()

        # Update flow control windows
        data_size = len(frame.payload)
        stream.flow_control_window -= data_size
        connection.connection_window -= data_size
        connection.bytes_received += data_size

        # Send WINDOW_UPDATE when window falls below 50% (BN-5.4)
        if stream.flow_control_window <= 32768:  # 50% of default 65536
            window_delta = 65536 - stream.flow_control_window
            window_update_frame = create_window_update_frame(
                stream.stream_id, window_delta
            )

            # Encrypt and send
            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    window_update_frame
                )
                await self._send_frame_data(connection, encrypted_frame)

            stream.flow_control_window += window_delta
            logger.debug(
                f"Sent WINDOW_UPDATE for stream {stream.stream_id}: +{window_delta}"
            )

        # Connection-level flow control
        if connection.connection_window <= 32768:
            connection_delta = 65536 - connection.connection_window
            conn_window_update = create_window_update_frame(0, connection_delta)

            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    conn_window_update
                )
                await self._send_frame_data(connection, encrypted_frame)

            connection.connection_window += connection_delta
            logger.debug(f"Sent connection WINDOW_UPDATE: +{connection_delta}")

    async def _handle_window_update_frame(
        self, connection: HTXConnection, frame: HTXFrame
    ) -> None:
        """Handle WINDOW_UPDATE frame (BN-5.4)."""
        if len(frame.payload) < 4:
            logger.warning("Invalid WINDOW_UPDATE frame: payload too short")
            return

        window_delta = struct.unpack(">I", frame.payload[:4])[0]

        if frame.stream_id == 0:
            # Connection-level window update
            connection.connection_window += window_delta
            logger.debug(
                f"Updated connection window: +{window_delta} (now {connection.connection_window})"
            )
        else:
            # Stream-level window update
            stream = connection.streams.get(frame.stream_id)
            if stream:
                stream.flow_control_window += window_delta
                logger.debug(
                    f"Updated stream {frame.stream_id} window: +{window_delta} (now {stream.flow_control_window})"
                )

    async def _handle_ping_frame(
        self, connection: HTXConnection, frame: HTXFrame
    ) -> None:
        """Handle PING frame - respond with same payload."""
        pong_frame = HTXFrameCodec.encode_frame(HTXFrameType.PING, 0, frame.payload)

        if connection.noise_protocol:
            encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                pong_frame
            )
            await self._send_frame_data(connection, encrypted_frame)

        logger.debug("Responded to PING")

    async def _handle_key_update_frame(
        self, connection: HTXConnection, frame: HTXFrame
    ) -> None:
        """Handle KEY_UPDATE frame (BN-5.3)."""
        if not connection.noise_protocol:
            logger.warning("Received KEY_UPDATE but no Noise protocol active")
            return

        success = connection.noise_protocol.process_key_update(frame.payload)
        if success:
            # Send ACK by finalizing the update
            connection.noise_protocol.finalize_key_update()
            logger.info("Key update completed successfully")
        else:
            logger.error("Key update failed")

    async def _send_window_update_if_needed(
        self, connection: HTXConnection, stream: HTXStream
    ) -> None:
        """Send WINDOW_UPDATE if stream window is low (BN-5.4)."""
        if stream.flow_control_window <= 32768:  # 50% threshold
            window_delta = 65536 - stream.flow_control_window
            window_update_frame = create_window_update_frame(
                stream.stream_id, window_delta
            )

            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    window_update_frame
                )
                await self._send_frame_data(connection, encrypted_frame)

            stream.flow_control_window += window_delta
            logger.debug(
                f"Sent WINDOW_UPDATE for stream {stream.stream_id}: +{window_delta}"
            )

    async def _send_frame_data(
        self, connection: HTXConnection, frame_data: bytes
    ) -> bool:
        """Send frame data over connection."""
        # Simulate sending (would write to actual socket)
        await asyncio.sleep(0.001)  # Simulate write latency
        return True

    async def _send_raw_data(self, connection: HTXConnection, data: bytes) -> None:
        """Send raw data during handshake."""
        # Simulate sending
        await asyncio.sleep(0.005)

    async def _receive_raw_data(
        self, connection: HTXConnection, expected_min_size: int
    ) -> bytes:
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

    async def _ping_loop(self) -> None:
        """H2/H3 PING cadence emulation (BN-5.5)."""
        import random

        while self.is_running:
            try:
                # Randomized interval [10-60]s ±10% (BN-5.5)
                min_interval = 10.0
                max_interval = 60.0
                base_interval = random.uniform(min_interval, max_interval)
                variance = base_interval * 0.1  # ±10%
                interval = random.uniform(
                    base_interval - variance, base_interval + variance
                )

                await asyncio.sleep(interval)

                if not self.is_running:
                    break

                # Send PING to all connected clients
                ping_tasks = []
                for connection in list(self.connections.values()):
                    if connection.state == HTXConnectionState.CONNECTED:
                        ping_tasks.append(self._send_ping_frame(connection))

                if ping_tasks:
                    await asyncio.gather(*ping_tasks, return_exceptions=True)

            except Exception as e:
                logger.exception(f"PING loop error: {e}")
                await asyncio.sleep(30.0)

    async def _idle_padding_loop(self) -> None:
        """H2/H3 idle padding emulation (BN-5.5)."""
        import random

        while self.is_running:
            try:
                # Check idle connections and send padding
                current_time = time.time()

                for connection in list(self.connections.values()):
                    if connection.state != HTXConnectionState.CONNECTED:
                        continue

                    # Check if connection has been idle
                    idle_time_ms = (current_time - connection.last_activity) * 1000
                    min_idle, max_idle = self.idle_timeout_range

                    # Randomized idle threshold [200-1200]ms (BN-5.5)
                    idle_threshold = random.uniform(min_idle, max_idle)

                    if idle_time_ms >= idle_threshold:
                        # Send idle padding [0-3 KiB] (BN-5.5)
                        padding_size = random.randint(*self.idle_padding_range)
                        if padding_size > 0:
                            await self._send_idle_padding(connection, padding_size)

                # Send PRIORITY frames at low rate [0.5-3%] (BN-5.5)
                if random.random() < self.priority_frame_rate:
                    connection_list = list(self.connections.values())
                    if connection_list:
                        connection = random.choice(connection_list)
                        if connection.state == HTXConnectionState.CONNECTED:
                            await self._send_priority_frame(connection)

                # Check every 100ms for responsive idle detection
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"Idle padding loop error: {e}")
                await asyncio.sleep(1.0)

    async def _send_ping_frame(self, connection: HTXConnection) -> None:
        """Send PING frame for keepalive (BN-5.5)."""
        try:
            ping_frame = HTXFrameCodec.encode_frame(
                HTXFrameType.PING, 0, secrets.token_bytes(8)
            )

            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    ping_frame
                )
                await self._send_frame_data(connection, encrypted_frame)

            logger.debug(f"Sent PING to connection {connection.connection_id}")

        except Exception as e:
            logger.warning(f"Failed to send PING to {connection.connection_id}: {e}")

    async def _send_idle_padding(
        self, connection: HTXConnection, padding_size: int
    ) -> None:
        """Send idle padding DATA frame (BN-5.5)."""
        try:
            # Create padding DATA on a dummy stream
            padding_data = secrets.token_bytes(padding_size)
            padding_frame = HTXFrameCodec.encode_frame(
                HTXFrameType.PADDING, 0, padding_data
            )

            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    padding_frame
                )
                await self._send_frame_data(connection, encrypted_frame)

            connection.last_activity = (
                time.time()
            )  # Update activity to prevent immediate re-padding
            logger.debug(
                f"Sent {padding_size} bytes idle padding to {connection.connection_id}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to send idle padding to {connection.connection_id}: {e}"
            )

    async def _send_priority_frame(self, connection: HTXConnection) -> None:
        """Send low-rate PRIORITY frame (BN-5.5)."""
        try:
            # Create minimal PRIORITY frame (stream dependency info)
            import random

            # Random stream ID for priority signal
            target_stream = (
                random.choice(list(connection.streams.keys()))
                if connection.streams
                else 1
            )
            priority_data = struct.pack(
                ">IB", target_stream, random.randint(1, 256)
            )  # stream + weight

            priority_frame = HTXFrameCodec.encode_frame(
                HTXFrameType.PRIORITY, 0, priority_data
            )

            if connection.noise_protocol:
                encrypted_frame = connection.noise_protocol.encrypt_transport_message(
                    priority_frame
                )
                await self._send_frame_data(connection, encrypted_frame)

            logger.debug(f"Sent PRIORITY frame to {connection.connection_id}")

        except Exception as e:
            logger.warning(
                f"Failed to send PRIORITY frame to {connection.connection_id}: {e}"
            )

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
            "fingerprint": connection.fingerprint.ja3_hash[:8] + "..."
            if connection.fingerprint
            else None,
            "noise_ready": connection.noise_protocol.state
            == NoiseHandshakeState.HANDSHAKE_COMPLETE
            if connection.noise_protocol
            else False,
            "has_ticket": connection.access_ticket is not None,
        }

    def get_transport_statistics(self) -> dict:
        """Get comprehensive transport statistics."""
        active_connections = len(self.connections)
        total_streams = sum(len(conn.streams) for conn in self.connections.values())
        total_bytes_sent = sum(conn.bytes_sent for conn in self.connections.values())
        total_bytes_received = sum(
            conn.bytes_received for conn in self.connections.values()
        )

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
