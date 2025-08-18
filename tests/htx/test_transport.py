"""
Comprehensive Test Suite for HTX Transport Coordinator - Betanet v1.1

Tests the main HTX transport implementation that coordinates all components:
- Connection establishment with uTLS fingerprinting
- Noise XK handshake coordination
- Access ticket authentication flow
- Stream management and flow control
- Message transmission and reception
- Integration between all modular components

Building on existing test patterns from the codebase.
"""

import os
import secrets
import sys
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path following existing pattern
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.p2p.htx.access_tickets import AccessTicketManager, TicketType, generate_issuer_keypair
from core.p2p.htx.transport import HTXConnection, HTXConnectionState, HTXStream, HTXStreamState, HTXTransport
from core.p2p.htx.utls_fingerprinting import uTLSFingerprintCalibrator


class TestHTXStreamState:
    """Test HTX stream state enumeration."""

    def test_stream_state_values(self):
        """Test stream state values."""
        assert HTXStreamState.IDLE.value == "idle"
        assert HTXStreamState.OPEN.value == "open"
        assert HTXStreamState.HALF_CLOSED_LOCAL.value == "half_closed_local"
        assert HTXStreamState.HALF_CLOSED_REMOTE.value == "half_closed_remote"
        assert HTXStreamState.CLOSED.value == "closed"


class TestHTXConnectionState:
    """Test HTX connection state enumeration."""

    def test_connection_state_values(self):
        """Test connection state values."""
        assert HTXConnectionState.DISCONNECTED.value == "disconnected"
        assert HTXConnectionState.TLS_HANDSHAKING.value == "tls_handshaking"
        assert HTXConnectionState.NOISE_HANDSHAKING.value == "noise_handshaking"
        assert HTXConnectionState.ACCESS_AUTHENTICATING.value == "access_authenticating"
        assert HTXConnectionState.CONNECTED.value == "connected"
        assert HTXConnectionState.ERROR.value == "error"
        assert HTXConnectionState.CLOSING.value == "closing"


class TestHTXStream:
    """Test HTX stream structure."""

    def test_stream_creation(self):
        """Test basic stream creation."""
        stream = HTXStream(stream_id=42)

        assert stream.stream_id == 42
        assert stream.state == HTXStreamState.IDLE
        assert stream.flow_control_window == 65536  # Default 64KB
        assert stream.data_received == b""
        assert stream.data_to_send == b""
        assert stream.created_at <= time.time()
        assert stream.last_activity <= time.time()

    def test_stream_with_custom_values(self):
        """Test stream with custom values."""
        stream = HTXStream(
            stream_id=100,
            state=HTXStreamState.OPEN,
            flow_control_window=131072,  # 128KB
        )

        assert stream.stream_id == 100
        assert stream.state == HTXStreamState.OPEN
        assert stream.flow_control_window == 131072


class TestHTXConnection:
    """Test HTX connection structure."""

    def test_connection_creation(self):
        """Test basic connection creation."""
        connection = HTXConnection(connection_id="test_conn_123", remote_address="example.com:443")

        assert connection.connection_id == "test_conn_123"
        assert connection.remote_address == "example.com:443"
        assert connection.state == HTXConnectionState.DISCONNECTED
        assert connection.fingerprint is None
        assert connection.noise_protocol is None
        assert connection.access_ticket is None
        assert len(connection.streams) == 0
        assert connection.next_stream_id == 1
        assert connection.connection_window == 65536
        assert connection.bytes_sent == 0
        assert connection.bytes_received == 0

    def test_connection_with_components(self):
        """Test connection with protocol components."""
        # Create mock components
        fingerprint = Mock()
        noise_protocol = Mock()
        access_ticket = Mock()

        connection = HTXConnection(
            connection_id="test_conn_456",
            remote_address="test.example.com:8443",
            state=HTXConnectionState.CONNECTED,
            fingerprint=fingerprint,
            noise_protocol=noise_protocol,
            access_ticket=access_ticket,
        )

        assert connection.state == HTXConnectionState.CONNECTED
        assert connection.fingerprint == fingerprint
        assert connection.noise_protocol == noise_protocol
        assert connection.access_ticket == access_ticket


class TestHTXTransport:
    """Test HTX transport coordinator."""

    def test_transport_initialization(self):
        """Test transport initialization."""
        transport = HTXTransport()

        assert isinstance(transport.fingerprint_calibrator, uTLSFingerprintCalibrator)
        assert isinstance(transport.ticket_manager, AccessTicketManager)
        assert transport.fingerprint_template == "chrome_120_windows"  # Default
        assert transport.known_server_key is None
        assert len(transport.connections) == 0
        assert len(transport.message_handlers) == 0
        assert len(transport.connection_handlers) == 0
        assert transport.is_running is False

    def test_transport_initialization_with_params(self):
        """Test transport initialization with parameters."""
        server_key = secrets.token_bytes(32)
        ticket_manager = AccessTicketManager()

        transport = HTXTransport(
            known_server_key=server_key,
            ticket_manager=ticket_manager,
            fingerprint_template="firefox_121_linux",
        )

        assert transport.known_server_key == server_key
        assert transport.ticket_manager == ticket_manager
        assert transport.fingerprint_template == "firefox_121_linux"

    @pytest.mark.asyncio
    async def test_transport_start_stop(self):
        """Test transport start and stop lifecycle."""
        transport = HTXTransport()

        # Start transport
        result = await transport.start()
        assert result is True
        assert transport.is_running is True
        assert transport.cleanup_task is not None

        # Stop transport
        await transport.stop()
        assert transport.is_running is False

        # Cleanup task should be cancelled
        if transport.cleanup_task:
            assert transport.cleanup_task.cancelled()

    @pytest.mark.asyncio
    async def test_transport_double_start(self):
        """Test starting transport twice."""
        transport = HTXTransport()

        # First start should succeed
        result1 = await transport.start()
        assert result1 is True

        # Second start should also succeed (no-op)
        result2 = await transport.start()
        assert result2 is True

        await transport.stop()

    @pytest.mark.asyncio
    async def test_connect_without_server_key(self):
        """Test connection without known server key."""
        transport = HTXTransport()
        await transport.start()

        # Should fail without server key for XK pattern
        connection_id = await transport.connect("example.com:443")
        assert connection_id is None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_connect_with_server_key(self):
        """Test connection with known server key."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Mock the internal connection methods
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock) as mock_tls,
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock) as mock_noise,
            patch.object(transport, "_authenticate_connection", new_callable=AsyncMock) as mock_auth,
        ):
            connection_id = await transport.connect("secure.example.com:443")

            # Should succeed with mocked methods
            assert connection_id is not None
            assert connection_id.startswith("client_")
            assert len(connection_id) > 10  # Should include random component

            # Should have called handshake methods
            mock_tls.assert_called_once()
            mock_noise.assert_called_once()
            # Auth should not be called without access ticket
            mock_auth.assert_not_called()

            # Connection should be in connections dict
            assert connection_id in transport.connections
            connection = transport.connections[connection_id]
            assert connection.state == HTXConnectionState.CONNECTED
            assert connection.remote_address == "secure.example.com:443"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_connect_with_access_ticket(self):
        """Test connection with access ticket."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Create access ticket
        ticket_manager = AccessTicketManager()
        private_key, public_key = generate_issuer_keypair()
        ticket_manager.add_trusted_issuer("test_issuer", public_key)

        access_ticket = ticket_manager.issue_ticket(
            "test_issuer", "test_subject", TicketType.PREMIUM, private_key=private_key
        )

        # Mock the internal methods
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock) as mock_tls,
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock) as mock_noise,
            patch.object(transport, "_authenticate_connection", new_callable=AsyncMock) as mock_auth,
        ):
            connection_id = await transport.connect("premium.example.com:443", access_ticket=access_ticket)

            assert connection_id is not None

            # All handshake methods should be called
            mock_tls.assert_called_once()
            mock_noise.assert_called_once()
            mock_auth.assert_called_once()

            # Connection should have the access ticket
            connection = transport.connections[connection_id]
            assert connection.access_ticket == access_ticket

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_message_no_connection(self):
        """Test sending message without connection."""
        transport = HTXTransport()
        await transport.start()

        # Should fail with non-existent connection
        result = await transport.send_message("nonexistent_conn", b"test data")
        assert result is False

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_message_with_connection(self):
        """Test sending message with valid connection."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Mock connection establishment
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
            patch.object(transport, "_send_frame_data", new_callable=AsyncMock, return_value=True) as mock_send,
        ):
            # Establish connection
            connection_id = await transport.connect("message.example.com:443")
            assert connection_id is not None

            # Send message
            test_data = b"Hello, HTX transport!"
            result = await transport.send_message(connection_id, test_data)

            assert result is True
            mock_send.assert_called_once()

            # Check connection statistics
            connection = transport.connections[connection_id]
            assert connection.bytes_sent > 0
            assert connection.frames_sent > 0

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_message_flow_control(self):
        """Test send message respects flow control."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Mock connection establishment
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
        ):
            connection_id = await transport.connect("flowcontrol.example.com:443")
            connection = transport.connections[connection_id]

            # Create stream with small flow control window
            stream = transport._get_or_create_stream(connection, 1)
            stream.flow_control_window = 100  # Small window

            # Try to send large message
            large_data = b"X" * 1000  # Larger than window
            result = await transport.send_message(connection_id, large_data, stream_id=1)

            # Should fail due to flow control
            assert result is False

        await transport.stop()

    @pytest.mark.asyncio
    async def test_receive_message(self):
        """Test receiving message from connection."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Mock connection establishment
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
        ):
            connection_id = await transport.connect("receive.example.com:443")
            connection = transport.connections[connection_id]

            # Simulate received data
            stream = transport._get_or_create_stream(connection, 5)
            stream.data_received = b"Received message data"

            # Receive message
            result = await transport.receive_message(connection_id)

            assert result is not None
            stream_id, data = result
            assert stream_id == 5
            assert data == b"Received message data"

            # Data should be consumed from stream
            assert stream.data_received == b""

        await transport.stop()

    @pytest.mark.asyncio
    async def test_receive_message_no_data(self):
        """Test receiving message when no data available."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
        ):
            connection_id = await transport.connect("nodata.example.com:443")

            # No data in streams
            result = await transport.receive_message(connection_id)
            assert result is None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test closing connection."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Mock connection establishment
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
        ):
            connection_id = await transport.connect("close.example.com:443")
            assert connection_id in transport.connections

            # Close connection
            result = await transport.close_connection(connection_id)
            assert result is True

            # Connection should be removed
            assert connection_id not in transport.connections

        await transport.stop()

    @pytest.mark.asyncio
    async def test_close_nonexistent_connection(self):
        """Test closing non-existent connection."""
        transport = HTXTransport()
        await transport.start()

        result = await transport.close_connection("nonexistent")
        assert result is False

        await transport.stop()

    def test_message_handler_registration(self):
        """Test message handler registration."""
        transport = HTXTransport()

        def test_handler(message):
            return f"Handled: {message}"

        transport.add_message_handler("test_message", test_handler)

        assert "test_message" in transport.message_handlers
        assert transport.message_handlers["test_message"] == test_handler

    def test_connection_handler_registration(self):
        """Test connection event handler registration."""
        transport = HTXTransport()

        def connection_handler(connection_id):
            return f"Connection event: {connection_id}"

        transport.add_connection_handler("connection_established", connection_handler)

        assert "connection_established" in transport.connection_handlers
        assert transport.connection_handlers["connection_established"] == connection_handler

    def test_get_or_create_stream(self):
        """Test stream creation and retrieval."""
        transport = HTXTransport()
        connection = HTXConnection("test_conn", "test.com:443")

        # Create new stream
        stream1 = transport._get_or_create_stream(connection, 10)
        assert stream1.stream_id == 10
        assert stream1.state == HTXStreamState.OPEN
        assert 10 in connection.streams

        # Get existing stream
        stream2 = transport._get_or_create_stream(connection, 10)
        assert stream2 == stream1  # Should be same object

        # Create different stream
        stream3 = transport._get_or_create_stream(connection, 20)
        assert stream3.stream_id == 20
        assert stream3 != stream1


class TestHTXTransportStatus:
    """Test HTX transport status and monitoring."""

    @pytest.mark.asyncio
    async def test_get_connection_status(self):
        """Test getting connection status."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Non-existent connection
        status = transport.get_connection_status("nonexistent")
        assert status is None

        # Mock connection establishment
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
        ):
            connection_id = await transport.connect("status.example.com:443")

            status = transport.get_connection_status(connection_id)
            assert status is not None
            assert status["connection_id"] == connection_id
            assert status["state"] == "connected"
            assert status["remote_address"] == "status.example.com:443"
            assert status["stream_count"] == 0
            assert status["bytes_sent"] >= 0
            assert status["bytes_received"] >= 0
            assert "uptime_seconds" in status
            assert "fingerprint" in status
            assert "noise_ready" in status

        await transport.stop()

    @pytest.mark.asyncio
    async def test_get_transport_statistics(self):
        """Test getting transport statistics."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key, fingerprint_template="safari_17_macos")
        await transport.start()

        stats = transport.get_transport_statistics()

        assert stats["transport_running"] is True
        assert stats["active_connections"] == 0  # No connections yet
        assert stats["total_streams"] == 0
        assert stats["total_bytes_sent"] == 0
        assert stats["total_bytes_received"] == 0
        assert stats["fingerprint_template"] == "safari_17_macos"
        assert stats["max_connections"] == 100  # Default
        assert stats["connection_timeout"] == 300.0  # Default
        assert "ticket_manager_stats" in stats
        assert "fingerprint_stats" in stats

        await transport.stop()

        # After stopping
        stats_stopped = transport.get_transport_statistics()
        assert stats_stopped["transport_running"] is False


class TestHTXTransportIntegration:
    """Integration tests for HTX transport system."""

    @pytest.mark.asyncio
    async def test_full_connection_workflow(self):
        """Test complete connection establishment workflow."""
        # Set up transport with all components
        server_key = secrets.token_bytes(32)
        ticket_manager = AccessTicketManager()
        private_key, public_key = generate_issuer_keypair()
        ticket_manager.add_trusted_issuer("integration_issuer", public_key)

        transport = HTXTransport(
            known_server_key=server_key,
            ticket_manager=ticket_manager,
            fingerprint_template="chrome_120_windows",
        )

        # Create access ticket
        access_ticket = ticket_manager.issue_ticket(
            "integration_issuer",
            "integration_subject",
            TicketType.PREMIUM,
            private_key=private_key,
        )

        await transport.start()

        # Mock all the network operations
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock) as mock_tls,
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock) as mock_noise,
            patch.object(transport, "_authenticate_connection", new_callable=AsyncMock) as mock_auth,
            patch.object(transport, "_send_frame_data", new_callable=AsyncMock, return_value=True),
        ):
            # 1. Establish connection
            connection_id = await transport.connect("integration.example.com:443", access_ticket=access_ticket)
            assert connection_id is not None

            # 2. Verify all handshake steps were performed
            mock_tls.assert_called_once()
            mock_noise.assert_called_once()
            mock_auth.assert_called_once()

            # 3. Send message
            message_data = b"Integration test message"
            send_result = await transport.send_message(connection_id, message_data)
            assert send_result is True

            # 4. Check connection status
            status = transport.get_connection_status(connection_id)
            assert status["state"] == "connected"
            assert status["has_ticket"] is True
            assert status["bytes_sent"] > 0

            # 5. Close connection
            close_result = await transport.close_connection(connection_id)
            assert close_result is True

            # 6. Verify cleanup
            assert connection_id not in transport.connections

        await transport.stop()

    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test managing multiple concurrent connections."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        connection_ids = []

        # Mock network operations
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock),
        ):
            # Establish multiple connections
            for i in range(5):
                connection_id = await transport.connect(f"multi{i}.example.com:443")
                assert connection_id is not None
                connection_ids.append(connection_id)

            # All connections should be active
            assert len(transport.connections) == 5

            # Check statistics
            stats = transport.get_transport_statistics()
            assert stats["active_connections"] == 5

            # Close all connections
            for connection_id in connection_ids:
                result = await transport.close_connection(connection_id)
                assert result is True

            # All connections should be closed
            assert len(transport.connections) == 0

        await transport.stop()

    @pytest.mark.asyncio
    async def test_component_integration(self):
        """Test integration between all modular components."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Mock network operations but verify component interactions
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock) as mock_tls,
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock) as mock_noise,
        ):
            connection_id = await transport.connect("component.example.com:443")
            connection = transport.connections[connection_id]

            # Verify component integration
            # 1. Fingerprint should be calibrated
            mock_tls.assert_called_once()
            tls_call_args = mock_tls.call_args[0]
            tls_connection = tls_call_args[0]
            assert tls_connection.fingerprint is not None
            assert tls_connection.fingerprint.browser_type == "chrome"  # Default template

            # 2. Noise protocol should be initialized
            mock_noise.assert_called_once()
            noise_call_args = mock_noise.call_args[0]
            noise_connection = noise_call_args[0]
            assert noise_connection.noise_protocol is not None
            assert noise_connection.noise_protocol.is_initiator is True
            assert noise_connection.noise_protocol.keys.remote_static == server_key

            # 3. Frame buffer should be initialized
            assert connection.frame_buffer is not None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in various scenarios."""
        server_key = secrets.token_bytes(32)
        transport = HTXTransport(known_server_key=server_key)
        await transport.start()

        # Test TLS handshake failure
        with patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock) as mock_tls:
            mock_tls.side_effect = RuntimeError("TLS handshake failed")

            connection_id = await transport.connect("error.example.com:443")
            assert connection_id is None  # Should fail

        # Test Noise handshake failure
        with (
            patch.object(transport, "_perform_tls_handshake", new_callable=AsyncMock),
            patch.object(transport, "_perform_noise_handshake", new_callable=AsyncMock) as mock_noise,
        ):
            mock_noise.side_effect = RuntimeError("Noise handshake failed")

            connection_id = await transport.connect("noise_error.example.com:443")
            assert connection_id is None  # Should fail

        await transport.stop()


def test_htx_transport_smoke_test():
    """Smoke test for HTX transport functionality."""
    print("Running HTX transport smoke test...")

    # Test transport initialization
    server_key = secrets.token_bytes(32)
    transport = HTXTransport(known_server_key=server_key, fingerprint_template="firefox_121_linux")

    assert transport.known_server_key == server_key
    assert transport.fingerprint_template == "firefox_121_linux"
    assert len(transport.connections) == 0
    print(f"  Transport initialized: template={transport.fingerprint_template}")

    # Test connection structure
    connection = HTXConnection("test_conn", "smoke.example.com:443")
    assert connection.connection_id == "test_conn"
    assert connection.state == HTXConnectionState.DISCONNECTED
    print(f"  Connection created: id={connection.connection_id}, state={connection.state.value}")

    # Test stream structure
    stream = HTXStream(stream_id=42, state=HTXStreamState.OPEN)
    assert stream.stream_id == 42
    assert stream.state == HTXStreamState.OPEN
    assert stream.flow_control_window == 65536
    print(f"  Stream created: id={stream.stream_id}, window={stream.flow_control_window} bytes")

    # Test component integration
    assert transport.fingerprint_calibrator is not None
    assert transport.ticket_manager is not None

    calibrator_stats = transport.fingerprint_calibrator.get_fingerprint_stats()
    ticket_stats = transport.ticket_manager.get_statistics()

    print(f"  Components: fingerprint_templates={len(calibrator_stats['available_templates'])}")
    print(f"             ticket_manager={ticket_stats['trusted_issuers']} issuers")

    # Test statistics
    stats = transport.get_transport_statistics()
    assert stats["transport_running"] is False
    assert stats["active_connections"] == 0
    print(f"  Statistics: running={stats['transport_running']}, connections={stats['active_connections']}")

    print("  HTX transport smoke test PASSED")


if __name__ == "__main__":
    # Run smoke test when executed directly
    test_htx_transport_smoke_test()
    print("\nTo run full test suite:")
    print("  pytest tests/htx/test_transport.py -v")
