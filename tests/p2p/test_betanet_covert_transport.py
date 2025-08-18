"""
Betanet HTX/H2/H3 Covert Transport Tests - Prompt 2

Tests for the covert transport capabilities including:
- HTTP/2 multiplexed channels
- HTTP/3 QUIC streams
- WebSocket persistent connections
- Server-Sent Events streaming
- Cover traffic generation
- Protocol auto-negotiation
"""

import os
import sys
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.p2p.betanet_covert_transport import (
    BetanetCovertTransport,
    CovertTrafficProfile,
    CovertTransportMode,
    HTTP2CovertChannel,
    HTTP3CovertChannel,
    ServerSentEventsCovertChannel,
    WebSocketCovertChannel,
    enhance_betanet_with_covert_transport,
)


class TestCovertTrafficProfile:
    """Test covert traffic profiling system."""

    def test_browser_profile_creation(self):
        """Test creating realistic browser traffic profile."""
        profile = CovertTrafficProfile.create_browser_profile()

        assert len(profile.user_agents) >= 4, "Should have multiple realistic user agents"
        assert len(profile.content_types) >= 3, "Should have multiple content types"
        assert len(profile.request_paths) >= 5, "Should have multiple API paths"

        # Check realistic user agents
        for ua in profile.user_agents:
            assert "Mozilla" in ua, "User agents should be realistic"
            assert any(browser in ua for browser in ["Chrome", "Firefox", "Safari"])

        # Check realistic paths
        assert any("/api/" in path for path in profile.request_paths)
        assert any("/cdn/" in path for path in profile.request_paths)

        # Check timing and payload ranges are reasonable
        assert 0.1 <= profile.timing_intervals[0] <= profile.timing_intervals[1] <= 10.0
        assert 64 <= profile.payload_size_range[0] <= profile.payload_size_range[1] <= 16384


class TestHTTP2CovertChannel:
    """Test HTTP/2 multiplexed covert channel."""

    def test_channel_initialization(self):
        """Test HTTP/2 channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        assert channel.profile == profile
        assert channel.connection is None
        assert channel.next_stream_id == 1
        assert len(channel.streams) == 0

    def test_realistic_headers_generation(self):
        """Test generation of realistic HTTP/2 headers."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        headers = channel._generate_realistic_headers(1024)

        # Check required HTTP/2 pseudo-headers
        pseudo_headers = {name for name, value in headers if name.startswith(":")}
        assert ":method" in pseudo_headers
        assert ":path" in pseudo_headers
        assert ":scheme" in pseudo_headers
        assert ":authority" in pseudo_headers

        # Check realistic headers
        header_dict = dict(headers)
        assert "user-agent" in header_dict
        assert "content-type" in header_dict
        assert header_dict["content-length"] == "1024"

        # Verify user agent is from profile
        assert header_dict["user-agent"] in profile.user_agents

    @patch("core.p2p.betanet_covert_transport.HTTP2_AVAILABLE", True)
    @patch("h2.connection.H2Connection")
    async def test_connection_establishment(self, mock_h2_connection):
        """Test HTTP/2 connection establishment."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        # Mock successful connection
        mock_conn = MagicMock()
        mock_h2_connection.return_value = mock_conn

        result = await channel.establish_connection("example.com", 443)

        assert result is True
        assert channel.connection is not None
        mock_conn.initiate_connection.assert_called_once()


class TestHTTP3CovertChannel:
    """Test HTTP/3 QUIC covert channel."""

    def test_channel_initialization(self):
        """Test HTTP/3 channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP3CovertChannel(profile)

        assert channel.profile == profile
        assert channel.connection is None
        assert channel.quic_connection is None

    def test_h3_headers_generation(self):
        """Test HTTP/3 headers generation."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP3CovertChannel(profile)

        headers = channel._generate_h3_headers(2048)

        # Check binary pseudo-headers for HTTP/3
        header_dict = {name.decode(): value.decode() for name, value in headers}

        assert ":method" in header_dict
        assert ":path" in header_dict
        assert ":scheme" in header_dict
        assert ":authority" in header_dict
        assert header_dict["content-length"] == "2048"


class TestWebSocketCovertChannel:
    """Test WebSocket persistent covert channel."""

    def test_channel_initialization(self):
        """Test WebSocket channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = WebSocketCovertChannel(profile)

        assert channel.profile == profile
        assert channel.websocket is None
        assert channel.is_connected is False

    async def test_connection_establishment_mock(self):
        """Test WebSocket connection establishment with mocking."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = WebSocketCovertChannel(profile)

        result = await channel.establish_connection("wss://example.com/ws")

        # Should succeed in test environment
        assert result is True
        assert channel.is_connected is True


class TestServerSentEventsCovertChannel:
    """Test SSE streaming covert channel."""

    def test_channel_initialization(self):
        """Test SSE channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = ServerSentEventsCovertChannel(profile)

        assert channel.profile == profile
        assert channel.event_stream is None

    async def test_sse_event_formatting(self):
        """Test Server-Sent Events formatting."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = ServerSentEventsCovertChannel(profile)

        test_data = b"test payload data"
        result = await channel.send_covert_event(test_data, "test_event")

        # Should format correctly (implementation detail test)
        assert result is True


class TestBetanetCovertTransport:
    """Test main covert transport orchestrator."""

    def test_transport_initialization(self):
        """Test covert transport initialization."""
        transport = BetanetCovertTransport(CovertTransportMode.HTTP2)

        assert transport.mode == CovertTransportMode.HTTP2
        assert transport.profile is not None
        assert len(transport.active_channels) == 0
        assert transport.cover_traffic_enabled is False

    def test_auto_mode_initialization(self):
        """Test auto mode initialization."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        assert transport.mode == CovertTransportMode.AUTO
        # Should have channels based on availability
        assert transport.websocket_channel is not None
        assert transport.sse_channel is not None

    async def test_channel_establishment_websocket(self):
        """Test establishing WebSocket channel."""
        transport = BetanetCovertTransport(CovertTransportMode.WEBSOCKET)

        result = await transport._establish_channel(CovertTransportMode.WEBSOCKET, "example.com", 443)

        assert result is True
        assert CovertTransportMode.WEBSOCKET in transport.active_channels

    async def test_covert_data_transmission(self):
        """Test sending data through covert channel."""
        transport = BetanetCovertTransport(CovertTransportMode.WEBSOCKET)

        # Establish WebSocket channel first
        await transport._establish_channel(CovertTransportMode.WEBSOCKET, "example.com", 443)

        test_data = b"covert message payload"
        result = await transport.send_covert_data(test_data)

        assert result is True

    def test_dummy_payload_generation(self):
        """Test cover traffic payload generation."""
        transport = BetanetCovertTransport()

        payload = transport._generate_dummy_payload(1024)

        assert len(payload) <= 1024  # Should be approximately the requested size
        assert b"timestamp" in payload  # Should contain realistic JSON structure
        assert b"session_id" in payload
        assert b"user_agent" in payload

    async def test_cover_traffic_lifecycle(self):
        """Test cover traffic start/stop lifecycle."""
        transport = BetanetCovertTransport()

        assert transport.cover_traffic_enabled is False

        await transport.start_cover_traffic()
        assert transport.cover_traffic_enabled is True
        assert transport.cover_traffic_task is not None

        await transport.stop_cover_traffic()
        assert transport.cover_traffic_enabled is False

    def test_status_reporting(self):
        """Test covert transport status reporting."""
        transport = BetanetCovertTransport(CovertTransportMode.HTTP3)

        status = transport.get_status()

        assert status["mode"] == "http3"
        assert "active_channels" in status
        assert "cover_traffic_enabled" in status
        assert "http2_available" in status
        assert "http3_available" in status
        assert status["channels_established"] == 0


class TestBetanetIntegration:
    """Test integration with existing Betanet transport."""

    def test_transport_enhancement(self):
        """Test enhancing existing transport with covert capabilities."""
        # Mock existing transport
        mock_transport = MagicMock()

        enhanced = enhance_betanet_with_covert_transport(mock_transport, CovertTransportMode.HTTP2)

        assert enhanced == mock_transport
        assert hasattr(mock_transport, "covert_transport")
        assert hasattr(mock_transport, "send_covert_message")
        assert hasattr(mock_transport, "start_covert_mode")
        assert isinstance(mock_transport.covert_transport, BetanetCovertTransport)


class TestCovertTransportIntegration:
    """Integration tests for covert transport system."""

    async def test_full_covert_workflow(self):
        """Test complete covert transport workflow."""
        # Initialize transport
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        # Start covert transport (will use WebSocket in test environment)
        result = await transport.start_covert_transport("example.com")
        assert result is True

        # Send covert data
        test_message = b"secret agent communication"
        send_result = await transport.send_covert_data(test_message)
        assert send_result is True

        # Verify cover traffic is active
        assert transport.cover_traffic_enabled is True

        # Clean shutdown
        await transport.stop()
        assert len(transport.active_channels) == 0
        assert transport.cover_traffic_enabled is False

    async def test_protocol_fallback(self):
        """Test automatic protocol fallback in AUTO mode."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        # Should fallback to available protocols
        result = await transport.start_covert_transport("example.com")

        # Should succeed with some protocol
        assert result is True
        assert len(transport.active_channels) > 0

    def test_realistic_traffic_patterns(self):
        """Test that generated traffic looks realistic."""
        profile = CovertTrafficProfile.create_browser_profile()
        transport = BetanetCovertTransport()

        # Generate multiple payloads
        payloads = [transport._generate_dummy_payload(size) for size in [512, 1024, 2048, 4096]]

        # Check they vary appropriately
        assert len(set(payloads)) == len(payloads), "Payloads should be unique"

        # Check realistic content
        for payload in payloads:
            assert b'"timestamp":' in payload
            assert b'"session_id":' in payload
            # Should contain one of the profile user agents
            assert any(ua.encode() in payload for ua in profile.user_agents)


def test_covert_transport_smoke_test():
    """Basic smoke test for covert transport functionality."""
    print("Running Betanet covert transport smoke test...")

    # Test profile creation
    profile = CovertTrafficProfile.create_browser_profile()
    assert len(profile.user_agents) > 0
    print(f"  Profile created with {len(profile.user_agents)} user agents")

    # Test transport initialization
    transport = BetanetCovertTransport(CovertTransportMode.AUTO)
    assert transport.mode == CovertTransportMode.AUTO
    print(f"  Transport initialized in {transport.mode.value} mode")

    # Test status reporting
    status = transport.get_status()
    assert "mode" in status
    assert "active_channels" in status
    print(f"  Status: {status['channels_established']} channels established")

    print("  Covert transport smoke test PASSED")


if __name__ == "__main__":
    # Run smoke test when executed directly
    test_covert_transport_smoke_test()
    print("\\nTo run full test suite:")
    print("  pytest tests/p2p/test_betanet_covert_transport.py -v")
