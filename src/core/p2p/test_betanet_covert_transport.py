"""Tests for BetanetCovertTransport implementations."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .betanet_covert_transport import (
    BetanetCovertTransport,
    CovertTrafficProfile,
    CovertTransportMode,
    HTTP2CovertChannel,
    ServerSentEventsCovertChannel,
    WebSocketCovertChannel,
)


class TestCovertTrafficProfile:
    """Test CovertTrafficProfile creation and configuration."""

    def test_browser_profile_creation(self):
        """Test creation of realistic browser profile."""
        profile = CovertTrafficProfile.create_browser_profile()

        assert len(profile.user_agents) > 0
        assert len(profile.content_types) > 0
        assert len(profile.request_paths) > 0
        assert len(profile.referrer_patterns) > 0
        assert profile.timing_intervals[0] < profile.timing_intervals[1]
        assert profile.payload_size_range[0] < profile.payload_size_range[1]
        assert profile.compression_enabled is True

    def test_profile_contains_realistic_data(self):
        """Test that profile contains realistic browser data."""
        profile = CovertTrafficProfile.create_browser_profile()

        # Check for common browser user agents
        assert any("Chrome" in ua for ua in profile.user_agents)
        assert any("Firefox" in ua for ua in profile.user_agents)

        # Check for common content types
        assert "application/json" in profile.content_types
        assert "text/plain" in profile.content_types

        # Check for realistic API paths
        assert any("/api/" in path for path in profile.request_paths)


class TestHTTP2CovertChannel:
    """Test HTTP/2 covert channel implementation."""

    def test_initialization(self):
        """Test HTTP/2 channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        assert channel.profile == profile
        assert channel.connection is None
        assert channel.streams == {}
        assert channel.next_stream_id == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_establish_connection_success(self, mock_client):
        """Test successful HTTP/2 connection establishment."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )

        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        result = await channel.establish_connection("example.com", 443)
        assert result is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_establish_connection_failure(self, mock_client):
        """Test HTTP/2 connection establishment failure."""
        # Setup mock for server error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )

        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        result = await channel.establish_connection("example.com", 443)
        assert result is False

    def test_header_generation(self):
        """Test realistic HTTP header generation."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        headers = channel._generate_realistic_headers(1024)

        # Check required headers
        header_dict = dict(headers)
        assert header_dict[":method"] == "POST"
        assert header_dict[":scheme"] == "https"
        assert header_dict["content-length"] == "1024"
        assert header_dict["user-agent"] in profile.user_agents
        assert header_dict["content-type"] in profile.content_types

    def test_payload_compression(self):
        """Test payload compression functionality."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = HTTP2CovertChannel(profile)

        test_data = b"This is test data that should compress well" * 10
        compressed = channel._compress_payload(test_data)

        assert len(compressed) < len(test_data)
        assert compressed != test_data


class TestWebSocketCovertChannel:
    """Test WebSocket covert channel implementation."""

    def test_initialization(self):
        """Test WebSocket channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = WebSocketCovertChannel(profile)

        assert channel.profile == profile
        assert channel.websocket is None
        assert channel.is_connected is False

    @pytest.mark.asyncio
    @patch("websockets.connect")
    async def test_establish_connection_success(self, mock_connect):
        """Test successful WebSocket connection."""
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket

        profile = CovertTrafficProfile.create_browser_profile()
        channel = WebSocketCovertChannel(profile)

        result = await channel.establish_connection("wss://example.com/ws")

        assert result is True
        assert channel.is_connected is True
        assert channel.websocket == mock_websocket

    @pytest.mark.asyncio
    async def test_send_binary_frame(self):
        """Test sending binary WebSocket frame."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = WebSocketCovertChannel(profile)

        # Mock websocket
        mock_websocket = AsyncMock()
        mock_websocket.closed = False
        channel.websocket = mock_websocket
        channel.is_connected = True

        test_data = b"binary test data"
        result = await channel.send_covert_frame(test_data, "binary")

        assert result is True
        mock_websocket.send.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_send_text_frame(self):
        """Test sending text WebSocket frame."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = WebSocketCovertChannel(profile)

        # Mock websocket
        mock_websocket = AsyncMock()
        mock_websocket.closed = False
        channel.websocket = mock_websocket
        channel.is_connected = True

        test_data = b"text test data"
        result = await channel.send_covert_frame(test_data, "text")

        assert result is True
        # Should send base64 encoded data for text frames
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        assert isinstance(sent_data, str)


class TestServerSentEventsCovertChannel:
    """Test Server-Sent Events covert channel implementation."""

    def test_initialization(self):
        """Test SSE channel initialization."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = ServerSentEventsCovertChannel(profile)

        assert channel.profile == profile
        assert channel.event_stream is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_establish_stream_success(self, mock_client_class):
        """Test successful SSE stream establishment."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.head.return_value = mock_response
        mock_client_class.return_value = mock_client

        profile = CovertTrafficProfile.create_browser_profile()
        channel = ServerSentEventsCovertChannel(profile)

        result = await channel.establish_stream("https://example.com/events")

        assert result is True
        assert channel.event_stream == "https://example.com/events"

    @pytest.mark.asyncio
    async def test_send_covert_event(self):
        """Test sending covert event via SSE."""
        profile = CovertTrafficProfile.create_browser_profile()
        channel = ServerSentEventsCovertChannel(profile)

        # Mock client and stream
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        channel.client = mock_client
        channel.event_stream = "https://example.com/events"

        test_data = b"test event data"
        result = await channel.send_covert_event(test_data, "message")

        assert result is True
        mock_client.post.assert_called_once()


class TestBetanetCovertTransport:
    """Test main BetanetCovertTransport orchestrator."""

    def test_initialization(self):
        """Test BetanetCovertTransport initialization."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        assert transport.mode == CovertTransportMode.AUTO
        assert isinstance(transport.profile, CovertTrafficProfile)
        assert transport.active_channels == {}
        assert transport.cover_traffic_enabled is False

    def test_initialization_with_specific_mode(self):
        """Test initialization with specific transport mode."""
        transport = BetanetCovertTransport(CovertTransportMode.WEBSOCKET)

        assert transport.mode == CovertTransportMode.WEBSOCKET
        assert transport.websocket_channel is not None

    @pytest.mark.asyncio
    async def test_auto_mode_channel_selection(self):
        """Test automatic channel selection in AUTO mode."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        # Mock channel establishment
        with patch.object(transport, "_establish_channel") as mock_establish:
            mock_establish.side_effect = [
                False,
                False,
                True,
            ]  # HTTP3 fails, HTTP2 fails, WebSocket succeeds

            result = await transport.start_covert_transport("example.com", 443)

            assert result is True
            assert transport.mode == CovertTransportMode.WEBSOCKET

    @pytest.mark.asyncio
    async def test_cover_traffic_generation(self):
        """Test cover traffic generation."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        # Mock active channels
        mock_channel = AsyncMock()
        transport.active_channels[CovertTransportMode.WEBSOCKET] = mock_channel

        # Start cover traffic
        await transport.start_cover_traffic()
        assert transport.cover_traffic_enabled is True
        assert transport.cover_traffic_task is not None

        # Stop cover traffic
        await transport.stop_cover_traffic()
        assert transport.cover_traffic_enabled is False

    def test_dummy_payload_generation(self):
        """Test dummy payload generation for cover traffic."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        payload = transport._generate_dummy_payload(1024)

        assert isinstance(payload, bytes)
        assert len(payload) >= 1000  # Should be close to requested size

        # Should be valid JSON
        data = json.loads(payload.decode())
        assert "timestamp" in data
        assert "session_id" in data
        assert "user_agent" in data

    def test_status_reporting(self):
        """Test status reporting functionality."""
        transport = BetanetCovertTransport(CovertTransportMode.WEBSOCKET)

        status = transport.get_status()

        assert status["mode"] == "websocket"
        assert "active_channels" in status
        assert "cover_traffic_enabled" in status
        assert "http2_available" in status
        assert "http3_available" in status
        assert "channels_established" in status

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown of covert transport."""
        transport = BetanetCovertTransport(CovertTransportMode.AUTO)

        # Start cover traffic
        await transport.start_cover_traffic()

        # Stop everything
        await transport.stop()

        assert transport.cover_traffic_enabled is False
        assert len(transport.active_channels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
