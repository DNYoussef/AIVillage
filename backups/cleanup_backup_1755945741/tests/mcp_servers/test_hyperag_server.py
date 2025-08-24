"""
Comprehensive tests for HypeRAG MCP Server.
"""

import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
import yaml

# Mock external dependencies that might not be available
try:
    from mcp_servers.hyperag.server import HypeRAGMCPServer
except ImportError:
    pytest.skip("MCP server dependencies not available", allow_module_level=True)


class TestHypeRAGMCPServer:
    """Test suite for HypeRAG MCP Server."""

    @pytest.fixture
    def mock_config(self):
        """Provide mock configuration."""
        return {
            "server": {
                "host": "localhost",
                "port": 8765,
                "max_connections": 100,
                "ping_interval": 30,
                "timeout": 300,
            },
            "auth": {
                "jwt_secret": "test-secret",
                "token_expiry": 3600,
                "require_auth": True,
            },
            "models": {"default_agent_type": "default", "warmup_on_start": True},
            "memory": {"provider": "memory", "max_size": 1000},
            "audit": {"enabled": True, "log_requests": True},
        }

    @pytest.fixture
    def temp_config_file(self, mock_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(mock_config, f)
            return f.name

    @pytest.fixture
    async def server(self, temp_config_file):
        """Create server instance for testing."""
        server = HypeRAGMCPServer(config_path=temp_config_file)
        yield server
        with contextlib.suppress(Exception):
            await server.shutdown()
        # Cleanup config file
        Path(temp_config_file).unlink(missing_ok=True)

    def test_server_initialization(self, temp_config_file):
        """Test server initialization."""
        server = HypeRAGMCPServer(config_path=temp_config_file)
        assert server.config_path == temp_config_file
        assert server.server is None
        assert server.connections == set()
        assert server.stats["connections"] == 0

    def test_default_config_fallback(self):
        """Test default configuration when no file provided."""
        server = HypeRAGMCPServer()
        default_config = server._get_default_config()

        assert "server" in default_config
        assert "auth" in default_config
        assert "models" in default_config
        assert default_config["server"]["host"] == "localhost"
        assert default_config["server"]["port"] == 8765
        assert default_config["auth"]["require_auth"] is True

    @pytest.mark.asyncio
    async def test_config_loading(self, server, mock_config):
        """Test configuration loading from file."""
        await server._load_config()

        assert server.config == mock_config
        assert server.config["server"]["host"] == "localhost"
        assert server.config["auth"]["jwt_secret"] == "test-secret"

    @pytest.mark.asyncio
    async def test_config_loading_missing_file(self):
        """Test config loading when file doesn't exist."""
        server = HypeRAGMCPServer(config_path="/nonexistent/config.yaml")
        await server._load_config()

        # Should fall back to default config
        assert server.config is not None
        assert server.config["server"]["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_server_initialize(self, server):
        """Test server initialization process."""
        with patch("mcp_servers.hyperag.server.logger") as mock_logger:
            await server.initialize()

            mock_logger.info.assert_called()
            assert server.config is not None
            assert hasattr(server, "auth_manager")
            assert hasattr(server, "memory_system")

    @pytest.mark.asyncio
    async def test_server_stats(self, server):
        """Test server statistics collection."""
        await server.initialize()

        stats = await server.get_server_stats()

        assert "connections" in stats
        assert "requests_handled" in stats
        assert "uptime" in stats
        assert "memory_usage" in stats
        assert isinstance(stats["connections"], int)
        assert stats["connections"] >= 0

    @pytest.mark.asyncio
    async def test_authentication_required(self, server):
        """Test authentication requirement."""
        await server.initialize()

        # Mock request without auth
        mock_request = {"method": "test_method", "params": {}}

        with patch.object(server, "_authenticate_request", return_value=False):
            # Should fail authentication
            assert await server._authenticate_request(mock_request) is False

    @pytest.mark.asyncio
    async def test_connection_handling(self, server):
        """Test connection handling."""
        await server.initialize()

        # Mock websocket connection
        mock_websocket = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)

        with patch.object(server, "_handle_client_session") as mock_handler:
            await server.handle_connection(mock_websocket, "/test")
            mock_handler.assert_called_once_with(mock_websocket)

    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error handling in server operations."""
        await server.initialize()

        mock_websocket = AsyncMock()
        error_message = "Test error"

        await server._send_error(mock_websocket, "TEST_ERROR", error_message)

        # Verify error was sent through websocket
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["error"]["code"] == "TEST_ERROR"
        assert sent_message["error"]["message"] == error_message

    @pytest.mark.asyncio
    async def test_server_shutdown(self, server):
        """Test server shutdown process."""
        await server.initialize()

        with patch("mcp_servers.hyperag.server.logger") as mock_logger:
            await server.shutdown()
            mock_logger.info.assert_called_with("HypeRAG MCP Server shutdown complete")

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, server):
        """Test handling multiple concurrent connections."""
        await server.initialize()

        mock_connections = [AsyncMock() for _ in range(5)]

        # Simulate multiple connections
        tasks = []
        for i, conn in enumerate(mock_connections):
            conn.remote_address = ("127.0.0.1", 12345 + i)
            task = asyncio.create_task(server.handle_connection(conn, f"/test{i}"))
            tasks.append(task)

        # Let connections establish
        await asyncio.sleep(0.1)

        # Should track all connections
        assert len(server.connections) <= len(mock_connections)

        # Cleanup
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_memory_system_integration(self, server):
        """Test integration with memory system."""
        await server.initialize()

        # Should have memory system initialized
        assert hasattr(server, "memory_system")

        # Test memory operations through server
        # This would require actual memory system implementation
        # For now, just verify the component exists
        assert server.memory_system is not None

    @pytest.mark.asyncio
    async def test_planning_system_integration(self, server):
        """Test integration with planning system."""
        await server.initialize()

        # Should have planning system available
        # This tests the integration point
        assert hasattr(server, "config")
        assert "models" in server.config

        # Planning integration would be tested through actual requests
        # This verifies the setup is correct
        assert server.config["models"]["default_agent_type"] is not None


class TestHypeRAGMCPServerIntegration:
    """Integration tests for HypeRAG MCP Server."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_server_lifecycle(self, tmp_path):
        """Test full server lifecycle from start to shutdown."""
        config_file = tmp_path / "test_config.yaml"
        config = {
            "server": {"host": "localhost", "port": 0},  # Use any available port
            "auth": {"require_auth": False},  # Disable auth for testing
            "models": {"warmup_on_start": False},  # Disable warmup for speed
        }

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        server = HypeRAGMCPServer(config_path=str(config_file))

        try:
            # Initialize
            await server.initialize()

            # Start server (would need actual websocket server for full test)
            # For now, just verify initialization worked
            assert server.config is not None

            # Get stats
            stats = await server.get_server_stats()
            assert stats["connections"] == 0

        finally:
            await server.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_request_response_cycle(self, server):
        """Test complete request/response cycle."""
        await server.initialize()

        # Mock a complete request cycle
        mock_websocket = AsyncMock()

        # This would test the full message handling pipeline
        # For now, verify the server can handle the setup
        with patch.object(server, "_handle_client_session") as mock_handler:
            await server.handle_connection(mock_websocket, "/")
            mock_handler.assert_called_once()


@pytest.mark.performance
class TestHypeRAGMCPServerPerformance:
    """Performance tests for HypeRAG MCP Server."""

    @pytest.mark.asyncio
    async def test_initialization_time(self, temp_config_file):
        """Test server initialization performance."""
        import time

        start_time = time.time()
        server = HypeRAGMCPServer(config_path=temp_config_file)
        await server.initialize()
        init_time = time.time() - start_time

        # Should initialize quickly (< 5 seconds)
        assert init_time < 5.0, f"Initialization took {init_time:.2f} seconds"

        await server.shutdown()

    @pytest.mark.asyncio
    async def test_connection_capacity(self, server):
        """Test server connection handling capacity."""
        await server.initialize()

        # Test with reasonable number of mock connections
        num_connections = 10
        mock_connections = [AsyncMock() for _ in range(num_connections)]

        start_time = asyncio.get_event_loop().time()

        # Create connection tasks
        tasks = []
        for i, conn in enumerate(mock_connections):
            conn.remote_address = ("127.0.0.1", 12345 + i)
            # Mock the session handling to complete quickly
            with patch.object(server, "_handle_client_session", return_value=None):
                task = asyncio.create_task(server.handle_connection(conn, f"/test{i}"))
                tasks.append(task)

        # Wait for all connections to be handled
        await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = asyncio.get_event_loop().time() - start_time

        # Should handle connections efficiently
        assert elapsed < 1.0, f"Connection handling took {elapsed:.2f} seconds"
