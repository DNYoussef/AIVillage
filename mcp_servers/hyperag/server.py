"""HypeRAG MCP Server Core

Main server implementation for HypeRAG Model Context Protocol server.
"""

import asyncio
import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol
import yaml

from .auth import (
    AuthContext,
    AuthenticationError,
    PermissionManager,
)
from .models import ModelRegistry
from .protocol import MCPProtocolHandler, MCPRequest

logger = logging.getLogger(__name__)


class HypeRAGMCPServer:
    """HypeRAG MCP Server implementation"""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or "config/hyperag_mcp.yaml"
        self.config = {}
        self.permission_manager: PermissionManager | None = None
        self.model_registry: ModelRegistry | None = None
        self.protocol_handler: MCPProtocolHandler | None = None
        self.server = None
        self.start_time = time.time()
        self.active_connections: dict[str, WebSocketServerProtocol] = {}
        self.connection_contexts: dict[str, AuthContext] = {}
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the server components"""
        logger.info("Initializing HypeRAG MCP Server...")

        # Load configuration
        await self._load_config()

        # Initialize permission manager
        jwt_secret = self.config.get("auth", {}).get("jwt_secret", "dev-secret-change-in-production")
        self.permission_manager = PermissionManager(
            jwt_secret=jwt_secret,
            enable_audit=self.config.get("audit", {}).get("enabled", True)
        )

        # Initialize model registry
        self.model_registry = ModelRegistry()

        # Initialize protocol handler
        self.protocol_handler = MCPProtocolHandler(
            permission_manager=self.permission_manager,
            model_registry=self.model_registry,
            storage_backend=None  # TODO: Initialize storage backend
        )

        # Set start time for metrics
        self.protocol_handler.start_time = self.start_time

        logger.info("HypeRAG MCP Server initialized successfully")

    async def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file) as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e!s}")
            self.config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration"""
        return {
            "server": {
                "host": "localhost",
                "port": 8765,
                "max_connections": 100,
                "ping_interval": 30,
                "ping_timeout": 10
            },
            "auth": {
                "jwt_secret": "dev-secret-change-in-production",
                "api_keys_enabled": True,
                "session_timeout_hours": 24
            },
            "audit": {
                "enabled": True,
                "max_entries": 10000
            },
            "models": {
                "default_agent_type": "default",
                "warmup_on_start": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    async def start(self) -> None:
        """Start the MCP server"""
        if not self.protocol_handler:
            await self.initialize()

        host = self.config.get("server", {}).get("host", "localhost")
        port = self.config.get("server", {}).get("port", 8765)

        logger.info(f"Starting HypeRAG MCP Server on {host}:{port}")

        # Set up signal handlers
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(self.shutdown())
                )

        # Start WebSocket server
        self.server = await websockets.serve(
            self.handle_connection,
            host,
            port,
            ping_interval=self.config.get("server", {}).get("ping_interval", 30),
            ping_timeout=self.config.get("server", {}).get("ping_timeout", 10),
            max_size=1024 * 1024,  # 1MB max message size
            compression=None
        )

        logger.info(f"HypeRAG MCP Server started on ws://{host}:{port}")

        # Wait for shutdown
        await self.shutdown_event.wait()

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new WebSocket connection"""
        connection_id = f"conn_{int(time.time() * 1000)}_{id(websocket)}"
        self.active_connections[connection_id] = websocket

        logger.info(f"New connection: {connection_id} from {websocket.remote_address}")

        try:
            await self._handle_client_session(connection_id, websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling connection {connection_id}: {e!s}")
        finally:
            # Clean up connection
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            if connection_id in self.connection_contexts:
                context = self.connection_contexts[connection_id]
                await self.permission_manager.invalidate_session(context.session_id)
                del self.connection_contexts[connection_id]

    async def _handle_client_session(self, connection_id: str, websocket: WebSocketServerProtocol) -> None:
        """Handle client session"""
        context: AuthContext | None = None

        async for message in websocket:
            try:
                # Parse JSON-RPC message
                data = json.loads(message)

                # Validate JSON-RPC format
                if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
                    await self._send_error(websocket, "INVALID_REQUEST", "Invalid JSON-RPC format")
                    continue

                method = data.get("method")
                params = data.get("params", {})
                request_id = data.get("id")

                if not method:
                    await self._send_error(websocket, "INVALID_REQUEST", "Missing method", request_id)
                    continue

                # Create MCP request
                request = MCPRequest(method=method, params=params, request_id=request_id)

                # Handle authentication
                if not context and method != "hyperag/health":
                    context = await self._authenticate_request(params, websocket.remote_address[0])
                    if context:
                        self.connection_contexts[connection_id] = context
                        logger.info(f"Authenticated connection {connection_id} as {context.user_id}")
                    else:
                        await self._send_error(websocket, "AUTH_REQUIRED", "Authentication required", request_id)
                        continue

                # Handle request
                response = await self.protocol_handler.handle_request(request, context)

                # Send response
                await websocket.send(json.dumps(response.to_dict()))

            except json.JSONDecodeError:
                await self._send_error(websocket, "PARSE_ERROR", "Invalid JSON")
            except Exception as e:
                logger.error(f"Error processing message from {connection_id}: {e!s}")
                await self._send_error(websocket, "INTERNAL_ERROR", f"Internal error: {e!s}")

    async def _authenticate_request(self, params: dict[str, Any], ip_address: str) -> AuthContext | None:
        """Authenticate a request"""
        try:
            # Check for JWT token
            auth_header = params.get("auth") or params.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                return await self.permission_manager.authenticate_jwt(token, ip_address)

            # Check for API key
            api_key = params.get("api_key") or params.get("x-api-key", "")
            if api_key:
                return await self.permission_manager.authenticate_api_key(api_key, ip_address)

            return None

        except AuthenticationError as e:
            logger.warning(f"Authentication failed from {ip_address}: {e!s}")
            return None

    async def _send_error(
        self,
        websocket: WebSocketServerProtocol,
        code: str,
        message: str,
        request_id: str | None = None
    ) -> None:
        """Send error response"""
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": request_id
        }
        await websocket.send(json.dumps(error_response))

    async def shutdown(self) -> None:
        """Shutdown the server"""
        logger.info("Shutting down HypeRAG MCP Server...")

        # Close all active connections
        if self.active_connections:
            logger.info(f"Closing {len(self.active_connections)} active connections...")
            for connection_id, websocket in list(self.active_connections.items()):
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {connection_id}: {e!s}")

        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Clean up model registry
        if self.model_registry:
            await self.model_registry.cleanup_all()

        # Signal shutdown complete
        self.shutdown_event.set()

        logger.info("HypeRAG MCP Server shutdown complete")

    async def get_server_stats(self) -> dict[str, Any]:
        """Get server statistics"""
        active_sessions = await self.permission_manager.get_active_sessions() if self.permission_manager else []
        model_stats = self.model_registry.get_model_stats() if self.model_registry else {}

        return {
            "uptime_seconds": time.time() - self.start_time,
            "active_connections": len(self.active_connections),
            "active_sessions": len(active_sessions),
            "registered_models": len(model_stats),
            "config_loaded": bool(self.config),
            "components": {
                "permission_manager": "initialized" if self.permission_manager else "not_initialized",
                "model_registry": "initialized" if self.model_registry else "not_initialized",
                "protocol_handler": "initialized" if self.protocol_handler else "not_initialized"
            }
        }


async def main():
    """Main entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and start server
    server = HypeRAGMCPServer()

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e!s}")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
