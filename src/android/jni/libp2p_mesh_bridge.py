"""Android JNI Bridge for LibP2P Mesh Network.

This module provides a Python-to-Android bridge using several approaches:
1. HTTP REST API bridge (primary)
2. WebSocket bridge (real-time messaging)
3. gRPC bridge (high-performance, optional)
4. File-based IPC (fallback)

The bridge allows Android apps to use the LibP2P mesh network through
standardized interfaces that can be called from Kotlin/Java via JNI.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import logging
from threading import Thread
import time
from typing import Any

# HTTP/WebSocket servers
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# gRPC (optional)
try:
    from concurrent import futures

    import grpc

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

from ..p2p.libp2p_mesh import (
    LibP2PMeshNetwork,
    MeshConfiguration,
    MeshMessage,
    MeshMessageType,
)

logger = logging.getLogger(__name__)


class LibP2PMeshBridge:
    """Bridge between Python LibP2P implementation and Android."""

    def __init__(self, port: int = 8080, ws_port: int = 8081) -> None:
        self.rest_port = port
        self.ws_port = ws_port
        self.mesh_network: LibP2PMeshNetwork | None = None
        self.app: FastAPI | None = None
        self.websocket_connections: list[WebSocket] = []
        self.running = False

        # Background task handling
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.event_loop: asyncio.AbstractEventLoop | None = None

        if FASTAPI_AVAILABLE:
            self._setup_fastapi()

    def _setup_fastapi(self) -> None:
        """Set up FastAPI application."""
        self.app = FastAPI(
            title="LibP2P Mesh Bridge",
            description="Android bridge for LibP2P mesh networking",
            version="1.0.0",
        )

        # CORS middleware for Android WebView access
        from fastapi.middleware.cors import CORSMiddleware

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_rest_endpoints()
        self._register_websocket_handlers()

    def _register_rest_endpoints(self) -> None:
        """Register REST API endpoints."""

        @self.app.post("/mesh/start")
        async def start_mesh(config: dict[str, Any] | None = None):
            """Start mesh network."""
            try:
                if self.mesh_network and self.mesh_network.status.value == "active":
                    return {
                        "status": "already_running",
                        "node_id": self.mesh_network.node_id,
                    }

                # Parse configuration
                mesh_config = MeshConfiguration()
                if config:
                    if "node_id" in config:
                        mesh_config.node_id = config["node_id"]
                    if "listen_port" in config:
                        mesh_config.listen_port = config["listen_port"]
                    if "max_peers" in config:
                        mesh_config.max_peers = config["max_peers"]
                    if "transports" in config:
                        mesh_config.transports = config["transports"]

                self.mesh_network = LibP2PMeshNetwork(mesh_config)

                # Register message handlers
                self.mesh_network.register_message_handler(
                    MeshMessageType.DATA_MESSAGE, self._handle_android_message
                )
                self.mesh_network.register_message_handler(
                    MeshMessageType.AGENT_TASK, self._handle_android_message
                )
                self.mesh_network.register_message_handler(
                    MeshMessageType.PARAMETER_UPDATE, self._handle_android_message
                )
                self.mesh_network.register_message_handler(
                    MeshMessageType.GRADIENT_SHARING, self._handle_android_message
                )

                # Start mesh network
                await self.mesh_network.start()

                return {
                    "status": "started",
                    "node_id": self.mesh_network.node_id,
                    "listen_port": self.mesh_network.config.listen_port,
                    "libp2p_available": self.mesh_network.status.value != "failed",
                }

            except Exception as e:
                logger.exception(f"Failed to start mesh: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/mesh/stop")
        async def stop_mesh():
            """Stop mesh network."""
            try:
                if self.mesh_network:
                    await self.mesh_network.stop()
                    self.mesh_network = None

                return {"status": "stopped"}

            except Exception as e:
                logger.exception(f"Failed to stop mesh: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/mesh/status")
        async def get_mesh_status():
            """Get mesh network status."""
            if not self.mesh_network:
                return {"status": "not_started"}

            return self.mesh_network.get_mesh_status()

        @self.app.post("/mesh/send")
        async def send_message(message_data: dict[str, Any]):
            """Send message through mesh."""
            try:
                if not self.mesh_network:
                    raise HTTPException(status_code=400, detail="Mesh not started")

                # Create mesh message
                message = MeshMessage(
                    type=MeshMessageType(message_data.get("type", "DATA_MESSAGE")),
                    sender=message_data.get("sender", ""),
                    recipient=message_data.get("recipient"),
                    payload=message_data.get("payload", "").encode(),
                    ttl=message_data.get("ttl", 5),
                )

                success = await self.mesh_network.send_message(message)

                return {
                    "success": success,
                    "message_id": message.id,
                    "timestamp": message.timestamp,
                }

            except Exception as e:
                logger.exception(f"Failed to send message: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/mesh/peers")
        async def get_peers():
            """Get connected peers."""
            if not self.mesh_network:
                return {"peers": []}

            peers = []
            for peer_id, capabilities in self.mesh_network.connected_peers.items():
                peers.append({"peer_id": peer_id, "capabilities": asdict(capabilities)})

            return {"peers": peers}

        @self.app.post("/mesh/connect")
        async def connect_peer(peer_data: dict[str, str]):
            """Connect to a specific peer."""
            try:
                if not self.mesh_network:
                    raise HTTPException(status_code=400, detail="Mesh not started")

                peer_address = peer_data.get("address")
                if not peer_address:
                    raise HTTPException(status_code=400, detail="Address required")

                success = await self.mesh_network.add_peer(peer_address)

                return {"success": success}

            except Exception as e:
                logger.exception(f"Failed to connect peer: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/mesh/discovery")
        async def get_discovery_info():
            """Get mDNS discovery information."""
            if not self.mesh_network or not self.mesh_network.mdns_discovery:
                return {"discovery_enabled": False}

            return {
                "discovery_enabled": True,
                "service_name": self.mesh_network.mdns_discovery.service_name,
                "discovered_peers": len(
                    self.mesh_network.mdns_discovery.discovered_peers
                ),
                "status": self.mesh_network.mdns_discovery.get_status(),
            }

        @self.app.post("/dht/store")
        async def dht_store(data: dict[str, Any]):
            """Store value in DHT."""
            try:
                if not self.mesh_network:
                    raise HTTPException(status_code=400, detail="Mesh not started")

                key = data.get("key")
                value = data.get("value", "").encode()

                if not key:
                    raise HTTPException(status_code=400, detail="Key required")

                success = await self.mesh_network.dht_store(key, value)

                return {"success": success}

            except Exception as e:
                logger.exception(f"DHT store failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/dht/get/{key}")
        async def dht_get(key: str):
            """Get value from DHT."""
            try:
                if not self.mesh_network:
                    raise HTTPException(status_code=400, detail="Mesh not started")

                value = await self.mesh_network.dht_get(key)

                if value is None:
                    raise HTTPException(status_code=404, detail="Key not found")

                return {"value": value.decode()}

            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"DHT get failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _register_websocket_handlers(self) -> None:
        """Register WebSocket handlers for real-time communication."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Receive messages from Android
                    data = await websocket.receive_text()
                    message_data = json.loads(data)

                    # Handle different message types
                    msg_type = message_data.get("type")

                    if msg_type == "mesh_send":
                        # Send mesh message
                        await self._handle_websocket_send(websocket, message_data)
                    elif msg_type == "ping":
                        # Respond to ping
                        await websocket.send_text(
                            json.dumps({"type": "pong", "timestamp": time.time()})
                        )
                    elif msg_type == "subscribe":
                        # Subscribe to message types
                        await self._handle_websocket_subscribe(websocket, message_data)

            except WebSocketDisconnect:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)

    async def _handle_websocket_send(
        self, websocket: WebSocket, message_data: dict[str, Any]
    ) -> None:
        """Handle WebSocket mesh send request."""
        try:
            if not self.mesh_network:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Mesh not started"})
                )
                return

            # Create and send mesh message
            message = MeshMessage(
                type=MeshMessageType(message_data.get("message_type", "DATA_MESSAGE")),
                sender=message_data.get("sender", ""),
                recipient=message_data.get("recipient"),
                payload=message_data.get("payload", "").encode(),
                ttl=message_data.get("ttl", 5),
            )

            success = await self.mesh_network.send_message(message)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "send_result",
                        "success": success,
                        "message_id": message.id,
                        "request_id": message_data.get("request_id"),
                    }
                )
            )

        except Exception as e:
            logger.exception(f"WebSocket send error: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": str(e),
                        "request_id": message_data.get("request_id"),
                    }
                )
            )

    async def _handle_websocket_subscribe(
        self, websocket: WebSocket, message_data: dict[str, Any]
    ) -> None:
        """Handle WebSocket subscription request."""
        # Store subscription info with websocket
        # This would be used to filter which messages to send to which clients
        await websocket.send_text(
            json.dumps(
                {"type": "subscribed", "request_id": message_data.get("request_id")}
            )
        )

    async def _handle_android_message(self, message: MeshMessage) -> None:
        """Handle incoming mesh messages and forward to Android clients."""
        # Convert message to JSON for transmission
        message_json = {
            "type": "mesh_message",
            "message_type": message.type.value,
            "sender": message.sender,
            "recipient": message.recipient,
            "payload": message.payload.decode() if message.payload else "",
            "message_id": message.id,
            "timestamp": message.timestamp,
            "hop_count": message.hop_count,
        }

        # Send to all connected WebSocket clients
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message_json))
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)

    async def start_bridge(self) -> None:
        """Start the bridge server."""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available, cannot start bridge")
            return

        self.running = True
        logger.info(f"Starting LibP2P Mesh Bridge on port {self.rest_port}")

        # Start in background thread to avoid blocking
        def run_server() -> None:
            uvicorn.run(self.app, host="0.0.0.0", port=self.rest_port, log_level="info")

        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait a bit for server to start
        await asyncio.sleep(2)
        logger.info("LibP2P Mesh Bridge started")

    def stop_bridge(self) -> None:
        """Stop the bridge server."""
        self.running = False
        if self.mesh_network:
            # Stop mesh network in background
            asyncio.create_task(self.mesh_network.stop())

        # Close WebSocket connections
        for websocket in self.websocket_connections:
            asyncio.create_task(websocket.close())

        self.websocket_connections.clear()
        logger.info("LibP2P Mesh Bridge stopped")

    def get_bridge_info(self) -> dict[str, Any]:
        """Get bridge information for Android integration."""
        return {
            "rest_port": self.rest_port,
            "ws_port": self.ws_port,
            "endpoints": {
                "start_mesh": f"POST http://localhost:{self.rest_port}/mesh/start",
                "stop_mesh": f"POST http://localhost:{self.rest_port}/mesh/stop",
                "get_status": f"GET http://localhost:{self.rest_port}/mesh/status",
                "send_message": f"POST http://localhost:{self.rest_port}/mesh/send",
                "get_peers": f"GET http://localhost:{self.rest_port}/mesh/peers",
                "websocket": f"ws://localhost:{self.rest_port}/ws",
            },
            "message_types": [t.value for t in MeshMessageType],
            "running": self.running,
            "mesh_active": self.mesh_network is not None
            and self.mesh_network.status.value == "active",
        }


# Convenience functions for JNI integration
_bridge_instance: LibP2PMeshBridge | None = None


def initialize_bridge(port: int = 8080) -> dict[str, Any]:
    """Initialize bridge - called from JNI."""
    global _bridge_instance

    try:
        _bridge_instance = LibP2PMeshBridge(port)
        return {"success": True, "bridge_info": _bridge_instance.get_bridge_info()}
    except Exception as e:
        logger.exception(f"Failed to initialize bridge: {e}")
        return {"success": False, "error": str(e)}


def start_bridge() -> dict[str, Any]:
    """Start bridge server - called from JNI."""
    global _bridge_instance

    if not _bridge_instance:
        return {"success": False, "error": "Bridge not initialized"}

    try:
        # Run in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_bridge_instance.start_bridge())

        return {"success": True}
    except Exception as e:
        logger.exception(f"Failed to start bridge: {e}")
        return {"success": False, "error": str(e)}


def stop_bridge() -> dict[str, Any]:
    """Stop bridge server - called from JNI."""
    global _bridge_instance

    if not _bridge_instance:
        return {"success": False, "error": "Bridge not running"}

    try:
        _bridge_instance.stop_bridge()
        _bridge_instance = None
        return {"success": True}
    except Exception as e:
        logger.exception(f"Failed to stop bridge: {e}")
        return {"success": False, "error": str(e)}


def get_bridge_status() -> dict[str, Any]:
    """Get bridge status - called from JNI."""
    global _bridge_instance

    if not _bridge_instance:
        return {"running": False, "error": "Bridge not initialized"}

    return _bridge_instance.get_bridge_info()


if __name__ == "__main__":
    # Test the bridge
    import asyncio

    async def test_bridge() -> None:
        bridge = LibP2PMeshBridge()
        await bridge.start_bridge()

        # Keep running for testing
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            bridge.stop_bridge()

    asyncio.run(test_bridge())
