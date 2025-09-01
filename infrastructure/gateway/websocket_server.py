#!/usr/bin/env python3
"""
WebSocket Server for Agent Forge Real-time Updates
Provides WebSocket connections for live progress updates.
"""

import asyncio
from datetime import datetime
import json
import logging
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Forge WebSocket Server")

# SECURITY: Secure WebSocket CORS configuration - NO WILDCARDS
import os

ws_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000").split(
    ","
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ws_cors_origins],  # SECURITY: No wildcards
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Limited methods for WebSocket
    allow_headers=["Content-Type", "Authorization", "Sec-WebSocket-Protocol"],
)


# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.subscriptions: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[client_id] = {"websocket": websocket, "channels": set()}
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_text(message)
        except:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: dict, channel: str = "global"):
        """Broadcast message to all subscribed clients."""
        disconnected = set()

        for client_id, sub_info in self.subscriptions.items():
            if channel == "global" or channel in sub_info["channels"]:
                try:
                    await sub_info["websocket"].send_json(message)
                except:
                    disconnected.add(sub_info["websocket"])

        # Clean up disconnected clients
        for ws in disconnected:
            self.active_connections.discard(ws)

    async def subscribe(self, client_id: str, channel: str):
        """Subscribe client to a channel."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id]["channels"].add(channel)
            logger.info(f"Client {client_id} subscribed to {channel}")

    async def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe client from a channel."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id]["channels"].discard(channel)
            logger.info(f"Client {client_id} unsubscribed from {channel}")


manager = ConnectionManager()


# Background task to send periodic updates
async def send_periodic_updates():
    """Send periodic system updates to all clients."""
    while True:
        await asyncio.sleep(5)

        # Send heartbeat
        await manager.broadcast(
            {
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "connections": len(manager.active_connections),
            },
            "system",
        )

        # Send mock phase updates
        if len(manager.active_connections) > 0:
            await manager.broadcast(
                {
                    "type": "system_metrics",
                    "cpu_usage": 45.2,
                    "memory_usage": 62.8,
                    "timestamp": datetime.now().isoformat(),
                },
                "metrics",
            )


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    logger.info("🚀 WebSocket Server starting...")
    asyncio.create_task(send_periodic_updates())
    logger.info("✅ WebSocket Server ready on port 8085!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Agent Forge WebSocket Server",
        "status": "running",
        "connections": len(manager.active_connections),
        "websocket_endpoint": "ws://localhost:8085/ws",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint."""
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "message": "Welcome to Agent Forge WebSocket Server",
            }
        )

        while True:
            # Receive messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "subscribe":
                    channel = message.get("channel", "global")
                    await manager.subscribe(client_id, channel)
                    await websocket.send_json({"type": "subscription_confirmed", "channel": channel})

                elif message.get("type") == "unsubscribe":
                    channel = message.get("channel", "global")
                    await manager.unsubscribe(client_id, channel)
                    await websocket.send_json({"type": "unsubscription_confirmed", "channel": channel})

                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

                elif message.get("type") == "phase_update":
                    # Broadcast phase updates to all clients
                    await manager.broadcast(
                        {
                            "type": "phase_update",
                            "phase_name": message.get("phase_name"),
                            "progress": message.get("progress", 0),
                            "message": message.get("message", ""),
                            "timestamp": datetime.now().isoformat(),
                        },
                        "phases",
                    )

                else:
                    # Echo unknown messages back
                    await websocket.send_json(
                        {"type": "echo", "original": message, "timestamp": datetime.now().isoformat()}
                    )

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket, client_id)


@app.post("/broadcast")
async def broadcast_message(message: dict):
    """HTTP endpoint to broadcast messages to WebSocket clients."""
    await manager.broadcast(message, message.get("channel", "global"))
    return {"status": "broadcast sent", "clients": len(manager.active_connections)}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting WebSocket Server on port 8085...")
    uvicorn.run(app, host="0.0.0.0", port=8085)
