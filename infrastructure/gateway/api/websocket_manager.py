#!/usr/bin/env python3
"""
WebSocket Manager for Real-Time Updates

Provides real-time updates for:
- Agent Forge phase progress
- System resource monitoring
- Model training metrics
- Chat interface events
"""

import asyncio
from datetime import datetime
import logging
from typing import Any
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Manages individual WebSocket connections."""

    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.connected_at = datetime.now()
        self.subscriptions: set[str] = set()
        self.last_ping = datetime.now()


class WebSocketManager:
    """Manages multiple WebSocket connections and broadcasts."""

    def __init__(self):
        self.connections: dict[str, WebSocketConnection] = {}
        self.subscriptions: dict[str, set[str]] = {}  # channel -> set of client_ids

    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept new WebSocket connection."""
        if client_id is None:
            client_id = str(uuid.uuid4())

        await websocket.accept()
        connection = WebSocketConnection(websocket, client_id)
        self.connections[client_id] = connection

        logger.info(f"WebSocket client connected: {client_id}")

        # Send welcome message
        await self.send_to_client(
            client_id,
            {"type": "connection_established", "client_id": client_id, "timestamp": datetime.now().isoformat()},
        )

        return client_id

    def disconnect(self, client_id: str):
        """Handle WebSocket disconnection."""
        if client_id in self.connections:
            connection = self.connections[client_id]

            # Remove from all subscriptions
            for channel in connection.subscriptions:
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(client_id)

            del self.connections[client_id]
            logger.info(f"WebSocket client disconnected: {client_id}")

    async def send_to_client(self, client_id: str, data: dict[str, Any]) -> bool:
        """Send data to specific client."""
        if client_id not in self.connections:
            return False

        try:
            connection = self.connections[client_id]
            await connection.websocket.send_json(data)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to client {client_id}: {e}")
            self.disconnect(client_id)
            return False

    async def broadcast_to_channel(self, channel: str, data: dict[str, Any]):
        """Broadcast data to all clients subscribed to channel."""
        if channel not in self.subscriptions:
            return

        # Add channel info to message
        message = {**data, "channel": channel, "timestamp": datetime.now().isoformat()}

        clients_to_remove = []

        for client_id in self.subscriptions[channel].copy():
            success = await self.send_to_client(client_id, message)
            if not success:
                clients_to_remove.append(client_id)

        # Clean up failed connections
        for client_id in clients_to_remove:
            self.subscriptions[channel].discard(client_id)

    def subscribe(self, client_id: str, channel: str):
        """Subscribe client to a channel."""
        if client_id not in self.connections:
            return False

        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()

        self.subscriptions[channel].add(client_id)
        self.connections[client_id].subscriptions.add(channel)

        logger.info(f"Client {client_id} subscribed to channel: {channel}")
        return True

    def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe client from a channel."""
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(client_id)

        if client_id in self.connections:
            self.connections[client_id].subscriptions.discard(channel)

        logger.info(f"Client {client_id} unsubscribed from channel: {channel}")

    def get_client_count(self, channel: str = None) -> int:
        """Get number of connected clients, optionally for specific channel."""
        if channel:
            return len(self.subscriptions.get(channel, set()))
        return len(self.connections)

    def get_channels(self) -> list[str]:
        """Get list of active channels."""
        return list(self.subscriptions.keys())


# Global WebSocket manager instance
ws_manager = WebSocketManager()

app = FastAPI(title="WebSocket Manager")

# SECURITY: Add secure WebSocket CORS middleware - NO WILDCARDS
try:
    from src.security.cors_config import WEBSOCKET_CORS_CONFIG

    app.add_middleware(CORSMiddleware, **WEBSOCKET_CORS_CONFIG)
except ImportError:
    # Fallback secure WebSocket configuration
    import os

    env = os.getenv("AIVILLAGE_ENV", "development")
    cors_origins = (
        ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
        if env != "production"
        else ["https://aivillage.app"]
    )
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    client_id = await ws_manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "subscribe":
                channel = data.get("channel")
                if channel:
                    ws_manager.subscribe(client_id, channel)
                    await ws_manager.send_to_client(client_id, {"type": "subscription_confirmed", "channel": channel})

            elif message_type == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    ws_manager.unsubscribe(client_id, channel)
                    await ws_manager.send_to_client(client_id, {"type": "unsubscription_confirmed", "channel": channel})

            elif message_type == "ping":
                await ws_manager.send_to_client(client_id, {"type": "pong", "timestamp": datetime.now().isoformat()})

            elif message_type == "get_status":
                await ws_manager.send_to_client(
                    client_id,
                    {
                        "type": "status",
                        "client_id": client_id,
                        "subscriptions": list(ws_manager.connections[client_id].subscriptions),
                        "connected_at": ws_manager.connections[client_id].connected_at.isoformat(),
                        "total_clients": len(ws_manager.connections),
                    },
                )

    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"WebSocket error for client {client_id}: {e}")
        ws_manager.disconnect(client_id)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "websocket_manager",
        "connected_clients": len(ws_manager.connections),
        "active_channels": len(ws_manager.subscriptions),
    }


@app.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    channel_stats = {}
    for channel, clients in ws_manager.subscriptions.items():
        channel_stats[channel] = len(clients)

    return {
        "total_connections": len(ws_manager.connections),
        "active_channels": len(ws_manager.subscriptions),
        "channel_stats": channel_stats,
        "channels": list(ws_manager.subscriptions.keys()),
    }


@app.post("/broadcast/{channel}")
async def broadcast_message(channel: str, message: dict[str, Any]):
    """Broadcast message to all clients in channel."""
    await ws_manager.broadcast_to_channel(channel, message)
    return {"success": True, "channel": channel, "client_count": ws_manager.get_client_count(channel)}


# Utility functions for other services to use
async def broadcast_phase_update(
    phase_name: str, status: str, progress: float, message: str, artifacts: dict[str, Any] = None
):
    """Broadcast Agent Forge phase updates."""
    await ws_manager.broadcast_to_channel(
        "agent_forge_phases",
        {
            "type": "phase_update",
            "phase_name": phase_name,
            "status": status,
            "progress": progress,
            "message": message,
            "artifacts": artifacts or {},
        },
    )


async def broadcast_system_metrics(metrics: dict[str, Any]):
    """Broadcast system resource metrics."""
    await ws_manager.broadcast_to_channel("system_metrics", {"type": "system_metrics", "metrics": metrics})


async def broadcast_model_update(model_id: str, event_type: str, data: dict[str, Any]):
    """Broadcast model-related updates."""
    await ws_manager.broadcast_to_channel(
        "model_updates",
        {
            "type": "model_update",
            "model_id": model_id,
            "event_type": event_type,  # "created", "loaded", "chat_response", etc.
            "data": data,
        },
    )


async def broadcast_training_metrics(phase_name: str, metrics: dict[str, Any]):
    """Broadcast training metrics during model training."""
    await ws_manager.broadcast_to_channel(
        "training_metrics", {"type": "training_metrics", "phase_name": phase_name, "metrics": metrics}
    )


# Background task for periodic updates
async def system_metrics_broadcaster():
    """Background task to broadcast system metrics periodically."""
    import psutil
    import torch

    while True:
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            metrics = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available_gb": memory.available / 1024**3,
            }

            # GPU metrics if available
            if torch.cuda.is_available():
                try:
                    metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**3
                    metrics["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except Exception as e:
                    logger.debug(f"Failed to collect GPU metrics: {e}")

            # Broadcast to subscribers
            await broadcast_system_metrics(metrics)

        except Exception:
            logger.exception("Error in system metrics broadcaster")

        await asyncio.sleep(5)  # Update every 5 seconds


# Background task will be started when the app runs
# asyncio.create_task(system_metrics_broadcaster())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8085)
