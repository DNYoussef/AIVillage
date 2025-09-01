"""
Example integration of WebSocketService with training and model services.
Demonstrates event-driven architecture for real-time updates.
"""

from datetime import datetime
from typing import Dict, Any

from .websocket_service import websocket_service, WebSocketMessage, MessageType


class TrainingEventHandler:
    """Example handler for training events that broadcasts via WebSocket."""

    def __init__(self):
        self.active_training_sessions = {}

    async def on_training_started(self, session_id: str, parameters: Dict[str, Any]):
        """Handle training started event."""
        self.active_training_sessions[session_id] = {
            "started_at": datetime.now(),
            "parameters": parameters,
            "progress": 0.0,
        }

        message = WebSocketMessage(
            type=MessageType.TRAINING_STARTED,
            data={
                "session_id": session_id,
                "parameters": parameters,
                "estimated_duration": parameters.get("estimated_duration"),
                "model_type": parameters.get("model_type"),
            },
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="training")

    async def on_training_progress(
        self, session_id: str, progress: float, current_epoch: int, loss: float, metrics: Dict[str, float]
    ):
        """Handle training progress updates."""
        if session_id in self.active_training_sessions:
            self.active_training_sessions[session_id]["progress"] = progress

        message = WebSocketMessage(
            type=MessageType.TRAINING_PROGRESS,
            data={
                "session_id": session_id,
                "progress": progress,
                "current_epoch": current_epoch,
                "loss": loss,
                "metrics": metrics,
                "estimated_time_remaining": self._estimate_time_remaining(session_id, progress),
            },
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="training")

    async def on_training_completed(self, session_id: str, results: Dict[str, Any]):
        """Handle training completion."""
        if session_id in self.active_training_sessions:
            del self.active_training_sessions[session_id]

        message = WebSocketMessage(
            type=MessageType.TRAINING_COMPLETED,
            data={
                "session_id": session_id,
                "results": results,
                "model_path": results.get("model_path"),
                "final_metrics": results.get("metrics", {}),
            },
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="training")

    async def on_training_error(self, session_id: str, error: str, traceback: str):
        """Handle training errors."""
        if session_id in self.active_training_sessions:
            del self.active_training_sessions[session_id]

        message = WebSocketMessage(
            type=MessageType.TRAINING_ERROR,
            data={"session_id": session_id, "error": error, "traceback": traceback},
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="training")

    def _estimate_time_remaining(self, session_id: str, progress: float) -> float:
        """Estimate remaining training time based on progress."""
        if session_id not in self.active_training_sessions or progress <= 0:
            return -1

        session = self.active_training_sessions[session_id]
        elapsed = (datetime.now() - session["started_at"]).total_seconds()
        total_estimated = elapsed / progress
        return total_estimated - elapsed


class ModelEventHandler:
    """Example handler for model lifecycle events."""

    async def on_model_created(self, model_id: str, model_info: Dict[str, Any]):
        """Handle model creation."""
        message = WebSocketMessage(
            type=MessageType.MODEL_CREATED,
            data={"model_id": model_id, "model_info": model_info, "created_at": datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="models")

    async def on_model_updated(self, model_id: str, updates: Dict[str, Any]):
        """Handle model updates."""
        message = WebSocketMessage(
            type=MessageType.MODEL_UPDATED,
            data={"model_id": model_id, "updates": updates, "updated_at": datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="models")

    async def on_model_deleted(self, model_id: str):
        """Handle model deletion."""
        message = WebSocketMessage(
            type=MessageType.MODEL_DELETED,
            data={"model_id": model_id, "deleted_at": datetime.now().isoformat()},
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="models")


class P2PFogEventHandler:
    """Example handler for P2P and Fog computing events."""

    async def on_p2p_status_update(self, status: Dict[str, Any]):
        """Handle P2P network status updates."""
        message = WebSocketMessage(
            type=MessageType.P2P_STATUS,
            data={
                "status": status,
                "peer_count": status.get("peer_count", 0),
                "network_health": status.get("network_health", "unknown"),
            },
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="p2p")

    async def on_fog_resources_update(self, resources: Dict[str, Any]):
        """Handle fog computing resource updates."""
        message = WebSocketMessage(
            type=MessageType.FOG_RESOURCES,
            data={
                "resources": resources,
                "available_nodes": resources.get("available_nodes", 0),
                "total_compute": resources.get("total_compute", 0),
            },
            timestamp=datetime.now().isoformat(),
        )

        await websocket_service.broadcast(message, topic="fog")


async def setup_websocket_integration():
    """Set up WebSocket service integration with event handlers."""

    # Initialize event handlers
    training_handler = TrainingEventHandler()
    model_handler = ModelEventHandler()
    p2p_fog_handler = P2PFogEventHandler()

    # Register custom message handlers
    async def handle_training_status_request(connection_id: str, data: Dict[str, Any]):
        """Handle training status requests."""
        await websocket_service.send_to_connection(
            connection_id,
            WebSocketMessage(
                type=MessageType.STATUS_UPDATE,
                data={
                    "type": "training_status",
                    "active_sessions": list(training_handler.active_training_sessions.keys()),
                    "session_details": training_handler.active_training_sessions,
                },
                timestamp=datetime.now().isoformat(),
            ),
        )

    websocket_service.register_event_handler(MessageType.CUSTOM, handle_training_status_request)

    # Start the WebSocket service
    await websocket_service.start()

    return {"training_handler": training_handler, "model_handler": model_handler, "p2p_fog_handler": p2p_fog_handler}


# Usage example for FastAPI integration
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


def create_websocket_endpoints(app: FastAPI):
    """Add WebSocket endpoints to FastAPI app."""

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for training updates."""
        connection_id = await websocket_service.connect(websocket, {"endpoint": "main"})

        # Auto-subscribe to all topics for main endpoint
        await websocket_service.subscribe(connection_id, "training")
        await websocket_service.subscribe(connection_id, "models")
        await websocket_service.subscribe(connection_id, "system")

        try:
            while True:
                data = await websocket.receive_text()
                await websocket_service.handle_message(connection_id, data)
        except WebSocketDisconnect:
            await websocket_service.disconnect(connection_id)

    @app.websocket("/ws/p2p-fog")
    async def p2p_fog_websocket(websocket: WebSocket):
        """WebSocket endpoint for P2P/Fog updates."""
        connection_id = await websocket_service.connect(websocket, {"endpoint": "p2p-fog"})

        # Subscribe to P2P and Fog topics
        await websocket_service.subscribe(connection_id, "p2p")
        await websocket_service.subscribe(connection_id, "fog")

        try:
            while True:
                data = await websocket.receive_text()
                await websocket_service.handle_message(connection_id, data)
        except WebSocketDisconnect:
            await websocket_service.disconnect(connection_id)

    @app.on_event("startup")
    async def startup_event():
        """Initialize WebSocket service on app startup."""
        await setup_websocket_integration()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup WebSocket service on app shutdown."""
        await websocket_service.stop()
