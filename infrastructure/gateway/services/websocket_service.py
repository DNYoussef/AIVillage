"""
WebSocket Service for AI Village Infrastructure
Manages WebSocket connections, broadcasting, and real-time updates.

This service extracts WebSocket functionality from the unified backend
to provide a clean, reusable WebSocket management system.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    """Standard WebSocket message types."""
    # Connection lifecycle
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    PING = "ping"
    PONG = "pong"
    
    # Training updates
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_ERROR = "training_error"
    
    # Model updates
    MODEL_CREATED = "model_created"
    MODEL_UPDATED = "model_updated"
    MODEL_DELETED = "model_deleted"
    
    # System status
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    
    # P2P/Fog specific
    P2P_STATUS = "p2p_status"
    P2P_PEER_UPDATE = "p2p_peer_update"
    FOG_RESOURCES = "fog_resources"
    FOG_MARKETPLACE = "fog_marketplace"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class WebSocketMessage:
    """Standardized WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str
    connection_id: Optional[str] = None
    target_connections: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "connection_id": self.connection_id,
            "target_connections": self.target_connections
        }


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    id: str
    websocket: WebSocket
    state: ConnectionState
    connected_at: datetime
    last_ping: Optional[datetime]
    subscription_topics: Set[str]
    metadata: Dict[str, Any]
    
    def is_alive(self) -> bool:
        """Check if connection is considered alive."""
        if self.state != ConnectionState.CONNECTED:
            return False
        if self.last_ping is None:
            return True
        return datetime.now() - self.last_ping < timedelta(minutes=5)


class WebSocketService:
    """
    Manages WebSocket connections and provides broadcasting capabilities.
    
    Features:
    - Connection lifecycle management
    - Message broadcasting and targeting
    - Topic-based subscriptions
    - Connection health monitoring
    - Event-driven architecture
    """
    
    def __init__(self, ping_interval: int = 30, cleanup_interval: int = 60):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.event_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.ping_interval = ping_interval
        self.cleanup_interval = cleanup_interval
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocketService initialized")
    
    async def start(self):
        """Start the WebSocket service background tasks."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_connections())
        self._ping_task = asyncio.create_task(self._ping_connections())
        logger.info("WebSocketService started")
    
    async def stop(self):
        """Stop the WebSocket service and cleanup."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()
            
        # Close all connections
        for conn_info in list(self.connections.values()):
            await self._disconnect_connection(conn_info.id)
            
        logger.info("WebSocketService stopped")
    
    async def connect(self, websocket: WebSocket, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket instance
            metadata: Optional metadata about the connection
            
        Returns:
            Connection ID
        """
        try:
            await websocket.accept()
            
            connection_id = str(uuid.uuid4())
            conn_info = ConnectionInfo(
                id=connection_id,
                websocket=websocket,
                state=ConnectionState.CONNECTED,
                connected_at=datetime.now(),
                last_ping=datetime.now(),
                subscription_topics=set(),
                metadata=metadata or {}
            )
            
            self.connections[connection_id] = conn_info
            
            # Send connection established message
            await self._send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.CONNECTION_ESTABLISHED,
                    data={
                        "connection_id": connection_id,
                        "server_time": datetime.now().isoformat(),
                        "features": [
                            "real_time_training_updates",
                            "model_lifecycle_events",
                            "p2p_network_status",
                            "fog_computing_metrics",
                            "topic_subscriptions",
                            "connection_health_monitoring"
                        ]
                    },
                    timestamp=datetime.now().isoformat(),
                    connection_id=connection_id
                )
            )
            
            logger.info(f"WebSocket connected: {connection_id} (total: {len(self.connections)})")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise
    
    async def disconnect(self, connection_id: str):
        """Disconnect a specific WebSocket connection."""
        await self._disconnect_connection(connection_id)
    
    async def _disconnect_connection(self, connection_id: str):
        """Internal method to disconnect a connection."""
        if connection_id not in self.connections:
            return
            
        conn_info = self.connections[connection_id]
        conn_info.state = ConnectionState.DISCONNECTING
        
        # Unsubscribe from all topics
        for topic in list(conn_info.subscription_topics):
            self.unsubscribe(connection_id, topic)
        
        try:
            await conn_info.websocket.close()
        except Exception:
            pass  # Connection might already be closed
        
        conn_info.state = ConnectionState.DISCONNECTED
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id} (remaining: {len(self.connections)})")
    
    async def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic."""
        if connection_id not in self.connections:
            return False
            
        conn_info = self.connections[connection_id]
        conn_info.subscription_topics.add(topic)
        self.topic_subscriptions[topic].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to topic: {topic}")
        return True
    
    def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic."""
        if connection_id not in self.connections:
            return False
            
        conn_info = self.connections[connection_id]
        conn_info.subscription_topics.discard(topic)
        self.topic_subscriptions[topic].discard(connection_id)
        
        # Clean up empty topic subscriptions
        if not self.topic_subscriptions[topic]:
            del self.topic_subscriptions[topic]
        
        logger.debug(f"Connection {connection_id} unsubscribed from topic: {topic}")
        return True
    
    async def broadcast(self, message: WebSocketMessage, topic: Optional[str] = None):
        """
        Broadcast a message to all connections or topic subscribers.
        
        Args:
            message: The message to broadcast
            topic: Optional topic to broadcast to (if None, broadcasts to all)
        """
        if topic:
            target_connections = self.topic_subscriptions.get(topic, set())
        else:
            target_connections = set(self.connections.keys())
        
        if not target_connections:
            return
        
        # Send to all target connections
        tasks = []
        for conn_id in target_connections:
            if conn_id in self.connections:
                tasks.append(self._send_to_connection(conn_id, message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_count = sum(1 for r in results if isinstance(r, Exception))
            if failed_count > 0:
                logger.warning(f"Failed to send message to {failed_count}/{len(tasks)} connections")
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send a message to a specific connection."""
        return await self._send_to_connection(connection_id, message)
    
    async def _send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Internal method to send message to a connection."""
        if connection_id not in self.connections:
            return False
            
        conn_info = self.connections[connection_id]
        if conn_info.state != ConnectionState.CONNECTED:
            return False
        
        try:
            message.connection_id = connection_id
            await conn_info.websocket.send_json(message.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            # Mark connection for cleanup
            conn_info.state = ConnectionState.ERROR
            return False
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Handle incoming message from a WebSocket connection."""
        if connection_id not in self.connections:
            return
        
        try:
            data = json.loads(raw_message)
            message_type = MessageType(data.get("type", "custom"))
            
            # Update last ping time
            conn_info = self.connections[connection_id]
            conn_info.last_ping = datetime.now()
            
            # Handle built-in message types
            if message_type == MessageType.PING:
                await self._handle_ping(connection_id, data)
            elif data.get("type") == "subscribe":
                await self._handle_subscribe(connection_id, data)
            elif data.get("type") == "unsubscribe":
                await self._handle_unsubscribe(connection_id, data)
            else:
                # Call registered event handlers
                handlers = self.event_handlers.get(message_type, [])
                for handler in handlers:
                    try:
                        await handler(connection_id, data)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
        
        except json.JSONDecodeError:
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"error": "Invalid JSON"},
                    timestamp=datetime.now().isoformat()
                )
            )
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]):
        """Handle ping message."""
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                type=MessageType.PONG,
                data={"server_time": datetime.now().isoformat()},
                timestamp=datetime.now().isoformat()
            )
        )
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription request."""
        topic = data.get("topic")
        if topic:
            success = await self.subscribe(connection_id, topic)
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.CUSTOM,
                    data={
                        "type": "subscription_response",
                        "topic": topic,
                        "success": success
                    },
                    timestamp=datetime.now().isoformat()
                )
            )
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription request."""
        topic = data.get("topic")
        if topic:
            success = self.unsubscribe(connection_id, topic)
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    type=MessageType.CUSTOM,
                    data={
                        "type": "unsubscription_response", 
                        "topic": topic,
                        "success": success
                    },
                    timestamp=datetime.now().isoformat()
                )
            )
    
    def register_event_handler(self, message_type: MessageType, handler: Callable):
        """Register an event handler for a specific message type."""
        self.event_handlers[message_type].append(handler)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len([c for c in self.connections.values() if c.state == ConnectionState.CONNECTED])
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get information about a specific connection."""
        return self.connections.get(connection_id)
    
    def get_topic_subscribers(self, topic: str) -> Set[str]:
        """Get connection IDs subscribed to a topic."""
        return self.topic_subscriptions.get(topic, set()).copy()
    
    async def _cleanup_connections(self):
        """Background task to cleanup dead connections."""
        while self._running:
            try:
                dead_connections = []
                for conn_id, conn_info in self.connections.items():
                    if not conn_info.is_alive() or conn_info.state == ConnectionState.ERROR:
                        dead_connections.append(conn_id)
                
                for conn_id in dead_connections:
                    await self._disconnect_connection(conn_id)
                
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(5)
    
    async def _ping_connections(self):
        """Background task to ping connections."""
        while self._running:
            try:
                ping_message = WebSocketMessage(
                    type=MessageType.PING,
                    data={"server_time": datetime.now().isoformat()},
                    timestamp=datetime.now().isoformat()
                )
                
                await self.broadcast(ping_message)
                await asyncio.sleep(self.ping_interval)
            except Exception as e:
                logger.error(f"Ping task error: {e}")
                await asyncio.sleep(5)


# Global WebSocket service instance
websocket_service = WebSocketService()