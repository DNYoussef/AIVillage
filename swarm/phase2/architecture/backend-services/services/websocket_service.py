"""
WebSocket Service - Real-time communication management

This service is responsible for:
- WebSocket connection lifecycle management
- Real-time message broadcasting and routing
- Topic-based subscription management
- Connection health monitoring and recovery
- Message queue management for offline clients

Size Target: <400 lines
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from interfaces.service_contracts import (
    IWebSocketService, WebSocketMessage, ConnectionInfo, Event
)

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a single WebSocket connection."""
    
    def __init__(self, websocket: WebSocket, connection_id: str, client_ip: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_ip = client_ip
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.subscriptions: Set[str] = set()
        self.user_agent: Optional[str] = None
        self.is_alive = True
        
    async def send_message(self, message: WebSocketMessage) -> bool:
        """Send message to this connection."""
        try:
            await self.websocket.send_json(message.dict())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            self.is_alive = False
            return False
    
    async def ping(self) -> bool:
        """Send ping to check connection health."""
        try:
            await self.websocket.ping()
            self.last_ping = datetime.now()
            return True
        except Exception:
            self.is_alive = False
            return False
    
    def is_subscribed_to(self, topic: str) -> bool:
        """Check if connection is subscribed to a topic."""
        return topic in self.subscriptions
    
    def subscribe(self, topic: str):
        """Subscribe to a topic."""
        self.subscriptions.add(topic)
    
    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic."""
        self.subscriptions.discard(topic)


class WebSocketService(IWebSocketService):
    """Implementation of the WebSocket Service."""
    
    def __init__(self, 
                 max_connections: int = 1000,
                 ping_interval: int = 30,
                 message_queue_size: int = 100):
        self.max_connections = max_connections
        self.ping_interval = ping_interval
        self.message_queue_size = message_queue_size
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> connection_ids
        self.message_queues: Dict[str, List[WebSocketMessage]] = {}  # connection_id -> messages
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.start_time = datetime.now()
        
        # Start background tasks
        self.ping_task = asyncio.create_task(self._ping_connections())
        self.cleanup_task = asyncio.create_task(self._cleanup_dead_connections())
    
    async def connect_websocket(self, websocket: WebSocket, client_ip: str) -> str:
        """Handle new WebSocket connection."""
        if len(self.connections) >= self.max_connections:
            logger.warning(f"Max connections reached ({self.max_connections})")
            await websocket.close(code=1013, reason="Server overloaded")
            return None
        
        # Accept connection
        await websocket.accept()
        
        # Create connection info
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id, client_ip)
        
        # Store connection
        self.connections[connection_id] = connection
        self.message_queues[connection_id] = []
        self.total_connections += 1
        
        logger.info(f"WebSocket connected: {connection_id} from {client_ip}")
        logger.info(f"Active connections: {len(self.connections)}")
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            type="connection_established",
            source_service="websocket_service",
            data={
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat()
            }
        )
        await connection.send_message(welcome_message)
        
        return connection_id
    
    async def disconnect_websocket(self, connection_id: str):
        """Handle WebSocket disconnection."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from all topic subscriptions
        for topic in list(connection.subscriptions):
            await self._unsubscribe_from_topic(connection_id, topic)
        
        # Cleanup
        del self.connections[connection_id]
        del self.message_queues[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
        logger.info(f"Active connections: {len(self.connections)}")
    
    async def handle_websocket_message(self, connection_id: str, message: dict):
        """Handle incoming WebSocket message from client."""
        if connection_id not in self.connections:
            return
        
        message_type = message.get("type")
        
        if message_type == "subscribe":
            topic = message.get("topic")
            if topic:
                await self.subscribe_to_topic(connection_id, topic)
        
        elif message_type == "unsubscribe":
            topic = message.get("topic")
            if topic:
                await self._unsubscribe_from_topic(connection_id, topic)
        
        elif message_type == "ping":
            # Respond with pong
            pong_message = WebSocketMessage(
                type="pong",
                source_service="websocket_service",
                data={"timestamp": datetime.now().isoformat()}
            )
            await self.send_to_connection(connection_id, pong_message)
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def broadcast_message(self, message: WebSocketMessage) -> bool:
        """Broadcast message to all connected clients."""
        if not self.connections:
            logger.debug("No active connections for broadcast")
            return True
        
        success_count = 0
        failed_connections = []
        
        for connection_id, connection in self.connections.items():
            try:
                success = await connection.send_message(message)
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
            except Exception as e:
                logger.error(f"Broadcast failed for connection {connection_id}: {e}")
                failed_connections.append(connection_id)
        
        # Mark failed connections for cleanup
        for conn_id in failed_connections:
            if conn_id in self.connections:
                self.connections[conn_id].is_alive = False
        
        self.total_messages_sent += success_count
        logger.debug(f"Broadcast to {success_count}/{len(self.connections)} connections")
        
        return len(failed_connections) == 0
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        success = await connection.send_message(message)
        
        if success:
            self.total_messages_sent += 1
        else:
            # Queue message for retry if connection is temporarily unavailable
            queue = self.message_queues.get(connection_id, [])
            if len(queue) < self.message_queue_size:
                queue.append(message)
                logger.debug(f"Queued message for connection {connection_id}")
        
        return success
    
    async def broadcast_to_topic(self, topic: str, message: WebSocketMessage) -> int:
        """Broadcast message to all subscribers of a topic."""
        subscribers = self.topic_subscribers.get(topic, set())
        if not subscribers:
            logger.debug(f"No subscribers for topic: {topic}")
            return 0
        
        success_count = 0
        
        for connection_id in subscribers.copy():  # Copy to avoid modification during iteration
            if connection_id in self.connections:
                success = await self.send_to_connection(connection_id, message)
                if success:
                    success_count += 1
        
        logger.debug(f"Topic broadcast '{topic}' to {success_count}/{len(subscribers)} subscribers")
        return success_count
    
    async def subscribe_to_topic(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to a topic."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscribe(topic)
        
        # Add to topic subscribers
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to topic '{topic}'")
        
        # Send subscription confirmation
        confirmation = WebSocketMessage(
            type="subscription_confirmed",
            source_service="websocket_service",
            data={"topic": topic, "connection_id": connection_id}
        )
        await self.send_to_connection(connection_id, confirmation)
        
        return True
    
    async def _unsubscribe_from_topic(self, connection_id: str, topic: str):
        """Unsubscribe connection from a topic."""
        if connection_id in self.connections:
            self.connections[connection_id].unsubscribe(topic)
        
        # Remove from topic subscribers
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(connection_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        
        logger.debug(f"Connection {connection_id} unsubscribed from topic '{topic}'")
    
    async def get_active_connections(self) -> List[ConnectionInfo]:
        """Get list of active connections."""
        return [
            ConnectionInfo(
                connection_id=conn.connection_id,
                connected_at=conn.connected_at,
                client_ip=conn.client_ip,
                user_agent=conn.user_agent,
                subscriptions=list(conn.subscriptions)
            )
            for conn in self.connections.values()
            if conn.is_alive
        ]
    
    async def get_statistics(self) -> Dict:
        """Get service statistics."""
        uptime = datetime.now() - self.start_time
        return {
            "active_connections": len(self.connections),
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "topic_count": len(self.topic_subscribers),
            "uptime_seconds": int(uptime.total_seconds()),
            "average_messages_per_connection": (
                self.total_messages_sent / max(self.total_connections, 1)
            )
        }
    
    async def _ping_connections(self):
        """Background task to ping connections and check health."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                ping_tasks = []
                for connection in self.connections.values():
                    if connection.is_alive:
                        # Check if connection needs ping
                        time_since_ping = datetime.now() - connection.last_ping
                        if time_since_ping > timedelta(seconds=self.ping_interval):
                            ping_tasks.append(connection.ping())
                
                if ping_tasks:
                    await asyncio.gather(*ping_tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    async def _cleanup_dead_connections(self):
        """Background task to cleanup dead connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                dead_connections = []
                for conn_id, connection in self.connections.items():
                    if not connection.is_alive:
                        dead_connections.append(conn_id)
                
                for conn_id in dead_connections:
                    await self.disconnect_websocket(conn_id)
                
                if dead_connections:
                    logger.info(f"Cleaned up {len(dead_connections)} dead connections")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def shutdown(self):
        """Shutdown the service gracefully."""
        logger.info("Shutting down WebSocket service")
        
        # Cancel background tasks
        self.ping_task.cancel()
        self.cleanup_task.cancel()
        
        # Close all connections
        for connection in self.connections.values():
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.warning(
                    "Failed to close WebSocket connection during shutdown",
                    extra={
                        "connection_id": connection.connection_id,
                        "client_id": getattr(connection, 'client_id', 'unknown'),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
        
        self.connections.clear()
        self.topic_subscribers.clear()
        self.message_queues.clear()


# Service factory
def create_websocket_service(max_connections: int = 1000) -> WebSocketService:
    """Create and configure the WebSocket Service."""
    return WebSocketService(max_connections=max_connections)