"""
WebSocket Transport Implementation

WebSocket transport implementation with circuit breaker support.
Consolidates existing WebSocket handler functionality.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

from .base_transport import BaseTransport, TransportState
from ..message_format import UnifiedMessage, MessageType, TransportType
from ..reliability.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class ConnectionInfo:
    """Information about a WebSocket connection"""
    
    def __init__(self, websocket: WebSocketServerProtocol, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = None
        self.node_id: Optional[str] = None  # Set during handshake
        self.authenticated = False
        self.message_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "node_id": self.node_id,
            "remote_address": f"{self.websocket.remote_address[0]}:{self.websocket.remote_address[1]}",
            "connected_at": self.connected_at.isoformat(),
            "authenticated": self.authenticated,
            "message_count": self.message_count
        }


class WebSocketTransport(BaseTransport):
    """WebSocket transport implementation with circuit breaker support"""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library is required for WebSocket transport")
        
        # Server configuration
        self.port = config.get("port", 8765)
        self.host = config.get("host", "0.0.0.0")
        self.max_connections = config.get("max_connections", 100)
        
        # WebSocket server
        self.server = None
        
        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}  # connection_id -> info
        self.node_connections: Dict[str, str] = {}  # node_id -> connection_id
        
        # Circuit breaker for connection resilience
        cb_config = config.get("circuit_breaker", {})
        self.circuit_breaker = CircuitBreaker(cb_config)
        
        # Authentication (optional)
        self.require_auth = config.get("require_auth", False)
        self.auth_tokens = config.get("auth_tokens", [])
        
        # Heartbeat settings
        self.heartbeat_interval = config.get("heartbeat_interval", 30)
        self.heartbeat_task = None
    
    async def start(self) -> None:
        """Start WebSocket server"""
        if self.running:
            logger.warning("WebSocket transport already running")
            return
        
        logger.info(f"Starting WebSocket transport on {self.host}:{self.port}")
        self.state = TransportState.STARTING
        
        try:
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=1024*1024,  # 1MB max message size
                ping_interval=self.heartbeat_interval,
                ping_timeout=self.heartbeat_interval * 2
            )
            
            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.running = True
            self.state = TransportState.RUNNING
            
            logger.info(f"WebSocket transport started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket transport: {e}")
            self.state = TransportState.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop WebSocket server"""
        if not self.running:
            logger.warning("WebSocket transport not running")
            return
        
        logger.info("Stopping WebSocket transport")
        self.state = TransportState.STOPPING
        
        try:
            # Stop heartbeat task
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Close all connections
            for connection_info in list(self.connections.values()):
                try:
                    await connection_info.websocket.close()
                except Exception:
                    pass
            
            self.connections.clear()
            self.node_connections.clear()
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.running = False
            self.state = TransportState.STOPPED
            
            logger.info("WebSocket transport stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket transport: {e}")
            self.state = TransportState.ERROR
            raise
    
    async def send(self, message: UnifiedMessage, target: str) -> bool:
        """Send message to specific WebSocket connection or node"""
        if not self.running:
            logger.error("Cannot send: WebSocket transport not running")
            return False
        
        try:
            # Resolve target to connection
            connection_info = self._resolve_target_connection(target)
            if not connection_info:
                logger.warning(f"Target not found: {target}")
                self._record_send_error()
                return False
            
            # Use circuit breaker for resilience
            success = await self.circuit_breaker.call(
                self._send_to_connection, connection_info, message
            )
            
            if success:
                self._record_send_success()
                connection_info.message_count += 1
            else:
                self._record_send_error()
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {target}: {e}")
            self._record_send_error()
            return False
    
    async def broadcast(self, message: UnifiedMessage) -> Dict[str, bool]:
        """Broadcast to all WebSocket connections"""
        if not self.running:
            logger.error("Cannot broadcast: WebSocket transport not running")
            return {}
        
        results = {}
        
        for connection_id, connection_info in list(self.connections.items()):
            try:
                success = await self.circuit_breaker.call(
                    self._send_to_connection, connection_info, message
                )
                
                target_id = connection_info.node_id or connection_id
                results[target_id] = success
                
                if success:
                    self._record_send_success()
                    connection_info.message_count += 1
                else:
                    self._record_send_error()
                    
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                results[connection_id] = False
                self._record_send_error()
        
        logger.info(f"WebSocket broadcast completed: {sum(results.values())}/{len(results)} successful")
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check WebSocket transport health"""
        connection_health = []
        for conn_info in self.connections.values():
            connection_health.append(conn_info.to_dict())
        
        return {
            "status": "healthy" if self.running else "stopped",
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "connections_active": len(self.connections),
            "max_connections": self.max_connections,
            "node_mappings": len(self.node_connections),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "connections": connection_health,
            "metrics": self.get_metrics()
        }
    
    def get_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active connections"""
        return {
            conn_id: conn_info.to_dict() 
            for conn_id, conn_info in self.connections.items()
        }
    
    def get_node_connection(self, node_id: str) -> Optional[str]:
        """Get connection ID for a node"""
        return self.node_connections.get(node_id)
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle new WebSocket connection"""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{id(websocket)}"
        connection_info = ConnectionInfo(websocket, connection_id)
        
        logger.info(f"New WebSocket connection: {connection_id}")
        
        if len(self.connections) >= self.max_connections:
            logger.warning(f"Max connections reached, rejecting: {connection_id}")
            await websocket.close(code=1013, reason="Server overloaded")
            return
        
        self.connections[connection_id] = connection_info
        self._update_connection_count(1)
        
        try:
            # Handle authentication if required
            if self.require_auth:
                await self._authenticate_connection(connection_info)
            
            # Main message loop
            async for raw_message in websocket:
                try:
                    await self._process_incoming_message(connection_info, raw_message)
                except Exception as e:
                    logger.error(f"Error processing message from {connection_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection {connection_id}: {e}")
            self._record_connection_error()
        finally:
            # Cleanup connection
            self._cleanup_connection(connection_info)
    
    async def _process_incoming_message(self, connection_info: ConnectionInfo, 
                                      raw_message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            # Parse message
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode('utf-8')
            
            message_data = json.loads(raw_message)
            
            # Check for special control messages
            if message_data.get("type") == "handshake":
                await self._handle_handshake(connection_info, message_data)
                return
            elif message_data.get("type") == "ping":
                await self._handle_ping(connection_info)
                return
            
            # Convert to unified message
            message = UnifiedMessage.from_dict(message_data)
            
            # Update connection info
            if not connection_info.node_id and message.source_id:
                connection_info.node_id = message.source_id
                self.node_connections[message.source_id] = connection_info.connection_id
                logger.info(f"Node registered: {message.source_id} -> {connection_info.connection_id}")
            
            # Handle the message
            await self.handle_incoming_message(message)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from {connection_info.connection_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing message from {connection_info.connection_id}: {e}")
    
    async def _handle_handshake(self, connection_info: ConnectionInfo, 
                              handshake_data: Dict[str, Any]) -> None:
        """Handle connection handshake"""
        node_id = handshake_data.get("node_id")
        auth_token = handshake_data.get("auth_token")
        
        # Authenticate if required
        if self.require_auth:
            if auth_token not in self.auth_tokens:
                logger.warning(f"Authentication failed for {connection_info.connection_id}")
                await connection_info.websocket.close(code=1008, reason="Authentication failed")
                return
        
        # Register node
        if node_id:
            connection_info.node_id = node_id
            connection_info.authenticated = True
            self.node_connections[node_id] = connection_info.connection_id
            
            # Send handshake response
            response = {
                "type": "handshake_ack",
                "node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await connection_info.websocket.send(json.dumps(response))
            logger.info(f"Handshake completed: {node_id} -> {connection_info.connection_id}")
    
    async def _handle_ping(self, connection_info: ConnectionInfo) -> None:
        """Handle ping message"""
        connection_info.last_ping = datetime.now(timezone.utc)
        
        # Send pong response
        pong_message = {
            "type": "pong",
            "timestamp": connection_info.last_ping.isoformat()
        }
        
        await connection_info.websocket.send(json.dumps(pong_message))
    
    async def _authenticate_connection(self, connection_info: ConnectionInfo) -> None:
        """Authenticate WebSocket connection"""
        # Wait for handshake message with auth token
        try:
            raw_message = await asyncio.wait_for(
                connection_info.websocket.recv(), timeout=10.0
            )
            
            message_data = json.loads(raw_message)
            if message_data.get("type") == "handshake":
                await self._handle_handshake(connection_info, message_data)
            else:
                logger.warning(f"Expected handshake, got {message_data.get('type')}")
                await connection_info.websocket.close(code=1002, reason="Protocol error")
                
        except asyncio.TimeoutError:
            logger.warning(f"Authentication timeout for {connection_info.connection_id}")
            await connection_info.websocket.close(code=1000, reason="Authentication timeout")
        except Exception as e:
            logger.error(f"Authentication error for {connection_info.connection_id}: {e}")
            await connection_info.websocket.close(code=1011, reason="Authentication error")
    
    async def _send_to_connection(self, connection_info: ConnectionInfo, 
                                message: UnifiedMessage) -> bool:
        """Send message to specific connection"""
        try:
            message_json = message.to_json()
            await connection_info.websocket.send(message_json)
            logger.debug(f"Message sent to {connection_info.connection_id}: {message.message_id}")
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed during send: {connection_info.connection_id}")
            self._cleanup_connection(connection_info)
            return False
        except Exception as e:
            logger.error(f"Error sending to {connection_info.connection_id}: {e}")
            return False
    
    def _resolve_target_connection(self, target: str) -> Optional[ConnectionInfo]:
        """Resolve target to connection info"""
        # Try as node ID first
        if target in self.node_connections:
            connection_id = self.node_connections[target]
            return self.connections.get(connection_id)
        
        # Try as connection ID
        return self.connections.get(target)
    
    def _cleanup_connection(self, connection_info: ConnectionInfo) -> None:
        """Clean up connection resources"""
        # Remove from connections
        if connection_info.connection_id in self.connections:
            del self.connections[connection_info.connection_id]
        
        # Remove node mapping
        if connection_info.node_id and connection_info.node_id in self.node_connections:
            del self.node_connections[connection_info.node_id]
        
        self._update_connection_count(-1)
        logger.info(f"Connection cleaned up: {connection_info.connection_id}")
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to monitor connections"""
        while self.running:
            try:
                # Check connection health
                current_time = datetime.now(timezone.utc)
                stale_connections = []
                
                for connection_info in self.connections.values():
                    if connection_info.last_ping:
                        time_since_ping = (current_time - connection_info.last_ping).total_seconds()
                        if time_since_ping > self.heartbeat_interval * 3:
                            stale_connections.append(connection_info)
                
                # Clean up stale connections
                for connection_info in stale_connections:
                    logger.warning(f"Closing stale connection: {connection_info.connection_id}")
                    try:
                        await connection_info.websocket.close()
                    except Exception:
                        pass
                    self._cleanup_connection(connection_info)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
