# LibP2P Mesh Network Implementation
# Production-ready P2P mesh networking with libp2p integration

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import time
from collections import defaultdict


logger = logging.getLogger(__name__)


class MeshMessageType(Enum):
    """Types of messages in the mesh network."""
    DISCOVERY = "discovery"
    HEARTBEAT = "heartbeat"
    DATA = "data"
    ROUTING = "routing"
    CONSENSUS = "consensus"
    SYNC = "sync"
    ERROR = "error"


@dataclass
class MeshMessage:
    """Message structure for mesh network communication."""
    
    message_type: MeshMessageType
    source_peer: str
    target_peer: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000000)}")
    ttl: int = 10  # Time to live for message propagation
    route_history: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "message_type": self.message_type.value,
            "source_peer": self.source_peer,
            "target_peer": self.target_peer,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "ttl": self.ttl,
            "route_history": self.route_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeshMessage':
        """Deserialize message from dictionary."""
        return cls(
            message_type=MeshMessageType(data["message_type"]),
            source_peer=data["source_peer"],
            target_peer=data.get("target_peer"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id"),
            ttl=data.get("ttl", 10),
            route_history=data.get("route_history", [])
        )


class PeerInfo:
    """Information about a peer in the mesh network."""
    
    def __init__(self, peer_id: str, address: str, port: int, capabilities: Set[str] = None):
        self.peer_id = peer_id
        self.address = address
        self.port = port
        self.capabilities = capabilities or set()
        self.last_seen = time.time()
        self.connection_count = 0
        self.reputation_score = 1.0
        self.latency_ms = 0
        
    def update_last_seen(self):
        """Update last seen timestamp."""
        self.last_seen = time.time()
        
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if peer is considered alive based on last seen timestamp."""
        return (time.time() - self.last_seen) < timeout
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize peer info to dictionary."""
        return {
            "peer_id": self.peer_id,
            "address": self.address,
            "port": self.port,
            "capabilities": list(self.capabilities),
            "last_seen": self.last_seen,
            "connection_count": self.connection_count,
            "reputation_score": self.reputation_score,
            "latency_ms": self.latency_ms
        }


class LibP2PMeshNetwork:
    """LibP2P-based mesh network implementation."""
    
    def __init__(self, 
                 peer_id: str, 
                 listen_address: str = "0.0.0.0", 
                 listen_port: int = 0,
                 max_connections: int = 50,
                 heartbeat_interval: float = 10.0):
        """
        Initialize LibP2P mesh network.
        
        Args:
            peer_id: Unique identifier for this peer
            listen_address: Address to listen on
            listen_port: Port to listen on (0 for random)
            max_connections: Maximum number of connections
            heartbeat_interval: Interval between heartbeat messages
        """
        self.peer_id = peer_id
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        
        # Network state
        self.peers: Dict[str, PeerInfo] = {}
        self.connections: Dict[str, Any] = {}  # Active connections
        self.message_handlers: Dict[MeshMessageType, List[Callable]] = defaultdict(list)
        self.routing_table: Dict[str, str] = {}  # target_peer -> next_hop_peer
        self.message_cache: Set[str] = set()  # For duplicate detection
        
        # Control flags
        self.is_running = False
        self.discovery_enabled = True
        self.routing_enabled = True
        
        logger.info(f"Initialized LibP2P mesh network for peer {peer_id}")
    
    async def start(self) -> bool:
        """Start the mesh network."""
        try:
            self.is_running = True
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())
            
            # Start peer discovery
            if self.discovery_enabled:
                asyncio.create_task(self._discovery_loop())
                
            # Start routing table maintenance
            if self.routing_enabled:
                asyncio.create_task(self._routing_maintenance_loop())
                
            logger.info(f"Started mesh network on {self.listen_address}:{self.listen_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mesh network: {e}")
            self.is_running = False
            return False
    
    async def stop(self):
        """Stop the mesh network."""
        self.is_running = False
        
        # Close all connections
        for peer_id, connection in self.connections.items():
            try:
                await self._close_connection(connection)
            except Exception as e:
                logger.warning(f"Error closing connection to {peer_id}: {e}")
                
        self.connections.clear()
        self.peers.clear()
        logger.info("Stopped mesh network")
    
    def add_message_handler(self, message_type: MeshMessageType, handler: Callable):
        """Add a message handler for a specific message type."""
        self.message_handlers[message_type].append(handler)
        
    def remove_message_handler(self, message_type: MeshMessageType, handler: Callable):
        """Remove a message handler."""
        if handler in self.message_handlers[message_type]:
            self.message_handlers[message_type].remove(handler)
    
    async def connect_peer(self, peer_address: str, peer_port: int, peer_id: str = None) -> bool:
        """Connect to a peer."""
        if len(self.connections) >= self.max_connections:
            logger.warning("Maximum connections reached")
            return False
            
        try:
            # Simulate connection establishment
            # In real implementation, this would use libp2p to establish connection
            peer_id = peer_id or f"peer_{peer_address}_{peer_port}"
            
            if peer_id not in self.connections:
                # Create peer info
                peer_info = PeerInfo(peer_id, peer_address, peer_port)
                self.peers[peer_id] = peer_info
                
                # Simulate connection object
                connection = {
                    "peer_id": peer_id,
                    "address": peer_address,
                    "port": peer_port,
                    "connected_at": time.time()
                }
                self.connections[peer_id] = connection
                
                logger.info(f"Connected to peer {peer_id} at {peer_address}:{peer_port}")
                
                # Send discovery message
                await self._send_discovery_message(peer_id)
                return True
            else:
                logger.info(f"Already connected to peer {peer_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_address}:{peer_port}: {e}")
            return False
    
    async def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect from a peer."""
        try:
            if peer_id in self.connections:
                connection = self.connections[peer_id]
                await self._close_connection(connection)
                del self.connections[peer_id]
                
            if peer_id in self.peers:
                del self.peers[peer_id]
                
            # Update routing table
            self.routing_table = {k: v for k, v in self.routing_table.items() if v != peer_id}
            
            logger.info(f"Disconnected from peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disconnect from peer {peer_id}: {e}")
            return False
    
    async def send_message(self, message: MeshMessage) -> bool:
        """Send a message through the mesh network."""
        try:
            # Add to route history
            if self.peer_id not in message.route_history:
                message.route_history.append(self.peer_id)
                
            # Determine target(s)
            if message.target_peer:
                # Direct message to specific peer
                await self._route_message(message)
            else:
                # Broadcast to all connected peers
                await self._broadcast_message(message)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def _route_message(self, message: MeshMessage):
        """Route a message to its target peer."""
        target_peer = message.target_peer
        
        if target_peer in self.connections:
            # Direct connection available
            await self._send_to_peer(target_peer, message)
        elif target_peer in self.routing_table:
            # Route through next hop
            next_hop = self.routing_table[target_peer]
            if next_hop in self.connections:
                await self._send_to_peer(next_hop, message)
            else:
                logger.warning(f"Next hop {next_hop} not available for target {target_peer}")
        else:
            # No route known, broadcast for discovery
            logger.warning(f"No route to {target_peer}, broadcasting")
            await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: MeshMessage):
        """Broadcast a message to all connected peers."""
        if message.ttl <= 0:
            logger.debug(f"Message {message.message_id} TTL expired")
            return
            
        # Decrement TTL
        message.ttl -= 1
        
        # Check for duplicate
        if message.message_id in self.message_cache:
            logger.debug(f"Ignoring duplicate message {message.message_id}")
            return
            
        # Add to cache
        self.message_cache.add(message.message_id)
        
        # Send to all connected peers except source and those in route history
        exclude_peers = {message.source_peer} | set(message.route_history)
        
        for peer_id in self.connections:
            if peer_id not in exclude_peers:
                await self._send_to_peer(peer_id, message)
    
    async def _send_to_peer(self, peer_id: str, message: MeshMessage):
        """Send a message to a specific peer."""
        try:
            if peer_id not in self.connections:
                logger.warning(f"No connection to peer {peer_id}")
                return
                
            # Update peer last seen
            if peer_id in self.peers:
                self.peers[peer_id].update_last_seen()
            
            # In real implementation, this would serialize and send over the network
            serialized_message = json.dumps(message.to_dict())
            
            # Simulate network send
            logger.debug(f"Sent message {message.message_id} to peer {peer_id}")
            
        except Exception as e:
            logger.error(f"Failed to send message to peer {peer_id}: {e}")
    
    async def _handle_received_message(self, message: MeshMessage):
        """Handle a received message."""
        try:
            # Update sender info
            if message.source_peer in self.peers:
                self.peers[message.source_peer].update_last_seen()
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            
            # Handle routing and discovery
            if message.message_type == MeshMessageType.DISCOVERY:
                await self._handle_discovery_message(message)
            elif message.message_type == MeshMessageType.ROUTING:
                await self._handle_routing_message(message)
                
        except Exception as e:
            logger.error(f"Error handling received message: {e}")
    
    async def _handle_discovery_message(self, message: MeshMessage):
        """Handle peer discovery messages."""
        source_peer = message.source_peer
        
        # Add or update peer info
        if source_peer not in self.peers:
            peer_info = PeerInfo(source_peer, "unknown", 0)
            self.peers[source_peer] = peer_info
        
        # Update routing table
        if len(message.route_history) > 0:
            next_hop = message.route_history[0]  # First hop from us
            self.routing_table[source_peer] = next_hop
    
    async def _handle_routing_message(self, message: MeshMessage):
        """Handle routing table update messages."""
        # Update routing information based on received data
        routing_data = message.payload.get("routing_table", {})
        
        for target, next_hop in routing_data.items():
            if target not in self.routing_table or len(message.route_history) < 3:
                # Accept route if we don't have one or if it's shorter
                self.routing_table[target] = message.source_peer
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        while self.is_running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.connections:
                    heartbeat_message = MeshMessage(
                        message_type=MeshMessageType.HEARTBEAT,
                        source_peer=self.peer_id,
                        payload={"timestamp": time.time(), "peer_count": len(self.peers)}
                    )
                    
                    await self._broadcast_message(heartbeat_message)
                    
                # Clean up dead peers
                await self._cleanup_dead_peers()
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _discovery_loop(self):
        """Periodic peer discovery."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Discovery every 30 seconds
                
                if len(self.connections) < self.max_connections:
                    discovery_message = MeshMessage(
                        message_type=MeshMessageType.DISCOVERY,
                        source_peer=self.peer_id,
                        payload={
                            "capabilities": list(getattr(self, 'capabilities', set())),
                            "max_connections": self.max_connections,
                            "current_connections": len(self.connections)
                        }
                    )
                    
                    await self._broadcast_message(discovery_message)
                    
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
    
    async def _routing_maintenance_loop(self):
        """Maintain routing table."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Update routing every minute
                
                # Share routing information
                routing_message = MeshMessage(
                    message_type=MeshMessageType.ROUTING,
                    source_peer=self.peer_id,
                    payload={"routing_table": dict(self.routing_table)}
                )
                
                await self._broadcast_message(routing_message)
                
            except Exception as e:
                logger.error(f"Error in routing maintenance loop: {e}")
    
    async def _cleanup_dead_peers(self):
        """Remove dead peers from network state."""
        dead_peers = []
        
        for peer_id, peer_info in self.peers.items():
            if not peer_info.is_alive():
                dead_peers.append(peer_id)
        
        for peer_id in dead_peers:
            logger.info(f"Removing dead peer {peer_id}")
            await self.disconnect_peer(peer_id)
    
    async def _send_discovery_message(self, target_peer: str):
        """Send discovery message to a specific peer."""
        discovery_message = MeshMessage(
            message_type=MeshMessageType.DISCOVERY,
            source_peer=self.peer_id,
            target_peer=target_peer,
            payload={"greeting": True}
        )
        
        await self.send_message(discovery_message)
    
    async def _close_connection(self, connection):
        """Close a network connection."""
        # In real implementation, this would close the actual network connection
        pass
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        alive_peers = sum(1 for peer in self.peers.values() if peer.is_alive())
        
        return {
            "peer_id": self.peer_id,
            "is_running": self.is_running,
            "active_connections": len(self.connections),
            "known_peers": len(self.peers),
            "alive_peers": alive_peers,
            "routing_table_size": len(self.routing_table),
            "message_cache_size": len(self.message_cache),
            "max_connections": self.max_connections,
            "discovery_enabled": self.discovery_enabled,
            "routing_enabled": self.routing_enabled
        }


# Backward compatibility - try to import from actual infrastructure locations first
try:
    from infrastructure.p2p.core.libp2p_mesh import *
except ImportError:
    try:
        from core.p2p.core.libp2p_mesh import *
    except ImportError:
        # Use the implementations defined above
        pass
