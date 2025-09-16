"""
P2P Transport Implementation

P2P transport implementation using existing LibP2P infrastructure.
Consolidates P2P communication functionality from multiple systems.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone

try:
    # Import existing P2P infrastructure
    from infrastructure.p2p.core.libp2p_transport import LibP2PTransport
    from infrastructure.p2p.communications.service_discovery import discover_services, ServiceInfo
    from infrastructure.p2p.communications.protocol import StandardCommunicationProtocol
    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False
    LibP2PTransport = None
    ServiceInfo = None

from .base_transport import BaseTransport, TransportState
from ..message_format import UnifiedMessage, MessageType, TransportType
from ..reliability.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class PeerInfo:
    """Information about a P2P peer"""
    
    def __init__(self, node_id: str, address: str, peer_id: str = None):
        self.node_id = node_id
        self.address = address
        self.peer_id = peer_id or node_id
        self.last_seen = datetime.now(timezone.utc)
        self.connected = False
        self.message_count = 0
        self.service_info: Optional[ServiceInfo] = None
    
    def update_last_seen(self) -> None:
        """Update last seen timestamp"""
        self.last_seen = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "peer_id": self.peer_id,
            "last_seen": self.last_seen.isoformat(),
            "connected": self.connected,
            "message_count": self.message_count
        }


class P2PTransport(BaseTransport):
    """P2P transport implementation using LibP2P"""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not P2P_AVAILABLE:
            raise ImportError("P2P infrastructure not available")
        
        # P2P configuration
        self.port = config.get("port", 0)  # 0 for auto-allocation
        self.bootstrap_peers = config.get("bootstrap_peers", [])
        self.enable_discovery = config.get("enable_discovery", True)
        
        # Initialize LibP2P transport
        self.libp2p = LibP2PTransport(node_id, {
            "port": self.port,
            "bootstrap_peers": self.bootstrap_peers
        })
        
        # Protocol for higher-level communication
        self.protocol = StandardCommunicationProtocol(node_id, self.port)
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}  # node_id -> PeerInfo
        self.connected_peers: Set[str] = set()
        
        # Discovery and monitoring
        self.discovery_task = None
        self.discovery_interval = config.get("discovery_interval", 30)
        
        # Circuit breaker for P2P resilience
        cb_config = config.get("circuit_breaker", {})
        self.circuit_breaker = CircuitBreaker(cb_config)
        
        # Message routing cache
        self.routing_cache: Dict[str, str] = {}  # target -> peer_id
        self.cache_ttl = config.get("cache_ttl", 300)  # 5 minutes
        
        logger.info(f"P2P transport initialized for node: {node_id}")
    
    async def start(self) -> None:
        """Start P2P transport with discovery"""
        if self.running:
            logger.warning("P2P transport already running")
            return
        
        logger.info(f"Starting P2P transport for node: {self.node_id}")
        self.state = TransportState.STARTING
        
        try:
            # Start LibP2P transport
            await self.libp2p.start()
            
            # Start protocol layer
            await self.protocol.start_server()
            
            # Set up message handlers
            self.protocol.register_handler("unified_message", self._handle_p2p_message)
            self.libp2p.set_message_handler(self._handle_libp2p_message)
            
            # Start peer discovery if enabled
            if self.enable_discovery:
                self.discovery_task = asyncio.create_task(self._discovery_loop())
            
            # Connect to bootstrap peers
            await self._connect_bootstrap_peers()
            
            self.running = True
            self.state = TransportState.RUNNING
            
            logger.info(f"P2P transport started on port: {self.protocol.port}")
            
        except Exception as e:
            logger.error(f"Failed to start P2P transport: {e}")
            self.state = TransportState.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop P2P transport"""
        if not self.running:
            logger.warning("P2P transport not running")
            return
        
        logger.info("Stopping P2P transport")
        self.state = TransportState.STOPPING
        
        try:
            # Stop discovery task
            if self.discovery_task:
                self.discovery_task.cancel()
                try:
                    await self.discovery_task
                except asyncio.CancelledError:
                    pass
            
            # Disconnect from peers
            await self._disconnect_all_peers()
            
            # Stop protocol and LibP2P
            await self.protocol.disconnect()
            await self.libp2p.stop()
            
            # Clear state
            self.peers.clear()
            self.connected_peers.clear()
            self.routing_cache.clear()
            
            self.running = False
            self.state = TransportState.STOPPED
            
            logger.info("P2P transport stopped")
            
        except Exception as e:
            logger.error(f"Error stopping P2P transport: {e}")
            self.state = TransportState.ERROR
            raise
    
    async def send(self, message: UnifiedMessage, target: str) -> bool:
        """Send P2P message to target peer"""
        if not self.running:
            logger.error("Cannot send: P2P transport not running")
            return False
        
        try:
            # Find target peer
            peer_info = await self._resolve_peer(target)
            if not peer_info:
                logger.warning(f"Cannot resolve peer: {target}")
                self._record_send_error()
                return False
            
            # Use circuit breaker for resilience
            success = await self.circuit_breaker.call(
                self._send_to_peer, peer_info, message
            )
            
            if success:
                self._record_send_success()
                peer_info.message_count += 1
                peer_info.update_last_seen()
            else:
                self._record_send_error()
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending P2P message to {target}: {e}")
            self._record_send_error()
            return False
    
    async def broadcast(self, message: UnifiedMessage) -> Dict[str, bool]:
        """Broadcast to all connected peers"""
        if not self.running:
            logger.error("Cannot broadcast: P2P transport not running")
            return {}
        
        results = {}
        
        for node_id, peer_info in list(self.peers.items()):
            if peer_info.connected and node_id != self.node_id:
                try:
                    success = await self.circuit_breaker.call(
                        self._send_to_peer, peer_info, message
                    )
                    
                    results[node_id] = success
                    
                    if success:
                        self._record_send_success()
                        peer_info.message_count += 1
                        peer_info.update_last_seen()
                    else:
                        self._record_send_error()
                        
                except Exception as e:
                    logger.error(f"Error broadcasting to {node_id}: {e}")
                    results[node_id] = False
                    self._record_send_error()
        
        logger.info(f"P2P broadcast completed: {sum(results.values())}/{len(results)} successful")
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check P2P transport health"""
        peer_health = {}
        for node_id, peer_info in self.peers.items():
            peer_health[node_id] = peer_info.to_dict()
        
        libp2p_health = {}
        try:
            libp2p_health = await self.libp2p.get_status()
        except Exception as e:
            libp2p_health = {"error": str(e)}
        
        return {
            "status": "healthy" if self.running else "stopped",
            "state": self.state,
            "port": getattr(self.protocol, 'port', 0),
            "peers_known": len(self.peers),
            "peers_connected": len(self.connected_peers),
            "discovery_enabled": self.enable_discovery,
            "libp2p_status": libp2p_health,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "peers": peer_health,
            "metrics": self.get_metrics()
        }
    
    def get_peers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about known peers"""
        return {
            node_id: peer_info.to_dict() 
            for node_id, peer_info in self.peers.items()
        }
    
    def get_connected_peers(self) -> Set[str]:
        """Get set of connected peer node IDs"""
        return self.connected_peers.copy()
    
    async def connect_peer(self, node_id: str, address: str) -> bool:
        """Manually connect to a peer"""
        try:
            # Create peer info
            peer_info = PeerInfo(node_id, address)
            
            # Attempt connection
            success = await self.protocol.connect(node_id, address)
            if success:
                peer_info.connected = True
                self.peers[node_id] = peer_info
                self.connected_peers.add(node_id)
                self._update_connection_count(1)
                logger.info(f"Connected to peer: {node_id} at {address}")
                return True
            else:
                logger.warning(f"Failed to connect to peer: {node_id} at {address}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to peer {node_id}: {e}")
            self._record_connection_error()
            return False
    
    async def disconnect_peer(self, node_id: str) -> bool:
        """Disconnect from a peer"""
        try:
            if node_id in self.connected_peers:
                await self.protocol.disconnect_peer(node_id)
                self.connected_peers.discard(node_id)
                
                if node_id in self.peers:
                    self.peers[node_id].connected = False
                
                self._update_connection_count(-1)
                logger.info(f"Disconnected from peer: {node_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error disconnecting from peer {node_id}: {e}")
            return False
    
    async def _connect_bootstrap_peers(self) -> None:
        """Connect to bootstrap peers"""
        for peer_address in self.bootstrap_peers:
            try:
                # Parse address format: /ip4/127.0.0.1/tcp/9000/p2p/QmNodeId
                # For now, use simple format: host:port
                if ":" in peer_address:
                    host, port = peer_address.split(":", 1)
                    peer_id = f"peer_{host}_{port}"
                    ws_address = f"ws://{host}:{port}/ws"
                    
                    await self.connect_peer(peer_id, ws_address)
                    
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer_address}: {e}")
    
    async def _discovery_loop(self) -> None:
        """Continuous peer discovery loop"""
        while self.running:
            try:
                # Discover services
                discovered_services = await discover_services()
                
                for service in discovered_services:
                    if service.node_id != self.node_id and service.node_id not in self.peers:
                        # Create peer info from service
                        address = f"ws://{service.host}:{service.port}/ws"
                        peer_info = PeerInfo(service.node_id, address)
                        peer_info.service_info = service
                        
                        self.peers[service.node_id] = peer_info
                        logger.info(f"Discovered peer: {service.node_id} at {address}")
                        
                        # Optionally auto-connect
                        if len(self.connected_peers) < 5:  # Maintain up to 5 connections
                            await self.connect_peer(service.node_id, address)
                
                await asyncio.sleep(self.discovery_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _resolve_peer(self, target: str) -> Optional[PeerInfo]:
        """Resolve target to peer info"""
        # Direct lookup
        if target in self.peers:
            return self.peers[target]
        
        # Check routing cache
        if target in self.routing_cache:
            peer_id = self.routing_cache[target]
            if peer_id in self.peers:
                return self.peers[peer_id]
        
        # Trigger discovery if not found
        if self.enable_discovery:
            try:
                services = await discover_services()
                for service in services:
                    if service.node_id == target:
                        address = f"ws://{service.host}:{service.port}/ws"
                        peer_info = PeerInfo(target, address)
                        peer_info.service_info = service
                        self.peers[target] = peer_info
                        
                        # Try to connect
                        if await self.connect_peer(target, address):
                            return peer_info
            except Exception as e:
                logger.error(f"Error resolving peer {target}: {e}")
        
        return None
    
    async def _send_to_peer(self, peer_info: PeerInfo, message: UnifiedMessage) -> bool:
        """Send message to specific peer"""
        try:
            # Convert to protocol-compatible format
            protocol_message = {
                "type": "unified_message",
                "data": message.to_dict()
            }
            
            # Send via protocol layer
            if peer_info.connected:
                success = await self.protocol.send_message(peer_info.node_id, protocol_message)
                if success:
                    logger.debug(f"P2P message sent to {peer_info.node_id}: {message.message_id}")
                    return True
            
            # Try LibP2P direct send as fallback
            success = await self.libp2p.send_message(peer_info.peer_id, message.to_dict())
            if success:
                logger.debug(f"LibP2P message sent to {peer_info.node_id}: {message.message_id}")
                return True
            
            logger.warning(f"Failed to send P2P message to {peer_info.node_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error sending to peer {peer_info.node_id}: {e}")
            return False
    
    async def _handle_p2p_message(self, sender_id: str, message_data: Dict[str, Any]) -> None:
        """Handle message from protocol layer"""
        try:
            if message_data.get("type") == "unified_message":
                unified_msg = UnifiedMessage.from_dict(message_data["data"])
                await self.handle_incoming_message(unified_msg)
                
                # Update peer info
                if sender_id in self.peers:
                    self.peers[sender_id].update_last_seen()
                    
        except Exception as e:
            logger.error(f"Error handling P2P message from {sender_id}: {e}")
    
    async def _handle_libp2p_message(self, peer_id: str, message_data: Dict[str, Any]) -> None:
        """Handle message from LibP2P layer"""
        try:
            # Convert LibP2P message to unified format
            unified_msg = UnifiedMessage.from_dict(message_data)
            await self.handle_incoming_message(unified_msg)
            
        except Exception as e:
            logger.error(f"Error handling LibP2P message from {peer_id}: {e}")
    
    async def _disconnect_all_peers(self) -> None:
        """Disconnect from all peers"""
        for node_id in list(self.connected_peers):
            try:
                await self.disconnect_peer(node_id)
            except Exception as e:
                logger.error(f"Error disconnecting from {node_id}: {e}")
