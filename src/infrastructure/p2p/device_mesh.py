"""Multi-protocol device mesh with automatic failover"""

import asyncio
import time
import platform
import subprocess
from typing import List, Optional, Tuple, Dict, Any
import logging

from .p2p_node import P2PNode, PeerCapabilities

logger = logging.getLogger(__name__)

class DeviceMesh:
    """Bluetooth + WiFi Direct mesh networking with store-and-forward"""
    
    def __init__(self, p2p_node: P2PNode):
        self.p2p_node = p2p_node
        self.bluetooth_enabled = self._check_bluetooth()
        self.wifi_direct_enabled = self._check_wifi_direct()
        self.hop_limit = 7  # Max hops for mesh routing
        self.store_forward_buffer: Dict[str, List[Dict]] = {}  # For offline nodes
        self.routing_table: Dict[str, str] = {}  # peer_id -> next_hop_id
        
        # Register mesh-specific message handlers
        self._register_mesh_handlers()
        
    def _check_bluetooth(self) -> bool:
        """Check Bluetooth availability"""
        try:
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["hciconfig"], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                return "UP RUNNING" in result.stdout
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["system_profiler", "SPBluetoothDataType"], 
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return "Bluetooth Low Energy Supported: Yes" in result.stdout
            elif platform.system() == "Windows":
                # Windows Bluetooth check
                result = subprocess.run(
                    ["powershell", "-Command", "Get-PnpDevice -Class Bluetooth"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return "OK" in result.stdout
            return False
        except Exception as e:
            logger.debug(f"Bluetooth check failed: {e}")
            return False
            
    def _check_wifi_direct(self) -> bool:
        """Check WiFi Direct availability"""
        try:
            if platform.system() == "Linux":
                # Check for WiFi Direct support
                result = subprocess.run(
                    ["iwconfig"], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                return "IEEE 802.11" in result.stdout
            elif platform.system() == "Windows":
                # Windows WiFi Direct check
                result = subprocess.run(
                    ["netsh", "wlan", "show", "profiles"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return len(result.stdout) > 0
            return False
        except Exception as e:
            logger.debug(f"WiFi Direct check failed: {e}")
            return False
            
    def _register_mesh_handlers(self):
        """Register mesh-specific message handlers"""
        self.p2p_node.register_handler('MESH_ROUTE', self._handle_mesh_route)
        self.p2p_node.register_handler('MESH_DISCOVER', self._handle_mesh_discover)
        self.p2p_node.register_handler('OFFLINE_DELIVERY', self._handle_offline_delivery)
        
    async def _handle_mesh_route(self, message: Dict, writer):
        """Handle mesh routing message"""
        target_peer = message.get('target_peer')
        hop_count = message.get('hop_count', 0)
        
        if hop_count >= self.hop_limit:
            logger.warning("Message exceeded hop limit, dropping")
            return
            
        if target_peer == self.p2p_node.node_id:
            # Message reached destination
            payload = message.get('payload', {})
            await self._handle_delivered_message(payload)
            return
            
        # Forward to next hop
        next_hop = self.routing_table.get(target_peer)
        if next_hop and next_hop in self.p2p_node.connections:
            message['hop_count'] = hop_count + 1
            await self.p2p_node.send_to_peer(next_hop, message)
        else:
            # No route found, store for later
            self.store_for_offline_peer(target_peer, message)
            
    async def _handle_mesh_discover(self, message: Dict, writer):
        """Handle mesh discovery message"""
        discovered_peers = message.get('peers', [])
        
        # Update routing table with discovered routes
        for peer_info in discovered_peers:
            peer_id = peer_info.get('peer_id')
            via_node = message.get('sender_id')
            hop_count = peer_info.get('hop_count', 1)
            
            if peer_id and peer_id != self.p2p_node.node_id:
                # Update route if this is a better path
                current_route = self.routing_table.get(peer_id)
                if not current_route or hop_count < self._get_route_cost(current_route):
                    self.routing_table[peer_id] = via_node
                    
    async def _handle_offline_delivery(self, message: Dict, writer):
        """Handle delivery of stored offline messages"""
        target_peer = message.get('target_peer')
        if target_peer in self.store_forward_buffer:
            messages = self.store_forward_buffer[target_peer]
            
            # Try to deliver stored messages
            for stored_msg in messages[:]:  # Copy list to modify during iteration
                try:
                    await self.p2p_node.send_to_peer(target_peer, stored_msg)
                    messages.remove(stored_msg)
                    logger.info(f"Delivered stored message to {target_peer}")
                except:
                    break  # Stop if delivery fails
                    
            # Clean up if all delivered
            if not messages:
                del self.store_forward_buffer[target_peer]
                
    async def _handle_delivered_message(self, payload: Dict):
        """Handle message that reached its destination"""
        msg_type = payload.get('type', 'UNKNOWN')
        logger.info(f"Mesh message delivered: {msg_type}")
        
        # Process the actual payload
        if msg_type in self.p2p_node.message_handlers:
            await self.p2p_node.message_handlers[msg_type](payload, None)
            
    def _get_route_cost(self, next_hop: str) -> int:
        """Get cost/distance for a route (simplified)"""
        # For now, all routes have cost 1
        # Could be enhanced with latency, bandwidth, etc.
        return 1
        
    async def discover_bluetooth_peers(self) -> List[Tuple[str, str]]:
        """Discover nearby Bluetooth devices"""
        if not self.bluetooth_enabled:
            logger.debug("Bluetooth not available for discovery")
            return []
            
        # Simplified Bluetooth discovery (would need bluetooth library)
        # For now, return empty list
        logger.debug("Bluetooth discovery not yet implemented")
        return []
        
    async def establish_mesh_connection(self, peer_addr: str, protocol: str = "auto") -> Optional[str]:
        """Establish connection with automatic protocol selection"""
        
        if protocol == "auto":
            # Try TCP first (most reliable)
            if await self._try_tcp_connect(peer_addr):
                return "tcp"
            # Could add Bluetooth, WiFi Direct here
            
        elif protocol == "tcp":
            return "tcp" if await self._try_tcp_connect(peer_addr) else None
        elif protocol == "bluetooth":
            return await self._try_bluetooth_connect(peer_addr)
        elif protocol == "wifi_direct":
            return await self._try_wifi_direct_connect(peer_addr)
            
        return None
        
    async def _try_tcp_connect(self, peer_addr: str) -> bool:
        """Try TCP connection to peer"""
        try:
            if ':' in peer_addr:
                host, port_str = peer_addr.rsplit(':', 1)
                port = int(port_str)
            else:
                host, port = peer_addr, 9000  # Default port
                
            return await self.p2p_node.connect_to_peer(host, port)
        except Exception as e:
            logger.debug(f"TCP connection to {peer_addr} failed: {e}")
            return False
            
    async def _try_bluetooth_connect(self, peer_addr: str) -> Optional[str]:
        """Try Bluetooth connection (placeholder)"""
        if not self.bluetooth_enabled:
            return None
            
        # Bluetooth connection would be implemented here
        # For now, return None as not implemented
        logger.debug("Bluetooth connection not yet implemented")
        return None
        
    async def _try_wifi_direct_connect(self, peer_addr: str) -> Optional[str]:
        """Try WiFi Direct connection (placeholder)"""
        if not self.wifi_direct_enabled:
            return None
            
        # WiFi Direct connection would be implemented here
        # For now, return None as not implemented
        logger.debug("WiFi Direct connection not yet implemented")
        return None
        
    def store_for_offline_peer(self, peer_id: str, message: Dict):
        """Store message for offline peer"""
        if peer_id not in self.store_forward_buffer:
            self.store_forward_buffer[peer_id] = []
            
        # Add timestamp and TTL
        stored_message = {
            "message": message,
            "timestamp": time.time(),
            "ttl": 3600,  # 1 hour TTL
            "attempts": 0
        }
        
        self.store_forward_buffer[peer_id].append(stored_message)
        
        # Limit buffer size per peer
        if len(self.store_forward_buffer[peer_id]) > 100:
            # Remove oldest messages
            self.store_forward_buffer[peer_id] = self.store_forward_buffer[peer_id][-100:]
            
        logger.info(f"Stored message for offline peer {peer_id}")
        
    def retrieve_stored_messages(self, peer_id: str) -> List[Dict]:
        """Retrieve stored messages for a peer"""
        if peer_id not in self.store_forward_buffer:
            return []
            
        current_time = time.time()
        valid_messages = []
        
        for stored_msg in self.store_forward_buffer[peer_id]:
            # Check TTL
            age = current_time - stored_msg["timestamp"]
            if age < stored_msg["ttl"]:
                valid_messages.append(stored_msg)
                
        # Update buffer with only valid messages
        self.store_forward_buffer[peer_id] = valid_messages
        
        return [msg["message"] for msg in valid_messages]
        
    async def send_mesh_message(self, target_peer: str, message: Dict, max_hops: int = 7):
        """Send message through mesh network"""
        
        # Direct connection available?
        if target_peer in self.p2p_node.connections:
            await self.p2p_node.send_to_peer(target_peer, message)
            return
            
        # Route through mesh
        next_hop = self.routing_table.get(target_peer)
        if next_hop and next_hop in self.p2p_node.connections:
            mesh_message = {
                'type': 'MESH_ROUTE',
                'target_peer': target_peer,
                'payload': message,
                'hop_count': 0,
                'max_hops': max_hops
            }
            await self.p2p_node.send_to_peer(next_hop, mesh_message)
        else:
            # Store for later delivery
            self.store_for_offline_peer(target_peer, message)
            
    async def broadcast_mesh_discovery(self):
        """Broadcast mesh topology for route discovery"""
        # Create discovery message with known peers
        peers_info = []
        for peer_id, capabilities in self.p2p_node.peer_registry.items():
            peers_info.append({
                'peer_id': peer_id,
                'hop_count': 1,  # Direct connections
                'capabilities': capabilities.__dict__
            })
            
        discovery_message = {
            'type': 'MESH_DISCOVER',
            'peers': peers_info,
            'timestamp': time.time()
        }
        
        await self.p2p_node.broadcast_to_peers('MESH_DISCOVER', discovery_message)
        
    async def start_mesh_maintenance(self):
        """Start background mesh maintenance tasks"""
        
        async def maintenance_loop():
            while True:
                try:
                    # Clean expired stored messages
                    await self._cleanup_expired_messages()
                    
                    # Update routing table
                    await self.broadcast_mesh_discovery()
                    
                    # Try to deliver stored messages
                    await self._attempt_offline_deliveries()
                    
                    await asyncio.sleep(60)  # Run every minute
                    
                except Exception as e:
                    logger.error(f"Mesh maintenance error: {e}")
                    await asyncio.sleep(10)
                    
        asyncio.create_task(maintenance_loop())
        
    async def _cleanup_expired_messages(self):
        """Remove expired messages from store-forward buffer"""
        current_time = time.time()
        
        for peer_id in list(self.store_forward_buffer.keys()):
            messages = self.store_forward_buffer[peer_id]
            valid_messages = [
                msg for msg in messages
                if (current_time - msg["timestamp"]) < msg["ttl"]
            ]
            
            if valid_messages:
                self.store_forward_buffer[peer_id] = valid_messages
            else:
                del self.store_forward_buffer[peer_id]
                
    async def _attempt_offline_deliveries(self):
        """Try to deliver stored messages to peers that came online"""
        for peer_id in list(self.store_forward_buffer.keys()):
            if peer_id in self.p2p_node.connections:
                # Peer is online, try delivery
                await self._handle_offline_delivery({'target_peer': peer_id}, None)
                
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get mesh network status"""
        return {
            'bluetooth_enabled': self.bluetooth_enabled,
            'wifi_direct_enabled': self.wifi_direct_enabled,
            'routing_table_size': len(self.routing_table),
            'stored_messages': sum(len(msgs) for msgs in self.store_forward_buffer.values()),
            'offline_peers': len(self.store_forward_buffer),
            'hop_limit': self.hop_limit
        }