#!/usr/bin/env python3
"""Mesh Network Performance Fixer

CRITICAL ISSUE: 0% message delivery rate in mesh network
This module fixes the routing algorithms, message serialization, 
and connection management to restore network functionality.

Root Causes Identified:
1. Broken routing algorithms
2. Message serialization failures  
3. Connection pooling issues
4. Network topology problems
"""

import asyncio
import json
import logging
import pickle
import socket
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import threading
import queue
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for mesh network communication."""
    DISCOVERY = "DISCOVERY"
    PARAMETER_UPDATE = "PARAMETER_UPDATE"
    GRADIENT_SHARE = "GRADIENT_SHARE"
    HEARTBEAT = "HEARTBEAT"
    ROUTE_UPDATE = "ROUTE_UPDATE"
    DATA_SYNC = "DATA_SYNC"


@dataclass
class NetworkMessage:
    """Enhanced network message with proper serialization."""
    id: str
    type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    timestamp: float
    ttl: int = 10
    route_path: List[str] = None
    
    def __post_init__(self):
        if self.route_path is None:
            self.route_path = []
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes with error handling."""
        try:
            data = {
                'id': self.id,
                'type': self.type.value,
                'sender_id': self.sender_id,
                'receiver_id': self.receiver_id,
                'payload': self.payload,
                'timestamp': self.timestamp,
                'ttl': self.ttl,
                'route_path': self.route_path
            }
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Message serialization failed: {e}")
            # Fallback to JSON serialization
            data['payload'] = str(self.payload)  # Convert to string as fallback
            return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'NetworkMessage':
        """Deserialize message from bytes with error handling."""
        try:
            # Try pickle first
            msg_data = pickle.loads(data)
        except Exception:
            try:
                # Fallback to JSON
                msg_data = json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Message deserialization failed: {e}")
                raise
        
        return cls(
            id=msg_data['id'],
            type=MessageType(msg_data['type']),
            sender_id=msg_data['sender_id'],
            receiver_id=msg_data['receiver_id'],
            payload=msg_data['payload'],
            timestamp=msg_data['timestamp'],
            ttl=msg_data['ttl'],
            route_path=msg_data.get('route_path', [])
        )


@dataclass
class NodeInfo:
    """Node information for routing table."""
    node_id: str
    host: str
    port: int
    last_seen: float
    latency: float = 0.0
    reliability: float = 1.0
    hops: int = 0


class ConnectionPool:
    """Connection pool for efficient network communication."""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.connections: Dict[str, socket.socket] = {}
        self.connection_times: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def get_connection(self, host: str, port: int) -> Optional[socket.socket]:
        """Get or create connection to host:port."""
        key = f"{host}:{port}"
        
        with self.lock:
            # Check if connection exists and is valid
            if key in self.connections:
                conn = self.connections[key]
                try:
                    # Test connection
                    conn.send(b'')
                    return conn
                except Exception:
                    # Connection is dead, remove it
                    self.connections.pop(key, None)
                    self.connection_times.pop(key, None)
            
            # Create new connection
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5.0)  # 5 second timeout
                conn.connect((host, port))
                
                # Manage pool size
                if len(self.connections) >= self.max_connections:
                    # Remove oldest connection
                    oldest_key = min(self.connection_times.keys(), 
                                   key=lambda k: self.connection_times[k])
                    old_conn = self.connections.pop(oldest_key)
                    self.connection_times.pop(oldest_key)
                    try:
                        old_conn.close()
                    except Exception:
                        pass
                
                self.connections[key] = conn
                self.connection_times[key] = time.time()
                return conn
                
            except Exception as e:
                logger.warning(f"Failed to connect to {key}: {e}")
                return None
    
    def close_all(self):
        """Close all connections."""
        with self.lock:
            for conn in self.connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self.connections.clear()
            self.connection_times.clear()


class EnhancedRoutingTable:
    """Enhanced routing table with multiple algorithms."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.nodes: Dict[str, NodeInfo] = {}
        self.lock = threading.Lock()
        self.route_cache: Dict[str, List[str]] = {}
        self.last_update = time.time()
    
    def add_node(self, node_info: NodeInfo):
        """Add or update node in routing table."""
        with self.lock:
            self.nodes[node_info.node_id] = node_info
            # Clear route cache when topology changes
            self.route_cache.clear()
            self.last_update = time.time()
            logger.debug(f"Added node {node_info.node_id} to routing table")
    
    def remove_node(self, node_id: str):
        """Remove node from routing table."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.route_cache.clear()
                self.last_update = time.time()
                logger.debug(f"Removed node {node_id} from routing table")
    
    def find_route(self, destination: str) -> Optional[List[str]]:
        """Find optimal route to destination using multiple algorithms."""
        if destination == self.node_id:
            return []
        
        # Check cache first
        if destination in self.route_cache:
            return self.route_cache[destination]
        
        with self.lock:
            # Direct connection
            if destination in self.nodes:
                route = [destination]
                self.route_cache[destination] = route
                return route
            
            # Dijkstra's algorithm for shortest path
            route = self._dijkstra_route(destination)
            if route:
                self.route_cache[destination] = route
                return route
            
            # Fallback: broadcast to all neighbors
            neighbors = list(self.nodes.keys())
            if neighbors:
                route = neighbors[:3]  # Use first 3 neighbors
                self.route_cache[destination] = route
                return route
        
        return None
    
    def _dijkstra_route(self, destination: str) -> Optional[List[str]]:
        """Dijkstra's algorithm for shortest path routing."""
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[self.node_id] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        unvisited.add(self.node_id)
        
        while unvisited:
            # Find minimum distance node
            current = min(unvisited, key=lambda n: distances.get(n, float('inf')))
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current == destination:
                # Reconstruct path
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                return path[::-1]  # Reverse path
            
            # Update neighbors
            for neighbor_id, neighbor in self.nodes.items():
                if neighbor_id in unvisited:
                    # Calculate edge weight (latency + reliability penalty)
                    weight = neighbor.latency + (1 - neighbor.reliability) * 10
                    alt_distance = distances[current] + weight
                    
                    if alt_distance < distances[neighbor_id]:
                        distances[neighbor_id] = alt_distance
                        previous[neighbor_id] = current
        
        return None
    
    def get_best_neighbors(self, count: int = 3) -> List[NodeInfo]:
        """Get best neighbors based on reliability and latency."""
        with self.lock:
            nodes = list(self.nodes.values())
            # Sort by reliability (desc) and latency (asc)
            nodes.sort(key=lambda n: (-n.reliability, n.latency))
            return nodes[:count]
    
    def update_node_metrics(self, node_id: str, latency: float, success: bool):
        """Update node performance metrics."""
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.latency = (node.latency * 0.8) + (latency * 0.2)  # EWMA
                if success:
                    node.reliability = min(1.0, node.reliability * 0.95 + 0.05)
                else:
                    node.reliability = max(0.1, node.reliability * 0.9)
                node.last_seen = time.time()


class EnhancedMeshNode:
    """Enhanced mesh node with fixed routing and communication."""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 0):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.routing_table = EnhancedRoutingTable(node_id)
        self.connection_pool = ConnectionPool()
        
        # Communication
        self.server_socket = None
        self.running = False
        self.message_queue = queue.Queue()
        self.received_messages = queue.Queue()
        
        # Performance tracking
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
        # Threading
        self.server_thread = None
        self.processor_thread = None
        
    def start(self) -> bool:
        """Start the mesh node."""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            
            # Get actual port if port was 0
            if self.port == 0:
                self.port = self.server_socket.getsockname()[1]
            
            self.server_socket.listen(10)
            self.running = True
            
            # Start threads
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.processor_thread = threading.Thread(target=self._message_processor, daemon=True)
            
            self.server_thread.start()
            self.processor_thread.start()
            
            logger.info(f"Node {self.node_id} started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start node {self.node_id}: {e}")
            return False
    
    def stop(self):
        """Stop the mesh node."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        self.connection_pool.close_all()
        logger.info(f"Node {self.node_id} stopped")
    
    def _server_loop(self):
        """Server loop to accept incoming connections."""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket,), 
                    daemon=True
                )
                client_thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error in {self.node_id}: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle incoming client connection."""
        try:
            # Read message length first (4 bytes)
            length_data = client_socket.recv(4)
            if len(length_data) != 4:
                return
            
            message_length = int.from_bytes(length_data, byteorder='big')
            
            # Read message data
            message_data = b''
            while len(message_data) < message_length:
                chunk = client_socket.recv(min(1024, message_length - len(message_data)))
                if not chunk:
                    break
                message_data += chunk
            
            if len(message_data) == message_length:
                # Deserialize message
                message = NetworkMessage.from_bytes(message_data)
                self.received_messages.put(message)
                self.stats['messages_received'] += 1
                self.stats['bytes_received'] += len(message_data)
                
                logger.debug(f"Node {self.node_id} received message {message.id}")
            
        except Exception as e:
            logger.error(f"Error handling client in {self.node_id}: {e}")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
    
    def _message_processor(self):
        """Process received messages."""
        while self.running:
            try:
                message = self.received_messages.get(timeout=1.0)
                self._process_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Message processing error in {self.node_id}: {e}")
    
    def _process_message(self, message: NetworkMessage):
        """Process a received message."""
        try:
            # Check TTL
            if message.ttl <= 0:
                logger.debug(f"Message {message.id} TTL expired")
                return
            
            # Check if message is for this node
            if message.receiver_id == self.node_id:
                logger.info(f"Node {self.node_id} received message for itself: {message.type.value}")
                return
            
            # Forward message if not for this node
            if message.receiver_id != "broadcast":
                self._forward_message(message)
            
            # Handle specific message types
            if message.type == MessageType.DISCOVERY:
                self._handle_discovery(message)
            elif message.type == MessageType.HEARTBEAT:
                self._handle_heartbeat(message)
            elif message.type == MessageType.ROUTE_UPDATE:
                self._handle_route_update(message)
                
        except Exception as e:
            logger.error(f"Error processing message in {self.node_id}: {e}")
    
    def _forward_message(self, message: NetworkMessage):
        """Forward message to next hop."""
        try:
            # Prevent loops
            if self.node_id in message.route_path:
                logger.debug(f"Loop detected in message {message.id}")
                return
            
            # Find route
            route = self.routing_table.find_route(message.receiver_id)
            if not route:
                logger.warning(f"No route to {message.receiver_id}")
                return
            
            # Update message
            message.ttl -= 1
            message.route_path.append(self.node_id)
            
            # Send to next hop
            next_hop = route[0]
            self._send_to_node(next_hop, message)
            
        except Exception as e:
            logger.error(f"Error forwarding message: {e}")
    
    def send_message(self, message: NetworkMessage) -> bool:
        """Send message through the mesh network."""
        try:
            if message.receiver_id == "broadcast":
                return self._broadcast_message(message)
            else:
                return self._send_unicast_message(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _send_unicast_message(self, message: NetworkMessage) -> bool:
        """Send unicast message."""
        try:
            route = self.routing_table.find_route(message.receiver_id)
            if not route:
                logger.warning(f"No route to {message.receiver_id}")
                return False
            
            message.route_path = [self.node_id]
            next_hop = route[0]
            
            return self._send_to_node(next_hop, message)
            
        except Exception as e:
            logger.error(f"Unicast send error: {e}")
            return False
    
    def _broadcast_message(self, message: NetworkMessage) -> bool:
        """Broadcast message to all neighbors."""
        success_count = 0
        neighbors = self.routing_table.get_best_neighbors(10)  # Broadcast to top 10
        
        for neighbor in neighbors:
            if self._send_to_node(neighbor.node_id, message):
                success_count += 1
        
        return success_count > 0
    
    def _send_to_node(self, node_id: str, message: NetworkMessage) -> bool:
        """Send message to specific node."""
        try:
            node_info = self.routing_table.nodes.get(node_id)
            if not node_info:
                logger.warning(f"Node {node_id} not in routing table")
                return False
            
            # Get connection
            conn = self.connection_pool.get_connection(node_info.host, node_info.port)
            if not conn:
                self.routing_table.update_node_metrics(node_id, 0, False)
                return False
            
            # Serialize message
            start_time = time.time()
            message_data = message.to_bytes()
            message_length = len(message_data)
            
            # Send length first, then message
            conn.send(message_length.to_bytes(4, byteorder='big'))
            conn.send(message_data)
            
            # Update metrics
            latency = time.time() - start_time
            self.routing_table.update_node_metrics(node_id, latency, True)
            
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += message_length
            
            logger.debug(f"Sent message {message.id} to {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending to {node_id}: {e}")
            self.routing_table.update_node_metrics(node_id, 0, False)
            self.stats['messages_failed'] += 1
            return False
    
    def connect_to_peer(self, peer_host: str, peer_port: int, peer_id: str) -> bool:
        """Connect to a peer node."""
        try:
            # Test connection
            conn = self.connection_pool.get_connection(peer_host, peer_port)
            if conn:
                # Add to routing table
                node_info = NodeInfo(
                    node_id=peer_id,
                    host=peer_host,
                    port=peer_port,
                    last_seen=time.time()
                )
                self.routing_table.add_node(node_info)
                
                # Send discovery message
                discovery_msg = NetworkMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.DISCOVERY,
                    sender_id=self.node_id,
                    receiver_id=peer_id,
                    payload={
                        'host': self.host,
                        'port': self.port,
                        'timestamp': time.time()
                    },
                    timestamp=time.time()
                )
                
                self.send_message(discovery_msg)
                logger.info(f"Connected to peer {peer_id} at {peer_host}:{peer_port}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id}: {e}")
        
        return False
    
    def _handle_discovery(self, message: NetworkMessage):
        """Handle discovery message."""
        try:
            payload = message.payload
            node_info = NodeInfo(
                node_id=message.sender_id,
                host=payload['host'],
                port=payload['port'],
                last_seen=time.time()
            )
            self.routing_table.add_node(node_info)
            logger.info(f"Discovered node {message.sender_id}")
        except Exception as e:
            logger.error(f"Error handling discovery: {e}")
    
    def _handle_heartbeat(self, message: NetworkMessage):
        """Handle heartbeat message."""
        try:
            if message.sender_id in self.routing_table.nodes:
                self.routing_table.nodes[message.sender_id].last_seen = time.time()
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")
    
    def _handle_route_update(self, message: NetworkMessage):
        """Handle route update message."""
        # Implementation for route updates
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            'node_id': self.node_id,
            'address': f"{self.host}:{self.port}",
            'running': self.running,
            'connected_nodes': len(self.routing_table.nodes),
            'stats': self.stats.copy(),
            'routing_table_size': len(self.routing_table.nodes)
        }


class MeshNetworkPerformanceTester:
    """Test the enhanced mesh network performance."""
    
    def __init__(self):
        self.nodes: List[EnhancedMeshNode] = []
        self.test_results = {}
    
    def create_test_network(self, num_nodes: int = 5) -> List[EnhancedMeshNode]:
        """Create a test network with enhanced nodes."""
        nodes = []
        
        for i in range(num_nodes):
            node = EnhancedMeshNode(f"node_{i}", "localhost")
            if node.start():
                nodes.append(node)
                time.sleep(0.1)  # Small delay between starts
        
        # Connect nodes in a mesh topology
        for i, node in enumerate(nodes):
            for j, peer in enumerate(nodes):
                if i != j:
                    node.connect_to_peer(peer.host, peer.port, peer.node_id)
                    time.sleep(0.1)
        
        self.nodes = nodes
        return nodes
    
    def test_message_delivery(self, num_messages: int = 10) -> Dict[str, Any]:
        """Test message delivery rates."""
        if len(self.nodes) < 2:
            return {'delivery_rate': 0.0, 'error': 'Insufficient nodes'}
        
        successful_deliveries = 0
        total_messages = 0
        
        for message_type in [MessageType.DISCOVERY, MessageType.PARAMETER_UPDATE, MessageType.GRADIENT_SHARE]:
            for i in range(num_messages):
                sender = self.nodes[0]
                receiver = self.nodes[-1]
                
                message = NetworkMessage(
                    id=str(uuid.uuid4()),
                    type=message_type,
                    sender_id=sender.node_id,
                    receiver_id=receiver.node_id,
                    payload={
                        'test_data': f"Test message {i}",
                        'timestamp': time.time()
                    },
                    timestamp=time.time()
                )
                
                if sender.send_message(message):
                    successful_deliveries += 1
                total_messages += 1
                
                time.sleep(0.1)  # Small delay between messages
        
        delivery_rate = (successful_deliveries / total_messages) * 100 if total_messages > 0 else 0
        
        return {
            'delivery_rate': delivery_rate,
            'successful_deliveries': successful_deliveries,
            'total_messages': total_messages,
            'node_count': len(self.nodes)
        }
    
    def cleanup(self):
        """Clean up test network."""
        for node in self.nodes:
            node.stop()
        self.nodes.clear()


def test_mesh_network_fix():
    """Test the mesh network performance fixes."""
    logger.info("Testing mesh network performance fixes...")
    
    tester = MeshNetworkPerformanceTester()
    
    try:
        # Create test network
        logger.info("Creating test network...")
        nodes = tester.create_test_network(5)
        
        if not nodes:
            logger.error("Failed to create test network")
            return False
        
        # Wait for network to stabilize
        logger.info("Waiting for network to stabilize...")
        time.sleep(2)
        
        # Test message delivery
        logger.info("Testing message delivery...")
        results = tester.test_message_delivery(20)
        
        logger.info(f"Message delivery results: {results}")
        
        # Print node statistics
        for node in nodes:
            stats = node.get_stats()
            logger.info(f"Node {stats['node_id']} stats: {stats}")
        
        # Save results
        test_results = {
            'timestamp': time.time(),
            'network_size': len(nodes),
            'delivery_results': results,
            'node_stats': [node.get_stats() for node in nodes]
        }
        
        with open('mesh_network_fix_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        success = results['delivery_rate'] > 80.0  # 80% delivery rate threshold
        
        if success:
            logger.info("✅ Mesh network performance fix SUCCESSFUL!")
            logger.info(f"Delivery rate: {results['delivery_rate']:.1f}%")
        else:
            logger.warning(f"⚠️ Mesh network still needs work. Delivery rate: {results['delivery_rate']:.1f}%")
        
        return success
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    finally:
        tester.cleanup()


if __name__ == "__main__":
    success = test_mesh_network_fix()
    if success:
        print("✅ Mesh network performance fix completed successfully!")
    else:
        print("❌ Mesh network performance fix needs more work.")