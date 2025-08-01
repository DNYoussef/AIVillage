"""P2P Node Implementation with Fallback for Windows Development."""

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
import hashlib
import uuid

# Use local encryption instead of libp2p for Windows compatibility
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node connection status."""
    OFFLINE = "offline"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class PeerInfo:
    """Information about a peer node."""
    peer_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    

class MessageType(Enum):
    """P2P message types."""
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    DATA = "data" 
    TENSOR_CHUNK = "tensor_chunk"
    SYNC_REQUEST = "sync_request"
    DISCOVERY = "discovery"
    ERROR = "error"


@dataclass
class P2PMessage:
    """P2P message structure."""
    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class P2PNode:
    """P2P Node for decentralized communication with Windows fallback."""
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        port: int = 8000,
        encryption_key: Optional[bytes] = None,
    ):
        self.node_id = node_id or str(uuid.uuid4())
        self.port = port
        self.status = NodeStatus.OFFLINE
        
        # Initialize encryption
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}
        self.known_addresses: Set[str] = set()
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Network components
        self.server: Optional[asyncio.Server] = None
        self.discovery_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connections_established": 0,
            "discovery_rounds": 0,
        }
        
        # Configuration
        self.config = {
            "heartbeat_interval": 30.0,  # seconds
            "discovery_interval": 60.0,  # seconds
            "connection_timeout": 10.0,  # seconds
            "max_message_size": 1024 * 1024,  # 1MB
            "max_peers": 100,
        }
        
        # Register default handlers
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.message_handlers[MessageType.HANDSHAKE] = self._handle_handshake
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.DISCOVERY] = self._handle_discovery
        
    async def start(self) -> None:
        """Start the P2P node."""
        try:
            logger.info(f"Starting P2P node {self.node_id} on port {self.port}")
            
            # Start TCP server
            self.server = await asyncio.start_server(
                self._handle_connection,
                '0.0.0.0',
                self.port
            )
            
            # Start background tasks
            self.discovery_task = asyncio.create_task(self._discovery_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.status = NodeStatus.CONNECTED
            logger.info(f"P2P node {self.node_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start P2P node: {e}")
            self.status = NodeStatus.ERROR
            raise
            
    async def stop(self) -> None:
        """Stop the P2P node."""
        logger.info(f"Stopping P2P node {self.node_id}")
        
        self.status = NodeStatus.OFFLINE
        
        # Cancel background tasks
        if self.discovery_task:
            self.discovery_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Clear peers
        self.peers.clear()
        
        logger.info(f"P2P node {self.node_id} stopped")
        
    async def connect_to_peer(self, address: str, port: int) -> bool:
        """Connect to a peer node."""
        try:
            logger.debug(f"Connecting to peer at {address}:{port}")
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=self.config["connection_timeout"]
            )
            
            # Send handshake
            handshake_msg = P2PMessage(
                message_type=MessageType.HANDSHAKE,
                sender_id=self.node_id,
                receiver_id="",
                payload={
                    "node_id": self.node_id,
                    "capabilities": self._get_capabilities(),
                    "timestamp": time.time(),
                }
            )
            
            await self._send_message_to_writer(writer, handshake_msg)
            
            # Wait for response
            response = await self._receive_message_from_reader(reader)
            
            if response and response.message_type == MessageType.HANDSHAKE:
                peer_id = response.payload.get("node_id")
                if peer_id:
                    peer_info = PeerInfo(
                        peer_id=peer_id,
                        address=address,
                        port=port,
                        status=NodeStatus.CONNECTED,
                        capabilities=response.payload.get("capabilities", {})
                    )
                    self.peers[peer_id] = peer_info
                    self.stats["connections_established"] += 1
                    
                    logger.info(f"Successfully connected to peer {peer_id}")
                    writer.close()
                    await writer.wait_closed()
                    return True
                    
            writer.close()
            await writer.wait_closed()
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {address}:{port}: {e}")
            return False
            
    async def send_message(
        self,
        peer_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> bool:
        """Send a message to a specific peer."""
        if peer_id not in self.peers:
            logger.warning(f"Peer {peer_id} not found")
            return False
            
        peer = self.peers[peer_id]
        if peer.status != NodeStatus.CONNECTED:
            logger.warning(f"Peer {peer_id} not connected")
            return False
            
        try:
            message = P2PMessage(
                message_type=message_type,
                sender_id=self.node_id,
                receiver_id=peer_id,
                payload=payload
            )
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.address, peer.port),
                timeout=self.config["connection_timeout"]
            )
            
            await self._send_message_to_writer(writer, message)
            self.stats["messages_sent"] += 1
            
            writer.close()
            await writer.wait_closed()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to peer {peer_id}: {e}")
            peer.status = NodeStatus.ERROR
            return False
            
    async def broadcast_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        exclude_peers: Optional[Set[str]] = None
    ) -> int:
        """Broadcast a message to all connected peers."""
        exclude_peers = exclude_peers or set()
        successful_sends = 0
        
        for peer_id in self.peers:
            if peer_id not in exclude_peers:
                if await self.send_message(peer_id, message_type, payload):
                    successful_sends += 1
                    
        return successful_sends
        
    async def query_peer(
        self,
        peer_id: str,
        query_type: str,
        query_data: Dict[str, Any],
        timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """Send a query to a peer and wait for response."""
        message_id = str(uuid.uuid4())
        
        # Set up response future
        response_future = asyncio.Future()
        self.pending_responses[message_id] = response_future
        
        try:
            # Send query
            payload = {
                "query_type": query_type,
                "query_data": query_data,
                "message_id": message_id,
                "expect_response": True,
            }
            
            success = await self.send_message(peer_id, MessageType.DATA, payload)
            
            if not success:
                return None
                
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Query to peer {peer_id} timed out")
            return None
        except Exception as e:
            logger.error(f"Query to peer {peer_id} failed: {e}")
            return None
        finally:
            self.pending_responses.pop(message_id, None)
            
    def add_known_address(self, address: str, port: int) -> None:
        """Add a known peer address for discovery."""
        self.known_addresses.add(f"{address}:{port}")
        
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[P2PMessage], Any]
    ) -> None:
        """Register a custom message handler."""
        self.message_handlers[message_type] = handler
        
    def get_peer_info(self, peer_id: str) -> Optional[PeerInfo]:
        """Get information about a specific peer."""
        return self.peers.get(peer_id)
        
    def get_connected_peers(self) -> List[PeerInfo]:
        """Get list of all connected peers."""
        return [
            peer for peer in self.peers.values()
            if peer.status == NodeStatus.CONNECTED
        ]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            **self.stats,
            "node_id": self.node_id,
            "status": self.status.value,
            "connected_peers": len(self.get_connected_peers()),
            "total_peers": len(self.peers),
            "uptime": time.time() - self.stats.get("start_time", time.time()),
        }
        
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming connection."""
        peer_address = writer.get_extra_info('peername')
        logger.debug(f"Incoming connection from {peer_address}")
        
        try:
            message = await self._receive_message_from_reader(reader)
            
            if message:
                await self._process_message(message, writer)
                
        except Exception as e:
            logger.error(f"Error handling connection from {peer_address}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            
    async def _send_message_to_writer(
        self,
        writer: asyncio.StreamWriter,
        message: P2PMessage
    ) -> None:
        """Send a message through a writer."""
        # Serialize and encrypt message
        message_data = json.dumps({
            "message_type": message.message_type.value,
            "sender_id": message.sender_id,
            "receiver_id": message.receiver_id,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "message_id": message.message_id,
        }).encode()
        
        encrypted_data = self.cipher.encrypt(message_data)
        
        # Send length prefix + encrypted data
        length = len(encrypted_data)
        writer.write(length.to_bytes(4, byteorder='big'))
        writer.write(encrypted_data)
        await writer.drain()
        
        self.stats["bytes_sent"] += 4 + length
        
    async def _receive_message_from_reader(
        self,
        reader: asyncio.StreamReader
    ) -> Optional[P2PMessage]:
        """Receive a message from a reader."""
        try:
            # Read length prefix
            length_data = await reader.readexactly(4)
            length = int.from_bytes(length_data, byteorder='big')
            
            if length > self.config["max_message_size"]:
                logger.warning(f"Message too large: {length} bytes")
                return None
                
            # Read encrypted message
            encrypted_data = await reader.readexactly(length)
            
            # Decrypt and deserialize
            message_data = self.cipher.decrypt(encrypted_data)
            message_dict = json.loads(message_data.decode())
            
            self.stats["bytes_received"] += 4 + length
            self.stats["messages_received"] += 1
            
            return P2PMessage(
                message_type=MessageType(message_dict["message_type"]),
                sender_id=message_dict["sender_id"],
                receiver_id=message_dict["receiver_id"],
                payload=message_dict["payload"],
                timestamp=message_dict["timestamp"],
                message_id=message_dict["message_id"],
            )
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
            
    async def _process_message(
        self,
        message: P2PMessage,
        writer: Optional[asyncio.StreamWriter] = None
    ) -> None:
        """Process an incoming message."""
        logger.debug(f"Processing {message.message_type.value} from {message.sender_id}")
        
        # Handle response to pending query
        if message.payload.get("response_to"):
            response_id = message.payload["response_to"]
            if response_id in self.pending_responses:
                self.pending_responses[response_id].set_result(message.payload)
                return
                
        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message, writer)
            except Exception as e:
                logger.error(f"Error in handler for {message.message_type.value}: {e}")
        else:
            logger.warning(f"No handler for message type {message.message_type.value}")
            
    async def _handle_handshake(
        self,
        message: P2PMessage,
        writer: Optional[asyncio.StreamWriter] = None
    ) -> None:
        """Handle handshake message."""
        sender_id = message.payload.get("node_id")
        
        if sender_id and sender_id != self.node_id:
            # Create peer info
            peer = PeerInfo(
                peer_id=sender_id,
                address="",  # Will be updated
                port=0,
                status=NodeStatus.CONNECTED,
                capabilities=message.payload.get("capabilities", {})
            )
            self.peers[sender_id] = peer
            
            # Send handshake response
            if writer:
                response = P2PMessage(
                    message_type=MessageType.HANDSHAKE,
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    payload={
                        "node_id": self.node_id,
                        "capabilities": self._get_capabilities(),
                        "timestamp": time.time(),
                    }
                )
                await self._send_message_to_writer(writer, response)
                
            logger.info(f"Handshake completed with peer {sender_id}")
            
    async def _handle_heartbeat(
        self,
        message: P2PMessage,
        writer: Optional[asyncio.StreamWriter] = None
    ) -> None:
        """Handle heartbeat message."""
        sender_id = message.sender_id
        
        if sender_id in self.peers:
            self.peers[sender_id].last_seen = time.time()
            self.peers[sender_id].status = NodeStatus.CONNECTED
            
    async def _handle_discovery(
        self,
        message: P2PMessage,
        writer: Optional[asyncio.StreamWriter] = None
    ) -> None:
        """Handle peer discovery message."""
        peer_list = message.payload.get("peers", [])
        
        for peer_info in peer_list:
            address = peer_info.get("address")
            port = peer_info.get("port")
            
            if address and port:
                self.add_known_address(address, port)
                
    async def _discovery_loop(self) -> None:
        """Background task for peer discovery."""
        while self.status == NodeStatus.CONNECTED:
            try:
                # Try to connect to known addresses
                for addr_port in list(self.known_addresses):
                    if len(self.peers) >= self.config["max_peers"]:
                        break
                        
                    address, port = addr_port.split(":")
                    port = int(port)
                    
                    # Skip if already connected
                    if any(p.address == address and p.port == port 
                          for p in self.peers.values()):
                        continue
                        
                    await self.connect_to_peer(address, port)
                    
                self.stats["discovery_rounds"] += 1
                await asyncio.sleep(self.config["discovery_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(10)  # Back off on error
                
    async def _heartbeat_loop(self) -> None:
        """Background task for sending heartbeats."""
        while self.status == NodeStatus.CONNECTED:
            try:
                # Send heartbeat to all connected peers
                heartbeat_payload = {
                    "timestamp": time.time(),
                    "status": self.status.value,
                }
                
                await self.broadcast_message(MessageType.HEARTBEAT, heartbeat_payload)
                
                # Check for stale peers
                current_time = time.time()
                stale_peers = []
                
                for peer_id, peer in self.peers.items():
                    if current_time - peer.last_seen > self.config["heartbeat_interval"] * 3:
                        stale_peers.append(peer_id)
                        
                # Remove stale peers
                for peer_id in stale_peers:
                    del self.peers[peer_id]
                    logger.info(f"Removed stale peer {peer_id}")
                    
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(10)
                
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get node capabilities."""
        return {
            "version": "1.0.0",
            "protocols": ["tensor_streaming", "mesh_routing"],
            "max_message_size": self.config["max_message_size"],
            "encryption": "fernet",
        }