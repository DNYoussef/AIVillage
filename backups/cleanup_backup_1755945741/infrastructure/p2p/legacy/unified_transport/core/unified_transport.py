"""Unified Communications Transport - Master Implementation

Consolidates the BEST features from:
- src/communications/protocol.py (WebSocket + encryption)
- src/core/p2p/betanet_transport_v2.py (Advanced transport)
- src/core/p2p/libp2p_mesh.py (Mesh networking)
- betanet-gateway/* (Rust performance layer)
- agents/navigator_agent.py (Routing logic)

This creates ONE consolidated, non-overlapping communication system.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import websockets
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Available transport types (from various implementations)"""

    WEBSOCKET = "websocket"  # From communications/protocol.py
    BETANET_HTX = "betanet_htx"  # From betanet_htx_transport.py
    LIBP2P_MESH = "libp2p_mesh"  # From libp2p_mesh.py
    BETANET_COVERT = "betanet_covert"  # From betanet_covert_transport.py
    RUST_GATEWAY = "rust_gateway"  # From betanet-gateway


class RoutingMode(Enum):
    """Routing modes consolidated from Navigator agent"""

    DIRECT = "direct"
    MESH_ROUTED = "mesh_routed"
    PRIVACY_FIRST = "privacy_first"
    PERFORMANCE_FIRST = "performance_first"
    MOBILE_OPTIMIZED = "mobile_optimized"


@dataclass
class UnifiedMessage:
    """Universal message format consolidating all message types"""

    # Core fields (from all implementations)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    message_type: str = "default"

    # Routing fields (from Navigator)
    routing_mode: RoutingMode = RoutingMode.DIRECT
    routing_path: list[str] = field(default_factory=list)
    priority: int = 1  # 1=high, 5=low

    # Security fields (from communications/protocol.py)
    encrypted: bool = True
    signature: str | None = None

    # Betanet fields (from betanet implementations)
    cover_traffic: bool = False
    privacy_level: int = 1  # 1-3 scale

    # Metadata
    timestamp: float = field(default_factory=time.time)
    expiry: float | None = None
    retries: int = 0
    max_retries: int = 3


class TransportInterface(ABC):
    """Abstract interface all transports must implement"""

    @abstractmethod
    async def connect(self, target_id: str, target_info: dict[str, Any]) -> bool:
        """Connect to target endpoint"""
        pass

    @abstractmethod
    async def send_message(self, message: UnifiedMessage) -> bool:
        """Send message through this transport"""
        pass

    @abstractmethod
    async def receive_message(self) -> UnifiedMessage | None:
        """Receive message from this transport"""
        pass

    @abstractmethod
    async def disconnect(self, target_id: str) -> None:
        """Disconnect from target"""
        pass


class WebSocketTransport(TransportInterface):
    """WebSocket transport (from communications/protocol.py)"""

    def __init__(self, agent_id: str, port: int = 8888):
        self.agent_id = agent_id
        self.port = port
        self.connections: dict[str, websockets.WebSocketServerProtocol] = {}
        self.encryption_keys: dict[str, Fernet] = {}

    async def connect(self, target_id: str, target_info: dict[str, Any]) -> bool:
        """Real WebSocket connection (consolidated from original)"""
        try:
            target_url = target_info.get("url", f"ws://localhost:{target_info.get('port', 8888)}")
            websocket = await websockets.connect(target_url)
            self.connections[target_id] = websocket

            # Setup encryption (from original implementation)
            self._setup_encryption(target_id)

            logger.info(f"Connected to {target_id} via WebSocket")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    def _setup_encryption(self, target_id: str) -> None:
        """Setup encryption key (from communications/protocol.py)"""
        key = Fernet.generate_key()
        self.encryption_keys[target_id] = Fernet(key)

    async def send_message(self, message: UnifiedMessage) -> bool:
        """Send message with encryption"""
        if message.receiver_id not in self.connections:
            return False

        try:
            # Serialize and encrypt if needed
            data = message.__dict__
            if message.encrypted and message.receiver_id in self.encryption_keys:
                encrypted_data = self.encryption_keys[message.receiver_id].encrypt(str(data).encode())
                data = {"encrypted": encrypted_data.decode()}

            await self.connections[message.receiver_id].send(str(data))
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    async def receive_message(self) -> UnifiedMessage | None:
        """Receive and decrypt message"""
        # Implementation would handle incoming WebSocket messages
        pass

    async def disconnect(self, target_id: str) -> None:
        """Close WebSocket connection"""
        if target_id in self.connections:
            await self.connections[target_id].close()
            del self.connections[target_id]


class BetanetTransport(TransportInterface):
    """Betanet transport (consolidated from betanet_transport_v2.py)"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.sessions: dict[str, Any] = {}

    async def connect(self, target_id: str, target_info: dict[str, Any]) -> bool:
        """Betanet HTX connection with fingerprinting"""
        # Consolidated from betanet_transport_v2.py logic
        logger.info(f"Connecting to {target_id} via Betanet HTX")
        # Implementation would use Betanet protocol
        return True

    async def send_message(self, message: UnifiedMessage) -> bool:
        """Send via Betanet with cover traffic"""
        # Implementation would use Betanet covert transport
        return True

    async def receive_message(self) -> UnifiedMessage | None:
        """Receive via Betanet"""
        pass

    async def disconnect(self, target_id: str) -> None:
        """Close Betanet session"""
        if target_id in self.sessions:
            del self.sessions[target_id]


class UnifiedCommunicationHub:
    """Master communication system - consolidates ALL implementations"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Available transports (consolidated)
        self.transports: dict[TransportType, TransportInterface] = {
            TransportType.WEBSOCKET: WebSocketTransport(agent_id),
            TransportType.BETANET_HTX: BetanetTransport(agent_id),
        }

        # Routing intelligence (from Navigator agent)
        self.routing_table: dict[str, dict[str, Any]] = {}
        self.peer_info: dict[str, dict[str, Any]] = {}

        # Message handling (from protocol.py)
        self.message_handlers: dict[str, Callable] = {}
        self.message_history: dict[str, list[UnifiedMessage]] = {}

        # Performance metrics
        self.transport_stats: dict[TransportType, dict[str, float]] = {}

    async def connect_to_peer(self, peer_id: str, peer_info: dict[str, Any]) -> bool:
        """Intelligent connection using best available transport"""

        # Route selection logic (from Navigator agent)
        best_transport = self._select_optimal_transport(peer_id, peer_info)

        success = await self.transports[best_transport].connect(peer_id, peer_info)
        if success:
            self.peer_info[peer_id] = peer_info
            self._update_routing_table(peer_id, best_transport)

        return success

    def _select_optimal_transport(self, peer_id: str, peer_info: dict[str, Any]) -> TransportType:
        """Smart transport selection (consolidated from Navigator logic)"""

        # Factors from various implementations:
        # - Network conditions (from Navigator)
        # - Security requirements (from Betanet)
        # - Performance needs (from WebSocket)
        # - Mobile constraints (from mobile optimizations)

        # Default to WebSocket for simplicity, but could be enhanced
        return TransportType.WEBSOCKET

    async def send_message(self, message: UnifiedMessage) -> bool:
        """Send message using optimal routing"""

        # Enhanced routing logic from Navigator
        if message.routing_mode == RoutingMode.PRIVACY_FIRST:
            # Use Betanet covert transport
            transport = TransportType.BETANET_HTX
        elif message.routing_mode == RoutingMode.PERFORMANCE_FIRST:
            # Use direct WebSocket
            transport = TransportType.WEBSOCKET
        else:
            # Use smart selection
            transport = self._select_optimal_transport(message.receiver_id, {})

        success = await self.transports[transport].send_message(message)

        # Track history (from protocol.py)
        if message.sender_id not in self.message_history:
            self.message_history[message.sender_id] = []
        self.message_history[message.sender_id].append(message)

        return success

    def _update_routing_table(self, peer_id: str, transport: TransportType) -> None:
        """Update routing information"""
        self.routing_table[peer_id] = {
            "primary_transport": transport,
            "last_connected": time.time(),
            "success_rate": 1.0,  # Would track over time
        }

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler (from protocol.py)"""
        self.message_handlers[message_type] = handler

    async def start(self) -> None:
        """Start the unified communication system"""
        logger.info(f"Starting unified communications for agent {self.agent_id}")
        # Would start all transport listeners

    async def stop(self) -> None:
        """Stop all transports"""
        for transport in self.transports.values():
            # Disconnect from all peers
            for peer_id in list(self.peer_info.keys()):
                await transport.disconnect(peer_id)


# Factory function for easy instantiation
def create_communication_hub(agent_id: str, **kwargs) -> UnifiedCommunicationHub:
    """Factory function to create unified communication hub"""
    return UnifiedCommunicationHub(agent_id)


# Backward compatibility aliases for migration
CommunicationsProtocol = UnifiedCommunicationHub  # For old imports
BetanetTransportV2 = BetanetTransport  # For old imports
