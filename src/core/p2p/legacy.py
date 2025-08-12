"""Legacy adapter around the core P2P node."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Any, Callable

from .p2p_node import P2PNode as CoreP2PNode


class NodeStatus(Enum):
    """Legacy node connection status."""

    OFFLINE = "offline"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MessageType(Enum):
    """Legacy message types for P2P communication."""

    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    DATA = "data"
    TENSOR_CHUNK = "tensor_chunk"
    SYNC_REQUEST = "sync_request"
    DISCOVERY = "discovery"
    ERROR = "error"


@dataclass
class P2PMessage:
    """Legacy P2P message structure."""

    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class PeerInfo:
    """Legacy peer information."""

    peer_id: str
    address: str = ""
    port: int = 0
    last_seen: float = field(default_factory=time.time)
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


class P2PNode(CoreP2PNode):
    """Adapter providing legacy P2PNode behaviour on top of the core implementation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._legacy_handlers: dict[str, Callable[[P2PMessage, asyncio.StreamWriter | None], Any]] = {}
        self.known_addresses: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------------------
    def add_known_address(self, address: str, port: int) -> None:
        """Manually add a peer address for discovery."""
        self.peer_discovery.add_known_peer(address, port)
        self.known_addresses.add((address, port))

    async def connect_to_peer(self, address: str, port: int) -> bool:
        """Legacy connect which hints discovery to contact the peer."""
        self.add_known_address(address, port)
        return True

    async def send_message(self, peer_id: str, message_type: MessageType, payload: dict[str, Any]) -> bool | None:
        """Send a legacy formatted message to a peer."""
        message = {"type": message_type.value, **payload}
        return await self.send_to_peer(peer_id, message)

    def register_handler(
        self, message_type: MessageType, handler: Callable[[P2PMessage, asyncio.StreamWriter | None], Any]
    ) -> None:
        """Register a handler for a legacy message type."""
        self._legacy_handlers[message_type.value] = handler

    @property
    def peers(self) -> dict[str, PeerInfo]:
        """Expose peers in legacy PeerInfo format."""
        peer_map: dict[str, PeerInfo] = {}
        for peer_id, capabilities in self.peer_registry.items():
            status = NodeStatus.CONNECTED if peer_id in self.connections else NodeStatus.OFFLINE
            peer_map[peer_id] = PeerInfo(
                peer_id=peer_id,
                status=status,
                capabilities=capabilities.__dict__,
                last_seen=capabilities.last_seen,
                latency_ms=capabilities.latency_ms,
            )
        return peer_map

    def get_connected_peers(self) -> list[PeerInfo]:
        """Return connected peers using legacy structures."""
        return [peer for peer in self.peers.values() if peer.status == NodeStatus.CONNECTED]

    def get_stats(self) -> dict[str, Any]:
        """Provide basic statistics matching legacy expectations."""
        return {
            "messages_sent": self.network_stats["messages_sent"],
            "messages_received": self.network_stats["messages_received"],
            "bytes_sent": self.network_stats["bytes_sent"],
            "bytes_received": self.network_stats["bytes_received"],
            "connected_peers": len(self.connections),
        }

    async def _handle_message(self, message: dict, writer: asyncio.StreamWriter) -> None:  # type: ignore[override]
        msg_type = message.get("type", "UNKNOWN")
        if msg_type in self._legacy_handlers:
            legacy_msg = P2PMessage(
                message_type=MessageType(msg_type),
                sender_id=message.get("sender_id", ""),
                receiver_id=message.get("recipient_id", ""),
                payload=message,
                timestamp=message.get("timestamp", time.time()),
                message_id=message.get("message_id", str(uuid.uuid4())),
            )
            handler = self._legacy_handlers[msg_type]
            if asyncio.iscoroutinefunction(handler):
                await handler(legacy_msg, writer)
            else:
                handler(legacy_msg, writer)
        else:
            await super()._handle_message(message, writer)


__all__ = ["P2PNode", "NodeStatus", "MessageType", "P2PMessage", "PeerInfo"]
