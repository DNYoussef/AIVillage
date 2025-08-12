"""Compatibility layer for production P2P components.

This module exposes lightweight message and peer data structures while
reusing the core :class:`P2PNode` implementation.  The adapter keeps a
subset of the legacy API to avoid widespread changes in production code.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.core.p2p.p2p_node import P2PNode as CoreP2PNode, NodeStatus


class MessageType(Enum):
    """Simplified message types for compatibility."""

    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    DATA = "data"
    TENSOR_CHUNK = "tensor_chunk"
    SYNC_REQUEST = "sync_request"
    DISCOVERY = "discovery"
    ERROR = "error"


@dataclass
class P2PMessage:
    """Lightweight P2P message structure."""

    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class PeerInfo:
    """Information about a peer node."""

    peer_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    status: NodeStatus = NodeStatus.DISCONNECTED
    capabilities: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


class P2PNode(CoreP2PNode):
    """Adapter around the core :class:`P2PNode` implementation.

    The adapter preserves a small subset of the previous production API
    by keeping attributes like ``peers`` and ``known_addresses`` while
    delegating all networking responsibilities to the core class.
    """

    def __init__(self, node_id: str | None = None, port: int = 0, **kwargs: Any) -> None:
        super().__init__(node_id=node_id, listen_port=port, **kwargs)
        self.port = self.listen_port
        self.peers: dict[str, PeerInfo] = {}
        self.known_addresses: set[str] = set()
        self.message_handlers: dict[MessageType, Callable[[P2PMessage], Any]] = {}
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connections_established": 0,
            "discovery_rounds": 0,
            "start_time": time.time(),
        }

    async def start(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - thin wrapper
        await super().start(*args, **kwargs)
        self.status = NodeStatus.ACTIVE

    async def stop(self) -> None:  # pragma: no cover - thin wrapper
        await super().stop()
        self.status = NodeStatus.DISCONNECTED

    def add_known_address(self, address: str, port: int) -> None:
        """Record a peer address for later discovery."""
        self.known_addresses.add(f"{address}:{port}")

    def register_handler(self, message_type: MessageType, handler: Callable[[P2PMessage], Any]) -> None:
        """Register a message handler for a given type."""
        self.message_handlers[message_type] = handler

    def get_peer_info(self, peer_id: str) -> PeerInfo | None:
        """Return information about a specific peer."""
        return self.peers.get(peer_id)

    def get_connected_peers(self) -> list[PeerInfo]:
        """Return peers currently marked as active."""
        return [p for p in self.peers.values() if p.status == NodeStatus.ACTIVE]

    def get_stats(self) -> dict[str, Any]:
        """Return basic node statistics."""
        return {
            **self.stats,
            "node_id": self.node_id,
            "status": self.status.value,
            "connected_peers": len(self.get_connected_peers()),
            "total_peers": len(self.peers),
            "uptime": time.time() - self.stats["start_time"],
        }
