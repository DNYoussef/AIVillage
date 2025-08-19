"""
P2P Mesh Integration for Global South Offline Coordinator.

This module integrates with the existing AIVillage P2P mesh network system
to provide Global South specific capabilities like offline-first operation,
store-and-forward messaging, and collaborative caching using BitChat and
BetaNet transports.

Integrates with:
- packages.p2p.core.transport_manager.TransportManager
- packages.p2p.bitchat.ble_transport.BitChatTransport
- packages.p2p.betanet.htx_transport.BetaNetTransport
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Import existing P2P infrastructure
try:
    from packages.p2p.bitchat.ble_transport import BitChatTransport, create_bitchat_transport
    from packages.p2p.core.message_types import MessagePriority, MessageType, UnifiedMessage
    from packages.p2p.core.transport_manager import (
        TransportCapabilities,
        TransportManager,
        TransportPriority,
        TransportType,
    )

    P2P_AVAILABLE = True
except ImportError:
    # Fallback if P2P system not available
    P2P_AVAILABLE = False

logger = logging.getLogger(__name__)


class PeerType(Enum):
    """Types of peers in the mesh network."""

    MOBILE = "mobile"  # Mobile device
    DESKTOP = "desktop"  # Desktop computer
    SERVER = "server"  # Local server
    GATEWAY = "gateway"  # Internet gateway
    RELAY = "relay"  # Message relay node


class ConnectionType(Enum):
    """Types of peer connections."""

    BLUETOOTH = "bluetooth"  # Bluetooth connection
    WIFI_DIRECT = "wifi_direct"  # WiFi Direct connection
    WIFI_HOTSPOT = "wifi_hotspot"  # WiFi hotspot connection
    ETHERNET = "ethernet"  # Ethernet connection
    USB = "usb"  # USB tethering


# Use MessageType from existing P2P infrastructure
# Additional mesh-specific message types if needed
class MeshMessageType(Enum):
    """Additional mesh-specific message types."""

    REQUEST = "request"  # Content request
    RESPONSE = "response"  # Content response
    ANNOUNCE = "announce"  # Service announcement
    GOSSIP = "gossip"  # Gossip protocol message


@dataclass
class PeerInfo:
    """Information about a mesh network peer."""

    peer_id: str
    peer_type: PeerType
    connection_type: ConnectionType
    address: str  # IP address or Bluetooth MAC
    port: int
    capabilities: set[str] = field(default_factory=set)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    is_online: bool = True
    battery_level: float | None = None
    storage_available_mb: float | None = None
    bandwidth_estimate_mbps: float | None = None
    reliability_score: float = 0.5  # 0.0-1.0

    def __post_init__(self):
        if isinstance(self.capabilities, list):
            self.capabilities = set(self.capabilities)

    def update_last_seen(self):
        """Update last seen timestamp."""
        self.last_seen = datetime.utcnow()
        self.is_online = True

    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if peer information is stale."""
        return (datetime.utcnow() - self.last_seen).total_seconds() > timeout_seconds


@dataclass
class MeshMessage:
    """Message in the mesh network."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str | None  # None for broadcast
    content: bytes
    timestamp: datetime
    ttl: int = 10  # Time to live (hop count)
    priority: int = 5  # 1-10, higher = more important
    compression_used: bool = False
    signature: bytes | None = None

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        header = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "priority": self.priority,
            "compression_used": self.compression_used,
            "content_length": len(self.content),
        }

        header_json = json.dumps(header).encode("utf-8")
        header_length = len(header_json)

        # Format: [header_length:4][header:N][content:M]
        return struct.pack("!I", header_length) + header_json + self.content

    @classmethod
    def from_bytes(cls, data: bytes) -> "MeshMessage":
        """Deserialize message from bytes."""
        if len(data) < 4:
            raise ValueError("Invalid message data")

        header_length = struct.unpack("!I", data[:4])[0]
        if len(data) < 4 + header_length:
            raise ValueError("Incomplete message data")

        header_json = data[4 : 4 + header_length]
        content = data[4 + header_length :]

        header = json.loads(header_json.decode("utf-8"))

        return cls(
            message_id=header["message_id"],
            message_type=MessageType(header["message_type"]),
            sender_id=header["sender_id"],
            recipient_id=header.get("recipient_id"),
            content=content,
            timestamp=datetime.fromisoformat(header["timestamp"]),
            ttl=header["ttl"],
            priority=header["priority"],
            compression_used=header["compression_used"],
        )


class P2PMeshIntegration:
    """
    P2P mesh integration for offline-first Global South scenarios.

    Integrates with existing AIVillage P2P infrastructure to provide Global South
    specific capabilities like offline-first operation, store-and-forward messaging,
    and collaborative caching using BitChat and BetaNet transports.
    """

    def __init__(
        self,
        device_id: str = None,
        peer_type: PeerType = PeerType.MOBILE,
        offline_coordinator=None,
        transport_priority: TransportPriority = TransportPriority.OFFLINE_FIRST,
    ):
        """Initialize P2P mesh integration with existing transport manager."""
        self.device_id = device_id or self._generate_device_id()
        self.peer_type = peer_type
        self.offline_coordinator = offline_coordinator

        # Initialize transport manager if P2P is available
        self.transport_manager: TransportManager | None = None
        self.bitchat_transport: BitChatTransport | None = None

        if P2P_AVAILABLE:
            # Create transport manager with Global South optimizations
            self.transport_manager = TransportManager(
                device_id=self.device_id,
                transport_priority=transport_priority,
                max_chunk_size=1024,  # Smaller chunks for Global South bandwidth constraints
                chunk_timeout_seconds=60,  # Longer timeout for poor connectivity
                max_retry_attempts=5,  # More retries for unreliable connections
            )

            # Create and register BitChat transport with Global South optimizations
            self.bitchat_transport = create_bitchat_transport(
                device_id=self.device_id,
                device_name=f"GlobalSouth-{self.device_id[:8]}",
                max_peers=50,  # Support more peers for mesh density
                discovery_interval=45,  # Less frequent discovery to save battery
                battery_optimization=True,
                enable_compression=True,  # Compress for bandwidth savings
            )

            # Configure BitChat capabilities for Global South
            bitchat_capabilities = TransportCapabilities(
                supports_broadcast=True,
                supports_multicast=True,
                supports_unicast=True,
                max_message_size=1024,  # Small message limit
                is_offline_capable=True,
                requires_internet=False,
                typical_latency_ms=2000,  # Higher latency expectation
                bandwidth_mbps=0.1,  # Very low bandwidth assumption
                provides_encryption=True,
                supports_forward_secrecy=False,
                has_built_in_auth=False,
                battery_impact="medium",
                data_cost_impact="low",
                works_on_cellular=True,
                works_on_wifi=True,
            )

            # Register BitChat transport
            self.transport_manager.register_transport(
                TransportType.BITCHAT, self.bitchat_transport, bitchat_capabilities
            )

            # Set device context for Global South scenarios
            self._configure_global_south_context()
        else:
            logger.warning("P2P infrastructure not available, running in fallback mode")

        # Peer management
        self.peers: dict[str, PeerInfo] = {}

        # Message handling for Global South specific messages
        self.mesh_message_handlers: dict[MeshMessageType, Callable] = {}
        self.pending_requests: dict[str, dict[str, Any]] = {}
        self.gossip_cache: dict[str, tuple[bytes, datetime]] = {}

        # Content caching with Global South optimizations
        self.local_cache: dict[str, tuple[bytes, datetime, int]] = {}  # key -> (data, timestamp, access_count)
        self.cache_capacity_mb = 50  # Reduced cache for limited storage
        self.current_cache_size = 0

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "peers_discovered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "relay_forwards": 0,
            "offline_periods": 0,
            "sync_events": 0,
        }

        # Background tasks
        self.sync_task: asyncio.Task | None = None
        self.cache_cleanup_task: asyncio.Task | None = None

        # Register Global South specific message handlers
        self._register_mesh_handlers()

        logger.info(
            f"P2P mesh integration initialized for Global South: device_id={self.device_id}, type={peer_type.value}"
        )

    def _generate_device_id(self) -> str:
        """Generate unique device ID for Global South scenarios."""
        import uuid

        return f"gs_{uuid.uuid4().hex[:12]}"

    def _configure_global_south_context(self):
        """Configure device context for Global South scenarios."""
        if not self.transport_manager:
            return

        # Set typical Global South device context
        self.transport_manager.update_device_context(
            battery_level=0.6,  # Assume moderate battery level
            is_charging=False,  # Assume not always charging
            power_save_mode=True,  # Enable power saving
            network_type="cellular",  # Primarily cellular connectivity
            has_internet=False,  # Assume offline-first
            is_metered_connection=True,  # Cellular data is expensive
            signal_strength=0.4,  # Assume poor signal in rural areas
            is_foreground=True,
            user_priority=MessagePriority.NORMAL,
            cost_budget_remaining=0.5,  # Limited data budget
            supports_bluetooth=True,
            supports_wifi_direct=False,  # Limited WiFi Direct support
            max_concurrent_connections=5,  # Limited by device constraints
        )

        logger.debug("Configured Global South device context")

    def _register_mesh_handlers(self):
        """Register Global South specific message handlers."""
        self.mesh_message_handlers[MeshMessageType.REQUEST] = self._handle_content_request
        self.mesh_message_handlers[MeshMessageType.RESPONSE] = self._handle_content_response
        self.mesh_message_handlers[MeshMessageType.ANNOUNCE] = self._handle_service_announce
        self.mesh_message_handlers[MeshMessageType.GOSSIP] = self._handle_gossip_message

        # Register unified message handler with transport manager
        if self.transport_manager:
            self.transport_manager.register_message_handler(self._handle_unified_message)

    async def start(self) -> bool:
        """Start P2P mesh networking using existing transport infrastructure."""
        try:
            if not P2P_AVAILABLE:
                logger.error("P2P infrastructure not available")
                return False

            # Start the transport manager (which will start BitChat and other transports)
            if self.transport_manager:
                success = await self.transport_manager.start()
                if not success:
                    logger.error("Failed to start transport manager")
                    return False

            # Start background tasks for Global South specific functionality
            self.sync_task = asyncio.create_task(self._sync_loop())
            self.cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())

            logger.info("P2P mesh integration started for Global South scenarios")
            return True

        except Exception as e:
            logger.error(f"Failed to start P2P mesh integration: {e}")
            return False

    async def stop(self) -> bool:
        """Stop P2P mesh networking."""
        try:
            # Stop transport manager
            if self.transport_manager:
                await self.transport_manager.stop()

            # Cancel background tasks
            for task in [self.sync_task, self.cache_cleanup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            logger.info("P2P mesh integration stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping P2P mesh integration: {e}")
            return False

    async def _handle_unified_message(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle incoming unified messages from the transport manager."""
        try:
            # Update statistics
            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(message.payload)

            # Check if it's a Global South specific message type
            if hasattr(message, "mesh_message_type"):
                mesh_type = getattr(message, "mesh_message_type")
                if mesh_type in self.mesh_message_handlers:
                    await self.mesh_message_handlers[mesh_type](message, transport_type)
                    return

            # Handle standard message types
            if message.message_type == MessageType.PING:
                await self._handle_ping_message(message, transport_type)
            elif message.message_type == MessageType.DATA:
                await self._handle_data_message(message, transport_type)
            else:
                logger.debug(f"Received message of type {message.message_type.value} via {transport_type.value}")

            # Forward to offline coordinator if available
            if self.offline_coordinator and message.message_type == MessageType.DATA:
                await self.offline_coordinator.store_message(
                    sender=message.metadata.sender_id or "unknown",
                    recipient=message.metadata.recipient_id or self.device_id,
                    content=message.payload,
                    priority=message.metadata.priority.value,
                )

        except Exception as e:
            logger.error(f"Error handling unified message: {e}")

    async def send_message(
        self, recipient: str, content: bytes, priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Send a message using the unified transport system."""
        if not self.transport_manager:
            logger.error("Transport manager not available")
            return False

        try:
            # Create unified message
            message = UnifiedMessage(message_type=MessageType.DATA, payload=content)
            message.metadata.recipient_id = recipient
            message.metadata.sender_id = self.device_id
            message.metadata.priority = priority
            message.metadata.max_hops = 7  # BitChat default

            # Send via transport manager (will select best transport)
            success = await self.transport_manager.send_message(message)

            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(content)
                logger.debug(f"Sent message to {recipient} ({len(content)} bytes)")
            else:
                logger.warning(f"Failed to send message to {recipient}")

            return success

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def _sync_loop(self):
        """Background task for syncing with offline coordinator and peers."""
        while True:
            try:
                await self._sync_with_offline_coordinator()
                await self._update_peer_status()
                await asyncio.sleep(30)  # Sync every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(10)

    async def _cache_cleanup_loop(self):
        """Background task for cleaning up cache."""
        while True:
            try:
                await self._cleanup_cache()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _sync_with_offline_coordinator(self):
        """Sync pending messages with offline coordinator."""
        if not self.offline_coordinator:
            return

        try:
            # Get pending message counts
            pending_counts = await self.offline_coordinator.get_pending_message_count()
            total_pending = sum(pending_counts.values())

            if total_pending > 0:
                logger.debug(f"Found {total_pending} pending messages to sync")
                self.stats["sync_events"] += 1

                # In a real implementation, we would:
                # 1. Retrieve pending messages from offline coordinator
                # 2. Send them via the mesh network when peers are available
                # 3. Mark them as sent when successful

        except Exception as e:
            logger.error(f"Error syncing with offline coordinator: {e}")

    async def _update_peer_status(self):
        """Update peer status from BitChat transport."""
        if not self.bitchat_transport:
            return

        try:
            # Get peer list from BitChat transport
            bitchat_peers = self.bitchat_transport.get_peers()

            # Update our peer tracking
            for peer_data in bitchat_peers:
                peer_id = peer_data["device_id"]
                if peer_id not in self.peers:
                    self.peers[peer_id] = PeerInfo(
                        peer_id=peer_id,
                        peer_type=PeerType.MOBILE,  # Default assumption
                        connection_type=ConnectionType.BLUETOOTH,
                        address="",  # BitChat doesn't expose addresses
                        port=0,
                        capabilities=set(),
                        last_seen=datetime.utcnow(),
                        is_online=peer_data["is_online"],
                    )
                    self.stats["peers_discovered"] += 1
                else:
                    self.peers[peer_id].is_online = peer_data["is_online"]
                    self.peers[peer_id].last_seen = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating peer status: {e}")

    async def request_content(self, content_key: str, timeout_seconds: int = 30) -> bytes | None:
        """Request content from mesh network using existing transport system."""
        # Check local cache first
        if content_key in self.local_cache:
            content_data, _, access_count = self.local_cache[content_key]
            self.local_cache[content_key] = (content_data, datetime.utcnow(), access_count + 1)
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for content key: {content_key}")
            return content_data

        # Request from peers via unified message system
        try:
            request_data = json.dumps({"key": content_key, "timestamp": time.time()}).encode("utf-8")

            # Broadcast request to all peers
            success = await self.send_message(
                recipient="broadcast", content=request_data, priority=MessagePriority.NORMAL
            )

            if not success:
                logger.warning(f"Failed to broadcast content request for {content_key}")
                self.stats["cache_misses"] += 1
                return None

            # Wait for response (simplified - in real implementation would track responses)
            await asyncio.sleep(min(timeout_seconds, 5))  # Give some time for responses

            # Check if content appeared in cache (from response handler)
            if content_key in self.local_cache:
                content_data, _, access_count = self.local_cache[content_key]
                self.stats["cache_hits"] += 1
                return content_data

        except Exception as e:
            logger.error(f"Error requesting content {content_key}: {e}")

        self.stats["cache_misses"] += 1
        return None

    async def _cleanup_cache(self):
        """Clean up old cache entries."""
        # Remove entries older than 1 hour with low access count
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        expired_keys = []

        for key, (data, timestamp, access_count) in self.local_cache.items():
            if timestamp < cutoff_time and access_count < 3:
                expired_keys.append(key)

        for key in expired_keys:
            data_size = len(self.local_cache[key][0])
            del self.local_cache[key]
            self.current_cache_size -= data_size

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} cache entries")

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status of P2P mesh integration."""
        transport_status = {}
        if self.transport_manager:
            transport_status = self.transport_manager.get_status()

        return {
            "device_id": self.device_id,
            "peer_type": self.peer_type.value,
            "p2p_available": P2P_AVAILABLE,
            "transport_manager_active": self.transport_manager is not None,
            "bitchat_transport_active": self.bitchat_transport is not None,
            "peer_count": len(self.peers),
            "cache_size_mb": self.current_cache_size / (1024 * 1024),
            "cache_items": len(self.local_cache),
            "offline_coordinator_active": self.offline_coordinator is not None,
            "statistics": self.stats.copy(),
            "transport_status": transport_status,
        }

    def get_peer_info(self) -> dict[str, Any]:
        """Get information about discovered peers."""
        return {
            "device_id": self.device_id,
            "peer_type": self.peer_type.value,
            "peers": {
                peer_id: {
                    "type": peer.peer_type.value,
                    "connection_type": peer.connection_type.value,
                    "address": peer.address,
                    "capabilities": list(peer.capabilities),
                    "last_seen": peer.last_seen.isoformat(),
                    "is_online": peer.is_online,
                    "battery_level": peer.battery_level,
                    "storage_available_mb": peer.storage_available_mb,
                }
                for peer_id, peer in self.peers.items()
            },
            "total_peers": len(self.peers),
        }

    def get_network_stats(self) -> dict[str, Any]:
        """Get comprehensive network statistics."""
        base_stats = self.stats.copy()

        # Add transport manager stats if available
        if self.transport_manager:
            transport_stats = self.transport_manager.get_status().get("statistics", {})
            base_stats.update({f"transport_{k}": v for k, v in transport_stats.items()})

        return {
            **base_stats,
            "cache_efficiency": self.stats["cache_hits"]
            / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1),
            "message_success_rate": self.stats["messages_sent"]
            / max(self.stats["messages_sent"] + self.stats.get("send_failures", 0), 1),
            "peer_discovery_rate": self.stats["peers_discovered"],
            "offline_periods": self.stats["offline_periods"],
            "sync_events": self.stats["sync_events"],
        }

    # Message handlers for Global South specific message types
    async def _handle_ping_message(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle ping messages from peers."""
        logger.debug(f"Received ping from {message.metadata.sender_id} via {transport_type.value}")

        # Update peer information if available
        sender_id = message.metadata.sender_id
        if sender_id and sender_id not in self.peers:
            self.peers[sender_id] = PeerInfo(
                peer_id=sender_id,
                peer_type=PeerType.MOBILE,  # Default
                connection_type=ConnectionType.BLUETOOTH
                if transport_type == TransportType.BITCHAT
                else ConnectionType.WIFI_HOTSPOT,
                address="",  # Transport layer handles addressing
                port=0,
                capabilities=set(),
                last_seen=datetime.utcnow(),
                is_online=True,
            )
            self.stats["peers_discovered"] += 1

    async def _handle_data_message(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle data messages from peers."""
        logger.debug(f"Received data message ({len(message.payload)} bytes) from {message.metadata.sender_id}")

        # Try to parse as content request
        try:
            request_data = json.loads(message.payload.decode("utf-8"))
            if "key" in request_data:
                await self._handle_content_request_data(request_data, message.metadata.sender_id)
                return
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # Handle as regular data message
        if self.offline_coordinator:
            await self.offline_coordinator.store_message(
                sender=message.metadata.sender_id or "unknown",
                recipient=message.metadata.recipient_id or self.device_id,
                content=message.payload,
                priority=message.metadata.priority.value,
            )

    async def _handle_content_request_data(self, request_data: dict[str, Any], sender_id: str):
        """Handle content request from peer."""
        content_key = request_data.get("key")
        if not content_key:
            return

        logger.debug(f"Content request for '{content_key}' from {sender_id}")

        # Check local cache
        if content_key in self.local_cache:
            content_data, _, access_count = self.local_cache[content_key]
            self.local_cache[content_key] = (content_data, datetime.utcnow(), access_count + 1)

            # Send response back
            response_data = json.dumps(
                {"key": content_key, "content": content_data.hex(), "timestamp": time.time()}
            ).encode("utf-8")

            await self.send_message(recipient=sender_id, content=response_data, priority=MessagePriority.NORMAL)

            self.stats["cache_hits"] += 1
            logger.debug(f"Responded to content request for '{content_key}'")
        else:
            self.stats["cache_misses"] += 1

    # Additional Global South specific message handlers
    async def _handle_content_request(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle content request messages."""
        # This would be called for mesh-specific content requests
        pass

    async def _handle_content_response(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle content response messages."""
        # This would be called for mesh-specific content responses
        pass

    async def _handle_service_announce(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle service announcement messages."""
        # This would be called for service announcements
        pass

    async def _handle_gossip_message(self, message: UnifiedMessage, transport_type: TransportType):
        """Handle gossip messages."""
        # This would be called for gossip protocol messages
        pass


async def create_p2p_mesh_integration(
    device_id: str = None,
    peer_type: PeerType = PeerType.MOBILE,
    offline_coordinator=None,
    transport_priority: TransportPriority = TransportPriority.OFFLINE_FIRST,
    start_immediately: bool = True,
) -> P2PMeshIntegration:
    """
    Create P2P mesh integration for Global South scenarios.

    This factory function creates a P2P mesh integration that leverages the existing
    AIVillage transport infrastructure (TransportManager, BitChat, etc.) to provide
    Global South specific capabilities.
    """
    mesh_integration = P2PMeshIntegration(
        device_id=device_id,
        peer_type=peer_type,
        offline_coordinator=offline_coordinator,
        transport_priority=transport_priority,
    )

    if start_immediately:
        success = await mesh_integration.start()
        if not success:
            logger.error("Failed to start P2P mesh integration")
            return None

    logger.info("P2P mesh integration created for Global South scenarios")
    return mesh_integration


if __name__ == "__main__":
    # Example usage for Global South scenarios
    async def main():
        # Create mesh integration with offline coordinator
        mesh = await create_p2p_mesh_integration(
            device_id="global-south-device-001",
            peer_type=PeerType.MOBILE,
            transport_priority=TransportPriority.OFFLINE_FIRST,
        )

        if mesh:
            try:
                # Run for demonstration
                print("P2P mesh integration running for Global South scenarios...")
                print(f"Status: {mesh.get_status()}")

                # Demonstrate content request
                content = await mesh.request_content("test-content-key", timeout_seconds=10)
                if content:
                    print(f"Retrieved content: {len(content)} bytes")
                else:
                    print("No content found in mesh network")

                # Wait for some peer activity
                await asyncio.sleep(60)

                # Show final statistics
                stats = mesh.get_network_stats()
                print(f"Final stats: {stats}")

            finally:
                await mesh.stop()
        else:
            print("Failed to create mesh integration")

    asyncio.run(main())
