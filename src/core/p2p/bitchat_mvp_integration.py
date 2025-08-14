"""BitChat MVP Integration Bridge

Integrates the new BitChat MVP (Android/iOS) with the existing AIVillage
dual-path transport system and navigator agent. Provides seamless interoperability
between mobile BitChat implementations and the Python-based infrastructure.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Import existing AIVillage components
try:
    from .bitchat_transport import BitChatMessage as LegacyBitChatMessage
    from .bitchat_transport import BitChatTransport
    from .dual_path_transport import DualPathMessage, DualPathTransport

    DUAL_PATH_AVAILABLE = True
except ImportError:
    DualPathTransport = None
    DUAL_PATH_AVAILABLE = False

# Import resource management for mobile optimization
try:
    from ...production.monitoring.mobile.resource_management import (
        BatteryThermalResourceManager,
        TransportPreference,
    )

    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    BatteryThermalResourceManager = None
    TransportPreference = None
    RESOURCE_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BitChatMVPMessage:
    """BitChat MVP message format (compatible with Android/iOS protobuf)"""

    msg_id: str
    created_at: int  # Unix timestamp milliseconds
    hop_count: int
    ttl: int
    original_sender: str
    message_type: str
    ciphertext_blob: bytes
    routing_metadata: dict[str, Any] | None = None
    priority: str = "PRIORITY_NORMAL"

    def to_protobuf_dict(self) -> dict[str, Any]:
        """Convert to protobuf format for Android/iOS interchange"""
        return {
            "msg_id": self.msg_id,
            "created_at": self.created_at,
            "hop_count": self.hop_count,
            "ttl": self.ttl,
            "original_sender": self.original_sender,
            "message_type": self.message_type,
            "ciphertext_blob": self.ciphertext_blob.hex(),
            "routing": self.routing_metadata,
            "priority": self.priority,
        }

    @classmethod
    def from_protobuf_dict(cls, data: dict[str, Any]) -> "BitChatMVPMessage":
        """Create from protobuf format"""
        return cls(
            msg_id=data["msg_id"],
            created_at=data["created_at"],
            hop_count=data["hop_count"],
            ttl=data["ttl"],
            original_sender=data["original_sender"],
            message_type=data["message_type"],
            ciphertext_blob=bytes.fromhex(data["ciphertext_blob"]),
            routing_metadata=data.get("routing"),
            priority=data.get("priority", "PRIORITY_NORMAL"),
        )

    def to_legacy_message(self) -> "LegacyBitChatMessage":
        """Convert to legacy BitChat format"""
        if LegacyBitChatMessage is None:
            raise RuntimeError("Legacy BitChat not available")

        return LegacyBitChatMessage(
            id=self.msg_id,
            sender=self.original_sender,
            recipient=self.routing_metadata.get("target_peer_id", "") if self.routing_metadata else "",
            payload=self.ciphertext_blob,
            ttl=self.ttl,
            hop_count=self.hop_count,
            timestamp=self.created_at / 1000.0,  # Convert to seconds
            priority=self._convert_priority_to_int(),
        )

    def _convert_priority_to_int(self) -> int:
        """Convert string priority to integer"""
        priority_map = {
            "PRIORITY_LOW": 3,
            "PRIORITY_NORMAL": 5,
            "PRIORITY_HIGH": 7,
            "PRIORITY_EMERGENCY": 10,
        }
        return priority_map.get(self.priority, 5)


class BitChatMVPIntegrationBridge:
    """Integration bridge between BitChat MVP and AIVillage infrastructure

    Provides:
    - Message format translation between mobile and Python implementations
    - Navigator agent integration for intelligent routing
    - Resource management integration for battery optimization
    - Dual-path transport coordination
    - Cross-platform message interchange
    """

    def __init__(
        self,
        peer_id: str = None,
        enable_legacy_bridge: bool = True,
        enable_dual_path: bool = True,
        enable_resource_management: bool = True,
    ):
        self.peer_id = peer_id or f"mvp_bridge_{uuid.uuid4().hex[:8]}"
        self.enable_legacy_bridge = enable_legacy_bridge
        self.enable_dual_path = enable_dual_path
        self.enable_resource_management = enable_resource_management

        # Component instances
        self.legacy_transport: BitChatTransport | None = None
        self.dual_path_transport: DualPathTransport | None = None
        self.resource_manager: BatteryThermalResourceManager | None = None

        # State management
        self.is_running = False
        self.mobile_peers: dict[str, dict[str, Any]] = {}  # Android/iOS peers
        self.message_handlers: list[Callable] = []
        self.routing_decisions: dict[str, str] = {}  # msg_id -> chosen_transport

        # Statistics
        self.stats = {
            "messages_bridged": 0,
            "mobile_to_legacy": 0,
            "legacy_to_mobile": 0,
            "dual_path_routed": 0,
            "resource_optimizations": 0,
        }

        logger.info(f"BitChat MVP Integration Bridge initialized: {self.peer_id}")

    async def start(self) -> bool:
        """Start the integration bridge"""
        if self.is_running:
            return True

        logger.info("Starting BitChat MVP Integration Bridge...")
        self.is_running = True

        try:
            # Initialize legacy BitChat transport if enabled
            if self.enable_legacy_bridge and BitChatTransport:
                self.legacy_transport = BitChatTransport(device_id=f"legacy_{self.peer_id}")

                # Register message handler for legacy transport
                self.legacy_transport.register_handler("default", self._handle_legacy_message)
                await self.legacy_transport.start()

                logger.info("Legacy BitChat transport bridge enabled")

            # Initialize dual-path transport if available
            if self.enable_dual_path and DUAL_PATH_AVAILABLE:
                # This would integrate with the dual-path system
                logger.info("Dual-path transport integration enabled")

            # Initialize resource management if available
            if self.enable_resource_management and RESOURCE_MANAGEMENT_AVAILABLE:
                self.resource_manager = BatteryThermalResourceManager()
                logger.info("Resource management integration enabled")

            logger.info("BitChat MVP Integration Bridge started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start integration bridge: {e}")
            self.is_running = False
            return False

    async def stop(self) -> None:
        """Stop the integration bridge"""
        logger.info("Stopping BitChat MVP Integration Bridge...")
        self.is_running = False

        # Stop legacy transport
        if self.legacy_transport:
            await self.legacy_transport.stop()
            self.legacy_transport = None

        # Clean up state
        self.mobile_peers.clear()
        self.routing_decisions.clear()

        logger.info("BitChat MVP Integration Bridge stopped")

    async def register_mobile_peer(self, peer_info: dict[str, Any]) -> None:
        """Register a mobile peer (Android/iOS) with the bridge"""
        peer_id = peer_info.get("peer_id")
        if not peer_id:
            logger.warning("Mobile peer registration missing peer_id")
            return

        self.mobile_peers[peer_id] = {
            "peer_id": peer_id,
            "platform": peer_info.get("platform", "unknown"),
            "capabilities": peer_info.get("capabilities", {}),
            "last_seen": time.time(),
            "battery_level": peer_info.get("battery_level", 1.0),
            "transport_types": peer_info.get("transport_types", []),
            "connection_quality": peer_info.get("connection_quality", 1.0),
        }

        logger.info(f"Registered mobile peer: {peer_id} ({peer_info.get('platform', 'unknown')})")

    async def send_to_mobile_peer(self, peer_id: str, message: BitChatMVPMessage, transport_hint: str = None) -> bool:
        """Send message to mobile peer (Android/iOS)

        Args:
            peer_id: Target mobile peer ID
            message: BitChat MVP message
            transport_hint: Preferred transport type

        Returns:
            True if message was sent successfully
        """
        if not self.is_running:
            logger.warning("Integration bridge not running")
            return False

        peer = self.mobile_peers.get(peer_id)
        if not peer:
            logger.warning(f"Mobile peer {peer_id} not registered")
            return False

        try:
            # Apply resource management optimizations if available
            if self.resource_manager:
                await self._apply_resource_optimizations(message, peer)

            # Determine best transport based on navigator logic
            chosen_transport = await self._select_transport_for_mobile(message, peer, transport_hint)

            # Convert message to mobile format
            mobile_data = message.to_protobuf_dict()

            # In production, this would send to actual mobile transport
            # For now, simulate successful transmission
            await self._simulate_mobile_transmission(peer_id, mobile_data, chosen_transport)

            self.stats["messages_bridged"] += 1
            self.routing_decisions[message.msg_id] = chosen_transport

            logger.debug(f"Sent message {message.msg_id[:8]} to mobile peer {peer_id} via {chosen_transport}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message to mobile peer {peer_id}: {e}")
            return False

    async def receive_from_mobile_peer(self, peer_id: str, message_data: dict[str, Any]) -> None:
        """Receive message from mobile peer and route appropriately"""
        try:
            # Convert from mobile format
            message = BitChatMVPMessage.from_protobuf_dict(message_data)

            logger.debug(f"Received message {message.msg_id[:8]} from mobile peer {peer_id}")

            # Update peer last seen
            if peer_id in self.mobile_peers:
                self.mobile_peers[peer_id]["last_seen"] = time.time()

            # Determine routing strategy
            routing_decision = await self._make_routing_decision(message, peer_id)

            # Route to appropriate destination
            if routing_decision == "legacy_bridge":
                await self._route_to_legacy(message)
            elif routing_decision == "dual_path":
                await self._route_to_dual_path(message)
            elif routing_decision == "mobile_relay":
                await self._route_to_mobile_peers(message, exclude_peer=peer_id)

            # Notify message handlers
            for handler in self.message_handlers:
                try:
                    await handler(message, peer_id)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")

            self.stats["messages_bridged"] += 1

        except Exception as e:
            logger.error(f"Error processing message from mobile peer {peer_id}: {e}")

    async def _apply_resource_optimizations(self, message: BitChatMVPMessage, peer: dict[str, Any]) -> None:
        """Apply resource management optimizations"""
        if not self.resource_manager:
            return

        try:
            # Mock device profile for optimization
            from ...production.monitoring.mobile.device_profiler import DeviceProfile

            device_profile = DeviceProfile(
                battery_percent=int(peer.get("battery_level", 1.0) * 100),
                cpu_temp_celsius=35.0,  # Default temperature
                network_type="wifi",  # Assume WiFi for BitChat
            )

            # Get resource state and optimization
            resource_state = await self.resource_manager.evaluate_and_adapt(device_profile)

            # Apply optimizations based on resource state
            if resource_state.transport_preference == TransportPreference.BITCHAT_ONLY:
                # Force BitChat-only routing for battery conservation
                message.routing_metadata = message.routing_metadata or {}
                message.routing_metadata["transport_preference"] = "bitchat_only"
                message.routing_metadata["reason"] = "battery_conservation"

            self.stats["resource_optimizations"] += 1

        except Exception as e:
            logger.debug(f"Resource optimization failed: {e}")

    async def _select_transport_for_mobile(
        self,
        message: BitChatMVPMessage,
        peer: dict[str, Any],
        transport_hint: str = None,
    ) -> str:
        """Select best transport for mobile message delivery"""

        # Check for explicit transport preference
        if transport_hint:
            return transport_hint

        # Check resource management constraints
        if message.routing_metadata and message.routing_metadata.get("transport_preference") == "bitchat_only":
            return "bitchat_mesh"

        # Consider peer capabilities
        available_transports = peer.get("transport_types", [])

        # Navigator logic simulation
        if "nearby_wifi" in available_transports:
            return "nearby_wifi"
        elif "multipeer_wifi" in available_transports:
            return "multipeer_wifi"
        elif "nearby_bluetooth" in available_transports:
            return "nearby_bluetooth"
        elif "ble_beacon" in available_transports:
            return "ble_beacon"
        else:
            return "bitchat_mesh"  # Fallback

    async def _make_routing_decision(self, message: BitChatMVPMessage, source_peer: str) -> str:
        """Make intelligent routing decision for received message"""

        # Check message type and routing metadata
        if message.message_type == "MESSAGE_TYPE_HEARTBEAT":
            return "local_processing"  # Don't route heartbeats

        # Check for specific target
        if message.routing_metadata:
            target_peer = message.routing_metadata.get("target_peer_id")
            if target_peer:
                # Check if target is a mobile peer
                if target_peer in self.mobile_peers:
                    return "mobile_relay"
                # Check if target is accessible via legacy transport
                elif self.legacy_transport and self.legacy_transport.is_peer_reachable(target_peer):
                    return "legacy_bridge"
                # Route via dual-path for unknown targets
                else:
                    return "dual_path"

        # Default: broadcast via all available paths
        return "broadcast_all"

    async def _route_to_legacy(self, message: BitChatMVPMessage) -> None:
        """Route message to legacy BitChat transport"""
        if not self.legacy_transport:
            logger.warning("Legacy transport not available for routing")
            return

        try:
            legacy_message = message.to_legacy_message()
            success = await self.legacy_transport.send_message(
                recipient=legacy_message.recipient,
                payload=legacy_message.payload,
                priority=legacy_message.priority,
                ttl=legacy_message.ttl,
            )

            if success:
                self.stats["legacy_to_mobile"] += 1
                logger.debug(f"Routed message {message.msg_id[:8]} to legacy transport")

        except Exception as e:
            logger.error(f"Failed to route to legacy transport: {e}")

    async def _route_to_dual_path(self, message: BitChatMVPMessage) -> None:
        """Route message to dual-path transport system"""
        if not DUAL_PATH_AVAILABLE:
            logger.warning("Dual-path transport not available")
            return

        try:
            # Convert to dual-path format
            DualPathMessage(
                id=message.msg_id,
                sender=message.original_sender,
                payload=message.ciphertext_blob,
                priority=self._convert_priority_to_dual_path(message.priority),
                content_type="application/bitchat-mvp",
            )

            # In production, would send via dual-path transport
            # For now, simulate
            logger.debug(f"Routed message {message.msg_id[:8]} to dual-path transport")
            self.stats["dual_path_routed"] += 1

        except Exception as e:
            logger.error(f"Failed to route to dual-path transport: {e}")

    async def _route_to_mobile_peers(self, message: BitChatMVPMessage, exclude_peer: str = None) -> None:
        """Route message to other mobile peers"""
        target_peers = [peer_id for peer_id in self.mobile_peers.keys() if peer_id != exclude_peer]

        for peer_id in target_peers:
            try:
                await self.send_to_mobile_peer(peer_id, message)
            except Exception as e:
                logger.error(f"Failed to route to mobile peer {peer_id}: {e}")

    async def _simulate_mobile_transmission(self, peer_id: str, message_data: dict[str, Any], transport: str) -> None:
        """Simulate transmission to mobile device (for testing)"""
        # In production, this would use actual mobile communication channels
        # For now, just simulate delay based on transport type

        delay_map = {
            "ble_beacon": 0.5,  # BLE is slower
            "nearby_bluetooth": 0.1,
            "nearby_wifi": 0.05,
            "multipeer_wifi": 0.05,
            "bitchat_mesh": 0.2,
        }

        delay = delay_map.get(transport, 0.1)
        await asyncio.sleep(delay)

        logger.debug(f"[SIM] Transmitted to {peer_id} via {transport} (delay: {delay}s)")

    async def _handle_legacy_message(self, legacy_message: "LegacyBitChatMessage") -> None:
        """Handle message received from legacy BitChat transport"""
        try:
            # Convert legacy message to MVP format
            mvp_message = BitChatMVPMessage(
                msg_id=legacy_message.id,
                created_at=int(legacy_message.timestamp * 1000),  # Convert to milliseconds
                hop_count=legacy_message.hop_count,
                ttl=legacy_message.ttl,
                original_sender=legacy_message.sender,
                message_type="MESSAGE_TYPE_DATA",
                ciphertext_blob=legacy_message.payload,
                routing_metadata={"target_peer_id": legacy_message.recipient} if legacy_message.recipient else None,
                priority=self._convert_int_priority_to_string(legacy_message.priority),
            )

            # Route to mobile peers if appropriate
            if not legacy_message.recipient or legacy_message.recipient in self.mobile_peers:
                await self._route_to_mobile_peers(mvp_message)

            self.stats["mobile_to_legacy"] += 1
            logger.debug(f"Bridged legacy message {legacy_message.id[:8]} to mobile peers")

        except Exception as e:
            logger.error(f"Error handling legacy message: {e}")

    def _convert_priority_to_dual_path(self, priority: str) -> int:
        """Convert MVP priority to dual-path integer priority"""
        priority_map = {
            "PRIORITY_LOW": 8,
            "PRIORITY_NORMAL": 5,
            "PRIORITY_HIGH": 2,
            "PRIORITY_EMERGENCY": 1,
        }
        return priority_map.get(priority, 5)

    def _convert_int_priority_to_string(self, priority: int) -> str:
        """Convert integer priority to MVP string priority"""
        if priority <= 3:
            return "PRIORITY_LOW"
        elif priority <= 6:
            return "PRIORITY_NORMAL"
        elif priority <= 8:
            return "PRIORITY_HIGH"
        else:
            return "PRIORITY_EMERGENCY"

    def add_message_handler(self, handler: Callable) -> None:
        """Add message handler for received messages"""
        self.message_handlers.append(handler)

    def get_integration_stats(self) -> dict[str, Any]:
        """Get integration bridge statistics"""
        return {
            "peer_id": self.peer_id,
            "is_running": self.is_running,
            "mobile_peers_connected": len(self.mobile_peers),
            "legacy_transport_active": self.legacy_transport is not None and self.legacy_transport.is_running,
            "dual_path_available": DUAL_PATH_AVAILABLE,
            "resource_management_active": self.resource_manager is not None,
            "statistics": self.stats.copy(),
            "mobile_peers": list(self.mobile_peers.keys()),
            "recent_routing_decisions": dict(list(self.routing_decisions.items())[-10:]),  # Last 10
        }

    def get_mobile_peer_info(self) -> list[dict[str, Any]]:
        """Get information about connected mobile peers"""
        return [
            {
                "peer_id": peer["peer_id"],
                "platform": peer["platform"],
                "last_seen": peer["last_seen"],
                "battery_level": peer["battery_level"],
                "transport_types": peer["transport_types"],
                "connection_quality": peer["connection_quality"],
                "capabilities": peer["capabilities"],
            }
            for peer in self.mobile_peers.values()
        ]


# Factory function for easy integration
async def create_bitchat_mvp_bridge(
    config: dict[str, Any] = None,
) -> BitChatMVPIntegrationBridge:
    """Create and start BitChat MVP integration bridge

    Args:
        config: Configuration dictionary

    Returns:
        Started integration bridge instance
    """
    config = config or {}

    bridge = BitChatMVPIntegrationBridge(
        peer_id=config.get("peer_id"),
        enable_legacy_bridge=config.get("enable_legacy_bridge", True),
        enable_dual_path=config.get("enable_dual_path", True),
        enable_resource_management=config.get("enable_resource_management", True),
    )

    await bridge.start()
    return bridge


# Example usage and testing
async def test_bitchat_mvp_integration():
    """Test the BitChat MVP integration bridge"""
    logger.info("Testing BitChat MVP Integration Bridge...")

    # Create integration bridge
    bridge = await create_bitchat_mvp_bridge()

    # Register mock mobile peers
    android_peer = {
        "peer_id": "android_test_device",
        "platform": "android",
        "battery_level": 0.8,
        "transport_types": ["nearby_wifi", "nearby_bluetooth", "ble_beacon"],
        "capabilities": {
            "supports_chunking": True,
            "max_chunk_size": 262144,
            "supports_encryption": True,
        },
    }

    ios_peer = {
        "peer_id": "ios_test_device",
        "platform": "ios",
        "battery_level": 0.6,
        "transport_types": ["multipeer_wifi", "multipeer_bluetooth"],
        "capabilities": {
            "supports_chunking": True,
            "max_chunk_size": 262144,
            "supports_background": False,
        },
    }

    await bridge.register_mobile_peer(android_peer)
    await bridge.register_mobile_peer(ios_peer)

    # Test message exchange
    test_message = BitChatMVPMessage(
        msg_id=f"test_{uuid.uuid4().hex[:8]}",
        created_at=int(time.time() * 1000),
        hop_count=0,
        ttl=7,
        original_sender="python_test_bridge",
        message_type="MESSAGE_TYPE_DATA",
        ciphertext_blob=b"Hello from BitChat MVP Integration Bridge!",
        priority="PRIORITY_NORMAL",
    )

    # Send to Android peer
    success = await bridge.send_to_mobile_peer("android_test_device", test_message)
    logger.info(f"Message to Android peer: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Send to iOS peer
    success = await bridge.send_to_mobile_peer("ios_test_device", test_message)
    logger.info(f"Message to iOS peer: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Simulate receiving message from mobile peer
    received_message_data = {
        "msg_id": f"mobile_{uuid.uuid4().hex[:8]}",
        "created_at": int(time.time() * 1000),
        "hop_count": 1,
        "ttl": 6,
        "original_sender": "android_test_device",
        "message_type": "MESSAGE_TYPE_DATA",
        "ciphertext_blob": b"Hello from Android BitChat!".hex(),
        "priority": "PRIORITY_HIGH",
    }

    await bridge.receive_from_mobile_peer("android_test_device", received_message_data)

    # Print statistics
    stats = bridge.get_integration_stats()
    logger.info(f"Integration stats: {stats}")

    # Clean up
    await bridge.stop()
    logger.info("BitChat MVP Integration Bridge test completed")


if __name__ == "__main__":
    # Run test if script is executed directly
    import asyncio

    asyncio.run(test_bitchat_mvp_integration())
