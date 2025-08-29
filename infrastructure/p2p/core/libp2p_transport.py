"""LibP2P Transport Integration.

Provides a transport wrapper that integrates the LibP2P mesh network
with the existing transport manager system, enabling seamless use of
LibP2P as a transport layer alongside other transport types.
"""

import asyncio
import json
import logging
import time
from typing import Any

from ..mobile_integration.libp2p_mesh import (
    LibP2PMeshNetwork,
    MeshConfiguration,
    MeshMessage,
    MeshMessageType,
)
from ..protocols.mesh_networking import DistanceVectorRouting, GossipProtocol, TopologyManager, TopologyType
from ..security.production_security import SecurityConfig, SecurityLevel, SecurityManager
from .message_delivery import DeliveryConfig, MessageDeliveryService, MessagePriority
from .message_types import MessageMetadata, MessageType, UnifiedMessage
from .transport_manager import TransportCapabilities

logger = logging.getLogger(__name__)


class LibP2PTransport:
    """Transport wrapper for LibP2P mesh network integration."""

    def __init__(
        self,
        node_id: str,
        config: MeshConfiguration | None = None,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        enable_delivery_guarantees: bool = True,
    ):
        self.node_id = node_id

        # Initialize LibP2P mesh network
        self.config = config or MeshConfiguration(node_id=node_id)
        self.mesh_network = LibP2PMeshNetwork(self.config)

        # Initialize security layer
        security_config = SecurityConfig(security_level=security_level)
        self.security_manager = SecurityManager(security_config)

        # Initialize advanced protocols
        self.gossip_protocol = GossipProtocol(node_id)
        self.routing_protocol = DistanceVectorRouting(node_id)
        self.topology_manager = TopologyManager(node_id, TopologyType.ADAPTIVE)

        # Initialize delivery service if enabled
        self.delivery_service: MessageDeliveryService | None = None
        if enable_delivery_guarantees:
            delivery_config = DeliveryConfig(
                max_retry_attempts=5,
                enable_persistence=True,
                concurrent_deliveries=10,
            )
            self.delivery_service = MessageDeliveryService(delivery_config)
            self.delivery_service.set_send_function(self._send_message_direct)

        # Transport state
        self.running = False
        self.capabilities = TransportCapabilities(
            supports_broadcast=True,
            supports_multicast=True,
            supports_unicast=True,
            max_message_size=1024 * 1024,  # 1MB
            is_offline_capable=True,
            requires_internet=False,
            typical_latency_ms=100,
            bandwidth_mbps=10.0,
            provides_encryption=True,
            supports_forward_secrecy=True,
            has_built_in_auth=True,
            battery_impact="medium",
            data_cost_impact="low",
            works_on_cellular=True,
            works_on_wifi=True,
        )

        # Message handlers
        self.message_handlers: list[Any] = []

        # Performance tracking
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "peer_connections": 0,
            "security_events": 0,
            "delivery_success_rate": 0.0,
        }

    async def start(self) -> bool:
        """Start the LibP2P transport."""
        if self.running:
            return True

        logger.info(f"Starting LibP2P transport for node {self.node_id}")

        try:
            # Start security manager
            await self.security_manager.start()

            # Start mesh network
            success = await self.mesh_network.start()
            if not success:
                logger.error("Failed to start LibP2P mesh network")
                return False

            # Register message handlers
            self.mesh_network.register_message_handler(MeshMessageType.DATA_MESSAGE, self._handle_incoming_message)
            self.mesh_network.register_message_handler(MeshMessageType.AGENT_TASK, self._handle_incoming_message)
            self.mesh_network.register_message_handler(MeshMessageType.PARAMETER_UPDATE, self._handle_incoming_message)
            self.mesh_network.register_message_handler(MeshMessageType.GRADIENT_SHARING, self._handle_incoming_message)

            # Start advanced protocols
            await self.gossip_protocol.start()
            await self.routing_protocol.start()
            await self.topology_manager.start()

            # Start delivery service if configured
            if self.delivery_service:
                await self.delivery_service.start()

            # Update capabilities based on mesh status
            self._update_capabilities()

            self.running = True
            logger.info("LibP2P transport started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start LibP2P transport: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the LibP2P transport."""
        if not self.running:
            return True

        logger.info("Stopping LibP2P transport")

        try:
            # Stop advanced protocols
            await self.topology_manager.stop()
            await self.routing_protocol.stop()
            await self.gossip_protocol.stop()

            # Stop delivery service
            if self.delivery_service:
                await self.delivery_service.stop()

            # Stop mesh network
            await self.mesh_network.stop()

            # Stop security manager
            await self.security_manager.stop()

            self.running = False
            self.capabilities.is_available = False
            self.capabilities.is_connected = False

            logger.info("LibP2P transport stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping LibP2P transport: {e}")
            return False

    async def send_message(self, message: UnifiedMessage) -> bool:
        """Send a message through the LibP2P transport."""
        if not self.running:
            logger.warning("Cannot send message: LibP2P transport not running")
            return False

        try:
            # Convert UnifiedMessage to MeshMessage
            mesh_message = self._unified_to_mesh_message(message)

            if self.delivery_service:
                # Use delivery service for guaranteed delivery
                priority = self._map_message_priority(message.metadata.priority)
                requires_ack = message.metadata.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]

                message_id = await self.delivery_service.send_message(
                    mesh_message,
                    priority=priority,
                    requires_ack=requires_ack,
                    ttl=message.metadata.ttl_seconds,
                )

                logger.debug(f"Message queued for delivery: {message_id}")
                return True
            else:
                # Send directly through mesh network
                success = await self._send_message_direct(mesh_message)
                return success

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def _send_message_direct(self, mesh_message: MeshMessage) -> bool:
        """Send message directly through mesh network."""
        try:
            # Apply security processing
            message_data = mesh_message.to_dict()
            processed_data = await self.security_manager.process_message(mesh_message.sender, message_data)

            if not processed_data:
                logger.warning(f"Message blocked by security layer from {mesh_message.sender}")
                return False

            # Send through mesh network
            success = await self.mesh_network.send_message(mesh_message)

            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(mesh_message.payload)

                # Use gossip protocol for broadcasts
                if not mesh_message.recipient:
                    await self.gossip_protocol.gossip_message(mesh_message.id, mesh_message.to_dict())

            return success

        except Exception as e:
            logger.error(f"Direct message send failed: {e}")
            return False

    async def _handle_incoming_message(self, mesh_message: MeshMessage):
        """Handle incoming messages from the mesh network."""
        try:
            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(mesh_message.payload)

            # Process through security layer
            message_data = mesh_message.to_dict()
            processed_data = await self.security_manager.process_message(mesh_message.sender, message_data)

            if not processed_data:
                logger.warning(f"Incoming message blocked by security layer from {mesh_message.sender}")
                return

            # Convert to UnifiedMessage
            unified_message = self._mesh_to_unified_message(mesh_message)

            # Forward to registered handlers
            for handler in self.message_handlers:
                try:
                    await handler(unified_message, "libp2p_mesh")
                except Exception as e:
                    logger.warning(f"Message handler error: {e}")

            # Send acknowledgment if required
            if mesh_message.type in [MeshMessageType.AGENT_TASK, MeshMessageType.PARAMETER_UPDATE]:
                await self._send_acknowledgment(mesh_message)

        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")

    async def _send_acknowledgment(self, original_message: MeshMessage):
        """Send acknowledgment for received message."""
        try:
            ack_message = MeshMessage(
                type=MeshMessageType.DATA_MESSAGE,
                sender=self.node_id,
                recipient=original_message.sender,
                payload=json.dumps(
                    {
                        "type": "acknowledgment",
                        "original_message_id": original_message.id,
                        "timestamp": time.time(),
                    }
                ).encode(),
                ttl=3,
            )

            await self.mesh_network.send_message(ack_message)
            logger.debug(f"Sent acknowledgment for message {original_message.id}")

        except Exception as e:
            logger.warning(f"Failed to send acknowledgment: {e}")

    def _unified_to_mesh_message(self, unified_msg: UnifiedMessage) -> MeshMessage:
        """Convert UnifiedMessage to MeshMessage."""
        # Map message types
        type_mapping = {
            MessageType.DATA: MeshMessageType.DATA_MESSAGE,
            MessageType.AGENT_TASK: MeshMessageType.AGENT_TASK,
            MessageType.PARAMETER_UPDATE: MeshMessageType.PARAMETER_UPDATE,
            MessageType.GRADIENT_SHARE: MeshMessageType.GRADIENT_SHARING,
            MessageType.HEARTBEAT: MeshMessageType.HEARTBEAT,
            MessageType.DISCOVERY: MeshMessageType.DISCOVERY,
            MessageType.ROUTING: MeshMessageType.ROUTING_UPDATE,
        }

        mesh_type = type_mapping.get(unified_msg.message_type, MeshMessageType.DATA_MESSAGE)

        return MeshMessage(
            type=mesh_type,
            sender=unified_msg.metadata.sender_id or self.node_id,
            recipient=unified_msg.metadata.recipient_id,
            payload=unified_msg.payload,
            ttl=unified_msg.metadata.max_hops,
            id=unified_msg.metadata.message_id,
            timestamp=unified_msg.metadata.timestamp,
            hop_count=unified_msg.metadata.hop_count,
            route_path=unified_msg.metadata.route_path.copy(),
        )

    def _mesh_to_unified_message(self, mesh_msg: MeshMessage) -> UnifiedMessage:
        """Convert MeshMessage to UnifiedMessage."""
        # Map message types back
        type_mapping = {
            MeshMessageType.DATA_MESSAGE: MessageType.DATA,
            MeshMessageType.AGENT_TASK: MessageType.AGENT_TASK,
            MeshMessageType.PARAMETER_UPDATE: MessageType.PARAMETER_UPDATE,
            MeshMessageType.GRADIENT_SHARING: MessageType.GRADIENT_SHARE,
            MeshMessageType.HEARTBEAT: MessageType.HEARTBEAT,
            MeshMessageType.DISCOVERY: MessageType.DISCOVERY,
            MeshMessageType.ROUTING_UPDATE: MessageType.ROUTING,
        }

        unified_type = type_mapping.get(mesh_msg.type, MessageType.DATA)

        metadata = MessageMetadata(
            message_id=mesh_msg.id,
            timestamp=mesh_msg.timestamp,
            sender_id=mesh_msg.sender,
            recipient_id=mesh_msg.recipient,
            hop_count=mesh_msg.hop_count,
            max_hops=mesh_msg.ttl,
            route_path=mesh_msg.route_path.copy(),
        )

        return UnifiedMessage(
            message_type=unified_type,
            payload=mesh_msg.payload,
            metadata=metadata,
        )

    def _map_message_priority(self, unified_priority: MessagePriority) -> MessagePriority:
        """Map UnifiedMessage priority to delivery service priority."""
        return unified_priority  # They use the same enum

    def register_message_handler(self, handler: Any):
        """Register a message handler."""
        self.message_handlers.append(handler)
        logger.debug("Message handler registered for LibP2P transport")

    def add_peer(self, peer_address: str) -> bool:
        """Add a peer to the network."""
        try:
            # This would typically be called by the mesh network
            # when peer discovery finds new peers
            asyncio.create_task(self.mesh_network.add_peer(peer_address))
            return True
        except Exception as e:
            logger.error(f"Failed to add peer {peer_address}: {e}")
            return False

    def _update_capabilities(self):
        """Update transport capabilities based on mesh status."""
        if not self.mesh_network:
            return

        mesh_status = self.mesh_network.get_mesh_status()

        self.capabilities.is_available = mesh_status["status"] in ["active", "degraded"]
        self.capabilities.is_connected = mesh_status["peer_count"] > 0
        self.capabilities.peer_count = mesh_status["peer_count"]
        self.capabilities.last_activity = time.time()

        # Update error rate based on delivery success
        if self.delivery_service:
            delivery_status = self.delivery_service.get_delivery_status()
            success_rate = delivery_status["performance_metrics"]["success_rate"]
            self.capabilities.error_rate = 1.0 - success_rate
            self.stats["delivery_success_rate"] = success_rate

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive transport status."""
        self._update_capabilities()

        mesh_status = self.mesh_network.get_mesh_status() if self.mesh_network else {}
        security_status = self.security_manager.get_security_status()
        delivery_status = self.delivery_service.get_delivery_status() if self.delivery_service else {}

        return {
            "transport_type": "libp2p_mesh",
            "node_id": self.node_id,
            "running": self.running,
            "capabilities": {
                "supports_broadcast": self.capabilities.supports_broadcast,
                "supports_multicast": self.capabilities.supports_multicast,
                "supports_unicast": self.capabilities.supports_unicast,
                "max_message_size": self.capabilities.max_message_size,
                "is_offline_capable": self.capabilities.is_offline_capable,
                "requires_internet": self.capabilities.requires_internet,
                "is_available": self.capabilities.is_available,
                "is_connected": self.capabilities.is_connected,
                "peer_count": self.capabilities.peer_count,
                "error_rate": self.capabilities.error_rate,
            },
            "mesh_status": mesh_status,
            "security_status": security_status,
            "delivery_status": delivery_status,
            "statistics": self.stats.copy(),
            "configuration": {
                "enable_dht": self.config.enable_dht,
                "enable_gossipsub": self.config.enable_gossipsub,
                "enable_mdns": self.config.enable_mdns,
                "max_peers": self.config.max_peers,
                "listen_port": self.config.listen_port,
            },
        }

    def get_capabilities(self) -> TransportCapabilities:
        """Get transport capabilities."""
        self._update_capabilities()
        return self.capabilities


# Integration helper functions
def create_libp2p_transport(
    node_id: str,
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    enable_delivery_guarantees: bool = True,
    **mesh_config_kwargs,
) -> LibP2PTransport:
    """Create a LibP2P transport with recommended configuration."""

    # Create mesh configuration
    mesh_config = MeshConfiguration(
        node_id=node_id, enable_dht=True, enable_gossipsub=True, enable_mdns=True, max_peers=50, **mesh_config_kwargs
    )

    return LibP2PTransport(
        node_id=node_id,
        config=mesh_config,
        security_level=security_level,
        enable_delivery_guarantees=enable_delivery_guarantees,
    )


def get_libp2p_capabilities() -> TransportCapabilities:
    """Get default LibP2P transport capabilities."""
    return TransportCapabilities(
        supports_broadcast=True,
        supports_multicast=True,
        supports_unicast=True,
        max_message_size=1024 * 1024,  # 1MB
        is_offline_capable=True,
        requires_internet=False,
        typical_latency_ms=100,
        bandwidth_mbps=10.0,
        provides_encryption=True,
        supports_forward_secrecy=True,
        has_built_in_auth=True,
        battery_impact="medium",
        data_cost_impact="low",
        works_on_cellular=True,
        works_on_wifi=True,
    )


# Example usage
if __name__ == "__main__":

    async def test_libp2p_transport():
        """Test LibP2P transport integration."""
        transport = create_libp2p_transport(
            "test_node",
            security_level=SecurityLevel.STANDARD,
            enable_delivery_guarantees=True,
        )

        # Register a message handler
        async def handle_message(message: UnifiedMessage, transport_type: str):
            print(f"Received message via {transport_type}: {message.payload.decode()}")

        transport.register_message_handler(handle_message)

        # Start transport
        success = await transport.start()
        if success:
            print("LibP2P transport started successfully")

            # Get status
            status = transport.get_status()
            print(f"Transport status: {json.dumps(status, indent=2)}")

            # Keep running for a bit
            await asyncio.sleep(5)

            await transport.stop()
            print("LibP2P transport stopped")
        else:
            print("Failed to start LibP2P transport")

    asyncio.run(test_libp2p_transport())
