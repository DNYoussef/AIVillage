"""
Edge Device P2P Integration Bridge

Integrates edge device management with the unified P2P transport system.
Provides seamless communication between edge devices using the consolidated
P2P infrastructure (BitChat, BetaNet, QUIC).
"""

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.edge_manager import EdgeDevice, EdgeManager

# Add P2P package to path for integration
p2p_path = Path(__file__).parent.parent.parent / "p2p"
if str(p2p_path) not in sys.path:
    sys.path.insert(0, str(p2p_path))

try:
    from core.message_types import MessagePriority, MessageType, UnifiedMessage
    from core.transport_manager import TransportManager, TransportPriority, TransportType

    P2P_AVAILABLE = True
except ImportError:
    # Fallback when P2P system not available
    P2P_AVAILABLE = False
    TransportManager = None
    TransportType = None

logger = logging.getLogger(__name__)


class EdgeP2PBridge:
    """
    Bridge between edge device system and P2P transport system

    Enables edge devices to communicate via the unified P2P transport layer
    with mobile-optimized routing decisions.
    """

    def __init__(self, edge_manager: "EdgeManager", transport_manager: TransportManager | None = None):
        self.edge_manager = edge_manager
        self.transport_manager = transport_manager
        self.device_transport_mappings: dict[str, str] = {}

        # Integration status
        self.p2p_available = P2P_AVAILABLE and transport_manager is not None

        if self.p2p_available:
            logger.info("Edge-P2P integration bridge initialized with full P2P support")
        else:
            logger.warning("Edge-P2P integration bridge initialized in fallback mode (P2P not available)")

    async def initialize_p2p_for_device(self, device_id: str) -> bool:
        """Initialize P2P transport for an edge device"""

        if not self.p2p_available:
            logger.warning(f"P2P not available for device {device_id}")
            return False

        device = self.edge_manager.devices.get(device_id)
        if not device:
            logger.error(f"Device {device_id} not found in edge manager")
            return False

        # Create device context for P2P transport selection
        device_context = self._create_device_context(device)

        # Update transport manager with device context
        self.transport_manager.update_device_context(**device_context)

        # Set transport priority based on device capabilities
        transport_priority = self._determine_transport_priority(device)
        self.transport_manager.transport_priority = transport_priority

        # Register device with transport manager
        transport_id = f"edge_{device_id}"
        self.device_transport_mappings[device_id] = transport_id

        logger.info(f"Initialized P2P transport for device {device_id} with priority {transport_priority.value}")
        return True

    def _create_device_context(self, device: "EdgeDevice") -> dict[str, Any]:
        """Create P2P device context from edge device capabilities"""

        caps = device.capabilities

        return {
            "battery_level": caps.battery_percent / 100.0 if caps.battery_percent else None,
            "is_charging": caps.battery_charging,
            "power_save_mode": caps.battery_percent and caps.battery_percent < 20,
            "network_type": caps.network_type,
            "has_internet": caps.has_internet,
            "is_metered_connection": caps.is_metered_connection,
            "is_foreground": True,  # Assume foreground for edge devices
            "supports_bluetooth": caps.supports_ble,
            "supports_wifi_direct": caps.supports_nearby,
            "max_concurrent_connections": caps.max_concurrent_tasks,
        }

    def _determine_transport_priority(self, device: "EdgeDevice") -> "TransportPriority":
        """Determine optimal transport priority for device"""

        if not P2P_AVAILABLE:
            return None

        caps = device.capabilities

        # Mobile devices prefer offline-first
        if device.device_type.value in ["smartphone", "tablet"]:
            return TransportPriority.OFFLINE_FIRST

        # Battery-powered devices prefer battery-aware routing
        if caps.battery_powered:
            if caps.battery_percent and caps.battery_percent < 30:
                return TransportPriority.BATTERY_AWARE
            return TransportPriority.OFFLINE_FIRST

        # High-performance devices can use performance-first
        if caps.cpu_cores >= 8 and caps.ram_total_mb >= 16384:
            return TransportPriority.PERFORMANCE_FIRST

        # Default to adaptive for other devices
        return TransportPriority.ADAPTIVE

    async def send_edge_message(
        self, from_device_id: str, to_device_id: str, message_type: str, payload: bytes, priority: str = "normal"
    ) -> bool:
        """Send message between edge devices via P2P transport"""

        if not self.p2p_available:
            logger.warning("Cannot send edge message: P2P not available")
            return False

        # Map priority to P2P message priority
        priority_mapping = {
            "low": MessagePriority.LOW,
            "normal": MessagePriority.NORMAL,
            "high": MessagePriority.HIGH,
            "critical": MessagePriority.CRITICAL,
        }

        p2p_priority = priority_mapping.get(priority, MessagePriority.NORMAL)

        # Create unified message with proper metadata
        from core.message_types import MessageMetadata

        metadata = MessageMetadata(
            priority=p2p_priority,
            timestamp=time.time(),
            sender_id=from_device_id,
            recipient_id=to_device_id,
            correlation_id=f"edge_{from_device_id}_{to_device_id}_{int(time.time())}",
        )

        message = UnifiedMessage(
            message_type=MessageType.DATA, payload=payload, metadata=metadata  # Edge messages are data messages
        )

        # Add edge-specific metadata as custom headers
        message.metadata.custom_headers = {
            "from_device": from_device_id,
            "to_device": to_device_id,
            "edge_message_type": message_type,
        }

        # Send via transport manager
        try:
            success = await self.transport_manager.send_message(message)
            if success:
                logger.info(f"Sent edge message from {from_device_id} to {to_device_id}")
            else:
                logger.warning(f"Failed to send edge message from {from_device_id} to {to_device_id}")
            return success
        except Exception as e:
            logger.error(f"Error sending edge message: {e}")
            return False

    async def broadcast_edge_message(
        self, from_device_id: str, message_type: str, payload: bytes, priority: str = "normal"
    ) -> bool:
        """Broadcast message from edge device to all connected devices"""

        if not self.p2p_available:
            logger.warning("Cannot broadcast edge message: P2P not available")
            return False

        # Create broadcast message with proper metadata
        from core.message_types import MessageMetadata

        metadata = MessageMetadata(
            priority=MessagePriority.NORMAL,
            timestamp=time.time(),
            sender_id=from_device_id,
            correlation_id=f"edge_broadcast_{from_device_id}_{int(time.time())}",
        )

        message = UnifiedMessage(message_type=MessageType.DATA, payload=payload, metadata=metadata)

        # Add edge-specific metadata as custom headers
        message.metadata.custom_headers = {
            "from_device": from_device_id,
            "edge_message_type": message_type,
            "broadcast": True,
        }

        try:
            success = await self.transport_manager.send_message(message)
            if success:
                logger.info(f"Broadcast edge message from {from_device_id}")
            return success
        except Exception as e:
            logger.error(f"Error broadcasting edge message: {e}")
            return False

    def register_edge_message_handler(self, handler_func):
        """Register handler for incoming edge messages"""

        if not self.p2p_available:
            logger.warning("Cannot register edge message handler: P2P not available")
            return

        async def edge_message_wrapper(message: UnifiedMessage, transport_type: TransportType):
            """Wrapper to handle edge-specific message processing"""

            # Check if this is an edge message
            if (
                message.metadata
                and hasattr(message.metadata, "custom_headers")
                and message.metadata.custom_headers
                and "edge_message_type" in message.metadata.custom_headers
                and "from_device" in message.metadata.custom_headers
            ):
                custom = message.metadata.custom_headers

                # Extract edge message details
                edge_data = {
                    "from_device": custom["from_device"],
                    "to_device": custom.get("to_device"),
                    "edge_message_type": custom["edge_message_type"],
                    "payload": message.payload,
                    "transport_used": transport_type.value,
                    "is_broadcast": custom.get("broadcast", False),
                }

                # Call the registered handler
                try:
                    await handler_func(edge_data)
                except Exception as e:
                    logger.error(f"Error in edge message handler: {e}")

        # Register wrapper with transport manager
        self.transport_manager.register_message_handler(edge_message_wrapper)
        logger.info("Registered edge message handler with P2P transport system")

    async def sync_device_states(self) -> dict[str, Any]:
        """Sync edge device states with P2P transport system"""

        if not self.p2p_available:
            return {"error": "P2P not available"}

        sync_results = {}

        for device_id, device in self.edge_manager.devices.items():
            # Update device context in transport manager
            device_context = self._create_device_context(device)
            self.transport_manager.update_device_context(**device_context)

            # Get transport status
            transport_status = self.transport_manager.get_status()

            sync_results[device_id] = {
                "device_context_updated": True,
                "available_transports": transport_status.get("available_transports", []),
                "transport_priority": transport_status.get("transport_priority"),
            }

        logger.info(f"Synced {len(sync_results)} edge devices with P2P transport system")
        return sync_results

    def get_integration_status(self) -> dict[str, Any]:
        """Get status of edge-P2P integration"""

        status = {
            "p2p_available": self.p2p_available,
            "integrated_devices": len(self.device_transport_mappings),
            "device_mappings": self.device_transport_mappings.copy(),
        }

        if self.p2p_available and self.transport_manager:
            transport_status = self.transport_manager.get_status()
            status.update(
                {
                    "transport_status": transport_status,
                    "active_transports": transport_status.get("available_transports", []),
                }
            )

        return status

    async def optimize_transport_for_deployment(self, deployment_id: str, device_id: str) -> dict[str, Any]:
        """Optimize P2P transport settings for a specific deployment"""

        if not self.p2p_available:
            return {"error": "P2P not available"}

        deployment = self.edge_manager.deployments.get(deployment_id)
        device = self.edge_manager.devices.get(device_id)

        if not deployment or not device:
            return {"error": "Deployment or device not found"}

        # Get mobile resource manager for optimization
        try:
            from ..mobile.resource_manager import MobileDeviceProfile, MobileResourceManager

            mobile_manager = MobileResourceManager()

            # Create mobile profile from device capabilities
            mobile_profile = MobileDeviceProfile(
                timestamp=time.time(),
                device_id=device_id,
                battery_percent=device.capabilities.battery_percent,
                battery_charging=device.capabilities.battery_charging,
                cpu_temp_celsius=device.capabilities.cpu_temp_celsius,
                cpu_percent=50.0,  # Estimate
                ram_used_mb=device.capabilities.ram_total_mb - device.capabilities.ram_available_mb,
                ram_available_mb=device.capabilities.ram_available_mb,
                ram_total_mb=device.capabilities.ram_total_mb,
                network_type=device.capabilities.network_type,
                device_type=device.device_type.value,
            )

            # Get transport routing decision
            routing_decision = await mobile_manager.get_transport_routing_decision(
                message_size_bytes=deployment.compressed_size_mb * 1024 * 1024,  # Convert to bytes
                priority=deployment.priority,
                profile=mobile_profile,
            )

            return {
                "deployment_id": deployment_id,
                "device_id": device_id,
                "recommended_transport": routing_decision["primary_transport"],
                "fallback_transport": routing_decision["fallback_transport"],
                "chunk_size": routing_decision["chunk_size"],
                "rationale": routing_decision["rationale"],
                "estimated_cost": routing_decision["estimated_cost"],
                "estimated_latency": routing_decision["estimated_latency"],
            }

        except ImportError:
            return {"error": "Mobile resource manager not available"}
        except Exception as e:
            return {"error": f"Optimization failed: {e}"}


# Compatibility exports for when P2P is not available
class FallbackTransportManager:
    """Fallback transport manager when P2P system not available"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.available = False
        logger.warning("Using fallback transport manager - P2P functionality limited")

    async def send_message(self, message) -> bool:
        logger.warning("Cannot send message: P2P system not available")
        return False

    def register_message_handler(self, handler):
        logger.warning("Cannot register handler: P2P system not available")

    def get_status(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "available_transports": [], "error": "P2P system not available"}


def create_edge_p2p_bridge(edge_manager: "EdgeManager") -> EdgeP2PBridge:
    """Factory function to create edge-P2P bridge with proper fallbacks"""

    if P2P_AVAILABLE:
        try:
            # Try to create transport manager
            transport_manager = TransportManager(device_id="edge_bridge", transport_priority=TransportPriority.ADAPTIVE)
            return EdgeP2PBridge(edge_manager, transport_manager)
        except Exception as e:
            logger.warning(f"Failed to create transport manager: {e}")

    # Fallback bridge without P2P
    return EdgeP2PBridge(edge_manager, None)
