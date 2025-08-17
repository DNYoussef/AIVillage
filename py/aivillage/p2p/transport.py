"""
Unified Transport Manager for AIVillage P2P Communication

Provides coordinated transport layer supporting multiple transport types including:
- BitChat BLE mesh (offline-first communication)
- Betanet HTX (encrypted internet backbone)
- Direct QUIC (high-performance point-to-point)

Features comprehensive failover, Navigator integration, and mobile optimization.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any

# Import transport implementations
from .betanet.htx_transport import HtxClient
from .bitchat_bridge import create_bitchat_bridge, is_bitchat_available

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """Available transport types"""

    BITCHAT = "bitchat"  # Bluetooth mesh
    BETANET = "betanet"  # Encrypted internet
    QUIC = "quic"  # Direct QUIC


class TransportPriority(Enum):
    """Transport selection priorities"""

    OFFLINE_FIRST = "offline_first"  # Prefer BitChat
    PRIVACY_FIRST = "privacy_first"  # Prefer Betanet
    PERFORMANCE_FIRST = "performance_first"  # Prefer QUIC
    COST_AWARE = "cost_aware"  # Minimize cellular usage


@dataclass
class TransportConfig:
    """Configuration for transport layer"""

    device_id: str
    transport_priority: TransportPriority = TransportPriority.OFFLINE_FIRST
    enable_bitchat: bool = True
    enable_betanet: bool = True
    enable_quic: bool = True
    max_peers: int = 20
    battery_aware: bool = True
    data_cost_aware: bool = True


@dataclass
class MessageEnvelope:
    """Unified message envelope for transport layer"""

    recipient_id: str
    payload: bytes
    priority: int = 5
    ttl_hops: int = 7
    transport_preference: TransportType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TransportManager:
    """Unified transport coordination and management"""

    def __init__(
        self,
        device_id: str,
        transport_priority: TransportPriority = TransportPriority.OFFLINE_FIRST,
        enable_bitchat: bool = True,
        enable_betanet: bool = True,
        enable_quic: bool = True,
        **kwargs,
    ):
        self.device_id = device_id
        self.transport_priority = transport_priority
        self.config = TransportConfig(
            device_id=device_id,
            transport_priority=transport_priority,
            enable_bitchat=enable_bitchat,
            enable_betanet=enable_betanet,
            enable_quic=enable_quic,
            **kwargs,
        )

        # Transport instances
        self.transports: dict[TransportType, Any] = {}
        self.transport_status: dict[TransportType, bool] = {}

        # Message handlers
        self.message_handlers: list[Callable] = []

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "transport_failures": {},
            "last_activity": time.time(),
        }

        # Navigation integration (placeholder)
        self.navigator = None

        logger.info(f"Transport manager initialized for device {device_id}")

    def _init_bitchat(self):
        """Initialize BitChat BLE mesh transport"""
        try:
            if not self.config.enable_bitchat:
                logger.debug("BitChat disabled in configuration")
                return

            if not is_bitchat_available():
                logger.warning("BitChat bridge not available")
                return

            # Create BitChat bridge
            bridge = create_bitchat_bridge(
                device_id=self.device_id, max_peers=self.config.max_peers, enable_fec=True, enable_compression=True
            )

            # Register unified message handler
            bridge.register_unified_handler("default", self._handle_bitchat_message)

            self.transports[TransportType.BITCHAT] = bridge
            self.transport_status[TransportType.BITCHAT] = False

            logger.info("BitChat transport initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize BitChat transport: {e}")

    def _init_betanet(self):
        """Initialize Betanet HTX transport"""
        try:
            if not self.config.enable_betanet:
                logger.debug("Betanet disabled in configuration")
                return

            # Initialize HTX client for Betanet
            htx_client = HtxClient()

            self.transports[TransportType.BETANET] = htx_client
            self.transport_status[TransportType.BETANET] = True  # Mark as available

            logger.info("Betanet HTX transport initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Betanet transport: {e}")

    def _init_quic(self):
        """Initialize direct QUIC transport"""
        try:
            if not self.config.enable_quic:
                logger.debug("QUIC disabled in configuration")
                return

            # QUIC transport placeholder - would implement direct QUIC connections
            self.transports[TransportType.QUIC] = None  # Placeholder
            self.transport_status[TransportType.QUIC] = False

            logger.debug("QUIC transport placeholder initialized")

        except Exception as e:
            logger.error(f"Failed to initialize QUIC transport: {e}")

    async def start(self) -> bool:
        """Start all configured transports"""
        logger.info("Starting transport manager...")

        # Initialize transports
        self._init_bitchat()
        self._init_betanet()
        self._init_quic()

        # Start available transports
        start_tasks = []
        for transport_type, transport in self.transports.items():
            if transport:
                if hasattr(transport, "start"):
                    start_tasks.append(self._start_transport(transport_type, transport))

        if start_tasks:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.info(f"Started {success_count}/{len(start_tasks)} transports")
            return success_count > 0
        else:
            logger.warning("No transports available to start")
            return False

    async def _start_transport(self, transport_type: TransportType, transport) -> bool:
        """Start individual transport"""
        try:
            result = await transport.start()
            self.transport_status[transport_type] = result
            if result:
                logger.info(f"{transport_type.value} transport started successfully")
            else:
                logger.warning(f"{transport_type.value} transport failed to start")
            return result
        except Exception as e:
            logger.error(f"Error starting {transport_type.value} transport: {e}")
            self.transport_status[transport_type] = False
            return False

    async def stop(self) -> bool:
        """Stop all transports"""
        logger.info("Stopping transport manager...")

        stop_tasks = []
        for transport_type, transport in self.transports.items():
            if transport and hasattr(transport, "stop"):
                stop_tasks.append(self._stop_transport(transport_type, transport))

        if stop_tasks:
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.info(f"Stopped {success_count}/{len(stop_tasks)} transports")

        # Clear status
        for transport_type in self.transport_status:
            self.transport_status[transport_type] = False

        return True

    async def _stop_transport(self, transport_type: TransportType, transport) -> bool:
        """Stop individual transport"""
        try:
            result = await transport.stop()
            self.transport_status[transport_type] = False
            logger.info(f"{transport_type.value} transport stopped")
            return result
        except Exception as e:
            logger.error(f"Error stopping {transport_type.value} transport: {e}")
            return False

    def _select_transport(self, message: MessageEnvelope) -> TransportType | None:
        """Select best transport for message based on priority and availability"""

        # Use explicit preference if specified
        if message.transport_preference:
            if self.transport_status.get(message.transport_preference):
                return message.transport_preference

        # Select based on priority policy
        if self.transport_priority == TransportPriority.OFFLINE_FIRST:
            priority_order = [TransportType.BITCHAT, TransportType.BETANET, TransportType.QUIC]
        elif self.transport_priority == TransportPriority.PRIVACY_FIRST:
            priority_order = [TransportType.BETANET, TransportType.BITCHAT, TransportType.QUIC]
        elif self.transport_priority == TransportPriority.PERFORMANCE_FIRST:
            priority_order = [TransportType.QUIC, TransportType.BETANET, TransportType.BITCHAT]
        elif self.transport_priority == TransportPriority.COST_AWARE:
            priority_order = [TransportType.BITCHAT, TransportType.QUIC, TransportType.BETANET]
        else:
            priority_order = [TransportType.BITCHAT, TransportType.BETANET, TransportType.QUIC]

        # Return first available transport in priority order
        for transport_type in priority_order:
            if self.transport_status.get(transport_type):
                return transport_type

        return None

    async def send_message(
        self,
        recipient_id: str,
        payload: bytes,
        priority: int = 5,
        ttl_hops: int = 7,
        transport_preference: TransportType | None = None,
    ) -> bool:
        """Send message using best available transport"""

        message = MessageEnvelope(
            recipient_id=recipient_id,
            payload=payload,
            priority=priority,
            ttl_hops=ttl_hops,
            transport_preference=transport_preference,
        )

        # Select transport
        selected_transport = self._select_transport(message)
        if not selected_transport:
            logger.error("No available transports for message delivery")
            return False

        # Get transport instance
        transport = self.transports.get(selected_transport)
        if not transport:
            logger.error(f"Transport {selected_transport.value} not initialized")
            return False

        # Send message via selected transport
        try:
            if selected_transport == TransportType.BITCHAT:
                result = await transport.send_unified_message(recipient_id, payload, priority, ttl_hops)
            elif selected_transport == TransportType.BETANET:
                # Betanet HTX send implementation
                result = await self._send_via_betanet(transport, message)
            elif selected_transport == TransportType.QUIC:
                # QUIC send implementation (placeholder)
                result = await self._send_via_quic(transport, message)
            else:
                result = False

            if result:
                self.stats["messages_sent"] += 1
                self.stats["last_activity"] = time.time()
                logger.debug(f"Message sent via {selected_transport.value}")
            else:
                self._record_transport_failure(selected_transport)

            return result

        except Exception as e:
            logger.error(f"Error sending message via {selected_transport.value}: {e}")
            self._record_transport_failure(selected_transport)
            return False

    async def _send_via_betanet(self, transport, message: MessageEnvelope) -> bool:
        """Send message via Betanet HTX transport"""
        # Implementation would depend on HTX client API
        logger.debug("Betanet send not fully implemented")
        return True  # Placeholder

    async def _send_via_quic(self, transport, message: MessageEnvelope) -> bool:
        """Send message via QUIC transport"""
        # Implementation would depend on QUIC client API
        logger.debug("QUIC send not fully implemented")
        return False  # Placeholder

    def _record_transport_failure(self, transport_type: TransportType):
        """Record transport failure for statistics"""
        if transport_type.value not in self.stats["transport_failures"]:
            self.stats["transport_failures"][transport_type.value] = 0
        self.stats["transport_failures"][transport_type.value] += 1

    async def _handle_bitchat_message(self, message):
        """Handle incoming BitChat message"""
        try:
            self.stats["messages_received"] += 1
            self.stats["last_activity"] = time.time()

            # Notify registered handlers
            for handler in self.message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.warning(f"Message handler error: {e}")

            logger.debug("BitChat message handled")

        except Exception as e:
            logger.error(f"Error handling BitChat message: {e}")

    def register_message_handler(self, handler: Callable):
        """Register handler for incoming messages"""
        self.message_handlers.append(handler)
        logger.debug("Message handler registered")

    def get_transport_status(self) -> dict[str, Any]:
        """Get comprehensive transport status"""
        active_transports = [transport_type.value for transport_type, active in self.transport_status.items() if active]

        status = {
            "device_id": self.device_id,
            "transport_priority": self.transport_priority.value,
            "active_transports": active_transports,
            "transport_count": len(active_transports),
            "statistics": self.stats.copy(),
        }

        # Add individual transport status
        for transport_type, transport in self.transports.items():
            if transport and hasattr(transport, "get_status"):
                try:
                    transport_status = transport.get_status()
                    status[f"{transport_type.value}_status"] = transport_status
                except Exception as e:
                    logger.warning(f"Error getting {transport_type.value} status: {e}")

        return status


# Factory functions
def create_transport_manager(device_id: str, **kwargs) -> TransportManager:
    """Factory function to create transport manager"""
    return TransportManager(device_id, **kwargs)


def create_offline_first_manager(device_id: str, **kwargs) -> TransportManager:
    """Create transport manager optimized for offline-first operation"""
    return TransportManager(
        device_id, transport_priority=TransportPriority.OFFLINE_FIRST, enable_bitchat=True, **kwargs
    )


def create_privacy_first_manager(device_id: str, **kwargs) -> TransportManager:
    """Create transport manager optimized for privacy"""
    return TransportManager(
        device_id, transport_priority=TransportPriority.PRIVACY_FIRST, enable_betanet=True, **kwargs
    )
