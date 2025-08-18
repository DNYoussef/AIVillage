"""
BitChat Bridge for Unified Transport System

Provides compatibility and bridge layer between the legacy BitChat transport
implementation and the new unified Python transport system.
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Add src to path for BitChat imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

try:
    # Try to import BitChat components directly
    # Avoid importing modules that depend on zeroconf
    import importlib.util

    # Check if the required modules exist and can be imported
    bitchat_transport_path = os.path.join(os.path.dirname(__file__), "../../../src/core/p2p/bitchat_transport.py")
    bitchat_framing_path = os.path.join(os.path.dirname(__file__), "../../../src/core/p2p/bitchat_framing.py")

    if os.path.exists(bitchat_transport_path) and os.path.exists(bitchat_framing_path):
        # Import directly to avoid __init__.py dependencies
        spec = importlib.util.spec_from_file_location("bitchat_transport", bitchat_transport_path)
        bitchat_transport_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bitchat_transport_module)

        spec = importlib.util.spec_from_file_location("bitchat_framing", bitchat_framing_path)
        bitchat_framing_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bitchat_framing_module)

        BitChatTransport = bitchat_transport_module.BitChatTransport
        BitChatMessage = bitchat_transport_module.BitChatMessage
        BitChatFrame = bitchat_framing_module.BitChatFrame
        BitChatFraming = bitchat_framing_module.BitChatFraming
        MessageType = bitchat_framing_module.MessageType

        BITCHAT_AVAILABLE = True
        logger.info("BitChat components imported successfully via direct module loading")
    else:
        raise ImportError("BitChat module files not found")

except Exception as e:
    logger.warning(f"BitChat components not available: {e}")

    # Fallback classes when BitChat is not available
    class BitChatTransport:
        def __init__(self, *args, **kwargs):
            pass

        async def send_message(self, *args, **kwargs):
            return False

    class BitChatMessage:
        def __init__(self, *args, **kwargs):
            pass

    class BitChatFrame:
        def __init__(self, *args, **kwargs):
            pass

    class BitChatFraming:
        def __init__(self, *args, **kwargs):
            pass

    class MessageType:
        DATA = 1
        CAPABILITY = 2
        HEARTBEAT = 3

    BITCHAT_AVAILABLE = False


@dataclass
class UnifiedMessage:
    """Unified message format for BitChat bridge"""

    recipient_id: str
    payload: bytes
    priority: int = 5
    ttl_hops: int = 7
    message_type: str = "data"
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BitChatTransportBridge:
    """Bridge between BitChat transport and unified transport system"""

    def __init__(self, device_id: str, **kwargs):
        self.device_id = device_id
        self.unified_handlers: dict[str, Callable] = {}

        # Initialize BitChat components if available
        if BITCHAT_AVAILABLE:
            self.transport = BitChatTransport(
                device_id=device_id,
                max_peers=kwargs.get("max_peers", 20),
            )

            # Initialize framing
            self.framing = BitChatFraming(
                enable_fec=kwargs.get("enable_fec", True), enable_compression=kwargs.get("enable_compression", True)
            )
        else:
            self.transport = None
            self.framing = None
            logger.warning("BitChat bridge created but BitChat not available")

    def register_unified_handler(self, message_type: str, handler: Callable):
        """Register handler for unified message format"""
        self.unified_handlers[message_type] = handler
        logger.debug(f"Registered unified handler for {message_type}")

    async def send_unified_message(
        self, recipient_id: str, payload: bytes, priority: int = 5, ttl_hops: int = 7
    ) -> bool:
        """Send message using unified message format"""
        if not BITCHAT_AVAILABLE or not self.transport:
            return False

        try:
            # Convert to BitChat format and send
            success = await self.transport.send_message(
                recipient=recipient_id, payload=payload, priority=priority, ttl=ttl_hops
            )
            return success
        except Exception as e:
            logger.error(f"Error sending unified message: {e}")
            return False

    async def start(self) -> bool:
        """Start the BitChat transport bridge"""
        if not BITCHAT_AVAILABLE or not self.transport:
            logger.warning("Cannot start BitChat bridge - BitChat not available")
            return False

        try:
            # Start BitChat transport
            result = await self.transport.start()
            if result:
                logger.info("BitChat bridge started successfully")
            return result
        except Exception as e:
            logger.error(f"Error starting BitChat bridge: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the BitChat transport bridge"""
        if not BITCHAT_AVAILABLE or not self.transport:
            return True

        try:
            result = await self.transport.stop()
            logger.info("BitChat bridge stopped")
            return result
        except Exception as e:
            logger.error(f"Error stopping BitChat bridge: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get bridge status information"""
        status = {
            "available": BITCHAT_AVAILABLE,
            "device_id": self.device_id,
            "transport_active": False,
            "registered_handlers": len(self.unified_handlers),
        }

        if BITCHAT_AVAILABLE and self.transport:
            try:
                transport_status = self.transport.get_status()
                status.update(
                    {
                        "transport_active": transport_status.get("active", False),
                        "peer_count": transport_status.get("peer_count", 0),
                        "message_count": transport_status.get("message_count", 0),
                    }
                )
            except Exception as e:
                logger.warning(f"Error getting transport status: {e}")

        return status


def create_bitchat_bridge(device_id: str, **kwargs) -> BitChatTransportBridge:
    """Factory function to create BitChat bridge instance"""
    return BitChatTransportBridge(device_id, **kwargs)


# Bridge availability check
def is_bitchat_available() -> bool:
    """Check if BitChat bridge is available"""
    return BITCHAT_AVAILABLE
