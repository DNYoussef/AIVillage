"""
Compatibility bridge for legacy P2P implementations.

Provides adapters and bridges to maintain compatibility with existing
P2P transport implementations while migrating to the unified system.
"""

import logging
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger(__name__)


class LegacyTransportBridge:
    """
    Bridge to legacy P2P transport implementations.

    Provides compatibility layer for existing implementations in:
    - src/core/p2p/
    - py/aivillage/p2p/
    - src/infrastructure/p2p/
    - src/production/communications/p2p/
    """

    def __init__(self, legacy_path: str, transport_type: str = "unknown"):
        self.legacy_path = legacy_path
        self.transport_type = transport_type
        self.legacy_module = None
        self.legacy_instance = None
        self.available = False

        logger.info(f"Legacy bridge initialized for {transport_type} at {legacy_path}")

    async def initialize(self) -> bool:
        """Initialize connection to legacy transport."""
        try:
            # Add legacy path to Python path
            legacy_dir = Path(self.legacy_path).parent
            if str(legacy_dir) not in sys.path:
                sys.path.insert(0, str(legacy_dir))

            # Try to import legacy module
            module_name = Path(self.legacy_path).stem
            self.legacy_module = __import__(module_name)

            # Try to find transport class
            transport_classes = [
                "TransportManager",
                "BitChatTransport",
                "BetanetTransport",
                "P2PTransport",
                "MeshNetwork",
            ]

            for class_name in transport_classes:
                if hasattr(self.legacy_module, class_name):
                    transport_class = getattr(self.legacy_module, class_name)
                    self.legacy_instance = transport_class()
                    self.available = True
                    logger.info(f"Connected to legacy {class_name} in {self.legacy_path}")
                    return True

            logger.warning(f"No compatible transport class found in {self.legacy_path}")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize legacy bridge: {e}")
            return False

    async def send_message(self, message_data: bytes, recipient: str = "") -> bool:
        """Send message via legacy transport."""
        if not self.available or not self.legacy_instance:
            return False

        try:
            # Try different legacy method signatures
            send_methods = ["send_message", "send", "broadcast", "transmit"]

            for method_name in send_methods:
                if hasattr(self.legacy_instance, method_name):
                    method = getattr(self.legacy_instance, method_name)

                    # Try different parameter combinations
                    try:
                        if recipient:
                            result = await method(message_data, recipient)
                        else:
                            result = await method(message_data)
                        return bool(result)
                    except TypeError:
                        # Try synchronous version
                        try:
                            if recipient:
                                result = method(message_data, recipient)
                            else:
                                result = method(message_data)
                            return bool(result)
                        except Exception:
                            continue

            logger.warning("No compatible send method found in legacy transport")
            return False

        except Exception as e:
            logger.error(f"Error sending via legacy transport: {e}")
            return False

    async def start(self) -> bool:
        """Start legacy transport."""
        if not self.available or not self.legacy_instance:
            return False

        try:
            start_methods = ["start", "initialize", "connect", "begin"]

            for method_name in start_methods:
                if hasattr(self.legacy_instance, method_name):
                    method = getattr(self.legacy_instance, method_name)
                    try:
                        result = await method()
                        return bool(result)
                    except TypeError:
                        # Try synchronous version
                        try:
                            result = method()
                            return bool(result)
                        except Exception:
                            continue

            # If no start method, assume already started
            return True

        except Exception as e:
            logger.error(f"Error starting legacy transport: {e}")
            return False

    async def stop(self) -> bool:
        """Stop legacy transport."""
        if not self.available or not self.legacy_instance:
            return True

        try:
            stop_methods = ["stop", "shutdown", "disconnect", "close"]

            for method_name in stop_methods:
                if hasattr(self.legacy_instance, method_name):
                    method = getattr(self.legacy_instance, method_name)
                    try:
                        result = await method()
                        return bool(result)
                    except TypeError:
                        # Try synchronous version
                        try:
                            result = method()
                            return bool(result)
                        except Exception:
                            continue

            return True

        except Exception as e:
            logger.error(f"Error stopping legacy transport: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get status from legacy transport."""
        status = {
            "available": self.available,
            "transport_type": self.transport_type,
            "legacy_path": self.legacy_path,
        }

        if self.available and self.legacy_instance:
            try:
                status_methods = ["get_status", "status", "info", "get_info"]

                for method_name in status_methods:
                    if hasattr(self.legacy_instance, method_name):
                        method = getattr(self.legacy_instance, method_name)
                        try:
                            legacy_status = method()
                            if isinstance(legacy_status, dict):
                                status.update(legacy_status)
                            break
                        except Exception:
                            continue

            except Exception as e:
                logger.warning(f"Error getting legacy status: {e}")

        return status


class LegacyAdapterManager:
    """Manager for multiple legacy transport adapters."""

    def __init__(self):
        self.adapters: dict[str, LegacyTransportBridge] = {}
        self.known_legacy_paths = [
            # BitChat implementations
            "src/core/p2p/bitchat_transport.py",
            "py/aivillage/p2p/bitchat_bridge.py",
            "src/core/p2p/bitchat_mvp_integration.py",
            # BetaNet implementations
            "src/core/p2p/betanet_htx_transport.py",
            "src/core/p2p/betanet_transport_v2.py",
            "src/hardware/protocols/betanet/htx_transport.py",
            # General P2P implementations
            "src/core/p2p/transport_manager_enhanced.py",
            "src/infrastructure/p2p/device_mesh.py",
            "src/production/communications/p2p/p2p_node.py",
        ]

    async def discover_legacy_transports(self) -> list[str]:
        """Discover available legacy transport implementations."""
        available_transports = []

        for legacy_path in self.known_legacy_paths:
            if Path(legacy_path).exists():
                # Determine transport type from path
                transport_type = "unknown"
                if "bitchat" in legacy_path.lower():
                    transport_type = "bitchat"
                elif "betanet" in legacy_path.lower():
                    transport_type = "betanet"
                elif "mesh" in legacy_path.lower():
                    transport_type = "mesh"
                elif "p2p" in legacy_path.lower():
                    transport_type = "p2p"

                # Create adapter
                adapter_id = f"{transport_type}_{Path(legacy_path).stem}"
                adapter = LegacyTransportBridge(legacy_path, transport_type)

                if await adapter.initialize():
                    self.adapters[adapter_id] = adapter
                    available_transports.append(adapter_id)
                    logger.info(f"Discovered legacy transport: {adapter_id}")

        return available_transports

    async def start_all(self) -> dict[str, bool]:
        """Start all available legacy transports."""
        results = {}

        for adapter_id, adapter in self.adapters.items():
            try:
                results[adapter_id] = await adapter.start()
            except Exception as e:
                logger.error(f"Error starting {adapter_id}: {e}")
                results[adapter_id] = False

        return results

    async def stop_all(self) -> dict[str, bool]:
        """Stop all legacy transports."""
        results = {}

        for adapter_id, adapter in self.adapters.items():
            try:
                results[adapter_id] = await adapter.stop()
            except Exception as e:
                logger.error(f"Error stopping {adapter_id}: {e}")
                results[adapter_id] = False

        return results

    def get_adapter(self, adapter_id: str) -> LegacyTransportBridge | None:
        """Get specific legacy adapter."""
        return self.adapters.get(adapter_id)

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status from all legacy adapters."""
        return {adapter_id: adapter.get_status() for adapter_id, adapter in self.adapters.items()}


# Factory functions
def create_legacy_bridge(legacy_path: str, transport_type: str = "unknown") -> LegacyTransportBridge:
    """Factory function to create legacy transport bridge."""
    return LegacyTransportBridge(legacy_path, transport_type)


def create_adapter_manager() -> LegacyAdapterManager:
    """Factory function to create legacy adapter manager."""
    return LegacyAdapterManager()


async def discover_and_bridge_legacy_transports() -> LegacyAdapterManager:
    """Convenience function to discover and initialize all legacy transports."""
    manager = LegacyAdapterManager()
    discovered = await manager.discover_legacy_transports()

    if discovered:
        logger.info(f"Discovered {len(discovered)} legacy transports: {discovered}")
        start_results = await manager.start_all()
        successful_starts = sum(1 for success in start_results.values() if success)
        logger.info(f"Started {successful_starts}/{len(discovered)} legacy transports")
    else:
        logger.info("No legacy transports discovered")

    return manager
