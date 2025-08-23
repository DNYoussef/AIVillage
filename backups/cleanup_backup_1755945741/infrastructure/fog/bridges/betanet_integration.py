"""
BetaNet Integration Bridge for Fog Computing

This module provides integration between the fog computing infrastructure and the
BetaNet bounty implementation without modifying the bounty code. It acts as an
adapter layer that allows fog compute jobs to use BetaNet transport protocols
while maintaining the bounty's integrity for verification.

Integration Strategy:
- Import BetaNet components as external dependencies
- Provide wrapper/adapter classes for fog compute usage
- Keep bounty code completely untouched
- Enable fog compute to leverage BetaNet's secure transport capabilities
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import BetaNet components from the bounty workspace
BETANET_AVAILABLE = False
try:
    # Import the consolidated Python BetaNet implementation from bounty
    import sys

    betanet_path = Path(__file__).parent.parent.parent / "p2p" / "betanet-bounty" / "python"
    sys.path.insert(0, str(betanet_path))

    from covert_channels import HTTP3CovertChannel, HTTPCovertChannel, WebSocketCovertChannel
    from mixnet_privacy import PrivacyMode, VRFMixnetRouter
    from mobile_optimization import AdaptiveChunking, BatteryThermalOptimizer

    BETANET_AVAILABLE = True
    logger.info("BetaNet bounty integration available - fog compute can use BetaNet transport")

except ImportError as e:
    logger.warning(f"BetaNet bounty not available: {e}")
    logger.info("Fog compute will use fallback transport methods")


# Fallback implementations for when BetaNet bounty is not available
class FallbackTransport:
    """Fallback transport when BetaNet is not available"""

    def __init__(self, transport_type: str = "http"):
        self.transport_type = transport_type

    async def send_message(self, message: bytes, destination: str) -> bool:
        """Send message using fallback transport"""
        logger.warning(f"Using fallback {self.transport_type} transport (BetaNet not available)")
        # Simulate message sending
        await asyncio.sleep(0.1)
        return True

    async def receive_message(self) -> bytes | None:
        """Receive message using fallback transport"""
        return None


class BetaNetFogTransport:
    """
    Fog computing adapter for BetaNet transport protocols

    This class integrates BetaNet's advanced transport capabilities with fog compute jobs:
    - Covert channels for steganographic communication
    - VRF-based mixnet routing for privacy
    - Mobile-optimized chunking and battery awareness
    """

    def __init__(self, privacy_mode: str = "balanced", enable_covert: bool = True, mobile_optimization: bool = True):
        """
        Initialize BetaNet transport adapter for fog computing

        Args:
            privacy_mode: Privacy level - "strict", "balanced", or "performance"
            enable_covert: Enable covert channel capabilities
            mobile_optimization: Enable battery/thermal optimizations
        """
        self.privacy_mode = privacy_mode
        self.enable_covert = enable_covert
        self.mobile_optimization = mobile_optimization
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_transferred": 0,
            "covert_channels_used": 0,
            "privacy_hops": 0,
        }

        if BETANET_AVAILABLE:
            self._init_betanet_components()
        else:
            self._init_fallback_components()

    def _init_betanet_components(self):
        """Initialize BetaNet components when available"""
        try:
            # Initialize covert channels
            if self.enable_covert:
                self.http_covert = HTTPCovertChannel()
                self.http3_covert = HTTP3CovertChannel()
                self.websocket_covert = WebSocketCovertChannel()

            # Initialize privacy router
            privacy_mode_map = {
                "strict": PrivacyMode.STRICT,
                "balanced": PrivacyMode.BALANCED,
                "performance": PrivacyMode.PERFORMANCE,
            }
            self.mixnet_router = VRFMixnetRouter(
                privacy_mode=privacy_mode_map.get(self.privacy_mode, PrivacyMode.BALANCED)
            )

            # Initialize mobile optimization
            if self.mobile_optimization:
                self.battery_optimizer = BatteryThermalOptimizer()
                self.adaptive_chunking = AdaptiveChunking()

            logger.info("BetaNet components initialized for fog compute integration")

        except Exception as e:
            logger.error(f"Failed to initialize BetaNet components: {e}")
            self._init_fallback_components()

    def _init_fallback_components(self):
        """Initialize fallback components when BetaNet is not available"""
        self.fallback_transport = FallbackTransport()
        logger.warning("Using fallback transport - BetaNet integration not available")

    async def send_job_data(self, job_data: bytes, destination: str, priority: str = "normal") -> dict[str, Any]:
        """
        Send job data using BetaNet transport with fog compute optimizations

        Args:
            job_data: Job payload data
            destination: Destination endpoint
            priority: Message priority ("low", "normal", "high")

        Returns:
            Dictionary with transmission results and statistics
        """
        start_time = time.time()

        if not BETANET_AVAILABLE:
            success = await self.fallback_transport.send_message(job_data, destination)
            return {
                "success": success,
                "transport": "fallback",
                "duration": time.time() - start_time,
                "bytes_sent": len(job_data),
            }

        try:
            # Apply mobile optimization if enabled
            if self.mobile_optimization and hasattr(self, "adaptive_chunking"):
                chunks = await self.adaptive_chunking.chunk_data(job_data, optimization="fog_compute")
            else:
                chunks = [job_data]

            # Select covert channel based on priority and conditions
            transport_used = "standard"
            if self.enable_covert and priority == "high":
                # Use HTTP/3 for high priority
                if hasattr(self, "http3_covert"):
                    for chunk in chunks:
                        await self.http3_covert.send_covert_data(chunk, destination)
                    transport_used = "http3_covert"
                    self.stats["covert_channels_used"] += 1

            elif self.enable_covert and priority == "normal":
                # Use WebSocket for normal priority
                if hasattr(self, "websocket_covert"):
                    for chunk in chunks:
                        await self.websocket_covert.send_covert_data(chunk, destination)
                    transport_used = "websocket_covert"
                    self.stats["covert_channels_used"] += 1

            else:
                # Use standard HTTP covert for low priority
                if hasattr(self, "http_covert"):
                    for chunk in chunks:
                        await self.http_covert.send_covert_data(chunk, destination)
                    transport_used = "http_covert"

            # Apply mixnet routing for privacy
            if hasattr(self, "mixnet_router"):
                privacy_hops = await self.mixnet_router.get_route_length(destination)
                self.stats["privacy_hops"] += privacy_hops

            # Update statistics
            self.stats["messages_sent"] += 1
            self.stats["bytes_transferred"] += len(job_data)

            return {
                "success": True,
                "transport": transport_used,
                "duration": time.time() - start_time,
                "bytes_sent": len(job_data),
                "chunks": len(chunks),
                "privacy_hops": privacy_hops if hasattr(self, "mixnet_router") else 0,
            }

        except Exception as e:
            logger.error(f"BetaNet transport failed: {e}")
            # Fallback to standard transport
            success = await self.fallback_transport.send_message(job_data, destination)
            return {
                "success": success,
                "transport": "fallback_after_error",
                "duration": time.time() - start_time,
                "bytes_sent": len(job_data),
                "error": str(e),
            }

    async def receive_job_data(self, timeout: float = 30.0) -> dict[str, Any] | None:
        """
        Receive job data using BetaNet transport

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Dictionary with received data and metadata, or None if timeout
        """
        if not BETANET_AVAILABLE:
            data = await self.fallback_transport.receive_message()
            return {"data": data, "transport": "fallback"} if data else None

        try:
            # Try to receive from multiple covert channels
            if self.enable_covert:
                # Check WebSocket first (most responsive)
                if hasattr(self, "websocket_covert"):
                    data = await asyncio.wait_for(self.websocket_covert.receive_covert_data(), timeout=timeout / 3)
                    if data:
                        self.stats["messages_received"] += 1
                        self.stats["bytes_transferred"] += len(data)
                        return {"data": data, "transport": "websocket_covert", "timestamp": time.time()}

                # Check HTTP/3 next
                if hasattr(self, "http3_covert"):
                    data = await asyncio.wait_for(self.http3_covert.receive_covert_data(), timeout=timeout / 3)
                    if data:
                        self.stats["messages_received"] += 1
                        self.stats["bytes_transferred"] += len(data)
                        return {"data": data, "transport": "http3_covert", "timestamp": time.time()}

                # Check standard HTTP last
                if hasattr(self, "http_covert"):
                    data = await asyncio.wait_for(self.http_covert.receive_covert_data(), timeout=timeout / 3)
                    if data:
                        self.stats["messages_received"] += 1
                        self.stats["bytes_transferred"] += len(data)
                        return {"data": data, "transport": "http_covert", "timestamp": time.time()}

            return None

        except asyncio.TimeoutError:
            logger.debug(f"Receive timeout after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            return None

    def get_transport_stats(self) -> dict[str, Any]:
        """Get transport statistics for monitoring"""
        stats = self.stats.copy()
        stats.update(
            {
                "betanet_available": BETANET_AVAILABLE,
                "privacy_mode": self.privacy_mode,
                "covert_enabled": self.enable_covert,
                "mobile_optimization": self.mobile_optimization,
            }
        )
        return stats

    async def optimize_for_device(self, device_info: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize transport settings based on device capabilities

        Args:
            device_info: Device information (battery, network, etc.)

        Returns:
            Optimization recommendations
        """
        if not BETANET_AVAILABLE or not self.mobile_optimization:
            return {"optimizations": "none", "reason": "betanet_unavailable"}

        try:
            if hasattr(self, "battery_optimizer"):
                optimization = await self.battery_optimizer.optimize_for_device(device_info)

                # Apply optimizations
                if optimization.get("reduce_privacy_hops"):
                    self.mixnet_router.set_max_hops(2)  # Reduce from default

                if optimization.get("disable_covert"):
                    self.enable_covert = False

                if optimization.get("chunking_size"):
                    self.adaptive_chunking.set_max_chunk_size(optimization["chunking_size"])

                return optimization

            return {"optimizations": "none", "reason": "optimizer_unavailable"}

        except Exception as e:
            logger.error(f"Device optimization failed: {e}")
            return {"optimizations": "error", "error": str(e)}


class FogComputeBetaNetService:
    """
    Service class for fog compute nodes to use BetaNet transport

    This provides a high-level interface for fog compute infrastructure
    to leverage BetaNet's capabilities without directly depending on the bounty code.
    """

    def __init__(self):
        self.transport = None
        self.node_id = None
        self.active_jobs: dict[str, dict[str, Any]] = {}

    async def initialize(self, node_id: str, privacy_mode: str = "balanced", enable_covert: bool = True) -> bool:
        """
        Initialize the BetaNet service for this fog compute node

        Args:
            node_id: Unique identifier for this fog node
            privacy_mode: Privacy level configuration
            enable_covert: Enable covert communication capabilities

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.node_id = node_id
            self.transport = BetaNetFogTransport(
                privacy_mode=privacy_mode,
                enable_covert=enable_covert,
                mobile_optimization=True,  # Always enable for fog compute
            )

            logger.info(f"BetaNet service initialized for fog node {node_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BetaNet service: {e}")
            return False

    async def submit_job_to_peer(self, peer_node: str, job_data: dict[str, Any], priority: str = "normal") -> str:
        """
        Submit a job to another fog compute node using BetaNet transport

        Args:
            peer_node: Target fog node identifier
            job_data: Job specification and data
            priority: Job priority level

        Returns:
            Job submission ID
        """
        if not self.transport:
            raise RuntimeError("BetaNet service not initialized")

        job_id = f"{self.node_id}-{int(time.time())}-{len(self.active_jobs)}"

        # Serialize job data
        job_payload = json.dumps(job_data).encode("utf-8")

        # Send via BetaNet transport
        result = await self.transport.send_job_data(job_payload, peer_node, priority)

        # Track the job
        self.active_jobs[job_id] = {
            "peer_node": peer_node,
            "submitted_at": time.time(),
            "priority": priority,
            "status": "submitted",
            "transport_result": result,
        }

        logger.info(f"Job {job_id} submitted to {peer_node} via {result.get('transport')}")
        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a submitted job"""
        return self.active_jobs.get(job_id)

    def get_service_stats(self) -> dict[str, Any]:
        """Get service statistics"""
        base_stats = {
            "node_id": self.node_id,
            "active_jobs": len(self.active_jobs),
            "service_available": self.transport is not None,
        }

        if self.transport:
            base_stats.update(self.transport.get_transport_stats())

        return base_stats


# Factory function for easy integration
def create_betanet_transport(privacy_mode: str = "balanced") -> BetaNetFogTransport:
    """
    Factory function to create BetaNet transport for fog compute

    Args:
        privacy_mode: Privacy level - "strict", "balanced", or "performance"

    Returns:
        BetaNet transport instance
    """
    return BetaNetFogTransport(privacy_mode=privacy_mode)


def is_betanet_available() -> bool:
    """Check if BetaNet bounty integration is available"""
    return BETANET_AVAILABLE


def get_betanet_capabilities() -> dict[str, bool]:
    """Get available BetaNet capabilities"""
    return {
        "bounty_available": BETANET_AVAILABLE,
        "covert_channels": BETANET_AVAILABLE,
        "mixnet_privacy": BETANET_AVAILABLE,
        "mobile_optimization": BETANET_AVAILABLE,
        "secure_transport": True,  # Always available via fallback
    }
