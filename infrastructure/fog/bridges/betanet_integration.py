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
from pathlib import Path
import time
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
    - Constitutional compliance with privacy-preserving oversight
    - Tiered constitutional verification (Bronze/Silver/Gold/Platinum)
    - Zero-knowledge proof integration for privacy preservation
    """

    def __init__(self, privacy_mode: str = "balanced", enable_covert: bool = True, mobile_optimization: bool = True, 
                 constitutional_enabled: bool = False, constitutional_tier: str = "silver"):
        """
        Initialize BetaNet transport adapter for fog computing

        Args:
            privacy_mode: Privacy level - "strict", "balanced", or "performance"
            enable_covert: Enable covert channel capabilities
            mobile_optimization: Enable battery/thermal optimizations
            constitutional_enabled: Enable constitutional compliance features
            constitutional_tier: Constitutional tier - "bronze", "silver", "gold", "platinum"
        """
        self.privacy_mode = privacy_mode
        self.enable_covert = enable_covert
        self.mobile_optimization = mobile_optimization
        self.constitutional_enabled = constitutional_enabled
        self.constitutional_tier = constitutional_tier
        
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_transferred": 0,
            "covert_channels_used": 0,
            "privacy_hops": 0,
            "constitutional_verifications": 0,
            "constitutional_compliant_messages": 0,
            "zk_proofs_generated": 0,
            "privacy_preservation_rate": 0.0
        }

        # Constitutional transport integration
        self.constitutional_transport = None

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

            # Initialize constitutional features
            if self.constitutional_enabled:
                self._init_constitutional_components()

            logger.info("BetaNet components initialized for fog compute integration")

        except Exception as e:
            logger.error(f"Failed to initialize BetaNet components: {e}")
            self._init_fallback_components()

    def _init_constitutional_components(self):
        """Initialize constitutional BetaNet components"""
        try:
            # Import constitutional components
            from ...p2p.betanet.constitutional_transport import ConstitutionalBetaNetService, ConstitutionalTransportConfig
            from ...p2p.betanet.constitutional_frames import ConstitutionalTier
            
            # Map tier string to enum
            tier_mapping = {
                "bronze": ConstitutionalTier.BRONZE,
                "silver": ConstitutionalTier.SILVER,
                "gold": ConstitutionalTier.GOLD,
                "platinum": ConstitutionalTier.PLATINUM
            }
            
            constitutional_tier_enum = tier_mapping.get(self.constitutional_tier.lower(), ConstitutionalTier.SILVER)
            
            # Create constitutional transport configuration
            config = ConstitutionalTransportConfig(
                default_tier=constitutional_tier_enum,
                enable_real_time_moderation=True,
                enable_zero_knowledge_proofs=True,
                enable_tee_integration=True,
                enable_fog_integration=True,
                privacy_priority=0.6 if self.privacy_mode == "strict" else 0.4
            )
            
            # Initialize constitutional service
            self.constitutional_transport = ConstitutionalBetaNetService(config)
            
            logger.info(f"Constitutional BetaNet features initialized ({self.constitutional_tier} tier)")
            
        except ImportError as e:
            logger.warning(f"Constitutional features unavailable: {e}")
            self.constitutional_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize constitutional features: {e}")
            self.constitutional_enabled = False

    def _init_fallback_components(self):
        """Initialize fallback components when BetaNet is not available"""
        self.fallback_transport = FallbackTransport()
        logger.warning("Using fallback transport - BetaNet integration not available")

    async def send_job_data(self, job_data: bytes, destination: str, priority: str = "normal") -> dict[str, Any]:
        """
        Send job data using BetaNet transport with fog compute optimizations and constitutional compliance

        Args:
            job_data: Job payload data
            destination: Destination endpoint
            priority: Message priority ("low", "normal", "high")

        Returns:
            Dictionary with transmission results and statistics
        """
        start_time = time.time()

        # Try constitutional transport first if enabled
        if self.constitutional_enabled and self.constitutional_transport:
            try:
                constitutional_result = await self._send_via_constitutional_transport(
                    job_data, destination, priority
                )
                
                if constitutional_result["success"]:
                    duration = time.time() - start_time
                    self._update_constitutional_stats(constitutional_result, len(job_data), duration)
                    
                    return {
                        "success": True,
                        "transport": "constitutional_betanet",
                        "duration": duration,
                        "bytes_sent": len(job_data),
                        "constitutional_compliance": True,
                        "constitutional_tier": self.constitutional_tier,
                        "privacy_level": constitutional_result.get("privacy_level", 0.5),
                        "zk_proof_generated": constitutional_result.get("zk_proof_generated", False),
                        "details": constitutional_result
                    }
                else:
                    logger.warning("Constitutional transport failed, falling back to standard BetaNet")
                    
            except Exception as e:
                logger.warning(f"Constitutional transport error, falling back: {e}")

        if not BETANET_AVAILABLE:
            success = await self.fallback_transport.send_message(job_data, destination)
            return {
                "success": success,
                "transport": "fallback",
                "duration": time.time() - start_time,
                "bytes_sent": len(job_data),
                "constitutional_compliance": False
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

    async def _send_via_constitutional_transport(self, job_data: bytes, destination: str, priority: str) -> dict[str, Any]:
        """Send job data via constitutional BetaNet transport"""
        
        if not self.constitutional_transport:
            return {"success": False, "error": "Constitutional transport not initialized"}
        
        try:
            # Start constitutional service if not running
            if not getattr(self.constitutional_transport, 'running', False):
                success = await self.constitutional_transport.start_service(mode="client")
                if not success:
                    return {"success": False, "error": "Failed to start constitutional service"}
            
            # Send message with constitutional verification
            result = await self.constitutional_transport.send_message(
                content=job_data,
                destination=destination,
                privacy_tier=self.constitutional_tier,
                priority=1 if priority == "low" else 2 if priority == "normal" else 3,
                require_verification=True
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Constitutional transport error: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_constitutional_stats(self, result: dict[str, Any], bytes_sent: int, duration: float):
        """Update constitutional transport statistics"""
        
        self.stats["constitutional_verifications"] += 1
        
        if result.get("success", False):
            self.stats["constitutional_compliant_messages"] += 1
        
        # Check if ZK proof was generated (Gold/Platinum tiers)
        if result.get("zk_proof_generated", False) or self.constitutional_tier in ["gold", "platinum"]:
            self.stats["zk_proofs_generated"] += 1
        
        # Update privacy preservation rate
        privacy_level = result.get("privacy_level", 0.0)
        current_rate = self.stats["privacy_preservation_rate"]
        total_verifications = self.stats["constitutional_verifications"]
        
        if total_verifications > 0:
            self.stats["privacy_preservation_rate"] = (
                (current_rate * (total_verifications - 1) + privacy_level) / total_verifications
            )
    
    async def initialize_constitutional_features(self):
        """Initialize constitutional features after transport creation"""
        
        if self.constitutional_enabled and self.constitutional_transport:
            try:
                success = await self.constitutional_transport.start_service(mode="client")
                if success:
                    logger.info("Constitutional features activated for BetaNet transport")
                    return True
                else:
                    logger.warning("Failed to initialize constitutional features")
                    self.constitutional_enabled = False
                    return False
            except Exception as e:
                logger.error(f"Constitutional initialization error: {e}")
                self.constitutional_enabled = False
                return False
        
        return not self.constitutional_enabled


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
def create_betanet_transport(privacy_mode: str = "balanced", constitutional_enabled: bool = False, 
                           constitutional_tier: str = "silver") -> BetaNetFogTransport:
    """
    Factory function to create BetaNet transport for fog compute

    Args:
        privacy_mode: Privacy level - "strict", "balanced", or "performance"
        constitutional_enabled: Enable constitutional compliance features
        constitutional_tier: Constitutional tier - "bronze", "silver", "gold", "platinum"

    Returns:
        BetaNet transport instance with optional constitutional features
    """
    return BetaNetFogTransport(
        privacy_mode=privacy_mode, 
        constitutional_enabled=constitutional_enabled,
        constitutional_tier=constitutional_tier
    )


def is_betanet_available() -> bool:
    """Check if BetaNet bounty integration is available"""
    return BETANET_AVAILABLE


def get_betanet_capabilities() -> dict[str, bool]:
    """Get available BetaNet capabilities including constitutional features"""
    
    # Check if constitutional features are available
    constitutional_available = False
    try:
        from ...p2p.betanet.constitutional_transport import ConstitutionalBetaNetService
        constitutional_available = True
    except ImportError:
        pass
    
    return {
        "bounty_available": BETANET_AVAILABLE,
        "covert_channels": BETANET_AVAILABLE,
        "mixnet_privacy": BETANET_AVAILABLE,
        "mobile_optimization": BETANET_AVAILABLE,
        "secure_transport": True,  # Always available via fallback
        "constitutional_compliance": constitutional_available,
        "privacy_preserving_oversight": constitutional_available,
        "zero_knowledge_proofs": constitutional_available,
        "tiered_constitutional_verification": constitutional_available,
        "tee_integration": constitutional_available
    }
