"""
SCION-aware Navigator integration for AIVillage transport selection.
Integrates SCION paths into the Navigator's transport decision-making process.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from packages.p2p.core.message_types import UnifiedMessage as Message
from packages.p2p.scion_gateway import GatewayConfig, SCIONGateway, SCIONGatewayError, SCIONPath

logger = logging.getLogger(__name__)


class TransportPriority(Enum):
    """Transport priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    FALLBACK = 5


class PathProtocol(Enum):
    """Path protocol types."""

    SCION = "scion"
    BETANET = "betanet"
    BITCHAT = "bitchat"
    STORE_FORWARD = "store_forward"


class RoutingPriority(Enum):
    """Routing priority modes."""

    PERFORMANCE_FIRST = "performance_first"
    COST_FIRST = "cost_first"
    PRIVACY_FIRST = "privacy_first"
    RELIABILITY_FIRST = "reliability_first"


class EnergyMode(Enum):
    """Energy management modes."""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    ULTRA_LOW_POWER = "ultra_low_power"


@dataclass
class MessageContext:
    """Message context for routing decisions."""

    size_bytes: int = 1024
    priority: int = 5
    requires_realtime: bool = False
    privacy_required: bool = False


@dataclass
class NetworkConditions:
    """Current network conditions."""

    internet_available: bool = True
    bluetooth_available: bool = False
    wifi_connected: bool = True
    cellular_connected: bool = False
    bandwidth_mbps: float = 100.0
    latency_ms: float = 50.0


@dataclass
class PeerInfo:
    """Information about discovered peers."""

    peer_id: str
    protocols: set[str]
    hop_distance: int
    bluetooth_rssi: float | None = None
    trust_score: float = 0.5


@dataclass
class RoutingReceipt:
    """Receipt for routing decisions."""

    chosen_path: str
    switch_latency_ms: float
    reason: str
    timestamp: float
    scion_available: bool
    scion_paths: int
    path_scores: dict[str, float]


@dataclass
class TransportCandidate:
    """Transport option with routing information."""

    transport_type: str
    endpoint: str
    priority: TransportPriority
    estimated_latency_ms: float
    reliability_score: float  # 0.0 - 1.0
    cost_factor: float  # 0.0 - 1.0 (lower is cheaper)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Navigation routing decision."""

    primary_transport: TransportCandidate
    backup_transports: list[TransportCandidate]
    decision_reason: str
    confidence_score: float  # 0.0 - 1.0
    estimated_total_time_ms: float
    route_metadata: dict[str, Any] = field(default_factory=dict)


class LinkChangeDetector:
    """Detects network link changes for fast switching."""

    def __init__(self, target_switch_time_ms: int = 500):
        self.target_switch_time_ms = target_switch_time_ms
        self.current_state = {}

    def update_link_state(self, new_state: dict[str, Any]) -> bool:
        """Update link state and return True if changed."""
        if self.current_state != new_state:
            self.current_state = new_state.copy()
            return True
        return False


class NavigatorAgent:
    """
    Enhanced Navigator Agent with SCION support and fast switching.
    """

    def __init__(
        self,
        agent_id: str = "navigator_agent",
        routing_priority: RoutingPriority = RoutingPriority.PERFORMANCE_FIRST,
        energy_mode: EnergyMode = EnergyMode.BALANCED,
    ):
        self.agent_id = agent_id
        self.routing_priority = routing_priority
        self.energy_mode = energy_mode

        # SCION components
        self.scion_gateway: SCIONGateway | None = None
        self.scion_enabled = False

        # Network state
        self.conditions = NetworkConditions()
        self.discovered_peers: dict[str, PeerInfo] = {}

        # Performance tracking
        self.path_switch_threshold_ms = 500
        self.fast_switch_enabled = True
        self.global_south_mode = False

        # Caching and history
        self.scion_path_cache: dict[str, list[SCIONPath]] = {}
        self.routing_decisions: dict[str, RoutingDecision] = {}
        self.path_rtt_ewma: dict[str, float] = {}
        self.receipts: list[RoutingReceipt] = []

        # Link change detection
        self.link_change_detector = LinkChangeDetector(target_switch_time_ms=500)

    async def select_path(
        self, destination: str, context: MessageContext, available_protocols: list[str]
    ) -> tuple[PathProtocol, dict[str, Any]]:
        """Select optimal path for given destination and context."""
        start_time = time.time()

        # Check SCION availability
        scion_available = False
        scion_paths = 0

        if "scion" in available_protocols and self.scion_enabled and self.scion_gateway:
            try:
                health = await self.scion_gateway.health_check()
                if health.get("status") == "healthy" and health.get("scion_connected", False):
                    paths = await self.scion_gateway.query_paths(destination)
                    scion_available = True
                    scion_paths = len(paths)
                    self.scion_path_cache[destination] = paths
                else:
                    scion_available = False
            except Exception as e:
                logger.debug(f"SCION query failed: {e}")
                scion_available = False

        # Score available protocols
        path_scores = {}

        if scion_available and context.requires_realtime:
            # Prefer SCION for high-performance needs
            protocol = PathProtocol.SCION
            path_scores["scion"] = 0.9
            path_scores["betanet"] = 0.6
            reason = "scion_high_performance"
        elif self.conditions.internet_available and "betanet" in available_protocols:
            # Fallback to Betanet
            protocol = PathProtocol.BETANET
            path_scores["betanet"] = 0.8
            path_scores["scion"] = 0.3 if scion_available else 0.0
            reason = "betanet_internet_available"
        elif not self.conditions.internet_available and "bitchat" in available_protocols:
            # Offline-first BitChat
            protocol = PathProtocol.BITCHAT
            path_scores["bitchat"] = 0.7
            path_scores["scion"] = 0.0
            reason = "bitchat_offline_first"
        else:
            # Default fallback
            protocol = PathProtocol.BETANET if "betanet" in available_protocols else PathProtocol.BITCHAT
            path_scores[protocol.value] = 0.5
            reason = "default_fallback"

        # Calculate switch latency
        switch_latency = (time.time() - start_time) * 1000

        # Create receipt
        receipt = RoutingReceipt(
            chosen_path=protocol.value,
            switch_latency_ms=switch_latency,
            reason=reason,
            timestamp=time.time(),
            scion_available=scion_available,
            scion_paths=scion_paths,
            path_scores=path_scores,
        )
        self.receipts.append(receipt)

        # Metadata
        metadata = {
            "scion_available": scion_available,
            "scion_paths": scion_paths,
            "offline_capable": protocol == PathProtocol.BITCHAT,
            "energy_efficient": protocol == PathProtocol.BITCHAT,
        }

        return protocol, metadata

    def get_receipts(self, count: int | None = None) -> list[RoutingReceipt]:
        """Get routing receipts for analysis."""
        if count is None:
            return self.receipts.copy()
        return self.receipts[-count:] if count > 0 else []

    def get_fast_switching_metrics(self) -> dict[str, Any]:
        """Get fast switching performance metrics."""
        return {
            "target_switch_time_ms": self.path_switch_threshold_ms,
            "fast_switch_enabled": self.fast_switch_enabled,
            "receipts_count": len(self.receipts),
            "avg_switch_time_ms": sum(r.switch_latency_ms for r in self.receipts) / max(1, len(self.receipts)),
        }

    def get_scion_statistics(self) -> dict[str, Any]:
        """Get SCION-specific statistics."""
        scion_selections = sum(1 for r in self.receipts if r.chosen_path == "scion")
        receipts_with_scion = sum(1 for r in self.receipts if r.scion_available)

        return {
            "scion_enabled": self.scion_enabled,
            "scion_selections": scion_selections,
            "receipts_with_scion": receipts_with_scion,
            "path_rtt_ewma_entries": len(self.path_rtt_ewma),
            "cached_destinations": len(self.scion_path_cache),
        }


class SCIONAwareNavigator:
    """
    Enhanced Navigator that integrates SCION paths into transport selection.
    Provides intelligent routing decisions based on path quality, cost, and availability.
    """

    def __init__(
        self,
        scion_config: GatewayConfig,
        transport_manager: Any,  # Mock transport manager
        enable_scion_preference: bool = True,
        scion_weight: float = 2.0,
    ):
        self.scion_config = scion_config
        self.transport_manager = transport_manager
        self.enable_scion_preference = enable_scion_preference
        self.scion_weight = scion_weight
        self._is_running = False

    async def start(self) -> None:
        """Start the SCION-aware Navigator."""
        self._is_running = True
        logger.info("SCION-aware Navigator started")

    async def stop(self) -> None:
        """Stop the Navigator and cleanup resources."""
        self._is_running = False
        logger.info("SCION-aware Navigator stopped")

    async def find_optimal_route(
        self,
        destination: str,
        message: Message,
        constraints: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Find optimal transport route to destination."""
        # Mock implementation for testing
        primary_transport = TransportCandidate(
            transport_type="scion",
            endpoint=destination,
            priority=TransportPriority.HIGH,
            estimated_latency_ms=30.0,
            reliability_score=0.95,
            cost_factor=0.2,
        )

        return RoutingDecision(
            primary_transport=primary_transport,
            backup_transports=[],
            decision_reason="scion_preferred",
            confidence_score=0.9,
            estimated_total_time_ms=50.0,
        )
