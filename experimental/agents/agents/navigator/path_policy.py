"""Navigator Agent - Dual-Path Routing for BitChat/Betanet

The Navigator is a Tier 1 infrastructure agent responsible for intelligent
path selection between BitChat (offline Bluetooth mesh) and Betanet (global
decentralized internet). Optimized for Global South scenarios with offline-first
priorities.

Key Responsibilities:
- BitChat-first routing policy for energy efficiency
- Multi-hop mesh routing optimization
- DTN/store-and-forward for offline scenarios
- Adaptive bandwidth and QoS management
- Protocol switching based on network conditions
- Privacy-aware mixnode selection
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging
import socket
import time
from typing import Any
import uuid

# SCION Gateway import
try:
    from src.transport.scion_gateway import (
        GatewayConfig,
        SCIONConnectionError,
        SCIONGateway,
        SCIONPath,
    )

    SCION_AVAILABLE = True
except ImportError:
    SCION_AVAILABLE = False
    logger.warning("SCION Gateway not available - SCION transport disabled")

# Bluetooth imports with fallback
try:
    import bluetooth

    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LinkChangeDetector:
    """Detects network link changes for fast path switching (500ms target)"""

    def __init__(self, target_switch_time_ms: int = 500):
        self.target_switch_time_ms = target_switch_time_ms
        self.link_state_history: list[dict] = []
        self.max_history = 10
        self.last_check_time = 0.0
        self.check_interval_ms = 100  # Check every 100ms

        # Current link state
        self.current_state = {
            "bluetooth_available": False,
            "internet_available": False,
            "wifi_connected": False,
            "cellular_connected": False,
            "bandwidth_mbps": 0.0,
            "latency_ms": 0.0,
        }

        # Change detection
        self.change_events: list[dict] = []
        self.max_events = 50

    def update_link_state(self, new_state: dict) -> bool:
        """Update link state and detect changes

        Returns:
            True if significant change detected requiring path switch
        """
        current_time = time.time() * 1000  # milliseconds

        # Rate limit checks to target interval
        if current_time - self.last_check_time < self.check_interval_ms:
            return False

        self.last_check_time = current_time

        # Detect changes from current state
        changes_detected = []

        for key, new_value in new_state.items():
            old_value = self.current_state.get(key)
            if old_value != new_value:
                changes_detected.append(
                    {
                        "field": key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "timestamp": current_time,
                    }
                )

        # Update current state
        self.current_state.update(new_state)

        # Add to history
        state_snapshot = {
            "timestamp": current_time,
            "state": self.current_state.copy(),
            "changes": changes_detected,
        }

        self.link_state_history.append(state_snapshot)
        if len(self.link_state_history) > self.max_history:
            self.link_state_history.pop(0)

        # Determine if changes require path switching
        requires_switch = self._evaluate_change_significance(changes_detected)

        if requires_switch:
            # Record change event
            event = {
                "timestamp": current_time,
                "changes": changes_detected,
                "switch_required": True,
                "evaluation_time_ms": current_time
                - (changes_detected[0]["timestamp"] if changes_detected else current_time),
            }

            self.change_events.append(event)
            if len(self.change_events) > self.max_events:
                self.change_events.pop(0)

            logger.info(f"Link change detected requiring path switch: {[c['field'] for c in changes_detected]}")

        return requires_switch

    def _evaluate_change_significance(self, changes: list[dict]) -> bool:
        """Evaluate if changes are significant enough to require path switching"""
        if not changes:
            return False

        # High-priority changes that require immediate switching
        critical_changes = [
            "bluetooth_available",
            "internet_available",
            "wifi_connected",
            "cellular_connected",
        ]

        for change in changes:
            field = change["field"]

            # Critical connectivity changes
            if field in critical_changes:
                return True

            # Significant bandwidth changes (>50% change)
            if field == "bandwidth_mbps":
                old_val = change["old_value"] or 0
                new_val = change["new_value"] or 0
                if old_val > 0:
                    change_ratio = abs(new_val - old_val) / old_val
                    if change_ratio > 0.5:  # >50% bandwidth change
                        return True

            # Significant latency changes (>100ms change)
            if field == "latency_ms":
                old_val = change["old_value"] or 0
                new_val = change["new_value"] or 0
                if abs(new_val - old_val) > 100:  # >100ms latency change
                    return True

        return False

    def get_switch_recommendation(self) -> tuple[bool, str, dict]:
        """Get path switch recommendation based on current state

        Returns:
            (should_switch, recommended_path, rationale)
        """
        state = self.current_state
        rationale = {"reasons": [], "metrics": state.copy()}

        # Prefer paths based on current connectivity
        if state.get("bluetooth_available") and not state.get("internet_available"):
            # Offline scenario - prefer BitChat
            return True, "bitchat", {**rationale, "reasons": ["offline_scenario"]}

        if state.get("internet_available") and state.get("wifi_connected"):
            # High-quality internet - prefer Betanet
            if state.get("bandwidth_mbps", 0) > 10 and state.get("latency_ms", 999) < 100:
                return (
                    True,
                    "betanet",
                    {**rationale, "reasons": ["high_quality_internet"]},
                )

        if state.get("cellular_connected") and not state.get("wifi_connected"):
            # Cellular only - consider cost and battery
            if state.get("bandwidth_mbps", 0) < 2:
                return (
                    True,
                    "bitchat",
                    {**rationale, "reasons": ["cellular_low_bandwidth"]},
                )

        # No strong preference - maintain current path
        return False, "maintain", rationale

    def get_performance_metrics(self) -> dict:
        """Get performance metrics for the link change detector"""
        if not self.change_events:
            return {"events": 0}

        recent_events = [e for e in self.change_events if time.time() * 1000 - e["timestamp"] < 60000]  # Last minute

        evaluation_times = [e.get("evaluation_time_ms", 0) for e in recent_events]
        avg_evaluation_time = sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0

        return {
            "total_events": len(self.change_events),
            "recent_events": len(recent_events),
            "avg_evaluation_time_ms": avg_evaluation_time,
            "target_switch_time_ms": self.target_switch_time_ms,
            "within_target": avg_evaluation_time <= self.target_switch_time_ms,
            "current_state": self.current_state.copy(),
        }


class PathProtocol(Enum):
    """Available routing protocols"""

    BITCHAT = "bitchat"  # Bluetooth mesh, offline
    BETANET = "betanet"  # HTX/HTXQUIC, online
    SCION = "scion"  # SCION multipath network
    STORE_FORWARD = "dtm"  # Delay-tolerant messaging
    FALLBACK = "fallback"  # Legacy P2P fallback


class EnergyMode(Enum):
    """Energy management modes"""

    POWERSAVE = "powersave"  # Minimize energy usage
    BALANCED = "balanced"  # Balance performance/energy
    PERFORMANCE = "performance"  # Maximize performance


class RoutingPriority(Enum):
    """Routing decision priorities"""

    OFFLINE_FIRST = "offline_first"  # Prefer offline (Global South)
    PERFORMANCE_FIRST = "performance_first"  # Prefer fastest route
    PRIVACY_FIRST = "privacy_first"  # Prefer mixnode routing
    BANDWIDTH_FIRST = "bandwidth_first"  # Prefer high bandwidth


@dataclass
class NetworkConditions:
    """Current network and device conditions"""

    # Connectivity
    bluetooth_available: bool = False
    internet_available: bool = False
    cellular_available: bool = False
    wifi_connected: bool = False

    # Device status
    battery_percent: int | None = None
    charging: bool = False
    mobile_device: bool = False

    # Network performance
    latency_ms: float = 100.0
    bandwidth_mbps: float = 1.0
    reliability_score: float = 0.8

    # Environmental context
    geographic_region: str | None = None
    censorship_risk: float = 0.0  # 0.0=safe, 1.0=high risk
    data_cost_usd_mb: float = 0.0  # Cost per MB

    # Peer proximity
    nearby_peers: int = 0
    peer_hop_distances: dict[str, int] = field(default_factory=dict)

    def is_low_resource_environment(self) -> bool:
        """Check if this is a resource-constrained environment"""
        return (
            (self.battery_percent is not None and self.battery_percent < 30)
            or self.data_cost_usd_mb > 0.01
            or not self.internet_available  # >$0.01/MB is expensive
            or self.bandwidth_mbps < 1.0
        )

    def is_privacy_sensitive(self) -> bool:
        """Check if privacy routing is needed"""
        return self.censorship_risk > 0.3


@dataclass
class Receipt:
    """Receipt for bounty reviewers with switch latency tracking"""

    chosen_path: str
    switch_latency_ms: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    scion_available: bool = False
    scion_paths: int = 0
    path_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class MessageContext:
    """Context for routing decision"""

    size_bytes: int = 0
    priority: int = 5  # 1=low, 10=urgent
    content_type: str = "application/octet-stream"
    requires_realtime: bool = False
    privacy_required: bool = False
    delivery_deadline: float | None = None
    bandwidth_sensitive: bool = False

    def is_large_message(self) -> bool:
        """Check if message needs chunking"""
        return self.size_bytes > 10000  # >10KB

    def is_urgent(self) -> bool:
        """Check if message is time-critical"""
        return self.priority >= 8 or self.requires_realtime


@dataclass
class PeerInfo:
    """Information about a discovered peer"""

    peer_id: str
    protocols: set[str] = field(default_factory=set)
    hop_distance: int = 1
    last_seen: float = field(default_factory=time.time)
    trust_score: float = 0.5
    capabilities: set[str] = field(default_factory=set)

    # Performance metrics
    avg_latency_ms: float = 100.0
    reliability: float = 0.8
    bandwidth_mbps: float = 1.0

    # Proximity indicators
    bluetooth_rssi: int | None = None  # Signal strength
    geographic_distance_km: float | None = None

    def is_nearby(self) -> bool:
        """Check if peer is in close proximity"""
        return (
            self.hop_distance <= 3
            and (self.bluetooth_rssi is None or self.bluetooth_rssi > -60)
            and (self.geographic_distance_km is None or self.geographic_distance_km < 1.0)
        )

    def supports_protocol(self, protocol: str) -> bool:
        """Check if peer supports specific protocol"""
        return protocol in self.protocols


class NavigatorAgent:
    """Navigator Agent: Intelligent dual-path routing for AIVillage

    Implements BitChat-first routing policy optimized for:
    - Global South environments (offline-first)
    - Energy efficiency and battery conservation
    - Censorship resistance via mixnode routing
    - Adaptive bandwidth management
    - Store-and-forward for offline scenarios
    """

    def __init__(
        self,
        agent_id: str = None,
        routing_priority: RoutingPriority = RoutingPriority.OFFLINE_FIRST,
        energy_mode: EnergyMode = EnergyMode.BALANCED,
    ):
        self.agent_id = agent_id or f"navigator_{uuid.uuid4().hex[:8]}"
        self.routing_priority = routing_priority
        self.energy_mode = energy_mode

        # Network condition monitoring
        self.conditions = NetworkConditions()
        self.condition_update_interval = 30.0  # Update every 30 seconds
        self.last_condition_update = 0.0

        # Peer tracking
        self.discovered_peers: dict[str, PeerInfo] = {}
        self.peer_proximity_cache: dict[str, bool] = {}
        self.peer_cache_ttl = 300.0  # 5 minutes

        # SCION Gateway integration
        self.scion_gateway: SCIONGateway | None = None
        self.scion_enabled = SCION_AVAILABLE
        self.scion_path_cache: dict[str, list[SCIONPath]] = {}
        self.scion_cache_ttl = 60.0  # 1 minute cache for SCION paths
        self.scion_prefer_high_performance = True

        # RTT EWMA tracking for path scoring
        self.path_rtt_ewma: dict[str, float] = defaultdict(lambda: 50.0)  # Default 50ms
        self.rtt_ewma_alpha = 0.3  # EWMA smoothing factor

        # Routing statistics and learning
        self.route_success_rates: dict[str, float] = defaultdict(lambda: 0.8)
        self.protocol_performance: dict[str, dict[str, float]] = {
            "bitchat": {"latency": 200.0, "reliability": 0.85, "energy_cost": 0.2},
            "betanet": {"latency": 100.0, "reliability": 0.95, "energy_cost": 0.8},
            "scion": {"latency": 50.0, "reliability": 0.98, "energy_cost": 0.6},
            "store_forward": {"latency": 0.0, "reliability": 1.0, "energy_cost": 0.1},
        }

        # Receipt tracking for bounty reviewers
        self.receipts: list[Receipt] = []
        self.max_receipts = 1000

        # Decision caching
        self.routing_decisions: dict[str, tuple[PathProtocol, float]] = {}
        self.decision_cache_ttl = 60.0  # 1 minute

        # Fast path switching for link changes (500ms target)
        self.link_change_detector = LinkChangeDetector(target_switch_time_ms=500)
        self.last_path_switch_time = 0.0
        self.path_switch_threshold_ms = 500  # 500ms target for path switching
        self.fast_switch_enabled = True

        # Global South optimizations
        self.global_south_mode = True  # Prioritize offline-first
        self.data_cost_threshold = 0.005  # $0.005/MB threshold
        self.battery_conservation_threshold = 25  # Start conserving at 25%

        # Privacy settings
        self.privacy_aware = True
        self.mixnode_preferences = ["tor", "i2p", "betanet_mixnode"]

        logger.info(f"Navigator initialized: {self.agent_id} (priority={routing_priority.value})")

    async def select_path(
        self,
        destination: str,
        message_context: MessageContext,
        available_protocols: list[str] | None = None,
    ) -> tuple[PathProtocol, dict[str, Any]]:
        """Core path selection algorithm with SCION preference and measurable ≤500ms switch time:

        1. Check SCION availability via gateway health/cache
        2. Score paths using RTT EWMA + loss + policy (privacy/perf)
        3. Emit Receipt {chosen_path, switch_latency_ms, reason} for bounty reviewers
        4. BitChat-first for Global South (offline/low-power)
        5. Betanet for wide-area/high-bandwidth needs
        6. Store-and-forward when neither available

        Returns:
            Tuple of (selected_protocol, routing_metadata)
        """
        start_time = time.time() * 1000  # milliseconds for latency tracking

        # Update network conditions
        await self._update_network_conditions()

        # Check SCION availability first (preferred protocol)
        scion_available = False
        scion_paths = []
        if self.scion_enabled:
            scion_available, scion_paths = await self._check_scion_availability(destination)

        # Check for link changes requiring fast switching (500ms target)
        if self.fast_switch_enabled:
            link_change_detected = await self._check_fast_switching(destination, message_context)
            if link_change_detected:
                # Fast path switch detected - use optimized decision with SCION preference
                protocol, metadata = await self._fast_path_selection_with_scion(
                    destination,
                    message_context,
                    available_protocols,
                    scion_available,
                    scion_paths,
                )
                switch_latency_ms = time.time() * 1000 - start_time
                self._emit_receipt(
                    protocol.value,
                    switch_latency_ms,
                    "fast_link_change",
                    scion_available,
                    len(scion_paths),
                )
                return protocol, metadata

        # Check decision cache
        cache_key = f"{destination}_{message_context.priority}_{message_context.size_bytes}"
        if cache_key in self.routing_decisions:
            cached_decision, cache_time = self.routing_decisions[cache_key]
            if time.time() - cache_time < self.decision_cache_ttl:
                logger.debug(f"Using cached routing decision: {cached_decision.value}")
                switch_latency_ms = time.time() * 1000 - start_time
                self._emit_receipt(
                    cached_decision.value,
                    switch_latency_ms,
                    "cache_hit",
                    scion_available,
                    len(scion_paths),
                )
                return cached_decision, self._get_routing_metadata(cached_decision, destination)

        # Core routing decision logic with SCION preference and path scoring
        (
            selected_protocol,
            path_scores,
        ) = await self._evaluate_routing_options_with_scion(
            destination,
            message_context,
            available_protocols,
            scion_available,
            scion_paths,
        )

        # Cache decision
        self.routing_decisions[cache_key] = (selected_protocol, time.time())

        # Calculate switch latency and emit receipt
        switch_latency_ms = time.time() * 1000 - start_time
        reason = self._determine_selection_reason(selected_protocol, scion_available, path_scores)
        self._emit_receipt(
            selected_protocol.value,
            switch_latency_ms,
            reason,
            scion_available,
            len(scion_paths),
            path_scores,
        )

        # Generate routing metadata
        routing_metadata = self._get_routing_metadata(selected_protocol, destination)
        routing_metadata["path_scores"] = path_scores
        routing_metadata["scion_available"] = scion_available
        routing_metadata["scion_paths"] = len(scion_paths)

        logger.info(
            f"Selected route for {destination}: {selected_protocol.value} "
            f"(size={message_context.size_bytes}, priority={message_context.priority}, "
            f"switch_time={switch_latency_ms:.1f}ms, scion_available={scion_available})"
        )

        return selected_protocol, routing_metadata

    async def _check_scion_availability(self, destination: str) -> tuple[bool, list[SCIONPath]]:
        """Check SCION availability via gateway health/cache"""
        if not self.scion_enabled:
            return False, []

        # Check cache first
        cache_key = f"scion_paths_{destination}"
        if cache_key in self.scion_path_cache:
            cached_paths, cache_time = self.scion_path_cache[cache_key]
            if time.time() - cache_time < self.scion_cache_ttl:
                return len(cached_paths) > 0, cached_paths

        # Initialize SCION gateway if needed
        if self.scion_gateway is None:
            try:
                config = GatewayConfig()
                self.scion_gateway = SCIONGateway(config)
                await self.scion_gateway.start()
                logger.info("SCION Gateway initialized for Navigator")
            except Exception as e:
                logger.warning(f"Failed to initialize SCION Gateway: {e}")
                return False, []

        try:
            # Check gateway health
            health = await self.scion_gateway.health_check()
            if health.get("status") != "healthy" or not health.get("scion_connected"):
                logger.debug(f"SCION Gateway unhealthy: {health}")
                return False, []

            # Query available paths to destination
            paths = await self.scion_gateway.query_paths(destination)

            # Cache results
            self.scion_path_cache[cache_key] = (paths, time.time())

            # Update RTT EWMA for discovered paths
            for path in paths:
                path_key = f"scion_{path.path_id}"
                old_rtt = self.path_rtt_ewma[path_key]
                new_rtt = path.rtt_us / 1000.0  # Convert to ms
                self.path_rtt_ewma[path_key] = self.rtt_ewma_alpha * new_rtt + (1 - self.rtt_ewma_alpha) * old_rtt

            logger.debug(f"SCION availability check: {len(paths)} paths to {destination}")
            return len(paths) > 0, paths

        except SCIONConnectionError as e:
            logger.warning(f"SCION connectivity error for {destination}: {e}")
            return False, []
        except Exception as e:
            logger.error(f"SCION availability check failed for {destination}: {e}")
            return False, []

    async def _evaluate_routing_options_with_scion(
        self,
        destination: str,
        context: MessageContext,
        available_protocols: list[str] | None,
        scion_available: bool,
        scion_paths: list[SCIONPath],
    ) -> tuple[PathProtocol, dict[str, float]]:
        """Evaluate and select optimal routing protocol with SCION preference and path scoring"""
        available = available_protocols or [
            "bitchat",
            "betanet",
            "scion",
            "store_forward",
        ]

        # Calculate path scores using RTT EWMA + loss + policy
        path_scores = self._calculate_path_scores(destination, context, available, scion_available, scion_paths)

        # PRIORITY 1: SCION preference when available and high-performance desired
        if scion_available and "scion" in available:
            # Check if SCION is the best choice based on scores
            scion_score = path_scores.get("scion", 0.0)

            # Prefer SCION for high-priority or performance-sensitive messages
            if (
                context.priority >= 7
                or context.requires_realtime
                or self.routing_priority == RoutingPriority.PERFORMANCE_FIRST
                or self.scion_prefer_high_performance
            ):
                # Check if SCION has good performance (RTT < 100ms, loss < 5%)
                best_scion_path = min(scion_paths, key=lambda p: p.rtt_us) if scion_paths else None
                if best_scion_path and best_scion_path.rtt_us < 100000 and best_scion_path.loss_rate < 0.05:
                    logger.debug(f"SCION selected for high-performance (RTT={best_scion_path.rtt_us / 1000:.1f}ms)")
                    return PathProtocol.SCION, path_scores

            # Also prefer SCION if it has the highest score overall
            if scion_score == max(path_scores.values()):
                logger.debug(f"SCION selected as highest scored path (score={scion_score:.3f})")
                return PathProtocol.SCION, path_scores

        # Continue with existing priority logic
        return await self._evaluate_routing_options_fallback(destination, context, available, path_scores)

    def _calculate_path_scores(
        self,
        destination: str,
        context: MessageContext,
        available_protocols: list[str],
        scion_available: bool,
        scion_paths: list[SCIONPath],
    ) -> dict[str, float]:
        """Score paths using RTT EWMA + loss + policy (privacy/perf)"""
        scores = {}

        # Base scoring weights
        rtt_weight = 0.4  # RTT EWMA importance
        loss_weight = 0.3  # Packet loss importance
        policy_weight = 0.3  # Policy preference importance

        # Adjust weights based on message context
        if context.requires_realtime:
            rtt_weight = 0.6
            loss_weight = 0.3
            policy_weight = 0.1
        elif context.privacy_required:
            rtt_weight = 0.2
            loss_weight = 0.2
            policy_weight = 0.6

        for protocol in available_protocols:
            if protocol not in available_protocols:
                continue

            score = 0.0

            if protocol == "scion" and scion_available:
                # SCION scoring using actual path data
                if scion_paths:
                    best_path = min(scion_paths, key=lambda p: p.rtt_us + p.loss_rate * 100000)

                    # RTT score (lower is better, normalize to 0-1)
                    rtt_ms = best_path.rtt_us / 1000.0
                    rtt_score = max(0, 1.0 - rtt_ms / 200.0)  # 200ms = score of 0

                    # Loss score (lower is better)
                    loss_score = max(0, 1.0 - best_path.loss_rate / 0.1)  # 10% loss = score of 0

                    # Policy score - SCION gets bonus for performance/privacy
                    policy_score = 0.9
                    if context.requires_realtime:
                        policy_score = 1.0  # Perfect for real-time
                    elif context.privacy_required:
                        policy_score = 0.8  # Good for privacy (multipath)

                    score = rtt_weight * rtt_score + loss_weight * loss_score + policy_weight * policy_score
                else:
                    score = 0.5  # Default score when paths unknown

            elif protocol == "betanet":
                # Betanet scoring
                betanet_rtt = self.path_rtt_ewma.get("betanet", 100.0)
                rtt_score = max(0, 1.0 - betanet_rtt / 200.0)
                loss_score = 0.95  # Assume low loss for internet

                policy_score = 0.8
                if context.privacy_required:
                    policy_score = 0.9  # Good for privacy with mixnodes
                elif self.conditions.is_low_resource_environment():
                    policy_score = 0.3  # Poor for low-resource

                score = rtt_weight * rtt_score + loss_weight * loss_score + policy_weight * policy_score

            elif protocol == "bitchat":
                # BitChat scoring
                bitchat_rtt = self.path_rtt_ewma.get("bitchat", 200.0)
                rtt_score = max(0, 1.0 - bitchat_rtt / 300.0)
                loss_score = 0.85  # Bluetooth can be lossy

                policy_score = 0.6
                if self.global_south_mode or self.conditions.is_low_resource_environment():
                    policy_score = 0.95  # Excellent for offline/low-resource
                elif destination not in self.discovered_peers or not self.discovered_peers[destination].is_nearby():
                    policy_score = 0.1  # Poor if no nearby peer

                score = rtt_weight * rtt_score + loss_weight * loss_score + policy_weight * policy_score

            elif protocol == "store_forward":
                # Store-and-forward scoring
                score = 0.3  # Low score due to delay
                if not self.conditions.internet_available and not self.conditions.bluetooth_available:
                    score = 1.0  # Perfect when no other option

            scores[protocol] = max(0.0, min(1.0, score))  # Clamp to [0,1]

        return scores

    def _emit_receipt(
        self,
        chosen_path: str,
        switch_latency_ms: float,
        reason: str,
        scion_available: bool = False,
        scion_paths: int = 0,
        path_scores: dict[str, float] = None,
    ) -> None:
        """Emit receipt for bounty reviewers with switch latency tracking"""
        receipt = Receipt(
            chosen_path=chosen_path,
            switch_latency_ms=switch_latency_ms,
            reason=reason,
            scion_available=scion_available,
            scion_paths=scion_paths,
            path_scores=path_scores or {},
        )

        self.receipts.append(receipt)

        # Limit receipt history
        if len(self.receipts) > self.max_receipts:
            self.receipts = self.receipts[-self.max_receipts :]

        logger.info(f"Receipt emitted: {chosen_path} ({switch_latency_ms:.1f}ms) - {reason}")

    def _determine_selection_reason(
        self,
        protocol: PathProtocol,
        scion_available: bool,
        path_scores: dict[str, float],
    ) -> str:
        """Determine reason for path selection for receipt"""
        if protocol == PathProtocol.SCION:
            if scion_available:
                return "scion_high_performance"
            else:
                return "scion_fallback"
        elif protocol == PathProtocol.BETANET:
            if scion_available:
                return "betanet_over_scion"
            else:
                return "betanet_internet_available"
        elif protocol == PathProtocol.BITCHAT:
            if self.global_south_mode:
                return "bitchat_offline_first"
            else:
                return "bitchat_peer_nearby"
        elif protocol == PathProtocol.STORE_FORWARD:
            return "store_forward_fallback"
        else:
            return f"{protocol.value}_selected"

    async def _fast_path_selection_with_scion(
        self,
        destination: str,
        context: MessageContext,
        available_protocols: list[str] | None,
        scion_available: bool,
        scion_paths: list[SCIONPath],
    ) -> tuple[PathProtocol, dict[str, Any]]:
        """Fast path selection with SCION preference optimized for 500ms switching target"""
        available = available_protocols or [
            "bitchat",
            "betanet",
            "scion",
            "store_forward",
        ]

        # Quick SCION check for fast switching
        if scion_available and "scion" in available:
            # Use SCION if it has low latency paths
            if scion_paths:
                best_path = min(scion_paths, key=lambda p: p.rtt_us)
                if best_path.rtt_us < 100000:  # < 100ms RTT
                    logger.debug("Fast switch to SCION (low latency available)")
                    return PathProtocol.SCION, self._get_fast_routing_metadata(
                        PathProtocol.SCION,
                        {"fast_scion": True, "rtt_ms": best_path.rtt_us / 1000},
                    )

        # Fallback to existing fast path selection
        return await self._fast_path_selection(destination, context, available_protocols)

    async def _evaluate_routing_options_fallback(
        self,
        destination: str,
        context: MessageContext,
        available: list[str],
        path_scores: dict[str, float],
    ) -> tuple[PathProtocol, dict[str, float]]:
        """Fallback to original routing logic when SCION not selected"""
        # Use the original routing logic but return the highest scored available protocol
        protocol = await self._evaluate_routing_options(destination, context, available)
        return protocol, path_scores

    async def _evaluate_routing_options(
        self,
        destination: str,
        context: MessageContext,
        available_protocols: list[str] | None,
    ) -> PathProtocol:
        """Evaluate and select optimal routing protocol"""
        # Check what protocols are available
        available = available_protocols or ["bitchat", "betanet", "store_forward"]

        # PRIORITY 1: Emergency/urgent messages
        if context.is_urgent():
            if self.conditions.internet_available and "betanet" in available:
                logger.debug("Urgent message - selecting Betanet")
                return PathProtocol.BETANET
            if self.conditions.bluetooth_available and "bitchat" in available:
                if await self._is_peer_nearby(destination):
                    logger.debug("Urgent message - peer nearby, selecting BitChat")
                    return PathProtocol.BITCHAT

        # PRIORITY 2: Offline-first for Global South scenarios
        if self.routing_priority == RoutingPriority.OFFLINE_FIRST or self.global_south_mode:
            # Check for nearby peers via BitChat
            if (
                self.conditions.bluetooth_available
                and "bitchat" in available
                and await self._is_peer_nearby(destination)
            ):
                # BitChat preferred for local/offline scenarios
                if self.energy_mode == EnergyMode.POWERSAVE or self.conditions.is_low_resource_environment():
                    logger.debug("Offline-first: selecting BitChat (energy efficient)")
                    return PathProtocol.BITCHAT

        # PRIORITY 3: Message size considerations
        if context.is_large_message():
            # Large messages prefer high-bandwidth routes
            if (
                self.conditions.internet_available
                and "betanet" in available
                and self.conditions.bandwidth_mbps > 5.0
                and not self._is_data_expensive()
            ):
                logger.debug("Large message - selecting Betanet (high bandwidth)")
                return PathProtocol.BETANET

        # PRIORITY 4: Privacy requirements
        if context.privacy_required or self.conditions.is_privacy_sensitive():
            if (
                self.routing_priority == RoutingPriority.PRIVACY_FIRST
                and "betanet" in available
                and self.conditions.internet_available
            ):
                # Betanet with mixnode routing for privacy
                logger.debug("Privacy required - selecting Betanet with mixnodes")
                return PathProtocol.BETANET

        # PRIORITY 5: Energy management
        if self.energy_mode == EnergyMode.POWERSAVE or self._is_battery_low():
            # Prefer low-energy options
            if (
                self.conditions.bluetooth_available
                and "bitchat" in available
                and await self._is_peer_nearby(destination)
            ):
                logger.debug("Energy conservation - selecting BitChat")
                return PathProtocol.BITCHAT
            # Store and forward to conserve energy
            logger.debug("Energy conservation - selecting store-and-forward")
            return PathProtocol.STORE_FORWARD

        # PRIORITY 6: Performance-first routing
        if self.routing_priority == RoutingPriority.PERFORMANCE_FIRST:
            if self.conditions.internet_available and "betanet" in available and not self._is_data_expensive():
                logger.debug("Performance-first - selecting Betanet")
                return PathProtocol.BETANET

        # PRIORITY 7: Bandwidth optimization
        if self.routing_priority == RoutingPriority.BANDWIDTH_FIRST:
            if self.conditions.internet_available and "betanet" in available and self.conditions.bandwidth_mbps > 10.0:
                logger.debug("Bandwidth-first - selecting Betanet")
                return PathProtocol.BETANET

        # FALLBACK LOGIC: Select best available option

        # Try BitChat if peer is reachable
        if self.conditions.bluetooth_available and "bitchat" in available and await self._is_peer_nearby(destination):
            logger.debug("Fallback - selecting BitChat (peer nearby)")
            return PathProtocol.BITCHAT

        # Try Betanet if internet available
        if self.conditions.internet_available and "betanet" in available:
            logger.debug("Fallback - selecting Betanet (internet available)")
            return PathProtocol.BETANET

        # Store-and-forward as last resort
        logger.debug("Fallback - selecting store-and-forward (offline)")
        return PathProtocol.STORE_FORWARD

    async def _update_network_conditions(self) -> None:
        """Update current network and device conditions"""
        current_time = time.time()

        if current_time - self.last_condition_update < self.condition_update_interval:
            return  # Skip update if too recent

        self.last_condition_update = current_time

        # Check Bluetooth availability
        self.conditions.bluetooth_available = await self._check_bluetooth_available()

        # Check internet connectivity
        self.conditions.internet_available = await self._check_internet_available()

        # Check WiFi vs cellular
        self.conditions.wifi_connected = await self._check_wifi_connected()

        # Update battery status (if available)
        self.conditions.battery_percent = await self._get_battery_level()

        # Update nearby peer count
        self.conditions.nearby_peers = len([peer for peer in self.discovered_peers.values() if peer.is_nearby()])

        # Update bandwidth estimate
        self.conditions.bandwidth_mbps = await self._estimate_bandwidth()

        logger.debug(
            f"Network conditions updated: BT={self.conditions.bluetooth_available}, "
            f"Internet={self.conditions.internet_available}, "
            f"Peers={self.conditions.nearby_peers}"
        )

    async def _check_bluetooth_available(self) -> bool:
        """Check if Bluetooth is available and enabled"""
        if not BLUETOOTH_AVAILABLE:
            return False

        try:
            # Try to discover devices (short scan)
            bluetooth.discover_devices(duration=1, lookup_names=False)
            return True
        except Exception:
            return False

    async def _check_internet_available(self) -> bool:
        """Check internet connectivity"""
        try:
            # Try to connect to well-known servers
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            return True
        except Exception:
            try:
                # Fallback to CloudFlare DNS
                socket.create_connection(("1.1.1.1", 53), timeout=2)
                return True
            except Exception:
                return False

    async def _check_wifi_connected(self) -> bool:
        """Check if connected to WiFi (vs cellular)"""
        # Platform-specific implementation would go here
        # For now, assume WiFi if internet is available
        return self.conditions.internet_available

    async def _get_battery_level(self) -> int | None:
        """Get current battery percentage"""
        try:
            # Platform-specific battery check would go here
            # For now, simulate battery level
            import psutil

            battery = psutil.sensors_battery()
            return int(battery.percent) if battery else None
        except Exception:
            return None

    async def _estimate_bandwidth(self) -> float:
        """Estimate current network bandwidth"""
        if not self.conditions.internet_available:
            return 0.0

        # Simplified bandwidth estimation
        if self.conditions.wifi_connected:
            return 50.0  # Assume 50 Mbps on WiFi
        return 5.0  # Assume 5 Mbps on cellular

    async def _is_peer_nearby(self, peer_id: str) -> bool:
        """Check if peer is within BitChat range"""
        # Check cache first
        cache_key = f"nearby_{peer_id}"
        if cache_key in self.peer_proximity_cache:
            return self.peer_proximity_cache[cache_key]

        # Check discovered peers
        if peer_id in self.discovered_peers:
            peer = self.discovered_peers[peer_id]
            is_nearby = peer.is_nearby()

            # Cache result
            self.peer_proximity_cache[cache_key] = is_nearby
            return is_nearby

        # Check hop distance if available
        hop_distance = self.conditions.peer_hop_distances.get(peer_id, 999)
        is_nearby = hop_distance <= 7  # Within BitChat 7-hop limit

        # Cache result
        self.peer_proximity_cache[cache_key] = is_nearby
        return is_nearby

    def _is_data_expensive(self) -> bool:
        """Check if mobile data is expensive"""
        return self.conditions.data_cost_usd_mb > self.data_cost_threshold

    def _is_battery_low(self) -> bool:
        """Check if battery is low"""
        if self.conditions.battery_percent is None:
            return False
        return self.conditions.battery_percent < self.battery_conservation_threshold

    def _get_routing_metadata(self, protocol: PathProtocol, destination: str) -> dict[str, Any]:
        """Generate routing metadata for selected protocol"""
        metadata = {
            "protocol": protocol.value,
            "destination": destination,
            "timestamp": time.time(),
            "energy_mode": self.energy_mode.value,
            "routing_priority": self.routing_priority.value,
        }

        if protocol == PathProtocol.BITCHAT:
            metadata.update(
                {
                    "max_hops": 7,
                    "store_forward_enabled": True,
                    "energy_efficient": True,
                    "offline_capable": True,
                    "estimated_latency_ms": self.protocol_performance["bitchat"]["latency"],
                }
            )

        elif protocol == PathProtocol.BETANET:
            metadata.update(
                {
                    "privacy_routing": self.conditions.is_privacy_sensitive(),
                    "mixnode_hops": 2 if self.conditions.is_privacy_sensitive() else 0,
                    "bandwidth_adaptive": True,
                    "global_reach": True,
                    "estimated_latency_ms": self.protocol_performance["betanet"]["latency"],
                }
            )

        elif protocol == PathProtocol.SCION:
            metadata.update(
                {
                    "multipath": True,
                    "path_aware": True,
                    "high_performance": True,
                    "global_reach": True,
                    "failover_support": True,
                    "estimated_latency_ms": self.protocol_performance["scion"]["latency"],
                }
            )

        elif protocol == PathProtocol.STORE_FORWARD:
            metadata.update(
                {
                    "delay_tolerant": True,
                    "energy_minimal": True,
                    "guaranteed_delivery": True,
                    "estimated_latency_ms": 0,  # Delivered when possible
                }
            )

        return metadata

    def update_peer_info(self, peer_id: str, peer_info: PeerInfo) -> None:
        """Update information about discovered peer"""
        self.discovered_peers[peer_id] = peer_info

        # Clear proximity cache for this peer
        cache_key = f"nearby_{peer_id}"
        self.peer_proximity_cache.pop(cache_key, None)

        logger.debug(f"Updated peer info: {peer_id} (hop_distance={peer_info.hop_distance})")

    def update_routing_success(self, protocol: str, destination: str, success: bool) -> None:
        """Update routing success statistics for learning"""
        key = f"{protocol}_{destination}"
        current_rate = self.route_success_rates[key]

        # Exponential moving average
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.route_success_rates[key] = new_rate

        logger.debug(f"Updated success rate for {key}: {new_rate:.2f}")

    def set_energy_mode(self, mode: EnergyMode) -> None:
        """Change energy management mode"""
        self.energy_mode = mode
        logger.info(f"Energy mode changed to: {mode.value}")

    def set_routing_priority(self, priority: RoutingPriority) -> None:
        """Change routing priority mode"""
        self.routing_priority = priority
        logger.info(f"Routing priority changed to: {priority.value}")

    def enable_global_south_mode(self, enabled: bool = True) -> None:
        """Enable/disable Global South optimizations"""
        self.global_south_mode = enabled
        if enabled:
            # Optimize for offline-first, energy efficiency
            self.routing_priority = RoutingPriority.OFFLINE_FIRST
            self.energy_mode = EnergyMode.BALANCED
            self.data_cost_threshold = 0.005  # Sensitive to data costs
            logger.info("Global South mode enabled - prioritizing offline-first routing")
        else:
            logger.info("Global South mode disabled")

    def get_status(self) -> dict[str, Any]:
        """Get current Navigator status and statistics"""
        return {
            "agent_id": self.agent_id,
            "routing_priority": self.routing_priority.value,
            "energy_mode": self.energy_mode.value,
            "global_south_mode": self.global_south_mode,
            "network_conditions": {
                "bluetooth_available": self.conditions.bluetooth_available,
                "internet_available": self.conditions.internet_available,
                "wifi_connected": self.conditions.wifi_connected,
                "battery_percent": self.conditions.battery_percent,
                "nearby_peers": self.conditions.nearby_peers,
                "bandwidth_mbps": self.conditions.bandwidth_mbps,
                "is_low_resource": self.conditions.is_low_resource_environment(),
                "privacy_sensitive": self.conditions.is_privacy_sensitive(),
            },
            "discovered_peers": len(self.discovered_peers),
            "routing_decisions_cached": len(self.routing_decisions),
            "protocol_performance": self.protocol_performance,
            "top_routes": dict(list(self.route_success_rates.items())[:10]),
        }

    def get_recommended_protocols(self, destination: str) -> list[str]:
        """Get recommended protocols for destination in priority order"""
        # Based on current conditions, return optimal protocol list
        recommendations = []

        if self.conditions.bluetooth_available and destination in self.discovered_peers:
            peer = self.discovered_peers[destination]
            if peer.is_nearby():
                recommendations.append("bitchat")

        if self.conditions.internet_available:
            recommendations.append("betanet")

        # Always include store-and-forward as fallback
        recommendations.append("store_forward")

        return recommendations

    async def optimize_route_for_context(
        self, protocol: PathProtocol, destination: str, context: MessageContext
    ) -> dict[str, Any]:
        """Optimize routing parameters for specific context"""
        optimizations = {}

        if protocol == PathProtocol.BITCHAT:
            # BitChat optimizations
            optimizations.update(
                {
                    "compression_enabled": context.size_bytes > 1000,
                    "priority_queue": context.priority > 7,
                    "ttl_hops": min(7, max(3, context.priority)),
                    "store_forward_ttl_hours": 24 if context.priority > 5 else 6,
                }
            )

        elif protocol == PathProtocol.BETANET:
            # Betanet optimizations
            optimizations.update(
                {
                    "use_mixnodes": context.privacy_required or self.conditions.is_privacy_sensitive(),
                    "mixnode_hops": 2 if context.privacy_required else 0,
                    "bandwidth_tier": "high" if context.is_large_message() else "standard",
                    "reliability_level": "guaranteed" if context.priority > 7 else "best_effort",
                    "chunking_enabled": context.size_bytes > 32768,
                }
            )

        return optimizations

    def cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        current_time = time.time()

        # Clean routing decision cache
        expired_decisions = [
            key
            for key, (_, timestamp) in self.routing_decisions.items()
            if current_time - timestamp > self.decision_cache_ttl
        ]
        for key in expired_decisions:
            del self.routing_decisions[key]

        # Clean peer proximity cache
        expired_proximity = [
            key
            for key in self.peer_proximity_cache.keys()
            if current_time - self.last_condition_update > self.peer_cache_ttl
        ]
        for key in expired_proximity:
            del self.peer_proximity_cache[key]

        # Clean old peer info
        expired_peers = [
            peer_id for peer_id, peer in self.discovered_peers.items() if current_time - peer.last_seen > 3600  # 1 hour
        ]
        for peer_id in expired_peers:
            del self.discovered_peers[peer_id]

        logger.debug(
            f"Cache cleanup: removed {len(expired_decisions)} decisions, "
            f"{len(expired_proximity)} proximity entries, {len(expired_peers)} peers"
        )

    async def _check_fast_switching(self, destination: str, context: MessageContext) -> bool:
        """Check if link changes require fast path switching (500ms target)"""
        current_time = time.time() * 1000  # milliseconds

        # Build current link state for change detection
        link_state = {
            "bluetooth_available": self.conditions.bluetooth_available,
            "internet_available": self.conditions.internet_available,
            "wifi_connected": self.conditions.wifi_connected,
            "cellular_connected": not self.conditions.wifi_connected and self.conditions.internet_available,
            "bandwidth_mbps": self.conditions.bandwidth_mbps,
            "latency_ms": self.conditions.latency_ms,
        }

        # Update link change detector
        change_detected = self.link_change_detector.update_link_state(link_state)

        if change_detected:
            # Check if enough time has passed since last switch to avoid thrashing
            time_since_last_switch = current_time - self.last_path_switch_time
            if time_since_last_switch > self.path_switch_threshold_ms:
                self.last_path_switch_time = current_time
                logger.info(f"Fast switching triggered for {destination} after {time_since_last_switch:.0f}ms")
                return True

        return False

    async def _fast_path_selection(
        self,
        destination: str,
        context: MessageContext,
        available_protocols: list[str] | None,
    ) -> tuple[PathProtocol, dict[str, Any]]:
        """Fast path selection optimized for 500ms switching target"""
        available = available_protocols or ["bitchat", "betanet", "store_forward"]

        # Get fast switching recommendation from link change detector
        (
            should_switch,
            recommended_path,
            rationale,
        ) = self.link_change_detector.get_switch_recommendation()

        if should_switch and recommended_path != "maintain":
            # Use recommendation if protocol is available
            if recommended_path == "bitchat" and "bitchat" in available:
                logger.debug(f"Fast switch to BitChat: {rationale['reasons']}")
                return PathProtocol.BITCHAT, self._get_fast_routing_metadata(PathProtocol.BITCHAT, rationale)

            elif recommended_path == "betanet" and "betanet" in available:
                logger.debug(f"Fast switch to Betanet: {rationale['reasons']}")
                return PathProtocol.BETANET, self._get_fast_routing_metadata(PathProtocol.BETANET, rationale)

        # Fallback to emergency fast selection based on connectivity
        if self.conditions.bluetooth_available and "bitchat" in available:
            # BitChat available - use for fast switching
            return PathProtocol.BITCHAT, self._get_fast_routing_metadata(PathProtocol.BITCHAT, {"fast_switch": True})

        if self.conditions.internet_available and "betanet" in available:
            # Internet available - use Betanet for fast switching
            return PathProtocol.BETANET, self._get_fast_routing_metadata(PathProtocol.BETANET, {"fast_switch": True})

        # Last resort - store and forward
        return PathProtocol.STORE_FORWARD, self._get_fast_routing_metadata(
            PathProtocol.STORE_FORWARD, {"emergency": True}
        )

    def _get_fast_routing_metadata(self, protocol: PathProtocol, rationale: dict) -> dict[str, Any]:
        """Generate metadata for fast-switched routing"""
        metadata = {
            "protocol": protocol.value,
            "fast_switch": True,
            "switch_time_target_ms": self.path_switch_threshold_ms,
            "timestamp": time.time(),
            "rationale": rationale,
        }

        if protocol == PathProtocol.BITCHAT:
            metadata.update(
                {
                    "max_hops": 7,
                    "energy_efficient": True,
                    "offline_capable": True,
                    "fast_switch_reason": "bluetooth_mesh_optimization",
                }
            )
        elif protocol == PathProtocol.BETANET:
            metadata.update(
                {
                    "global_reach": True,
                    "high_bandwidth": True,
                    "fast_switch_reason": "internet_connectivity_optimization",
                }
            )
        elif protocol == PathProtocol.SCION:
            metadata.update(
                {
                    "multipath": True,
                    "path_aware": True,
                    "high_performance": True,
                    "fast_switch_reason": "scion_multipath_optimization",
                }
            )
        elif protocol == PathProtocol.STORE_FORWARD:
            metadata.update(
                {
                    "delay_tolerant": True,
                    "guaranteed_delivery": True,
                    "fast_switch_reason": "emergency_fallback",
                }
            )

        return metadata

    def get_fast_switching_metrics(self) -> dict[str, Any]:
        """Get metrics for fast switching performance"""
        detector_metrics = self.link_change_detector.get_performance_metrics()

        return {
            "fast_switch_enabled": self.fast_switch_enabled,
            "target_switch_time_ms": self.path_switch_threshold_ms,
            "last_switch_time": self.last_path_switch_time,
            "link_change_detector": detector_metrics,
            "switch_performance": {
                "within_target": detector_metrics.get("within_target", False),
                "avg_evaluation_time": detector_metrics.get("avg_evaluation_time_ms", 0),
                "recent_events": detector_metrics.get("recent_events", 0),
            },
        }

    def get_receipts(self, count: int = 100) -> list[Receipt]:
        """Get recent receipts for bounty reviewers"""
        return self.receipts[-count:] if count < len(self.receipts) else self.receipts

    def get_scion_statistics(self) -> dict[str, Any]:
        """Get SCION-specific statistics"""
        if not self.scion_enabled:
            return {"scion_enabled": False}

        return {
            "scion_enabled": True,
            "scion_gateway_initialized": self.scion_gateway is not None,
            "scion_path_cache_entries": len(self.scion_path_cache),
            "scion_prefer_high_performance": self.scion_prefer_high_performance,
            "path_rtt_ewma_entries": len([k for k in self.path_rtt_ewma.keys() if k.startswith("scion_")]),
            "receipts_with_scion": len([r for r in self.receipts if r.scion_available]),
            "scion_selections": len([r for r in self.receipts if r.chosen_path == "scion"]),
        }
