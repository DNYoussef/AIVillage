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

import logging
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Bluetooth imports with fallback
try:
    import bluetooth

    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PathProtocol(Enum):
    """Available routing protocols"""

    BITCHAT = "bitchat"  # Bluetooth mesh, offline
    BETANET = "betanet"  # HTX/HTXQUIC, online
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
            and (
                self.geographic_distance_km is None or self.geographic_distance_km < 1.0
            )
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

        # Routing statistics and learning
        self.route_success_rates: dict[str, float] = defaultdict(lambda: 0.8)
        self.protocol_performance: dict[str, dict[str, float]] = {
            "bitchat": {"latency": 200.0, "reliability": 0.85, "energy_cost": 0.2},
            "betanet": {"latency": 100.0, "reliability": 0.95, "energy_cost": 0.8},
            "store_forward": {"latency": 0.0, "reliability": 1.0, "energy_cost": 0.1},
        }

        # Decision caching
        self.routing_decisions: dict[str, tuple[PathProtocol, float]] = {}
        self.decision_cache_ttl = 60.0  # 1 minute

        # Global South optimizations
        self.global_south_mode = True  # Prioritize offline-first
        self.data_cost_threshold = 0.005  # $0.005/MB threshold
        self.battery_conservation_threshold = 25  # Start conserving at 25%

        # Privacy settings
        self.privacy_aware = True
        self.mixnode_preferences = ["tor", "i2p", "betanet_mixnode"]

        logger.info(
            f"Navigator initialized: {self.agent_id} (priority={routing_priority.value})"
        )

    async def select_path(
        self,
        destination: str,
        message_context: MessageContext,
        available_protocols: list[str] | None = None,
    ) -> tuple[PathProtocol, dict[str, Any]]:
        """Core path selection algorithm implementing AIVillage routing priorities:

        1. BitChat-first for Global South (offline/low-power)
        2. Betanet for wide-area/high-bandwidth needs
        3. Store-and-forward when neither available
        4. Adaptive based on energy, privacy, and performance

        Returns:
            Tuple of (selected_protocol, routing_metadata)
        """
        # Update network conditions
        await self._update_network_conditions()

        # Check decision cache
        cache_key = (
            f"{destination}_{message_context.priority}_{message_context.size_bytes}"
        )
        if cache_key in self.routing_decisions:
            cached_decision, cache_time = self.routing_decisions[cache_key]
            if time.time() - cache_time < self.decision_cache_ttl:
                logger.debug(f"Using cached routing decision: {cached_decision.value}")
                return cached_decision, self._get_routing_metadata(
                    cached_decision, destination
                )

        # Core routing decision logic
        selected_protocol = await self._evaluate_routing_options(
            destination, message_context, available_protocols
        )

        # Cache decision
        self.routing_decisions[cache_key] = (selected_protocol, time.time())

        # Generate routing metadata
        routing_metadata = self._get_routing_metadata(selected_protocol, destination)

        logger.info(
            f"Selected route for {destination}: {selected_protocol.value} "
            f"(size={message_context.size_bytes}, priority={message_context.priority})"
        )

        return selected_protocol, routing_metadata

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
        if (
            self.routing_priority == RoutingPriority.OFFLINE_FIRST
            or self.global_south_mode
        ):
            # Check for nearby peers via BitChat
            if (
                self.conditions.bluetooth_available
                and "bitchat" in available
                and await self._is_peer_nearby(destination)
            ):
                # BitChat preferred for local/offline scenarios
                if (
                    self.energy_mode == EnergyMode.POWERSAVE
                    or self.conditions.is_low_resource_environment()
                ):
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
            if (
                self.conditions.internet_available
                and "betanet" in available
                and not self._is_data_expensive()
            ):
                logger.debug("Performance-first - selecting Betanet")
                return PathProtocol.BETANET

        # PRIORITY 7: Bandwidth optimization
        if self.routing_priority == RoutingPriority.BANDWIDTH_FIRST:
            if (
                self.conditions.internet_available
                and "betanet" in available
                and self.conditions.bandwidth_mbps > 10.0
            ):
                logger.debug("Bandwidth-first - selecting Betanet")
                return PathProtocol.BETANET

        # FALLBACK LOGIC: Select best available option

        # Try BitChat if peer is reachable
        if (
            self.conditions.bluetooth_available
            and "bitchat" in available
            and await self._is_peer_nearby(destination)
        ):
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
        self.conditions.nearby_peers = len(
            [peer for peer in self.discovered_peers.values() if peer.is_nearby()]
        )

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

    def _get_routing_metadata(
        self, protocol: PathProtocol, destination: str
    ) -> dict[str, Any]:
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
                    "estimated_latency_ms": self.protocol_performance["bitchat"][
                        "latency"
                    ],
                }
            )

        elif protocol == PathProtocol.BETANET:
            metadata.update(
                {
                    "privacy_routing": self.conditions.is_privacy_sensitive(),
                    "mixnode_hops": 2 if self.conditions.is_privacy_sensitive() else 0,
                    "bandwidth_adaptive": True,
                    "global_reach": True,
                    "estimated_latency_ms": self.protocol_performance["betanet"][
                        "latency"
                    ],
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

        logger.debug(
            f"Updated peer info: {peer_id} (hop_distance={peer_info.hop_distance})"
        )

    def update_routing_success(
        self, protocol: str, destination: str, success: bool
    ) -> None:
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
            logger.info(
                "Global South mode enabled - prioritizing offline-first routing"
            )
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
                    "use_mixnodes": context.privacy_required
                    or self.conditions.is_privacy_sensitive(),
                    "mixnode_hops": 2 if context.privacy_required else 0,
                    "bandwidth_tier": "high"
                    if context.is_large_message()
                    else "standard",
                    "reliability_level": "guaranteed"
                    if context.priority > 7
                    else "best_effort",
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
            peer_id
            for peer_id, peer in self.discovered_peers.items()
            if current_time - peer.last_seen > 3600  # 1 hour
        ]
        for peer_id in expired_peers:
            del self.discovered_peers[peer_id]

        logger.debug(
            f"Cache cleanup: removed {len(expired_decisions)} decisions, "
            f"{len(expired_proximity)} proximity entries, {len(expired_peers)} peers"
        )
