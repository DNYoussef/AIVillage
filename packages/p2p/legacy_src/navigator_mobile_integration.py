"""
Navigator Policy + Mobile Resource Manager Integration - Prompt 3

This module wires together the Navigator's transport selection with the Mobile
Resource Manager to create intelligent, constraint-aware routing decisions.

Key integration features:
- Battery-aware transport selection (BitChat-first under low power)
- Thermal throttling integration with transport decisions
- Memory-constrained chunk sizing coordination
- Network cost-aware routing preferences
- Real-time policy adaptation based on device state
- Seamless BitChat ↔ Betanet handoff based on device constraints

Integration Point: Multi-protocol routing with resource awareness
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from navigation.scion_navigator import RoutingDecision, SCIONAwareNavigator, TransportCandidate, TransportPriority
    from production.monitoring.mobile.resource_management import (
        BatteryThermalResourceManager,
        PowerMode,
        ResourceState,
        TransportPreference,
    )
except ImportError:
    # Mock classes for development/testing
    from dataclasses import dataclass, field
    from enum import Enum

    class PowerMode(Enum):
        PERFORMANCE = "performance"
        BALANCED = "balanced"
        POWER_SAVE = "power_save"
        CRITICAL = "critical"

    class TransportPreference(Enum):
        BITCHAT_ONLY = "bitchat_only"
        BITCHAT_PREFERRED = "bitchat_preferred"
        BALANCED = "balanced"
        BETANET_PREFERRED = "betanet_preferred"
        BETANET_ONLY = "betanet_only"

    class TransportPriority(Enum):
        CRITICAL = 1
        HIGH = 2
        NORMAL = 3
        LOW = 4
        FALLBACK = 5

    @dataclass
    class TransportCandidate:
        transport_type: str
        endpoint: str
        priority: TransportPriority
        estimated_latency_ms: float
        reliability_score: float
        cost_factor: float
        metadata: dict = field(default_factory=dict)

    @dataclass
    class RoutingDecision:
        primary_transport: TransportCandidate
        backup_transports: list[TransportCandidate]
        decision_reason: str
        confidence_score: float
        estimated_total_time_ms: float
        route_metadata: dict = field(default_factory=dict)

    @dataclass
    class ResourceState:
        power_mode: PowerMode = PowerMode.BALANCED
        transport_preference: TransportPreference = TransportPreference.BALANCED
        active_policies: list[str] = field(default_factory=list)

    class SCIONAwareNavigator:
        async def get_available_transports(self, target: str) -> list[TransportCandidate]:
            return []

    class BatteryThermalResourceManager:
        async def get_current_state(self) -> ResourceState:
            return ResourceState()


logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of device constraints affecting transport selection."""

    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    THERMAL_HIGH = "thermal_high"
    THERMAL_CRITICAL = "thermal_critical"
    MEMORY_LOW = "memory_low"
    CELLULAR_DATA = "cellular_data"
    NETWORK_POOR = "network_poor"
    MEMORY_CONSTRAINED = "memory_constrained"
    BATTERY_SAVER = "battery_saver"


@dataclass
class TransportConstraints:
    """Device constraints that affect transport selection."""

    battery_level: int | None = None  # Percentage
    cpu_temp: float | None = None  # Celsius
    available_memory_gb: float | None = None
    network_type: str = "wifi"  # wifi, cellular, etc.
    data_cost_sensitive: bool = False
    active_constraints: list[ConstraintType] = field(default_factory=list)


@dataclass
class MobileAwareRoutingDecision(RoutingDecision):
    """Enhanced routing decision with mobile-specific context."""

    applied_constraints: list[ConstraintType] = field(default_factory=list)
    resource_state: ResourceState | None = None
    power_efficiency_score: float = 1.0  # Higher = more power efficient
    data_usage_estimate_mb: float = 0.0
    expected_battery_drain_percent: float = 0.0


class MobileResourceNavigator:
    """
    Navigator that integrates Mobile Resource Manager constraints into routing decisions.

    This creates the integration between transport selection and device constraints,
    enabling intelligent BitChat ↔ Betanet handoff based on real device state.
    """

    def __init__(
        self,
        base_navigator: SCIONAwareNavigator | None = None,
        resource_manager: BatteryThermalResourceManager | None = None,
        policy_config: dict[str, Any] | None = None,
    ):
        self.base_navigator = base_navigator
        self.resource_manager = resource_manager
        self.policy_config = policy_config or {}

        # Transport efficiency profiles (power consumption estimates)
        self.transport_efficiency = {
            "bitchat": {
                "power_factor": 0.3,
                "data_efficiency": 0.9,
            },  # Low power, high compression
            "betanet": {
                "power_factor": 0.7,
                "data_efficiency": 0.6,
            },  # Higher power, more features
            "scion": {"power_factor": 0.5, "data_efficiency": 0.8},  # Balanced
            "websocket": {
                "power_factor": 0.6,
                "data_efficiency": 0.4,
            },  # Higher power, low compression
        }

        logger.info("MobileResourceNavigator initialized with resource-aware routing")

    async def route_with_constraints(
        self,
        target: str,
        message_size_bytes: int,
        priority: TransportPriority = TransportPriority.NORMAL,
    ) -> MobileAwareRoutingDecision:
        """
        Make routing decision considering both transport availability and device constraints.

        This is the core integration method that combines Navigator transport selection
        with Mobile Resource Manager constraint evaluation.
        """
        # Step 1: Get current device resource state
        constraints = await self._evaluate_device_constraints()
        resource_state = None

        if self.resource_manager:
            try:
                resource_state = await self.resource_manager.get_current_state()
                logger.debug(
                    f"Resource state: {resource_state.power_mode}, transport pref: {resource_state.transport_preference}"
                )
            except Exception as e:
                logger.warning(f"Failed to get resource state: {e}")

        # Step 2: Get available transports from base navigator
        available_transports = []
        if self.base_navigator:
            try:
                available_transports = await self.base_navigator.get_available_transports(target)
            except Exception as e:
                logger.warning(f"Failed to get base transports: {e}")

        # Step 3: Add mobile-optimized transport options
        mobile_transports = self._generate_mobile_transport_candidates(target, message_size_bytes, constraints)
        all_transports = available_transports + mobile_transports

        # Step 4: Apply constraint-based filtering and scoring
        filtered_transports = self._apply_constraint_filtering(all_transports, constraints)
        scored_transports = self._score_transports_for_constraints(filtered_transports, constraints, message_size_bytes)

        # Step 5: Select optimal transport with mobile considerations
        if not scored_transports:
            # Fallback transport selection
            primary_transport = self._get_fallback_transport(target, constraints)
            backup_transports = []
            decision_reason = "No suitable transports found, using fallback"
            confidence_score = 0.3
        else:
            # Sort by composite score (includes power efficiency)
            scored_transports.sort(key=lambda x: x.metadata.get("composite_score", 0), reverse=True)
            primary_transport = scored_transports[0]
            backup_transports = scored_transports[1:3]  # Top 2 backups
            decision_reason = self._generate_decision_reason(constraints, primary_transport)
            confidence_score = primary_transport.metadata.get("confidence", 0.8)

        # Step 6: Calculate mobile-specific metrics
        power_efficiency = self._calculate_power_efficiency(primary_transport, message_size_bytes)
        data_usage_estimate = self._estimate_data_usage(primary_transport, message_size_bytes)
        battery_drain = self._estimate_battery_drain(primary_transport, message_size_bytes, constraints)

        return MobileAwareRoutingDecision(
            primary_transport=primary_transport,
            backup_transports=backup_transports,
            decision_reason=decision_reason,
            confidence_score=confidence_score,
            estimated_total_time_ms=primary_transport.estimated_latency_ms,
            route_metadata={
                "constraints_applied": [c.value for c in constraints.active_constraints],
                "resource_optimization": True,
                "mobile_aware": True,
            },
            applied_constraints=constraints.active_constraints,
            resource_state=resource_state,
            power_efficiency_score=power_efficiency,
            data_usage_estimate_mb=data_usage_estimate,
            expected_battery_drain_percent=battery_drain,
        )

    async def _evaluate_device_constraints(self) -> TransportConstraints:
        """Evaluate current device constraints that affect transport selection."""
        constraints = TransportConstraints()
        active_constraints = []

        try:
            # Get device profile from resource manager if available, otherwise use environment
            import os

            battery_level = int(os.getenv("BATTERY_LEVEL", "80"))
            cpu_temp = float(os.getenv("CPU_TEMP", "35.0"))
            network_type = os.getenv("NETWORK_TYPE", "wifi")

            constraints.battery_level = battery_level
            constraints.cpu_temp = cpu_temp
            constraints.network_type = network_type

            # Evaluate constraint conditions
            if battery_level <= 15:
                active_constraints.append(ConstraintType.BATTERY_CRITICAL)
            elif battery_level <= 30:
                active_constraints.append(ConstraintType.BATTERY_LOW)

            if cpu_temp > 60:
                active_constraints.append(ConstraintType.THERMAL_CRITICAL)
            elif cpu_temp > 45:
                active_constraints.append(ConstraintType.THERMAL_HIGH)

            if network_type == "cellular":
                active_constraints.append(ConstraintType.CELLULAR_DATA)
                constraints.data_cost_sensitive = True

            # If resource manager is available, enhance with real data
            if self.resource_manager:
                try:
                    # Get current resource metrics from resource manager
                    metrics = self.resource_manager.get_current_metrics()
                    if metrics.get("memory_pressure", 0) > 0.8:
                        active_constraints.append(ConstraintType.MEMORY_CONSTRAINED)
                    if metrics.get("battery_level", 1.0) < 0.3:
                        active_constraints.append(ConstraintType.BATTERY_SAVER)
                except Exception as e:
                    logger.warning(f"Resource manager integration error: {e}")

        except Exception as e:
            logger.warning(f"Error evaluating device constraints: {e}")

        constraints.active_constraints = active_constraints
        logger.debug(f"Active constraints: {[c.value for c in active_constraints]}")

        return constraints

    def _generate_mobile_transport_candidates(
        self, target: str, message_size: int, constraints: TransportConstraints
    ) -> list[TransportCandidate]:
        """Generate mobile-optimized transport candidates."""
        candidates = []

        # BitChat - optimized for low power scenarios
        bitchat_priority = TransportPriority.HIGH
        bitchat_reliability = 0.85

        # Boost BitChat priority under power/thermal constraints
        if ConstraintType.BATTERY_LOW in constraints.active_constraints:
            bitchat_priority = TransportPriority.CRITICAL
            bitchat_reliability = 0.90

        candidates.append(
            TransportCandidate(
                transport_type="bitchat",
                endpoint=f"bluetooth://{target}",
                priority=bitchat_priority,
                estimated_latency_ms=150.0,  # Typical Bluetooth mesh latency
                reliability_score=bitchat_reliability,
                cost_factor=0.1,  # Very low cost (no data usage)
                metadata={
                    "power_efficient": True,
                    "mesh_capable": True,
                    "compression": True,
                    "offline_capable": True,
                },
            )
        )

        # Betanet - optimized for performance scenarios
        betanet_priority = TransportPriority.NORMAL
        betanet_reliability = 0.92

        # Reduce Betanet priority under severe constraints
        if ConstraintType.BATTERY_CRITICAL in constraints.active_constraints:
            betanet_priority = TransportPriority.LOW
        elif ConstraintType.CELLULAR_DATA in constraints.active_constraints:
            betanet_priority = TransportPriority.LOW

        candidates.append(
            TransportCandidate(
                transport_type="betanet",
                endpoint=f"https://{target}:8443",
                priority=betanet_priority,
                estimated_latency_ms=80.0,
                reliability_score=betanet_reliability,
                cost_factor=0.6,  # Moderate data usage
                metadata={
                    "high_performance": True,
                    "covert_capable": True,
                    "http_compatible": True,
                },
            )
        )

        return candidates

    def _apply_constraint_filtering(
        self, transports: list[TransportCandidate], constraints: TransportConstraints
    ) -> list[TransportCandidate]:
        """Filter transports based on active constraints."""
        filtered = []

        for transport in transports:
            should_include = True

            # Critical battery: only BitChat and other low-power transports
            if ConstraintType.BATTERY_CRITICAL in constraints.active_constraints:
                if transport.transport_type not in ["bitchat", "ble"]:
                    should_include = False
                    logger.debug(f"Filtered out {transport.transport_type} due to critical battery")

            # High thermal: avoid CPU-intensive transports
            if ConstraintType.THERMAL_CRITICAL in constraints.active_constraints:
                if transport.transport_type in ["websocket", "http2"]:
                    should_include = False
                    logger.debug(f"Filtered out {transport.transport_type} due to thermal critical")

            # Cellular data: prefer low-cost transports
            if ConstraintType.CELLULAR_DATA in constraints.active_constraints:
                if transport.cost_factor > 0.5:  # High data usage
                    should_include = False
                    logger.debug(f"Filtered out {transport.transport_type} due to cellular data cost")

            if should_include:
                filtered.append(transport)

        return filtered

    def _score_transports_for_constraints(
        self,
        transports: list[TransportCandidate],
        constraints: TransportConstraints,
        message_size: int,
    ) -> list[TransportCandidate]:
        """Score transports considering mobile constraints."""
        for transport in transports:
            base_score = transport.reliability_score

            # Power efficiency bonus
            power_efficiency = self._calculate_power_efficiency(transport, message_size)
            power_bonus = power_efficiency * 0.3

            # Cost efficiency bonus (inverse of cost factor)
            cost_bonus = (1.0 - transport.cost_factor) * 0.2

            # Constraint-specific bonuses
            constraint_bonus = 0.0

            if ConstraintType.BATTERY_LOW in constraints.active_constraints:
                if transport.transport_type == "bitchat":
                    constraint_bonus += 0.4  # Strong preference for BitChat

            if ConstraintType.CELLULAR_DATA in constraints.active_constraints:
                if transport.cost_factor < 0.3:
                    constraint_bonus += 0.3

            # Calculate composite score
            composite_score = base_score + power_bonus + cost_bonus + constraint_bonus

            transport.metadata["power_efficiency"] = power_efficiency
            transport.metadata["cost_bonus"] = cost_bonus
            transport.metadata["constraint_bonus"] = constraint_bonus
            transport.metadata["composite_score"] = min(1.0, composite_score)
            transport.metadata["confidence"] = min(1.0, composite_score * 0.9)

        return transports

    def _get_fallback_transport(self, target: str, constraints: TransportConstraints) -> TransportCandidate:
        """Get fallback transport when no others are suitable."""
        # Always fallback to BitChat for maximum compatibility
        return TransportCandidate(
            transport_type="bitchat",
            endpoint=f"bluetooth://{target}",
            priority=TransportPriority.FALLBACK,
            estimated_latency_ms=200.0,
            reliability_score=0.75,
            cost_factor=0.1,
            metadata={"fallback": True, "reason": "no_suitable_transports"},
        )

    def _calculate_power_efficiency(self, transport: TransportCandidate, message_size: int) -> float:
        """Calculate power efficiency score for transport."""
        transport_type = transport.transport_type
        if transport_type in self.transport_efficiency:
            base_efficiency = self.transport_efficiency[transport_type]["power_factor"]

            # Adjust for message size (larger messages reduce per-byte efficiency)
            size_factor = min(1.0, 1024.0 / max(message_size, 64))  # Normalize to 1KB baseline

            return min(1.0, base_efficiency * (1.0 + size_factor * 0.2))

        return 0.5  # Default efficiency

    def _estimate_data_usage(self, transport: TransportCandidate, message_size: int) -> float:
        """Estimate data usage in MB."""
        transport_type = transport.transport_type
        if transport_type in self.transport_efficiency:
            efficiency = self.transport_efficiency[transport_type]["data_efficiency"]
            # Account for protocol overhead
            overhead_factor = 1.2 if transport_type in ["betanet", "websocket"] else 1.05
            estimated_bytes = message_size * overhead_factor / efficiency
            return estimated_bytes / (1024 * 1024)  # Convert to MB

        return message_size / (1024 * 1024) * 1.3  # Default with 30% overhead

    def _estimate_battery_drain(
        self,
        transport: TransportCandidate,
        message_size: int,
        constraints: TransportConstraints,
    ) -> float:
        """Estimate battery drain percentage."""
        power_efficiency = self._calculate_power_efficiency(transport, message_size)
        base_drain = 0.1  # 0.1% for baseline transmission

        # Scale by power efficiency (lower efficiency = higher drain)
        efficiency_multiplier = 2.0 - power_efficiency  # Range: 1.0-2.0

        # Scale by message size
        size_multiplier = min(3.0, message_size / 1024.0)  # Larger messages drain more

        estimated_drain = base_drain * efficiency_multiplier * size_multiplier

        # Apply constraint multipliers
        if ConstraintType.THERMAL_HIGH in constraints.active_constraints:
            estimated_drain *= 1.3  # Heat increases power consumption

        return min(5.0, estimated_drain)  # Cap at 5% drain per message

    def _generate_decision_reason(
        self, constraints: TransportConstraints, selected_transport: TransportCandidate
    ) -> str:
        """Generate human-readable decision reason."""
        reasons = []

        if ConstraintType.BATTERY_CRITICAL in constraints.active_constraints:
            reasons.append("critical battery level")
        elif ConstraintType.BATTERY_LOW in constraints.active_constraints:
            reasons.append("low battery")

        if ConstraintType.THERMAL_HIGH in constraints.active_constraints:
            reasons.append("high thermal load")

        if ConstraintType.CELLULAR_DATA in constraints.active_constraints:
            reasons.append("cellular data cost optimization")

        transport_reason = f"selected {selected_transport.transport_type} transport"

        if reasons:
            constraint_text = ", ".join(reasons)
            return f"{transport_reason} due to {constraint_text}"
        else:
            return f"{transport_reason} for optimal performance"

    def get_integration_status(self) -> dict[str, Any]:
        """Get status of Navigator-ResourceManager integration."""
        return {
            "navigator_available": self.base_navigator is not None,
            "resource_manager_available": self.resource_manager is not None,
            "transport_profiles": list(self.transport_efficiency.keys()),
            "policy_config": self.policy_config,
            "integration_active": True,
        }


# Integration helper functions
async def create_mobile_aware_navigator(
    scion_config: Any | None = None, enable_resource_management: bool = True
) -> MobileResourceNavigator:
    """Factory function to create integrated mobile-aware navigator."""
    base_navigator = None
    resource_manager = None

    try:
        if scion_config:
            # Initialize SCION-aware navigator if config provided
            from core.transport_manager import TransportManager
            from navigation.scion_navigator import SCIONAwareNavigator

            transport_manager = TransportManager()
            base_navigator = SCIONAwareNavigator(scion_config, transport_manager)
    except ImportError:
        logger.warning("SCION navigator not available, using basic routing")

    try:
        if enable_resource_management:
            # Initialize resource manager
            from production.monitoring.mobile.resource_management import BatteryThermalResourceManager

            resource_manager = BatteryThermalResourceManager()
    except ImportError:
        logger.warning("Mobile resource manager not available, using constraint-only routing")

    navigator = MobileResourceNavigator(base_navigator=base_navigator, resource_manager=resource_manager)

    logger.info("Mobile-aware navigator created successfully")
    return navigator


def enhance_navigator_with_mobile_constraints(navigator: Any) -> Any:
    """Enhance existing navigator with mobile resource management."""
    if not hasattr(navigator, "_mobile_resource_navigator"):
        navigator._mobile_resource_navigator = MobileResourceNavigator()

        # Add method to original navigator
        async def route_with_mobile_constraints(target: str, message_size: int, priority=TransportPriority.NORMAL):
            return await navigator._mobile_resource_navigator.route_with_constraints(target, message_size, priority)

        navigator.route_with_mobile_constraints = route_with_mobile_constraints
        logger.info("Enhanced navigator with mobile resource management capabilities")

    return navigator
