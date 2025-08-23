"""
AIVillage P2P Transport Cost Optimizer

This module provides intelligent P2P transport cost optimization and budget alerts
by integrating with the existing transport management system and cost tracking.

Key features:
- Smart transport selection based on cost, battery, and network conditions
- Dynamic routing optimization for cost savings
- Budget alerts and threshold management
- Mobile-first cost optimizations (BitChat priority for offline scenarios)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import existing AIVillage infrastructure
try:
    from ...p2p.core.transport_manager import TransportType
    from .distributed_cost_tracker import CostCategory, DistributedCostTracker

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    logging.warning("P2P infrastructure not available - running in standalone mode")
    INFRASTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CostOptimizationStrategy(Enum):
    """Cost optimization strategies for P2P transport."""

    MINIMIZE_COST = "minimize_cost"  # Lowest cost priority
    BATTERY_AWARE = "battery_aware"  # Optimize for battery life
    BALANCED = "balanced"  # Balance cost, performance, reliability
    PERFORMANCE_FIRST = "performance_first"  # Performance over cost
    OFFLINE_FIRST = "offline_first"  # BitChat priority for offline scenarios


@dataclass
class TransportCostProfile:
    """Cost profile for different transport types under various conditions."""

    transport_type: TransportType
    base_cost_per_mb: float  # Base cost per MB
    battery_cost_per_mb: float  # Battery drain cost per MB
    cellular_multiplier: float = 2.0  # Cellular cost multiplier
    wifi_multiplier: float = 1.0  # WiFi cost multiplier
    reliability_score: float = 0.8  # Transport reliability (0-1)
    setup_latency_ms: int = 100  # Connection setup latency
    throughput_mbps: float = 1.0  # Expected throughput


@dataclass
class RoutingDecision:
    """P2P routing decision with cost justification."""

    selected_transport: TransportType
    estimated_cost: float
    estimated_battery_drain: float
    decision_reasons: list[str]
    alternative_options: list[tuple[TransportType, float]]  # (transport, cost)
    optimization_savings: float = 0.0  # Savings compared to default


@dataclass
class CostBudgetAlert:
    """Cost budget alert for transport usage."""

    transport_type: TransportType
    current_usage_usd: float
    budget_limit_usd: float
    usage_percentage: float
    time_window_hours: int
    alert_level: str  # INFO, WARNING, CRITICAL
    recommendations: list[str]


class P2PTransportOptimizer:
    """
    P2P transport cost optimizer that integrates with existing transport management
    to provide intelligent, cost-aware routing decisions.
    """

    def __init__(
        self,
        transport_manager: Any | None = None,
        cost_tracker: DistributedCostTracker | None = None,
        edge_manager: Any | None = None,
    ):
        """
        Initialize P2P transport optimizer.

        Args:
            transport_manager: TransportManager instance
            cost_tracker: DistributedCostTracker instance
            edge_manager: EdgeManager for device context
        """
        self.transport_manager = transport_manager
        self.cost_tracker = cost_tracker
        self.edge_manager = edge_manager

        # Cost profiles for different transports
        self.cost_profiles = self._initialize_cost_profiles()

        # Optimization settings
        self.optimization_strategy = CostOptimizationStrategy.BALANCED
        self.transport_budgets: dict[TransportType, float] = {}
        self.budget_alerts: list[CostBudgetAlert] = []

        # Performance tracking
        self.routing_decisions: list[RoutingDecision] = []
        self.cost_savings_total: float = 0.0

        # Configuration
        self.config = {
            "cost_optimization_enabled": True,
            "battery_threshold_percent": 20,  # Switch to BitChat below 20%
            "cellular_cost_awareness": True,  # Factor in cellular costs
            "budget_alert_threshold": 0.8,  # Alert at 80% budget
            "max_routing_history": 1000,  # Keep last 1000 decisions
            "cost_calculation_interval": 60,  # Recalculate costs every minute
        }

        logger.info("P2P transport optimizer initialized")

    def _initialize_cost_profiles(self) -> dict[TransportType, TransportCostProfile]:
        """Initialize cost profiles for different transport types."""
        profiles = {}

        # BitChat (Bluetooth mesh) - battery cost, no data charges
        profiles[TransportType.BITCHAT] = TransportCostProfile(
            transport_type=TransportType.BITCHAT,
            base_cost_per_mb=0.0,  # No direct data cost
            battery_cost_per_mb=0.001,  # Minimal battery per MB
            cellular_multiplier=1.0,  # Not cellular dependent
            wifi_multiplier=1.0,  # Not wifi dependent
            reliability_score=0.7,  # Medium reliability (mesh dependent)
            setup_latency_ms=2000,  # Higher setup latency
            throughput_mbps=0.1,  # Lower throughput
        )

        # BetaNet (Encrypted internet) - data cost, reliable
        profiles[TransportType.BETANET] = TransportCostProfile(
            transport_type=TransportType.BETANET,
            base_cost_per_mb=0.05,  # Base encrypted bandwidth cost
            battery_cost_per_mb=0.0002,  # Lower battery drain
            cellular_multiplier=10.0,  # High cellular cost penalty
            wifi_multiplier=1.0,  # Standard wifi cost
            reliability_score=0.95,  # High reliability
            setup_latency_ms=500,  # Medium setup latency
            throughput_mbps=5.0,  # Good throughput
        )

        # QUIC (Direct connection) - standard data cost, best performance
        profiles[TransportType.QUIC] = TransportCostProfile(
            transport_type=TransportType.QUIC,
            base_cost_per_mb=0.02,  # Standard connection cost
            battery_cost_per_mb=0.0001,  # Minimal battery drain
            cellular_multiplier=5.0,  # Moderate cellular penalty
            wifi_multiplier=1.0,  # Standard wifi cost
            reliability_score=0.98,  # Very high reliability
            setup_latency_ms=100,  # Low setup latency
            throughput_mbps=10.0,  # Best throughput
        )

        return profiles

    async def optimize_transport_selection(
        self,
        message_size_bytes: int,
        destination_id: str,
        priority: str = "normal",
        device_context: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """
        Optimize transport selection based on cost, device context, and conditions.

        Args:
            message_size_bytes: Size of message to transmit
            destination_id: Target device/node ID
            priority: Message priority (low, normal, high, critical)
            device_context: Current device state (battery, network, etc.)

        Returns:
            RoutingDecision with selected transport and justification
        """
        message_size_mb = message_size_bytes / (1024 * 1024)

        # Get device context
        if not device_context and self.edge_manager:
            device_context = await self._get_device_context()
        device_context = device_context or {}

        # Calculate costs for each transport option
        transport_costs = {}
        transport_reasons = {}

        for transport_type, profile in self.cost_profiles.items():
            cost_info = await self._calculate_transport_cost(
                transport_type=transport_type, message_size_mb=message_size_mb, device_context=device_context
            )
            transport_costs[transport_type] = cost_info
            transport_reasons[transport_type] = cost_info.get("reasons", [])

        # Apply optimization strategy
        selected_transport = await self._apply_optimization_strategy(
            transport_costs=transport_costs,
            message_size_mb=message_size_mb,
            priority=priority,
            device_context=device_context,
        )

        # Calculate decision details
        selected_cost_info = transport_costs[selected_transport]
        alternative_options = [
            (t, info["total_cost"]) for t, info in transport_costs.items() if t != selected_transport
        ]
        alternative_options.sort(key=lambda x: x[1])

        # Calculate optimization savings
        default_transport = TransportType.QUIC  # Assume QUIC as default
        default_cost = transport_costs.get(default_transport, {}).get("total_cost", 0)
        optimization_savings = max(0, default_cost - selected_cost_info["total_cost"])

        # Create routing decision
        decision = RoutingDecision(
            selected_transport=selected_transport,
            estimated_cost=selected_cost_info["total_cost"],
            estimated_battery_drain=selected_cost_info["battery_cost"],
            decision_reasons=selected_cost_info["reasons"],
            alternative_options=alternative_options,
            optimization_savings=optimization_savings,
        )

        # Record decision for analytics
        self.routing_decisions.append(decision)
        if len(self.routing_decisions) > self.config["max_routing_history"]:
            self.routing_decisions.pop(0)

        # Track cost savings
        self.cost_savings_total += optimization_savings

        # Log decision
        logger.info(
            f"Transport optimization: {selected_transport.value} selected for "
            f"{message_size_mb:.2f}MB (cost: ${selected_cost_info['total_cost']:.4f}, "
            f"savings: ${optimization_savings:.4f})"
        )

        return decision

    async def _calculate_transport_cost(
        self, transport_type: TransportType, message_size_mb: float, device_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comprehensive cost for a transport option."""
        profile = self.cost_profiles[transport_type]

        # Base transport cost
        base_cost = profile.base_cost_per_mb * message_size_mb

        # Battery cost
        battery_cost = profile.battery_cost_per_mb * message_size_mb

        # Network type multiplier
        network_type = device_context.get("network_type", "wifi")
        if network_type == "cellular":
            network_multiplier = profile.cellular_multiplier
        else:
            network_multiplier = profile.wifi_multiplier

        data_cost = base_cost * network_multiplier

        # Device context adjustments
        battery_percent = device_context.get("battery_percent", 100)
        thermal_temp = device_context.get("cpu_temp_celsius", 25)

        # Battery penalty for low battery (except BitChat which saves battery)
        battery_penalty = 0.0
        if battery_percent < self.config["battery_threshold_percent"] and transport_type != TransportType.BITCHAT:
            battery_penalty = data_cost * 0.5  # 50% penalty for low battery

        # Thermal penalty for high temperature
        thermal_penalty = 0.0
        if thermal_temp > 45:  # Above 45°C
            thermal_penalty = data_cost * 0.2  # 20% penalty for thermal throttling

        total_cost = data_cost + battery_cost + battery_penalty + thermal_penalty

        # Generate reasoning
        reasons = []
        if transport_type == TransportType.BITCHAT:
            reasons.append("BitChat: Battery-efficient mesh networking")
            if battery_percent < self.config["battery_threshold_percent"]:
                reasons.append(f"Low battery ({battery_percent}%) - BitChat saves power")

        if network_type == "cellular":
            reasons.append(f"Cellular network - {network_multiplier}x cost multiplier applied")

        if battery_penalty > 0:
            reasons.append(f"Low battery penalty: +${battery_penalty:.4f}")

        if thermal_penalty > 0:
            reasons.append(f"High temperature penalty: +${thermal_penalty:.4f}")

        return {
            "total_cost": total_cost,
            "data_cost": data_cost,
            "battery_cost": battery_cost,
            "battery_penalty": battery_penalty,
            "thermal_penalty": thermal_penalty,
            "network_multiplier": network_multiplier,
            "reliability_score": profile.reliability_score,
            "reasons": reasons,
        }

    async def _apply_optimization_strategy(
        self,
        transport_costs: dict[TransportType, dict[str, Any]],
        message_size_mb: float,
        priority: str,
        device_context: dict[str, Any],
    ) -> TransportType:
        """Apply the configured optimization strategy to select transport."""

        if self.optimization_strategy == CostOptimizationStrategy.MINIMIZE_COST:
            # Select lowest total cost
            return min(transport_costs.keys(), key=lambda t: transport_costs[t]["total_cost"])

        elif self.optimization_strategy == CostOptimizationStrategy.BATTERY_AWARE:
            # Prioritize battery savings, especially for low battery
            battery_percent = device_context.get("battery_percent", 100)
            if battery_percent < self.config["battery_threshold_percent"]:
                # Force BitChat for low battery
                return TransportType.BITCHAT
            else:
                # Select lowest battery + data cost combination
                return min(
                    transport_costs.keys(),
                    key=lambda t: transport_costs[t]["battery_cost"] + transport_costs[t]["data_cost"] * 0.5,
                )

        elif self.optimization_strategy == CostOptimizationStrategy.PERFORMANCE_FIRST:
            # Select based on reliability and performance, cost secondary
            def performance_score(transport_type: TransportType) -> float:
                profile = self.cost_profiles[transport_type]
                cost_info = transport_costs[transport_type]

                # Higher is better: reliability * throughput / (cost + setup_latency)
                perf = (profile.reliability_score * profile.throughput_mbps) / max(
                    0.001, cost_info["total_cost"] + profile.setup_latency_ms / 1000
                )
                return perf

            return max(transport_costs.keys(), key=performance_score)

        elif self.optimization_strategy == CostOptimizationStrategy.OFFLINE_FIRST:
            # Prioritize BitChat for offline scenarios, then cost-optimize
            network_type = device_context.get("network_type", "wifi")
            if network_type == "offline" or message_size_mb < 1.0:
                return TransportType.BITCHAT
            else:
                # Fall back to cost optimization
                return min(transport_costs.keys(), key=lambda t: transport_costs[t]["total_cost"])

        else:  # BALANCED strategy
            # Balance cost, performance, and reliability
            def balanced_score(transport_type: TransportType) -> float:
                profile = self.cost_profiles[transport_type]
                cost_info = transport_costs[transport_type]

                # Weighted score: reliability (40%) + cost efficiency (40%) + performance (20%)
                cost_efficiency = 1.0 / max(0.001, cost_info["total_cost"])
                performance = profile.throughput_mbps / max(1, profile.setup_latency_ms / 1000)

                score = profile.reliability_score * 0.4 + cost_efficiency * 0.4 + performance * 0.2
                return score

            return max(transport_costs.keys(), key=balanced_score)

    async def _get_device_context(self) -> dict[str, Any]:
        """Get current device context from edge manager."""
        context = {"battery_percent": 100, "cpu_temp_celsius": 25, "network_type": "wifi", "data_budget_mb": 1000}

        if self.edge_manager and hasattr(self.edge_manager, "get_device_status"):
            try:
                status = await self.edge_manager.get_device_status()
                context.update(status)
            except Exception as e:
                logger.warning(f"Could not get device context: {e}")

        return context

    def set_transport_budget(self, transport_type: TransportType, budget_usd: float, hours: int = 24):
        """Set budget limit for a transport type."""
        self.transport_budgets[transport_type] = {"budget_usd": budget_usd, "hours": hours, "start_time": time.time()}
        logger.info(f"Set budget for {transport_type.value}: ${budget_usd} per {hours}h")

    async def check_budget_alerts(self) -> list[CostBudgetAlert]:
        """Check for budget alerts and return any active alerts."""
        alerts = []
        current_time = time.time()

        for transport_type, budget_info in self.transport_budgets.items():
            # Calculate usage in budget window
            budget_start = budget_info["start_time"]
            budget_window_hours = budget_info["hours"]
            budget_limit = budget_info["budget_usd"]

            # Get recent cost events for this transport
            if self.cost_tracker:
                recent_usage = await self._calculate_recent_transport_usage(transport_type, budget_start, current_time)
            else:
                recent_usage = 0.0

            usage_percentage = (recent_usage / budget_limit) * 100

            # Determine alert level
            alert_level = "INFO"
            recommendations = []

            if usage_percentage >= 95:
                alert_level = "CRITICAL"
                recommendations.extend(
                    [
                        f"CRITICAL: {transport_type.value} usage at {usage_percentage:.1f}% of budget",
                        "Consider switching to BitChat for cost savings",
                        "Review message priorities and reduce non-critical traffic",
                    ]
                )
            elif usage_percentage >= self.config["budget_alert_threshold"] * 100:
                alert_level = "WARNING"
                recommendations.extend(
                    [
                        f"WARNING: {transport_type.value} approaching budget limit",
                        "Monitor usage closely",
                        "Consider cost optimization strategies",
                    ]
                )

            if alert_level in ["WARNING", "CRITICAL"]:
                alert = CostBudgetAlert(
                    transport_type=transport_type,
                    current_usage_usd=recent_usage,
                    budget_limit_usd=budget_limit,
                    usage_percentage=usage_percentage,
                    time_window_hours=budget_window_hours,
                    alert_level=alert_level,
                    recommendations=recommendations,
                )
                alerts.append(alert)

        # Store alerts for monitoring
        self.budget_alerts = alerts

        # Log critical alerts
        for alert in alerts:
            if alert.alert_level == "CRITICAL":
                logger.critical(
                    f"Budget alert: {alert.transport_type.value} at "
                    f"{alert.usage_percentage:.1f}% of ${alert.budget_limit_usd} budget"
                )
            elif alert.alert_level == "WARNING":
                logger.warning(
                    f"Budget warning: {alert.transport_type.value} at " f"{alert.usage_percentage:.1f}% of budget"
                )

        return alerts

    async def _calculate_recent_transport_usage(
        self, transport_type: TransportType, start_time: float, end_time: float
    ) -> float:
        """Calculate recent transport usage costs."""
        if not self.cost_tracker:
            return 0.0

        # Filter cost events for this transport in time window
        usage_cost = 0.0
        for event in self.cost_tracker.cost_events:
            if (
                event.timestamp >= start_time
                and event.timestamp <= end_time
                and event.category == CostCategory.P2P_TRANSPORT
                and event.transport_type == transport_type.value
            ):
                usage_cost += event.cost_amount

        return usage_cost

    def get_cost_optimization_report(self) -> dict[str, Any]:
        """Generate comprehensive cost optimization report."""
        if not self.routing_decisions:
            return {"message": "No routing decisions recorded yet"}

        # Calculate aggregate statistics
        total_decisions = len(self.routing_decisions)
        transport_usage = {}
        total_estimated_cost = 0.0

        for decision in self.routing_decisions:
            transport = decision.selected_transport.value
            transport_usage[transport] = transport_usage.get(transport, 0) + 1
            total_estimated_cost += decision.estimated_cost

        # Transport distribution
        transport_distribution = {
            transport: (count / total_decisions) * 100 for transport, count in transport_usage.items()
        }

        # Recent decisions analysis
        recent_decisions = (
            self.routing_decisions[-100:] if len(self.routing_decisions) > 100 else self.routing_decisions
        )
        recent_savings = sum(d.optimization_savings for d in recent_decisions)

        # Budget status
        budget_status = []
        for alert in self.budget_alerts:
            budget_status.append(
                {
                    "transport": alert.transport_type.value,
                    "usage_percent": alert.usage_percentage,
                    "alert_level": alert.alert_level,
                }
            )

        return {
            "optimization_summary": {
                "total_decisions": total_decisions,
                "total_estimated_cost_usd": total_estimated_cost,
                "total_savings_usd": self.cost_savings_total,
                "average_savings_per_decision": self.cost_savings_total / max(1, total_decisions),
            },
            "transport_distribution": transport_distribution,
            "recent_performance": {
                "recent_decisions_analyzed": len(recent_decisions),
                "recent_savings_usd": recent_savings,
                "optimization_strategy": self.optimization_strategy.value,
            },
            "budget_alerts": budget_status,
            "recommendations": self._generate_optimization_recommendations(),
        }

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on usage patterns."""
        recommendations = []

        if not self.routing_decisions:
            return ["No data available for recommendations"]

        # Analyze transport usage patterns
        transport_usage = {}
        high_cost_decisions = []

        for decision in self.routing_decisions[-100:]:  # Recent decisions
            transport = decision.selected_transport.value
            transport_usage[transport] = transport_usage.get(transport, 0) + 1

            if decision.estimated_cost > 0.1:  # High cost threshold
                high_cost_decisions.append(decision)

        # Generate specific recommendations
        bitchat_usage = transport_usage.get("bitchat", 0)
        total_usage = sum(transport_usage.values())

        if total_usage > 0 and bitchat_usage / total_usage < 0.3:
            recommendations.append("Consider increasing BitChat usage for cost savings, especially during low battery")

        if len(high_cost_decisions) > 10:
            recommendations.append(
                f"Found {len(high_cost_decisions)} high-cost routing decisions - review cellular usage"
            )

        if len(self.budget_alerts) > 0:
            recommendations.append("Active budget alerts detected - consider switching optimization strategy")

        cellular_heavy = any("cellular" in str(d.decision_reasons) for d in self.routing_decisions[-20:])
        if cellular_heavy:
            recommendations.append("High cellular usage detected - prioritize WiFi connections when available")

        if not recommendations:
            recommendations.append("Transport cost optimization is working well - continue current strategy")

        return recommendations

    def set_optimization_strategy(self, strategy: CostOptimizationStrategy):
        """Update the optimization strategy."""
        old_strategy = self.optimization_strategy
        self.optimization_strategy = strategy
        logger.info(f"Optimization strategy changed: {old_strategy.value} -> {strategy.value}")


# Helper functions for integration
async def create_transport_optimizer_with_infrastructure() -> P2PTransportOptimizer:
    """Create P2P transport optimizer with all available infrastructure."""
    transport_manager = None
    edge_manager = None
    cost_tracker = None

    if INFRASTRUCTURE_AVAILABLE:
        try:
            from ...p2p.core.transport_manager import TransportManager, TransportPriority

            transport_manager = TransportManager(
                device_id="cost_optimizer", transport_priority=TransportPriority.ADAPTIVE
            )
        except Exception as e:
            logger.warning(f"Could not initialize transport manager: {e}")

        try:
            from ...edge.core.edge_manager import EdgeManager

            edge_manager = EdgeManager()
        except Exception as e:
            logger.warning(f"Could not initialize edge manager: {e}")

        try:
            from .distributed_cost_tracker import create_cost_tracker_with_infrastructure

            cost_tracker = await create_cost_tracker_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize cost tracker: {e}")

    return P2PTransportOptimizer(
        transport_manager=transport_manager, cost_tracker=cost_tracker, edge_manager=edge_manager
    )


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create optimizer
        optimizer = await create_transport_optimizer_with_infrastructure()

        # Set up budgets
        optimizer.set_transport_budget(TransportType.BETANET, budget_usd=10.0, hours=24)
        optimizer.set_transport_budget(TransportType.QUIC, budget_usd=5.0, hours=24)

        # Set optimization strategy
        optimizer.set_optimization_strategy(CostOptimizationStrategy.BATTERY_AWARE)

        # Simulate routing decisions
        test_scenarios = [
            {
                "message_size_bytes": 1024,  # Small message
                "device_context": {"battery_percent": 85, "network_type": "wifi"},
            },
            {
                "message_size_bytes": 1024 * 1024,  # Large message
                "device_context": {"battery_percent": 15, "network_type": "cellular"},  # Low battery + cellular
            },
            {
                "message_size_bytes": 512 * 1024,  # Medium message
                "device_context": {"battery_percent": 60, "network_type": "wifi"},
            },
        ]

        print("🚀 P2P Transport Cost Optimization Demo")
        print("=" * 60)

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📱 Scenario {i}: {scenario['message_size_bytes']/1024:.1f}KB message")
            print(
                f"   Battery: {scenario['device_context']['battery_percent']}%, "
                f"Network: {scenario['device_context']['network_type']}"
            )

            decision = await optimizer.optimize_transport_selection(
                message_size_bytes=scenario["message_size_bytes"],
                destination_id="test-device",
                device_context=scenario["device_context"],
            )

            print(f"   ✅ Selected: {decision.selected_transport.value}")
            print(f"   💰 Cost: ${decision.estimated_cost:.4f}")
            print(f"   🔋 Battery: {decision.estimated_battery_drain:.4f}")
            print(f"   💡 Savings: ${decision.optimization_savings:.4f}")
            print(f"   📝 Reasons: {', '.join(decision.decision_reasons)}")

        # Check budget alerts
        alerts = await optimizer.check_budget_alerts()
        if alerts:
            print(f"\n⚠️  Budget Alerts: {len(alerts)}")
            for alert in alerts:
                print(
                    f"   {alert.alert_level}: {alert.transport_type.value} at "
                    f"{alert.usage_percentage:.1f}% of budget"
                )

        # Generate optimization report
        report = optimizer.get_cost_optimization_report()
        print("\n📊 Optimization Report:")
        print(f"   Total decisions: {report['optimization_summary']['total_decisions']}")
        print(f"   Total savings: ${report['optimization_summary']['total_savings_usd']:.4f}")
        print(f"   Transport distribution: {report['transport_distribution']}")
        print(f"   Recommendations: {len(report['recommendations'])}")
        for rec in report["recommendations"]:
            print(f"   - {rec}")

    asyncio.run(main())
