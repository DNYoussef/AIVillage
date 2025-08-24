"""
AIVillage Distributed Cost Tracking System

Tracks costs across the entire distributed infrastructure including:
- Fog compute nodes running Agent Forge phases
- P2P transport costs (BitChat/BetaNet/QUIC data usage)
- Edge device participation costs (battery/thermal/data)
- Multi-cloud resource costs (AWS, Azure, GCP)

Integrates with existing AIVillage infrastructure including fog compute orchestrator,
P2P transport manager, and edge device management systems.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any

# Import existing AIVillage infrastructure
try:
    from ...p2p.core.transport_manager import TransportType

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    logging.warning("Infrastructure components not available - running in standalone mode")
    INFRASTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs in the distributed AIVillage system."""

    FOG_COMPUTE = "fog_compute"  # Fog node compute costs
    P2P_TRANSPORT = "p2p_transport"  # Network/data transport costs
    EDGE_DEVICE = "edge_device"  # Edge device participation costs
    CLOUD_RESOURCE = "cloud_resource"  # Multi-cloud infrastructure costs
    AGENT_FORGE = "agent_forge"  # Agent Forge phase execution costs
    DATA_STORAGE = "data_storage"  # Distributed storage costs


class CostUnit(Enum):
    """Units for cost measurement."""

    USD_PER_HOUR = "usd_per_hour"  # Hourly costs (compute, storage)
    USD_PER_GB = "usd_per_gb"  # Data transfer costs
    USD_PER_REQUEST = "usd_per_request"  # API/transaction costs
    BATTERY_PERCENT = "battery_percent"  # Battery drain costs
    WATTS_HOUR = "watts_hour"  # Energy consumption


@dataclass
class CostEvent:
    """Individual cost tracking event."""

    timestamp: float
    category: CostCategory
    cost_amount: float
    cost_unit: CostUnit
    resource_id: str  # Node ID, device ID, etc.
    resource_type: str  # fog_node, edge_device, cloud_vm
    phase: str | None = None  # Agent Forge phase if applicable
    transport_type: str | None = None  # P2P transport type if applicable
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostBudget:
    """Budget configuration for cost management."""

    category: CostCategory
    budget_amount: float
    budget_unit: CostUnit
    time_window_hours: int = 24  # Budget time window
    alert_threshold: float = 0.8  # Alert when 80% of budget used
    hard_limit: bool = False  # Whether to enforce hard limits


@dataclass
class CostSummary:
    """Cost summary for reporting."""

    total_cost_usd: float
    cost_by_category: dict[str, float]
    cost_by_resource: dict[str, float]
    cost_by_phase: dict[str, float]
    top_cost_drivers: list[tuple[str, float]]
    budget_status: dict[str, dict[str, Any]]
    recommendations: list[str]


class DistributedCostTracker:
    """
    Main cost tracking system for AIVillage distributed infrastructure.

    Integrates with existing fog compute, P2P, and edge device systems
    to provide comprehensive cost visibility and management.
    """

    def __init__(
        self,
        config_path: str | None = None,
        fog_orchestrator: Any | None = None,
        edge_manager: Any | None = None,
        transport_manager: Any | None = None,
    ):
        """
        Initialize the distributed cost tracker.

        Args:
            config_path: Path to cost tracking configuration
            fog_orchestrator: FogComputeOrchestrator instance
            edge_manager: EdgeManager instance
            transport_manager: TransportManager instance
        """
        self.config_path = config_path or "config/cost_tracking.json"

        # Infrastructure components
        self.fog_orchestrator = fog_orchestrator
        self.edge_manager = edge_manager
        self.transport_manager = transport_manager

        # Cost tracking state
        self.cost_events: list[CostEvent] = []
        self.cost_budgets: dict[str, CostBudget] = {}
        self.active_sessions: dict[str, dict[str, Any]] = {}

        # Configuration
        self.config = self._load_config()
        self.cost_rates = self._load_cost_rates()

        # Monitoring
        self.last_alert_time: dict[str, float] = {}
        self.alert_cooldown = 300  # 5 minutes between alerts

        logger.info("Distributed cost tracker initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load cost tracking configuration."""
        default_config = {
            "tracking_enabled": True,
            "cost_storage_retention_days": 30,
            "budget_check_interval_seconds": 300,
            "cost_aggregation_interval_seconds": 60,
            "currency": "USD",
            "fog_compute_enabled": True,
            "p2p_transport_enabled": True,
            "edge_device_enabled": True,
            "cloud_resource_enabled": True,
        }

        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")

        return default_config

    def _load_cost_rates(self) -> dict[str, dict[str, float]]:
        """Load cost rates for different resource types."""
        return {
            "fog_compute": {
                # Per vCPU hour rates
                "cpu_cost_per_vcpu_hour": 0.02,  # $0.02 per vCPU hour
                "memory_cost_per_gb_hour": 0.004,  # $0.004 per GB hour
                "gpu_cost_per_hour": 0.50,  # $0.50 per GPU hour
                "storage_cost_per_gb_month": 0.10,  # $0.10 per GB per month
            },
            "p2p_transport": {
                # Data transfer costs
                "bitchat_battery_cost_per_mb": 0.001,  # Battery equivalent cost
                "betanet_bandwidth_cost_per_gb": 0.05,  # Encrypted bandwidth cost
                "quic_data_cost_per_gb": 0.02,  # Direct connection cost
                "cellular_data_cost_per_gb": 2.00,  # Mobile data premium
            },
            "edge_device": {
                # Device participation costs
                "battery_drain_cost_per_percent": 0.01,  # $0.01 per battery %
                "thermal_impact_cost_per_degree": 0.005,  # Thermal wear cost
                "device_wear_cost_per_hour": 0.001,  # General device wear
                "mobile_data_cost_per_gb": 5.00,  # Mobile data cost
            },
            "cloud_resource": {
                # Multi-cloud costs (examples)
                "aws_ec2_t3_medium": 0.0416,  # Per hour
                "azure_b2s": 0.0408,  # Per hour
                "gcp_e2_medium": 0.0335,  # Per hour
                "storage_per_gb_month": 0.023,  # Standard storage
            },
        }

    async def track_fog_compute_cost(
        self,
        node_id: str,
        phase: str,
        cpu_cores: int,
        memory_gb: float,
        gpu_hours: float = 0,
        duration_seconds: int = 0,
    ) -> CostEvent:
        """Track costs for fog compute node usage during Agent Forge phase."""

        if not self.config.get("fog_compute_enabled", True):
            return None

        rates = self.cost_rates["fog_compute"]
        duration_hours = duration_seconds / 3600.0 if duration_seconds > 0 else 1.0

        # Calculate compute costs
        cpu_cost = cpu_cores * rates["cpu_cost_per_vcpu_hour"] * duration_hours
        memory_cost = memory_gb * rates["memory_cost_per_gb_hour"] * duration_hours
        gpu_cost = gpu_hours * rates["gpu_cost_per_hour"]

        total_cost = cpu_cost + memory_cost + gpu_cost

        cost_event = CostEvent(
            timestamp=time.time(),
            category=CostCategory.FOG_COMPUTE,
            cost_amount=total_cost,
            cost_unit=CostUnit.USD_PER_HOUR,
            resource_id=node_id,
            resource_type="fog_node",
            phase=phase,
            metadata={
                "cpu_cores": cpu_cores,
                "memory_gb": memory_gb,
                "gpu_hours": gpu_hours,
                "duration_hours": duration_hours,
                "cpu_cost": cpu_cost,
                "memory_cost": memory_cost,
                "gpu_cost": gpu_cost,
                "agent_forge_phase": phase,
            },
        )

        await self._record_cost_event(cost_event)
        logger.debug(f"Tracked fog compute cost: ${total_cost:.4f} for {phase} on {node_id}")

        return cost_event

    async def track_p2p_transport_cost(
        self, transport_type: TransportType, data_mb: float, device_id: str, is_cellular: bool = False
    ) -> CostEvent:
        """Track costs for P2P transport data usage."""

        if not self.config.get("p2p_transport_enabled", True):
            return None

        rates = self.cost_rates["p2p_transport"]

        # Calculate transport-specific costs
        if transport_type == TransportType.BITCHAT:
            # BitChat uses battery, convert to equivalent cost
            cost = data_mb * rates["bitchat_battery_cost_per_mb"]
            cost_unit = CostUnit.BATTERY_PERCENT
        elif transport_type == TransportType.BETANET:
            # BetaNet encrypted bandwidth
            cost_per_gb = rates["cellular_data_cost_per_gb"] if is_cellular else rates["betanet_bandwidth_cost_per_gb"]
            cost = (data_mb / 1024) * cost_per_gb
            cost_unit = CostUnit.USD_PER_GB
        elif transport_type == TransportType.QUIC:
            # Direct QUIC connection
            cost_per_gb = rates["cellular_data_cost_per_gb"] if is_cellular else rates["quic_data_cost_per_gb"]
            cost = (data_mb / 1024) * cost_per_gb
            cost_unit = CostUnit.USD_PER_GB
        else:
            cost = 0
            cost_unit = CostUnit.USD_PER_GB

        cost_event = CostEvent(
            timestamp=time.time(),
            category=CostCategory.P2P_TRANSPORT,
            cost_amount=cost,
            cost_unit=cost_unit,
            resource_id=device_id,
            resource_type="p2p_transport",
            transport_type=transport_type.value,
            metadata={
                "data_mb": data_mb,
                "is_cellular": is_cellular,
                "transport_type": transport_type.value,
                "cost_calculation": f"{data_mb}MB via {transport_type.value}",
            },
        )

        await self._record_cost_event(cost_event)
        logger.debug(f"Tracked P2P transport cost: ${cost:.4f} for {data_mb}MB via {transport_type.value}")

        return cost_event

    async def track_edge_device_cost(
        self,
        device_id: str,
        battery_drain_percent: float,
        thermal_impact_celsius: float = 0,
        participation_hours: float = 1.0,
        mobile_data_gb: float = 0,
    ) -> CostEvent:
        """Track costs for edge device participation."""

        if not self.config.get("edge_device_enabled", True):
            return None

        rates = self.cost_rates["edge_device"]

        # Calculate edge device costs
        battery_cost = battery_drain_percent * rates["battery_drain_cost_per_percent"]
        thermal_cost = thermal_impact_celsius * rates["thermal_impact_cost_per_degree"] * participation_hours
        wear_cost = participation_hours * rates["device_wear_cost_per_hour"]
        data_cost = mobile_data_gb * rates["mobile_data_cost_per_gb"]

        total_cost = battery_cost + thermal_cost + wear_cost + data_cost

        cost_event = CostEvent(
            timestamp=time.time(),
            category=CostCategory.EDGE_DEVICE,
            cost_amount=total_cost,
            cost_unit=CostUnit.USD_PER_HOUR,
            resource_id=device_id,
            resource_type="edge_device",
            metadata={
                "battery_drain_percent": battery_drain_percent,
                "thermal_impact_celsius": thermal_impact_celsius,
                "participation_hours": participation_hours,
                "mobile_data_gb": mobile_data_gb,
                "battery_cost": battery_cost,
                "thermal_cost": thermal_cost,
                "wear_cost": wear_cost,
                "data_cost": data_cost,
            },
        )

        await self._record_cost_event(cost_event)
        logger.debug(f"Tracked edge device cost: ${total_cost:.4f} for device {device_id}")

        return cost_event

    async def track_agent_forge_phase_cost(self, phase: str, node_id: str, phase_result: Any) -> CostEvent:
        """Track costs for specific Agent Forge phase execution."""

        # Extract resource usage from phase result
        if hasattr(phase_result, "metrics"):
            metrics = phase_result.metrics

            cpu_hours = metrics.get("cpu_hours", 1.0)
            memory_gb_hours = metrics.get("memory_gb_hours", 4.0)
            gpu_hours = metrics.get("gpu_hours", 0.0)
            duration_seconds = metrics.get("duration_seconds", 3600)

            # Use fog compute cost tracking
            cost_event = await self.track_fog_compute_cost(
                node_id=node_id,
                phase=phase,
                cpu_cores=int(cpu_hours),
                memory_gb=memory_gb_hours,
                gpu_hours=gpu_hours,
                duration_seconds=duration_seconds,
            )

            if cost_event:
                cost_event.category = CostCategory.AGENT_FORGE
                cost_event.metadata["agent_forge_phase"] = phase

            return cost_event

        return None

    async def _record_cost_event(self, cost_event: CostEvent):
        """Record a cost event and check budgets."""
        self.cost_events.append(cost_event)

        # Check budget alerts
        await self._check_budget_alerts(cost_event)

        # Persist cost data
        await self._persist_cost_data()

    async def _check_budget_alerts(self, cost_event: CostEvent):
        """Check if cost event triggers budget alerts."""
        category_key = cost_event.category.value

        if category_key not in self.cost_budgets:
            return

        budget = self.cost_budgets[category_key]

        # Calculate recent costs in budget window
        cutoff_time = time.time() - (budget.time_window_hours * 3600)
        recent_costs = [
            event
            for event in self.cost_events
            if event.timestamp > cutoff_time and event.category == cost_event.category
        ]

        total_cost = sum(event.cost_amount for event in recent_costs)
        budget_usage = total_cost / budget.budget_amount

        # Check alert threshold
        if budget_usage >= budget.alert_threshold:
            alert_key = f"{category_key}_budget_alert"

            # Respect alert cooldown
            if (
                alert_key not in self.last_alert_time
                or time.time() - self.last_alert_time[alert_key] > self.alert_cooldown
            ):
                await self._send_budget_alert(cost_event.category, budget, total_cost, budget_usage)
                self.last_alert_time[alert_key] = time.time()

    async def _send_budget_alert(
        self, category: CostCategory, budget: CostBudget, current_cost: float, usage_percent: float
    ):
        """Send budget alert notification."""
        alert_message = (
            f"🚨 BUDGET ALERT: {category.value} costs at {usage_percent*100:.1f}% "
            f"of budget (${current_cost:.2f} / ${budget.budget_amount:.2f})"
        )

        logger.warning(alert_message)

        # Here you would integrate with your notification system
        # e.g., send to monitoring dashboard, Slack, email, etc.

    async def _persist_cost_data(self):
        """Persist cost data to storage."""
        # In production, this would write to database
        # For now, we'll write to JSON file
        try:
            cost_data = {
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "category": event.category.value,
                        "cost_amount": event.cost_amount,
                        "cost_unit": event.cost_unit.value,
                        "resource_id": event.resource_id,
                        "resource_type": event.resource_type,
                        "phase": event.phase,
                        "transport_type": event.transport_type,
                        "metadata": event.metadata,
                    }
                    for event in self.cost_events[-100:]  # Keep last 100 events
                ],
                "last_updated": time.time(),
            }

            cost_file = Path("data/cost_tracking.json")
            cost_file.parent.mkdir(parents=True, exist_ok=True)

            with open(cost_file, "w") as f:
                json.dump(cost_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist cost data: {e}")

    def add_budget(self, budget: CostBudget):
        """Add a cost budget for monitoring."""
        self.cost_budgets[budget.category.value] = budget
        logger.info(f"Added budget: {budget.category.value} = ${budget.budget_amount}")

    def get_cost_summary(self, hours_back: int = 24) -> CostSummary:
        """Get comprehensive cost summary."""
        cutoff_time = time.time() - (hours_back * 3600)
        recent_events = [e for e in self.cost_events if e.timestamp > cutoff_time]

        if not recent_events:
            return CostSummary(
                total_cost_usd=0,
                cost_by_category={},
                cost_by_resource={},
                cost_by_phase={},
                top_cost_drivers=[],
                budget_status={},
                recommendations=[],
            )

        # Calculate totals
        total_cost = sum(self._convert_to_usd(event) for event in recent_events)

        # Cost by category
        cost_by_category = {}
        for event in recent_events:
            category = event.category.value
            cost_by_category[category] = cost_by_category.get(category, 0) + self._convert_to_usd(event)

        # Cost by resource
        cost_by_resource = {}
        for event in recent_events:
            resource = event.resource_id
            cost_by_resource[resource] = cost_by_resource.get(resource, 0) + self._convert_to_usd(event)

        # Cost by phase
        cost_by_phase = {}
        for event in recent_events:
            if event.phase:
                cost_by_phase[event.phase] = cost_by_phase.get(event.phase, 0) + self._convert_to_usd(event)

        # Top cost drivers
        top_drivers = sorted(cost_by_resource.items(), key=lambda x: x[1], reverse=True)[:5]

        # Budget status
        budget_status = {}
        for category, budget in self.cost_budgets.items():
            category_cost = cost_by_category.get(category, 0)
            budget_status[category] = {
                "budget": budget.budget_amount,
                "spent": category_cost,
                "remaining": budget.budget_amount - category_cost,
                "usage_percent": (category_cost / budget.budget_amount) * 100,
            }

        # Generate recommendations
        recommendations = self._generate_cost_recommendations(cost_by_category, recent_events)

        return CostSummary(
            total_cost_usd=total_cost,
            cost_by_category=cost_by_category,
            cost_by_resource=cost_by_resource,
            cost_by_phase=cost_by_phase,
            top_cost_drivers=top_drivers,
            budget_status=budget_status,
            recommendations=recommendations,
        )

    def _convert_to_usd(self, event: CostEvent) -> float:
        """Convert cost event to USD for aggregation."""
        if event.cost_unit == CostUnit.USD_PER_HOUR or event.cost_unit == CostUnit.USD_PER_GB:
            return event.cost_amount
        elif event.cost_unit == CostUnit.BATTERY_PERCENT:
            # Convert battery percent to USD equivalent
            return event.cost_amount * self.cost_rates["edge_device"]["battery_drain_cost_per_percent"]
        else:
            return event.cost_amount

    def _generate_cost_recommendations(
        self, cost_by_category: dict[str, float], recent_events: list[CostEvent]
    ) -> list[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # High fog compute costs
        if cost_by_category.get("fog_compute", 0) > 10.0:
            recommendations.append("Consider optimizing Agent Forge phase distribution - fog compute costs are high")

        # High P2P transport costs
        if cost_by_category.get("p2p_transport", 0) > 5.0:
            recommendations.append("Review P2P transport routing - consider BitChat for cost savings on mobile")

        # High edge device costs
        if cost_by_category.get("edge_device", 0) > 3.0:
            recommendations.append("Edge device participation costs are high - review battery/thermal policies")

        # Transport-specific recommendations
        cellular_events = [e for e in recent_events if e.metadata.get("is_cellular")]
        if len(cellular_events) > 10:
            recommendations.append("High cellular data usage detected - prioritize BitChat/WiFi when available")

        return recommendations

    async def start_monitoring(self):
        """Start background cost monitoring."""
        logger.info("Starting distributed cost monitoring")

        # Start background tasks
        if INFRASTRUCTURE_AVAILABLE:
            tasks = []

            if self.fog_orchestrator:
                tasks.append(asyncio.create_task(self._monitor_fog_compute()))

            if self.transport_manager:
                tasks.append(asyncio.create_task(self._monitor_p2p_transport()))

            if self.edge_manager:
                tasks.append(asyncio.create_task(self._monitor_edge_devices()))

            # Run monitoring tasks
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitor_fog_compute(self):
        """Monitor fog compute resource usage and costs."""
        while True:
            try:
                # Get active Agent Forge phases from orchestrator
                if hasattr(self.fog_orchestrator, "active_tasks"):
                    for task_id, task_info in self.fog_orchestrator.active_tasks.items():
                        node_id = task_info.get("node_id")
                        phase = task_info.get("phase")
                        resources = task_info.get("resources", {})

                        await self.track_fog_compute_cost(
                            node_id=node_id,
                            phase=phase,
                            cpu_cores=resources.get("cpu_cores", 2),
                            memory_gb=resources.get("memory_gb", 4),
                            gpu_hours=resources.get("gpu_hours", 0),
                            duration_seconds=self.config.get("cost_aggregation_interval_seconds", 60),
                        )

                await asyncio.sleep(self.config.get("cost_aggregation_interval_seconds", 60))

            except Exception as e:
                logger.error(f"Error in fog compute monitoring: {e}")
                await asyncio.sleep(60)

    async def _monitor_p2p_transport(self):
        """Monitor P2P transport usage and costs."""
        while True:
            try:
                # Get transport statistics
                if hasattr(self.transport_manager, "stats"):
                    stats = self.transport_manager.stats

                    # Track data usage by transport type
                    bytes_sent = stats.get("bytes_sent", 0)
                    if bytes_sent > 0:
                        # Estimate costs based on transport statistics
                        for transport_type in [TransportType.BITCHAT, TransportType.BETANET, TransportType.QUIC]:
                            routing_decisions = stats.get("routing_decisions", {})
                            usage = routing_decisions.get(transport_type.value, 0)

                            if usage > 0:
                                # Estimate data per transport based on usage percentage
                                data_mb = (bytes_sent / (1024 * 1024)) * (
                                    usage / max(sum(routing_decisions.values()), 1)
                                )

                                await self.track_p2p_transport_cost(
                                    transport_type=transport_type,
                                    data_mb=data_mb,
                                    device_id=self.transport_manager.device_id,
                                    is_cellular=False,  # Would need device context
                                )

                await asyncio.sleep(self.config.get("cost_aggregation_interval_seconds", 60))

            except Exception as e:
                logger.error(f"Error in P2P transport monitoring: {e}")
                await asyncio.sleep(60)

    async def _monitor_edge_devices(self):
        """Monitor edge device participation costs."""
        while True:
            try:
                # Get active edge devices
                if hasattr(self.edge_manager, "devices"):
                    for device_id, device in self.edge_manager.devices.items():
                        # Track device participation costs
                        capabilities = getattr(device, "capabilities", {})

                        battery_level = capabilities.get("battery_percent", 100)
                        thermal_temp = capabilities.get("cpu_temp_celsius", 25)

                        # Estimate costs based on device activity
                        if battery_level < 95:  # Device has been active
                            await self.track_edge_device_cost(
                                device_id=device_id,
                                battery_drain_percent=5,  # Estimated drain
                                thermal_impact_celsius=max(0, thermal_temp - 25),
                                participation_hours=self.config.get("cost_aggregation_interval_seconds", 60) / 3600,
                            )

                await asyncio.sleep(self.config.get("cost_aggregation_interval_seconds", 60))

            except Exception as e:
                logger.error(f"Error in edge device monitoring: {e}")
                await asyncio.sleep(60)


# Helper functions for integration
async def create_cost_tracker_with_infrastructure() -> DistributedCostTracker:
    """Create cost tracker with all available infrastructure components."""
    fog_orchestrator = None
    edge_manager = None
    transport_manager = None

    if INFRASTRUCTURE_AVAILABLE:
        try:
            from ...agent_forge.core.unified_pipeline import UnifiedConfig
            from ...agent_forge.integration.fog_burst import FogBurstConfig, FogBurstOrchestrator

            # Create fog compute orchestrator
            UnifiedConfig()
            fog_config = FogBurstConfig()
            fog_orchestrator = FogBurstOrchestrator(fog_config)

        except Exception as e:
            logger.warning(f"Could not initialize fog orchestrator: {e}")

        try:
            from ...edge.core.edge_manager import EdgeManager

            edge_manager = EdgeManager()
        except Exception as e:
            logger.warning(f"Could not initialize edge manager: {e}")

        try:
            from ...p2p.core.transport_manager import TransportManager, TransportPriority

            transport_manager = TransportManager(
                device_id="cost_tracker", transport_priority=TransportPriority.ADAPTIVE
            )
        except Exception as e:
            logger.warning(f"Could not initialize transport manager: {e}")

    return DistributedCostTracker(
        fog_orchestrator=fog_orchestrator, edge_manager=edge_manager, transport_manager=transport_manager
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create cost tracker
        tracker = await create_cost_tracker_with_infrastructure()

        # Add some budgets
        tracker.add_budget(
            CostBudget(
                category=CostCategory.FOG_COMPUTE,
                budget_amount=50.0,  # $50 per day
                budget_unit=CostUnit.USD_PER_HOUR,
                time_window_hours=24,
            )
        )

        tracker.add_budget(
            CostBudget(
                category=CostCategory.P2P_TRANSPORT,
                budget_amount=10.0,  # $10 per day
                budget_unit=CostUnit.USD_PER_GB,
                time_window_hours=24,
            )
        )

        # Track some example costs
        await tracker.track_fog_compute_cost(
            node_id="fog-node-1", phase="forge_training", cpu_cores=4, memory_gb=8, gpu_hours=1.0, duration_seconds=3600
        )

        await tracker.track_p2p_transport_cost(
            transport_type=TransportType.BETANET, data_mb=100.0, device_id="mobile-device-1", is_cellular=True
        )

        # Get cost summary
        summary = tracker.get_cost_summary()
        print(f"Total cost: ${summary.total_cost_usd:.2f}")
        print("Recommendations:", summary.recommendations)

    asyncio.run(main())
