"""
AIVillage Edge Device Cost-Aware Resource Allocation

This module provides intelligent cost-aware resource allocation for edge device
participation in the AIVillage distributed computing network.

Key features:
- Battery/thermal-aware participation decisions
- Data cost optimization for mobile networks
- Fair cost distribution across edge devices
- Incentive mechanisms for edge device participation
- Real-time resource allocation adjustments
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any

# AIVillage infrastructure imports
try:
    from .distributed_cost_tracker import DistributedCostTracker
    from .p2p_transport_optimizer import P2PTransportOptimizer

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    logging.warning("Edge infrastructure not available - running in standalone mode")
    INFRASTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceParticipationLevel(Enum):
    """Edge device participation levels."""

    MINIMAL = "minimal"  # Basic P2P relay only
    STANDARD = "standard"  # Standard compute participation
    INTENSIVE = "intensive"  # High-compute Agent Forge phases
    STORAGE_ONLY = "storage_only"  # Storage/caching only
    OFFLINE = "offline"  # No participation


class ResourceConstraint(Enum):
    """Types of resource constraints for edge devices."""

    BATTERY_CRITICAL = "battery_critical"  # <15% battery
    THERMAL_LIMIT = "thermal_limit"  # >60°C temperature
    DATA_BUDGET = "data_budget"  # Limited data allowance
    MEMORY_PRESSURE = "memory_pressure"  # High memory usage
    NETWORK_COST = "network_cost"  # Expensive cellular network
    USER_PREFERENCE = "user_preference"  # User-defined limits


@dataclass
class DeviceCapabilities:
    """Edge device capabilities and current state."""

    device_id: str
    device_type: str  # mobile, laptop, desktop, server

    # Hardware specs
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    gpu_available: bool = False

    # Current state
    battery_percent: float | None = None
    cpu_temp_celsius: float = 25.0
    memory_usage_percent: float = 50.0
    network_type: str = "wifi"  # wifi, cellular, ethernet

    # Cost factors
    electricity_cost_per_kwh: float = 0.12
    data_cost_per_gb: float = 0.0  # 0 for unlimited plans
    device_depreciation_per_hour: float = 0.001

    # Participation preferences
    max_participation_level: DeviceParticipationLevel = DeviceParticipationLevel.STANDARD
    allow_cellular_participation: bool = False
    min_battery_threshold: float = 20.0
    max_thermal_threshold: float = 55.0


@dataclass
class ResourceAllocation:
    """Resource allocation for specific edge device."""

    device_id: str
    participation_level: DeviceParticipationLevel
    allocated_cpu_percent: float  # % of CPU to allocate
    allocated_memory_gb: float  # Memory allocation
    allocated_storage_gb: float  # Storage allocation

    # Time constraints
    max_participation_hours: float = 24.0
    estimated_duration_minutes: int = 60

    # Cost analysis
    estimated_cost_usd: float = 0.0
    estimated_reward_usd: float = 0.0
    net_benefit_usd: float = 0.0

    # Task assignment
    assigned_tasks: list[str] = field(default_factory=list)
    agent_forge_phases: list[str] = field(default_factory=list)
    p2p_relay_nodes: set[str] = field(default_factory=set)


@dataclass
class ParticipationIncentive:
    """Incentive structure for edge device participation."""

    base_reward_per_hour: float = 0.10  # Base hourly reward
    cpu_bonus_per_core_hour: float = 0.02  # Bonus per CPU core hour
    gpu_bonus_per_hour: float = 0.50  # GPU usage bonus
    storage_bonus_per_gb_day: float = 0.001  # Storage provision bonus
    p2p_relay_bonus_per_gb: float = 0.01  # P2P data relay bonus

    # Quality multipliers
    uptime_multiplier: float = 1.0  # Based on uptime %
    performance_multiplier: float = 1.0  # Based on task completion
    loyalty_multiplier: float = 1.0  # Long-term participation bonus

    # Cost offsets
    covers_electricity: bool = True  # Whether rewards cover electricity
    covers_wear_tear: bool = True  # Whether rewards cover depreciation
    covers_data_costs: bool = False  # Whether rewards cover cellular data


class EdgeCostAllocator:
    """
    Cost-aware resource allocation system for edge devices.

    Manages fair and efficient allocation of computing tasks across edge devices
    while considering device constraints, costs, and participation incentives.
    """

    def __init__(
        self,
        edge_manager: Any | None = None,
        cost_tracker: DistributedCostTracker | None = None,
        transport_optimizer: P2PTransportOptimizer | None = None,
    ):
        """
        Initialize edge cost allocator.

        Args:
            edge_manager: EdgeManager instance
            cost_tracker: DistributedCostTracker instance
            transport_optimizer: P2PTransportOptimizer instance
        """
        self.edge_manager = edge_manager
        self.cost_tracker = cost_tracker
        self.transport_optimizer = transport_optimizer

        # Device tracking
        self.registered_devices: dict[str, DeviceCapabilities] = {}
        self.current_allocations: dict[str, ResourceAllocation] = {}
        self.participation_history: dict[str, list[dict[str, Any]]] = {}

        # Incentive system
        self.incentive_structure = ParticipationIncentive()
        self.total_rewards_paid: float = 0.0
        self.total_compute_hours: float = 0.0

        # Configuration
        self.config = self._load_config()

        # Task queue and scheduling
        self.pending_tasks: list[dict[str, Any]] = []
        self.task_requirements: dict[str, dict[str, Any]] = {}

        logger.info("Edge cost allocator initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load edge allocation configuration."""
        return {
            "allocation_enabled": True,
            "cost_optimization_enabled": True,
            "incentive_system_enabled": True,
            "fair_allocation_weight": 0.6,  # Weight for fairness vs efficiency
            "cost_awareness_weight": 0.4,  # Weight for cost considerations
            "min_task_duration_minutes": 5,  # Minimum task duration
            "max_devices_per_task": 10,  # Maximum devices per distributed task
            "battery_critical_threshold": 15,  # Critical battery level
            "thermal_warning_threshold": 55,  # Thermal throttling threshold
            "reallocation_interval_minutes": 15,  # How often to recheck allocations
            "participation_timeout_hours": 24,  # Max participation time
        }

    async def register_device(self, capabilities: DeviceCapabilities) -> bool:
        """
        Register edge device for participation.

        Args:
            capabilities: Device capabilities and preferences

        Returns:
            True if device registered successfully
        """
        device_id = capabilities.device_id

        # Validate device capabilities
        if not self._validate_device_capabilities(capabilities):
            logger.warning(f"Device {device_id} failed capability validation")
            return False

        # Store device capabilities
        self.registered_devices[device_id] = capabilities

        # Initialize participation history
        if device_id not in self.participation_history:
            self.participation_history[device_id] = []

        # Record registration
        self.participation_history[device_id].append(
            {
                "timestamp": time.time(),
                "event": "device_registered",
                "capabilities": capabilities.device_type,
                "max_participation": capabilities.max_participation_level.value,
            }
        )

        logger.info(
            f"Device {device_id} registered: {capabilities.device_type} with "
            f"{capabilities.cpu_cores} cores, {capabilities.memory_gb}GB RAM"
        )

        return True

    def _validate_device_capabilities(self, capabilities: DeviceCapabilities) -> bool:
        """Validate device capabilities for minimum requirements."""
        # Minimum hardware requirements
        if capabilities.cpu_cores < 1:
            return False
        if capabilities.memory_gb < 1.0:
            return False
        if capabilities.storage_gb < 5.0:
            return False

        # Safety thresholds
        if (
            capabilities.battery_percent is not None
            and capabilities.battery_percent < self.config["battery_critical_threshold"]
        ):
            return False

        if capabilities.cpu_temp_celsius > 70:  # Critical temperature
            return False

        return True

    async def allocate_resources_for_task(
        self, task_id: str, task_requirements: dict[str, Any], preferred_devices: list[str] | None = None
    ) -> list[ResourceAllocation]:
        """
        Allocate edge device resources for specific task.

        Args:
            task_id: Unique task identifier
            task_requirements: Resource requirements for task
            preferred_devices: Optional list of preferred device IDs

        Returns:
            List of resource allocations for selected devices
        """
        # Store task requirements
        self.task_requirements[task_id] = task_requirements

        # Get available devices
        available_devices = await self._get_available_devices()

        # Filter by preferences if specified
        if preferred_devices:
            available_devices = [d for d in available_devices if d.device_id in preferred_devices]

        if not available_devices:
            logger.warning(f"No available devices for task {task_id}")
            return []

        # Calculate resource allocations
        allocations = await self._calculate_optimal_allocation(
            task_id=task_id, task_requirements=task_requirements, available_devices=available_devices
        )

        # Apply allocations
        for allocation in allocations:
            self.current_allocations[allocation.device_id] = allocation

            # Update device participation history
            device_id = allocation.device_id
            if device_id in self.participation_history:
                self.participation_history[device_id].append(
                    {
                        "timestamp": time.time(),
                        "event": "task_allocated",
                        "task_id": task_id,
                        "participation_level": allocation.participation_level.value,
                        "estimated_cost": allocation.estimated_cost_usd,
                        "estimated_reward": allocation.estimated_reward_usd,
                    }
                )

        logger.info(f"Allocated resources for task {task_id}: {len(allocations)} devices")

        return allocations

    async def _get_available_devices(self) -> list[DeviceCapabilities]:
        """Get list of currently available devices."""
        available = []
        time.time()

        for device_id, capabilities in self.registered_devices.items():
            # Check if device has active allocation
            if device_id in self.current_allocations:
                allocation = self.current_allocations[device_id]
                if allocation.estimated_duration_minutes > 0:
                    continue  # Device busy

            # Check device constraints
            constraints = await self._check_device_constraints(capabilities)
            if not constraints:  # No blocking constraints
                available.append(capabilities)
            elif self.config.get("cost_optimization_enabled", True):
                # Consider devices with non-critical constraints
                non_critical = [c for c in constraints if c != ResourceConstraint.BATTERY_CRITICAL]
                if len(non_critical) == len(constraints):
                    available.append(capabilities)

        return available

    async def _check_device_constraints(self, capabilities: DeviceCapabilities) -> list[ResourceConstraint]:
        """Check for resource constraints on device."""
        constraints = []

        # Battery constraint
        if (
            capabilities.battery_percent is not None
            and capabilities.battery_percent < capabilities.min_battery_threshold
        ):
            if capabilities.battery_percent < self.config["battery_critical_threshold"]:
                constraints.append(ResourceConstraint.BATTERY_CRITICAL)

        # Thermal constraint
        if capabilities.cpu_temp_celsius > capabilities.max_thermal_threshold:
            constraints.append(ResourceConstraint.THERMAL_LIMIT)

        # Memory pressure
        if capabilities.memory_usage_percent > 85:
            constraints.append(ResourceConstraint.MEMORY_PRESSURE)

        # Data cost constraint (cellular network)
        if capabilities.network_type == "cellular" and not capabilities.allow_cellular_participation:
            constraints.append(ResourceConstraint.NETWORK_COST)

        # Data budget constraint
        if capabilities.data_cost_per_gb > 1.0:  # Expensive data
            constraints.append(ResourceConstraint.DATA_BUDGET)

        return constraints

    async def _calculate_optimal_allocation(
        self, task_id: str, task_requirements: dict[str, Any], available_devices: list[DeviceCapabilities]
    ) -> list[ResourceAllocation]:
        """Calculate optimal resource allocation across devices."""

        # Extract task requirements
        required_cpu_cores = task_requirements.get("cpu_cores", 2)
        required_memory_gb = task_requirements.get("memory_gb", 4)
        required_storage_gb = task_requirements.get("storage_gb", 10)
        estimated_duration_minutes = task_requirements.get("duration_minutes", 60)
        task_priority = task_requirements.get("priority", "normal")

        # Calculate device scores and costs
        device_scores = []
        for device in available_devices:
            score_info = await self._calculate_device_score(device=device, task_requirements=task_requirements)
            device_scores.append((device, score_info))

        # Sort by score (higher is better)
        device_scores.sort(key=lambda x: x[1]["total_score"], reverse=True)

        # Select devices for allocation
        allocations = []
        remaining_cpu = required_cpu_cores
        remaining_memory = required_memory_gb
        remaining_storage = required_storage_gb

        max_devices = min(len(device_scores), self.config.get("max_devices_per_task", 10))

        for device, score_info in device_scores[:max_devices]:
            if remaining_cpu <= 0 and remaining_memory <= 0 and remaining_storage <= 0:
                break

            # Calculate allocation for this device
            device_cpu = min(remaining_cpu, device.cpu_cores * 0.8)  # Use up to 80% CPU
            device_memory = min(remaining_memory, device.memory_gb * 0.6)  # Use up to 60% memory
            device_storage = min(remaining_storage, device.storage_gb * 0.3)  # Use up to 30% storage

            # Determine participation level
            participation_level = self._determine_participation_level(
                device=device,
                cpu_allocation=device_cpu,
                memory_allocation=device_memory,
                constraints=score_info["constraints"],
            )

            # Calculate costs and rewards
            cost_info = await self._calculate_participation_cost(
                device=device,
                participation_level=participation_level,
                duration_minutes=estimated_duration_minutes,
                cpu_percent=device_cpu / device.cpu_cores * 100,
                memory_gb=device_memory,
            )

            # Create allocation
            allocation = ResourceAllocation(
                device_id=device.device_id,
                participation_level=participation_level,
                allocated_cpu_percent=device_cpu / device.cpu_cores * 100,
                allocated_memory_gb=device_memory,
                allocated_storage_gb=device_storage,
                estimated_duration_minutes=estimated_duration_minutes,
                estimated_cost_usd=cost_info["total_cost"],
                estimated_reward_usd=cost_info["total_reward"],
                net_benefit_usd=cost_info["total_reward"] - cost_info["total_cost"],
                assigned_tasks=[task_id],
            )

            # Only allocate if net benefit is positive (or task is critical)
            if allocation.net_benefit_usd > 0 or task_priority == "critical":
                allocations.append(allocation)
                remaining_cpu -= device_cpu
                remaining_memory -= device_memory
                remaining_storage -= device_storage

        # If we couldn't satisfy requirements, try with lower participation levels
        if (remaining_cpu > 0 or remaining_memory > 0) and not allocations:
            logger.warning(f"Could not fully satisfy requirements for task {task_id}")
            # Could implement fallback strategies here

        return allocations

    async def _calculate_device_score(
        self, device: DeviceCapabilities, task_requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate scoring for device selection."""

        # Check constraints
        constraints = await self._check_device_constraints(device)

        # Base capability score (0-100)
        cpu_score = min(100, (device.cpu_cores / 4) * 100)  # Normalize to 4 cores
        memory_score = min(100, (device.memory_gb / 8) * 100)  # Normalize to 8GB
        storage_score = min(100, (device.storage_gb / 100) * 100)  # Normalize to 100GB

        capability_score = (cpu_score + memory_score + storage_score) / 3

        # Cost efficiency score (0-100, higher is better/cheaper)
        hourly_cost = (
            device.electricity_cost_per_kwh * 0.5
            + device.device_depreciation_per_hour  # Estimated 0.5kWh usage
            + device.data_cost_per_gb * 0.1
        )  # Estimated 0.1GB usage

        cost_score = max(0, 100 - (hourly_cost * 1000))  # Scale to 0-100

        # Availability score (0-100)
        availability_score = 100

        # Penalize constraints
        for constraint in constraints:
            if constraint == ResourceConstraint.BATTERY_CRITICAL:
                availability_score -= 50  # Major penalty
            elif constraint == ResourceConstraint.THERMAL_LIMIT:
                availability_score -= 30
            elif constraint == ResourceConstraint.NETWORK_COST:
                cost_score -= 20
            else:
                availability_score -= 10

        # Participation history bonus
        device_history = self.participation_history.get(device.device_id, [])
        successful_tasks = len([h for h in device_history if h.get("event") == "task_completed"])
        history_bonus = min(20, successful_tasks * 2)  # Up to 20% bonus

        # Weighted total score
        fair_weight = self.config.get("fair_allocation_weight", 0.6)
        cost_weight = self.config.get("cost_awareness_weight", 0.4)

        total_score = (
            capability_score * fair_weight + cost_score * cost_weight + availability_score * 0.3 + history_bonus
        )

        return {
            "total_score": total_score,
            "capability_score": capability_score,
            "cost_score": cost_score,
            "availability_score": availability_score,
            "history_bonus": history_bonus,
            "constraints": constraints,
            "hourly_cost": hourly_cost,
        }

    def _determine_participation_level(
        self,
        device: DeviceCapabilities,
        cpu_allocation: float,
        memory_allocation: float,
        constraints: list[ResourceConstraint],
    ) -> DeviceParticipationLevel:
        """Determine appropriate participation level for device."""

        # Check for critical constraints
        if ResourceConstraint.BATTERY_CRITICAL in constraints:
            return DeviceParticipationLevel.OFFLINE

        # Check user preferences
        if device.max_participation_level == DeviceParticipationLevel.MINIMAL:
            return DeviceParticipationLevel.MINIMAL
        elif device.max_participation_level == DeviceParticipationLevel.STORAGE_ONLY:
            return DeviceParticipationLevel.STORAGE_ONLY

        # Determine based on resource allocation
        cpu_percent = (cpu_allocation / device.cpu_cores) * 100
        memory_percent = (memory_allocation / device.memory_gb) * 100

        if cpu_percent > 60 or memory_percent > 50:
            if device.max_participation_level == DeviceParticipationLevel.INTENSIVE:
                return DeviceParticipationLevel.INTENSIVE
            else:
                return DeviceParticipationLevel.STANDARD
        elif cpu_percent > 30 or memory_percent > 30:
            return DeviceParticipationLevel.STANDARD
        else:
            return DeviceParticipationLevel.MINIMAL

    async def _calculate_participation_cost(
        self,
        device: DeviceCapabilities,
        participation_level: DeviceParticipationLevel,
        duration_minutes: int,
        cpu_percent: float,
        memory_gb: float,
    ) -> dict[str, float]:
        """Calculate costs and rewards for device participation."""

        duration_hours = duration_minutes / 60.0

        # Calculate costs
        electricity_cost = (
            device.electricity_cost_per_kwh * (cpu_percent / 100) * 0.5 * duration_hours  # Estimated 0.5kWh at 100% CPU
        )

        wear_cost = device.device_depreciation_per_hour * duration_hours

        data_cost = 0.0
        if device.network_type == "cellular":
            estimated_data_gb = 0.1 * duration_hours  # Estimated data usage
            data_cost = estimated_data_gb * device.data_cost_per_gb

        total_cost = electricity_cost + wear_cost + data_cost

        # Calculate rewards based on participation level
        base_reward = self.incentive_structure.base_reward_per_hour * duration_hours

        cpu_bonus = (
            self.incentive_structure.cpu_bonus_per_core_hour * (cpu_percent / 100) * device.cpu_cores * duration_hours
        )

        gpu_bonus = 0.0
        if device.gpu_available and participation_level == DeviceParticipationLevel.INTENSIVE:
            gpu_bonus = self.incentive_structure.gpu_bonus_per_hour * duration_hours

        storage_bonus = self.incentive_structure.storage_bonus_per_gb_day * device.storage_gb * (duration_hours / 24)

        # Apply multipliers
        uptime_multiplier = self._calculate_uptime_multiplier(device.device_id)

        total_reward = (
            (base_reward + cpu_bonus + gpu_bonus + storage_bonus)
            * uptime_multiplier
            * self.incentive_structure.performance_multiplier
            * self.incentive_structure.loyalty_multiplier
        )

        return {
            "electricity_cost": electricity_cost,
            "wear_cost": wear_cost,
            "data_cost": data_cost,
            "total_cost": total_cost,
            "base_reward": base_reward,
            "cpu_bonus": cpu_bonus,
            "gpu_bonus": gpu_bonus,
            "storage_bonus": storage_bonus,
            "total_reward": total_reward,
            "net_benefit": total_reward - total_cost,
        }

    def _calculate_uptime_multiplier(self, device_id: str) -> float:
        """Calculate uptime multiplier based on device history."""
        history = self.participation_history.get(device_id, [])
        if len(history) < 2:
            return 1.0

        # Calculate success rate from recent tasks
        recent_tasks = [h for h in history[-10:] if h.get("event") in ["task_allocated", "task_completed"]]
        if len(recent_tasks) < 2:
            return 1.0

        completed_tasks = len([h for h in recent_tasks if h.get("event") == "task_completed"])
        allocated_tasks = len([h for h in recent_tasks if h.get("event") == "task_allocated"])

        if allocated_tasks == 0:
            return 1.0

        success_rate = completed_tasks / allocated_tasks
        return 0.5 + (success_rate * 0.5)  # 0.5 to 1.0 multiplier

    async def update_device_status(self, device_id: str, status_update: dict[str, Any]) -> bool:
        """Update device status and check for reallocation needs."""
        if device_id not in self.registered_devices:
            return False

        device = self.registered_devices[device_id]

        # Update device capabilities with new status
        if "battery_percent" in status_update:
            device.battery_percent = status_update["battery_percent"]
        if "cpu_temp_celsius" in status_update:
            device.cpu_temp_celsius = status_update["cpu_temp_celsius"]
        if "memory_usage_percent" in status_update:
            device.memory_usage_percent = status_update["memory_usage_percent"]
        if "network_type" in status_update:
            device.network_type = status_update["network_type"]

        # Check if reallocation needed
        if device_id in self.current_allocations:
            constraints = await self._check_device_constraints(device)

            # Critical constraint - stop participation
            if ResourceConstraint.BATTERY_CRITICAL in constraints:
                await self._deallocate_device(device_id, reason="battery_critical")
            elif len(constraints) > 2:  # Multiple constraints
                await self._reduce_device_allocation(device_id, reason="resource_constraints")

        return True

    async def _deallocate_device(self, device_id: str, reason: str):
        """Remove device from current allocations."""
        if device_id in self.current_allocations:
            allocation = self.current_allocations.pop(device_id)

            # Record deallocation
            if device_id in self.participation_history:
                self.participation_history[device_id].append(
                    {
                        "timestamp": time.time(),
                        "event": "task_deallocated",
                        "reason": reason,
                        "tasks_affected": allocation.assigned_tasks,
                    }
                )

            logger.info(f"Deallocated device {device_id}: {reason}")

    async def _reduce_device_allocation(self, device_id: str, reason: str):
        """Reduce device allocation due to constraints."""
        if device_id not in self.current_allocations:
            return

        allocation = self.current_allocations[device_id]

        # Reduce allocation by 50%
        allocation.allocated_cpu_percent *= 0.5
        allocation.allocated_memory_gb *= 0.5
        allocation.participation_level = DeviceParticipationLevel.MINIMAL

        # Recalculate costs and rewards
        device = self.registered_devices[device_id]
        cost_info = await self._calculate_participation_cost(
            device=device,
            participation_level=allocation.participation_level,
            duration_minutes=allocation.estimated_duration_minutes,
            cpu_percent=allocation.allocated_cpu_percent,
            memory_gb=allocation.allocated_memory_gb,
        )

        allocation.estimated_cost_usd = cost_info["total_cost"]
        allocation.estimated_reward_usd = cost_info["total_reward"]
        allocation.net_benefit_usd = cost_info["total_reward"] - cost_info["total_cost"]

        logger.info(f"Reduced allocation for device {device_id}: {reason}")

    def get_allocation_report(self) -> dict[str, Any]:
        """Generate comprehensive allocation report."""

        # Active allocations summary
        active_allocations = len(self.current_allocations)
        total_estimated_cost = sum(a.estimated_cost_usd for a in self.current_allocations.values())
        total_estimated_reward = sum(a.estimated_reward_usd for a in self.current_allocations.values())

        # Device participation distribution
        participation_levels = {}
        for allocation in self.current_allocations.values():
            level = allocation.participation_level.value
            participation_levels[level] = participation_levels.get(level, 0) + 1

        # Device type distribution
        device_types = {}
        for device in self.registered_devices.values():
            device_type = device.device_type
            device_types[device_type] = device_types.get(device_type, 0) + 1

        # Cost efficiency metrics
        total_cpu_allocated = sum(a.allocated_cpu_percent for a in self.current_allocations.values())
        cost_per_cpu_percent = total_estimated_cost / max(1, total_cpu_allocated)

        return {
            "allocation_summary": {
                "active_allocations": active_allocations,
                "registered_devices": len(self.registered_devices),
                "total_estimated_cost": total_estimated_cost,
                "total_estimated_reward": total_estimated_reward,
                "net_system_cost": total_estimated_cost - total_estimated_reward,
                "participation_distribution": participation_levels,
                "device_type_distribution": device_types,
            },
            "efficiency_metrics": {
                "cost_per_cpu_percent": cost_per_cpu_percent,
                "average_net_benefit": (total_estimated_reward - total_estimated_cost) / max(1, active_allocations),
                "resource_utilization": total_cpu_allocated / max(1, len(self.current_allocations) * 100),
            },
            "incentive_summary": {
                "total_rewards_paid": self.total_rewards_paid,
                "total_compute_hours": self.total_compute_hours,
                "average_reward_per_hour": self.total_rewards_paid / max(1, self.total_compute_hours),
            },
        }


# Helper functions
async def create_edge_cost_allocator_with_infrastructure() -> EdgeCostAllocator:
    """Create edge cost allocator with available infrastructure."""
    edge_manager = None
    cost_tracker = None
    transport_optimizer = None

    if INFRASTRUCTURE_AVAILABLE:
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

        try:
            from .p2p_transport_optimizer import create_transport_optimizer_with_infrastructure

            transport_optimizer = await create_transport_optimizer_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize transport optimizer: {e}")

    return EdgeCostAllocator(
        edge_manager=edge_manager, cost_tracker=cost_tracker, transport_optimizer=transport_optimizer
    )


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create edge cost allocator
        allocator = await create_edge_cost_allocator_with_infrastructure()

        # Example device registrations
        mobile_device = DeviceCapabilities(
            device_id="mobile-pixel-7",
            device_type="mobile",
            cpu_cores=8,
            memory_gb=8,
            storage_gb=128,
            battery_percent=75,
            network_type="wifi",
            max_participation_level=DeviceParticipationLevel.STANDARD,
            allow_cellular_participation=False,
            min_battery_threshold=25.0,
        )

        laptop_device = DeviceCapabilities(
            device_id="laptop-macbook-pro",
            device_type="laptop",
            cpu_cores=12,
            memory_gb=32,
            storage_gb=1024,
            network_type="wifi",
            max_participation_level=DeviceParticipationLevel.INTENSIVE,
            electricity_cost_per_kwh=0.15,
        )

        print("🎯 AIVillage Edge Cost Allocation Demo")
        print("=" * 60)

        # Register devices
        await allocator.register_device(mobile_device)
        await allocator.register_device(laptop_device)
        print(f"✅ Registered {len(allocator.registered_devices)} devices")

        # Example task requirements
        task_requirements = {
            "task_type": "agent_forge_phase",
            "phase": "quietstar",
            "cpu_cores": 6,
            "memory_gb": 12,
            "storage_gb": 20,
            "duration_minutes": 120,
            "priority": "normal",
        }

        # Allocate resources
        allocations = await allocator.allocate_resources_for_task(
            task_id="quietstar-training-001", task_requirements=task_requirements
        )

        print("\n📋 Task Allocation Results:")
        for allocation in allocations:
            device = allocator.registered_devices[allocation.device_id]
            print(f"   Device: {allocation.device_id} ({device.device_type})")
            print(f"   Level: {allocation.participation_level.value}")
            print(f"   CPU: {allocation.allocated_cpu_percent:.1f}%")
            print(f"   Memory: {allocation.allocated_memory_gb:.1f}GB")
            print(f"   Cost: ${allocation.estimated_cost_usd:.4f}")
            print(f"   Reward: ${allocation.estimated_reward_usd:.4f}")
            print(f"   Net: ${allocation.net_benefit_usd:.4f}")
            print()

        # Update device status (simulate battery drain)
        await allocator.update_device_status(
            "mobile-pixel-7", {"battery_percent": 18, "cpu_temp_celsius": 45}  # Low battery
        )
        print("📱 Updated mobile device status (low battery)")

        # Generate allocation report
        report = allocator.get_allocation_report()
        print("📊 Allocation Report:")
        print(f"   Active allocations: {report['allocation_summary']['active_allocations']}")
        print(f"   Total cost: ${report['allocation_summary']['total_estimated_cost']:.4f}")
        print(f"   Total reward: ${report['allocation_summary']['total_estimated_reward']:.4f}")
        print(f"   System efficiency: {report['efficiency_metrics']['resource_utilization']:.1%}")

    asyncio.run(main())
