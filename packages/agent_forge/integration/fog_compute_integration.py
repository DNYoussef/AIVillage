"""
Agent Forge Fog Compute Integration

Integrates the Agent Forge pipeline with fog computing infrastructure
to enable distributed processing, load balancing, and resource optimization
across edge devices and fog nodes.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import fog compute infrastructure
try:
    from ...edge.core.edge_manager import EdgeManager
    from ...edge.fog_compute.fog_coordinator import FogCoordinator
    from ...edge.fog_compute.fog_node import FogNode
    from ...edge.fog_compute.task_distributor import TaskDistributor

    FOG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Fog compute system not available: {e}")
    FOG_AVAILABLE = False

# Import edge device management
try:
    from ...edge.core.device_registry import DeviceRegistry
    from ...edge.mobile.resource_manager import BatteryThermalResourceManager

    EDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Edge device system not available: {e}")
    EDGE_AVAILABLE = False

# Import unified pipeline
from ..core.phase_controller import PhaseResult
from ..core.unified_pipeline import UnifiedConfig, UnifiedPipeline

logger = logging.getLogger(__name__)


@dataclass
class FogComputeConfig:
    """Configuration for fog compute integration."""

    # Fog network configuration
    enable_fog_coordination: bool = True
    enable_dynamic_load_balancing: bool = True
    enable_resource_optimization: bool = True
    enable_fault_tolerance: bool = True

    # Node management
    min_fog_nodes: int = 2
    max_fog_nodes: int = 20
    node_discovery_timeout: int = 30
    node_health_check_interval: int = 60

    # Task distribution
    task_scheduling_algorithm: str = "priority_weighted"  # "round_robin", "load_balanced", "priority_weighted"
    load_balancing_threshold: float = 0.8
    task_timeout: int = 3600  # seconds
    max_retries: int = 3

    # Resource allocation
    cpu_allocation_strategy: str = "adaptive"  # "adaptive", "fixed", "proportional"
    memory_allocation_strategy: str = "adaptive"
    gpu_allocation_priority: list[str] = field(default_factory=lambda: ["adas", "forge_training", "final_compression"])

    # Edge device integration
    enable_edge_devices: bool = True
    edge_device_roles: list[str] = field(default_factory=lambda: ["compute", "storage", "relay"])
    mobile_resource_management: bool = True
    battery_threshold: int = 20  # Minimum battery percentage for participation

    # Performance optimization
    enable_caching: bool = True
    enable_prefetching: bool = True
    cache_size_gb: int = 10
    compression_level: int = 1  # 0-3, higher = more compression

    # Monitoring and analytics
    enable_monitoring: bool = True
    metrics_collection_interval: int = 30
    performance_logging: bool = True
    resource_usage_tracking: bool = True


class FogComputeOrchestrator:
    """
    Orchestrates Agent Forge pipeline execution across fog compute infrastructure.
    """

    def __init__(self, base_config: UnifiedConfig, fog_config: FogComputeConfig):
        self.base_config = base_config
        self.fog_config = fog_config
        self.logger = logging.getLogger(__name__)

        # Infrastructure components
        self.fog_coordinator = None
        self.task_distributor = None
        self.edge_manager = None
        self.device_registry = None
        self.resource_manager = None

        # State tracking
        self.fog_nodes = []
        self.edge_devices = []
        self.active_tasks = {}
        self.performance_metrics = {}
        self.resource_usage = {}

        # Initialize infrastructure
        self._initialize_fog_infrastructure()

    def _initialize_fog_infrastructure(self):
        """Initialize fog compute infrastructure components."""
        try:
            if FOG_AVAILABLE:
                self.fog_coordinator = FogCoordinator()
                self.task_distributor = TaskDistributor()
                self.logger.info("Fog compute coordinator initialized")

            if EDGE_AVAILABLE:
                self.edge_manager = EdgeManager()
                self.device_registry = DeviceRegistry()
                self.resource_manager = BatteryThermalResourceManager()
                self.logger.info("Edge device management initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize fog infrastructure: {e}")
            raise

    async def discover_fog_infrastructure(self) -> dict[str, Any]:
        """Discover available fog nodes and edge devices."""
        infrastructure = {
            "fog_nodes": [],
            "edge_devices": [],
            "total_compute_power": 0,
            "total_memory_gb": 0,
            "total_storage_gb": 0,
            "gpu_nodes": 0,
        }

        try:
            # Discover fog nodes
            if self.fog_coordinator:
                self.logger.info("Discovering fog compute nodes...")
                fog_nodes = await self.fog_coordinator.discover_nodes(timeout=self.fog_config.node_discovery_timeout)

                for node in fog_nodes:
                    node_info = {
                        "node_id": node["node_id"],
                        "type": "fog_node",
                        "compute_power": node.get("compute_power", 1.0),
                        "cpu_cores": node.get("cpu_cores", 4),
                        "memory_gb": node.get("memory_gb", 8),
                        "storage_gb": node.get("storage_gb", 50),
                        "gpu_available": node.get("gpu_available", False),
                        "gpu_memory_gb": node.get("gpu_memory_gb", 0),
                        "network_bandwidth_mbps": node.get("network_bandwidth_mbps", 100),
                        "availability": node.get("availability", 1.0),
                        "current_load": node.get("current_load", 0.0),
                    }
                    infrastructure["fog_nodes"].append(node_info)
                    infrastructure["total_compute_power"] += node_info["compute_power"]
                    infrastructure["total_memory_gb"] += node_info["memory_gb"]
                    infrastructure["total_storage_gb"] += node_info["storage_gb"]
                    if node_info["gpu_available"]:
                        infrastructure["gpu_nodes"] += 1

            # Discover edge devices
            if self.edge_manager and self.fog_config.enable_edge_devices:
                self.logger.info("Discovering edge devices...")
                edge_devices = await self.edge_manager.discover_devices()

                for device in edge_devices:
                    # Check device eligibility
                    battery_level = device.get("battery_level", 100)
                    thermal_state = device.get("thermal_state", "normal")

                    if battery_level >= self.fog_config.battery_threshold and thermal_state in ["normal", "warm"]:

                        device_info = {
                            "device_id": device["device_id"],
                            "type": "edge_device",
                            "device_type": device.get("device_type", "mobile"),
                            "compute_power": device.get("compute_power", 0.5),
                            "cpu_cores": device.get("cpu_cores", 2),
                            "memory_gb": device.get("memory_gb", 4),
                            "storage_gb": device.get("storage_gb", 32),
                            "battery_level": battery_level,
                            "thermal_state": thermal_state,
                            "network_type": device.get("network_type", "wifi"),
                            "capabilities": device.get("capabilities", []),
                            "role": self._assign_device_role(device),
                        }
                        infrastructure["edge_devices"].append(device_info)

            self.fog_nodes = infrastructure["fog_nodes"]
            self.edge_devices = infrastructure["edge_devices"]

            total_nodes = len(infrastructure["fog_nodes"]) + len(infrastructure["edge_devices"])
            self.logger.info(
                f"Discovered fog infrastructure: {len(infrastructure['fog_nodes'])} fog nodes, "
                f"{len(infrastructure['edge_devices'])} edge devices "
                f"(Total: {total_nodes} compute nodes, {infrastructure['gpu_nodes']} with GPU)"
            )

            return infrastructure

        except Exception as e:
            self.logger.error(f"Infrastructure discovery failed: {e}")
            return infrastructure

    def _assign_device_role(self, device: dict[str, Any]) -> str:
        """Assign role to edge device based on capabilities."""
        compute_power = device.get("compute_power", 0)
        memory_gb = device.get("memory_gb", 0)
        storage_gb = device.get("storage_gb", 0)
        battery_level = device.get("battery_level", 100)

        # High-performance devices for compute
        if compute_power > 1.0 and memory_gb >= 8 and battery_level > 50:
            return "compute"
        # Devices with good storage for caching
        elif storage_gb > 32 and battery_level > 30:
            return "storage"
        # Other devices for relay/coordination
        else:
            return "relay"

    async def optimize_phase_distribution(self, phases: list[str]) -> dict[str, Any]:
        """Optimize distribution of Agent Forge phases across fog infrastructure."""
        optimization_plan = {
            "phase_assignments": {},
            "resource_allocation": {},
            "load_distribution": {},
            "estimated_completion_time": 0,
        }

        try:
            # Calculate resource requirements for each phase
            phase_requirements = self._calculate_phase_requirements(phases)

            # Get available resources
            available_resources = self._calculate_available_resources()

            # Optimize assignments using scheduling algorithm
            if self.fog_config.task_scheduling_algorithm == "priority_weighted":
                assignments = self._priority_weighted_scheduling(phase_requirements, available_resources)
            elif self.fog_config.task_scheduling_algorithm == "load_balanced":
                assignments = self._load_balanced_scheduling(phase_requirements, available_resources)
            else:  # round_robin
                assignments = self._round_robin_scheduling(phase_requirements, available_resources)

            optimization_plan["phase_assignments"] = assignments
            optimization_plan["resource_allocation"] = self._calculate_resource_allocation(assignments)
            optimization_plan["load_distribution"] = self._calculate_load_distribution(assignments)
            optimization_plan["estimated_completion_time"] = self._estimate_completion_time(assignments)

            self.logger.info(
                f"Optimized phase distribution across {len(assignments)} nodes. "
                f"Estimated completion: {optimization_plan['estimated_completion_time']:.1f}s"
            )

            return optimization_plan

        except Exception as e:
            self.logger.error(f"Phase distribution optimization failed: {e}")
            return optimization_plan

    def _calculate_phase_requirements(self, phases: list[str]) -> dict[str, dict[str, Any]]:
        """Calculate resource requirements for each phase."""
        requirements = {}

        # Define base requirements for each phase type
        phase_profiles = {
            "evomerge": {
                "cpu_cores": 4,
                "memory_gb": 8,
                "storage_gb": 10,
                "gpu_required": False,
                "compute_intensity": 0.8,
                "duration_estimate": 1800,  # 30 minutes
            },
            "quietstar": {
                "cpu_cores": 2,
                "memory_gb": 6,
                "storage_gb": 5,
                "gpu_required": False,
                "compute_intensity": 0.6,
                "duration_estimate": 2400,  # 40 minutes
            },
            "bitnet_compression": {
                "cpu_cores": 2,
                "memory_gb": 4,
                "storage_gb": 8,
                "gpu_required": False,
                "compute_intensity": 0.5,
                "duration_estimate": 600,  # 10 minutes
            },
            "forge_training": {
                "cpu_cores": 6,
                "memory_gb": 12,
                "storage_gb": 15,
                "gpu_required": True,
                "compute_intensity": 1.0,
                "duration_estimate": 7200,  # 2 hours
            },
            "tool_persona_baking": {
                "cpu_cores": 3,
                "memory_gb": 6,
                "storage_gb": 8,
                "gpu_required": False,
                "compute_intensity": 0.7,
                "duration_estimate": 3600,  # 1 hour
            },
            "adas": {
                "cpu_cores": 8,
                "memory_gb": 16,
                "storage_gb": 12,
                "gpu_required": True,
                "compute_intensity": 1.0,
                "duration_estimate": 5400,  # 1.5 hours
            },
            "final_compression": {
                "cpu_cores": 4,
                "memory_gb": 8,
                "storage_gb": 20,
                "gpu_required": False,
                "compute_intensity": 0.8,
                "duration_estimate": 1800,  # 30 minutes
            },
        }

        for phase in phases:
            if phase in phase_profiles:
                requirements[phase] = phase_profiles[phase].copy()

                # Adjust based on configuration
                if phase == "forge_training":
                    # Scale based on training steps
                    training_steps = self.base_config.training_steps
                    scale_factor = training_steps / 100000  # Base 100k steps
                    requirements[phase]["duration_estimate"] *= scale_factor
                    requirements[phase]["memory_gb"] *= min(scale_factor, 2.0)

        return requirements

    def _calculate_available_resources(self) -> dict[str, dict[str, Any]]:
        """Calculate available resources across all nodes."""
        resources = {}

        # Fog nodes
        for node in self.fog_nodes:
            resources[node["node_id"]] = {
                "type": "fog_node",
                "cpu_cores": node["cpu_cores"],
                "memory_gb": node["memory_gb"],
                "storage_gb": node["storage_gb"],
                "gpu_available": node["gpu_available"],
                "compute_power": node["compute_power"],
                "current_load": node.get("current_load", 0.0),
                "availability": node.get("availability", 1.0),
            }

        # Edge devices
        for device in self.edge_devices:
            resources[device["device_id"]] = {
                "type": "edge_device",
                "cpu_cores": device["cpu_cores"],
                "memory_gb": device["memory_gb"],
                "storage_gb": device["storage_gb"],
                "gpu_available": False,  # Most edge devices don't have discrete GPU
                "compute_power": device["compute_power"],
                "current_load": 0.0,  # Assume fresh for edge devices
                "availability": 1.0,
                "battery_level": device.get("battery_level", 100),
                "thermal_state": device.get("thermal_state", "normal"),
                "role": device.get("role", "compute"),
            }

        return resources

    def _priority_weighted_scheduling(
        self, phase_requirements: dict, available_resources: dict
    ) -> dict[str, list[str]]:
        """Schedule phases using priority-weighted algorithm."""
        assignments = {}

        # Define phase priorities (higher = more important)
        phase_priorities = {
            "evomerge": 10,
            "quietstar": 8,
            "bitnet_compression": 6,
            "forge_training": 9,
            "tool_persona_baking": 7,
            "adas": 9,
            "final_compression": 5,
        }

        # Sort phases by priority
        sorted_phases = sorted(phase_requirements.keys(), key=lambda p: phase_priorities.get(p, 0), reverse=True)

        # Sort nodes by capability
        sorted_nodes = sorted(
            available_resources.items(),
            key=lambda item: (item[1]["compute_power"], item[1]["gpu_available"], -item[1]["current_load"]),
            reverse=True,
        )

        # Assign phases to nodes
        for phase in sorted_phases:
            requirements = phase_requirements[phase]

            # Find best matching node
            best_node = None
            best_score = -1

            for node_id, resources in sorted_nodes:
                # Check hard requirements
                if (
                    resources["cpu_cores"] >= requirements["cpu_cores"]
                    and resources["memory_gb"] >= requirements["memory_gb"]
                    and resources["storage_gb"] >= requirements["storage_gb"]
                    and (not requirements["gpu_required"] or resources["gpu_available"])
                ):

                    # Calculate fit score
                    cpu_fit = min(resources["cpu_cores"] / requirements["cpu_cores"], 2.0)
                    memory_fit = min(resources["memory_gb"] / requirements["memory_gb"], 2.0)
                    load_penalty = 1.0 - resources["current_load"]

                    score = (cpu_fit + memory_fit) * load_penalty * resources["compute_power"]

                    # Bonus for GPU availability when needed
                    if requirements["gpu_required"] and resources["gpu_available"]:
                        score *= 1.5

                    # Penalty for edge devices on heavy tasks
                    if resources["type"] == "edge_device" and requirements["compute_intensity"] > 0.8:
                        score *= 0.7

                    if score > best_score:
                        best_score = score
                        best_node = node_id

            if best_node:
                if best_node not in assignments:
                    assignments[best_node] = []
                assignments[best_node].append(phase)

                # Update node load
                node_resources = available_resources[best_node]
                load_increase = requirements["compute_intensity"] * 0.3
                node_resources["current_load"] = min(node_resources["current_load"] + load_increase, 1.0)

        return assignments

    def _load_balanced_scheduling(self, phase_requirements: dict, available_resources: dict) -> dict[str, list[str]]:
        """Schedule phases using load balancing algorithm."""
        assignments = {node_id: [] for node_id in available_resources.keys()}
        node_loads = {node_id: 0.0 for node_id in available_resources.keys()}

        for phase in phase_requirements.keys():
            requirements = phase_requirements[phase]

            # Find node with lowest load that can handle the phase
            eligible_nodes = []
            for node_id, resources in available_resources.items():
                if (
                    resources["cpu_cores"] >= requirements["cpu_cores"]
                    and resources["memory_gb"] >= requirements["memory_gb"]
                    and resources["storage_gb"] >= requirements["storage_gb"]
                    and (not requirements["gpu_required"] or resources["gpu_available"])
                ):
                    eligible_nodes.append(node_id)

            if eligible_nodes:
                # Choose node with lowest current load
                best_node = min(eligible_nodes, key=lambda n: node_loads[n])
                assignments[best_node].append(phase)
                node_loads[best_node] += requirements["compute_intensity"]

        return assignments

    def _round_robin_scheduling(self, phase_requirements: dict, available_resources: dict) -> dict[str, list[str]]:
        """Schedule phases using round-robin algorithm."""
        assignments = {}
        available_nodes = list(available_resources.keys())

        for i, phase in enumerate(phase_requirements.keys()):
            node_id = available_nodes[i % len(available_nodes)]
            if node_id not in assignments:
                assignments[node_id] = []
            assignments[node_id].append(phase)

        return assignments

    def _calculate_resource_allocation(self, assignments: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
        """Calculate detailed resource allocation for assignments."""
        allocation = {}

        for node_id, phases in assignments.items():
            total_cpu = sum(self._get_phase_requirement(p, "cpu_cores") for p in phases)
            total_memory = sum(self._get_phase_requirement(p, "memory_gb") for p in phases)
            total_storage = sum(self._get_phase_requirement(p, "storage_gb") for p in phases)
            gpu_needed = any(self._get_phase_requirement(p, "gpu_required") for p in phases)

            allocation[node_id] = {
                "phases": phases,
                "cpu_cores_needed": total_cpu,
                "memory_gb_needed": total_memory,
                "storage_gb_needed": total_storage,
                "gpu_needed": gpu_needed,
                "estimated_duration": (
                    max(self._get_phase_requirement(p, "duration_estimate") for p in phases) if phases else 0
                ),
            }

        return allocation

    def _calculate_load_distribution(self, assignments: dict[str, list[str]]) -> dict[str, float]:
        """Calculate load distribution across nodes."""
        load_distribution = {}

        for node_id, phases in assignments.items():
            total_load = sum(self._get_phase_requirement(p, "compute_intensity") for p in phases)
            load_distribution[node_id] = min(total_load, 1.0)

        return load_distribution

    def _estimate_completion_time(self, assignments: dict[str, list[str]]) -> float:
        """Estimate total completion time for the pipeline."""
        max_time = 0

        for node_id, phases in assignments.items():
            node_time = sum(self._get_phase_requirement(p, "duration_estimate") for p in phases)
            max_time = max(max_time, node_time)

        return max_time

    def _get_phase_requirement(self, phase: str, requirement: str) -> Any:
        """Get specific requirement for a phase."""
        requirements = self._calculate_phase_requirements([phase])
        return requirements.get(phase, {}).get(requirement, 0)

    async def execute_distributed_pipeline(self, optimization_plan: dict[str, Any]) -> list[PhaseResult]:
        """Execute the Agent Forge pipeline across distributed fog infrastructure."""
        self.logger.info("Starting distributed Agent Forge pipeline execution")

        assignments = optimization_plan["phase_assignments"]
        results = []

        try:
            # Start tasks on each node
            tasks = {}
            for node_id, phases in assignments.items():
                if phases:  # Only if node has assigned phases
                    task = asyncio.create_task(self._execute_node_phases(node_id, phases))
                    tasks[node_id] = task
                    self.logger.info(f"Started phases {phases} on node {node_id}")

            # Wait for all tasks to complete
            completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Process results
            for i, (node_id, result) in enumerate(zip(tasks.keys(), completed_tasks)):
                if isinstance(result, Exception):
                    self.logger.error(f"Node {node_id} failed: {result}")
                    # Create failure result
                    failure_result = PhaseResult(
                        success=False,
                        model=None,
                        phase_name=f"FogNode_{node_id}",
                        error=str(result),
                        metrics={"node_id": node_id},
                        duration_seconds=0,
                    )
                    results.append(failure_result)
                else:
                    results.extend(result)
                    self.logger.info(f"Node {node_id} completed successfully")

            self.logger.info(
                f"Distributed pipeline execution completed. "
                f"{len([r for r in results if r.success])}/{len(results)} phases successful"
            )

            return results

        except Exception as e:
            self.logger.error(f"Distributed pipeline execution failed: {e}")
            raise

    async def _execute_node_phases(self, node_id: str, phases: list[str]) -> list[PhaseResult]:
        """Execute assigned phases on a specific node."""
        node_results = []

        try:
            # Create node-specific pipeline configuration
            node_config = self._create_node_config(node_id, phases)

            # Create pipeline for this node
            node_pipeline = UnifiedPipeline(node_config)

            # Execute pipeline
            pipeline_result = await node_pipeline.run_pipeline()

            if pipeline_result.success:
                # Extract individual phase results
                if "phase_results" in pipeline_result.artifacts:
                    for phase_data in pipeline_result.artifacts["phase_results"]:
                        phase_result = PhaseResult(
                            success=phase_data["success"],
                            model=pipeline_result.model,  # Share the final model
                            phase_name=phase_data["phase_name"],
                            metrics={
                                **phase_data.get("metrics", {}),
                                "node_id": node_id,
                                "node_type": self._get_node_type(node_id),
                            },
                            error=phase_data.get("error"),
                            duration_seconds=phase_data.get("duration", 0),
                        )
                        node_results.append(phase_result)
                else:
                    # Single result
                    pipeline_result.metrics["node_id"] = node_id
                    pipeline_result.metrics["node_type"] = self._get_node_type(node_id)
                    node_results.append(pipeline_result)
            else:
                # Pipeline failed
                pipeline_result.metrics = {"node_id": node_id, "node_type": self._get_node_type(node_id)}
                node_results.append(pipeline_result)

        except Exception as e:
            self.logger.error(f"Node {node_id} execution failed: {e}")
            failure_result = PhaseResult(
                success=False,
                model=None,
                phase_name=f"Node_{node_id}_Execution",
                error=str(e),
                metrics={"node_id": node_id, "node_type": self._get_node_type(node_id)},
                duration_seconds=0,
            )
            node_results.append(failure_result)

        return node_results

    def _create_node_config(self, node_id: str, phases: list[str]) -> UnifiedConfig:
        """Create pipeline configuration for a specific node."""
        # Copy base configuration
        config_dict = self.base_config.__dict__.copy()

        # Enable only assigned phases
        config_dict.update(
            {
                "enable_evomerge": "evomerge" in phases,
                "enable_quietstar": "quietstar" in phases,
                "enable_initial_compression": "bitnet_compression" in phases,
                "enable_training": "forge_training" in phases,
                "enable_tool_baking": "tool_persona_baking" in phases,
                "enable_adas": "adas" in phases,
                "enable_final_compression": "final_compression" in phases,
            }
        )

        # Adjust configuration based on node type
        node_type = self._get_node_type(node_id)
        if node_type == "edge_device":
            # Edge device optimizations
            config_dict["batch_size"] = min(config_dict["batch_size"], 8)
            config_dict["training_steps"] = min(config_dict["training_steps"], 1000)
        elif node_type == "fog_node":
            # Fog node optimizations
            node_info = next((n for n in self.fog_nodes if n["node_id"] == node_id), {})
            compute_power = node_info.get("compute_power", 1.0)
            if compute_power > 2.0:
                config_dict["batch_size"] = int(config_dict["batch_size"] * min(compute_power, 4))

        # Set output directory for this node
        config_dict["output_dir"] = Path(f"./fog_output/{node_id}")
        config_dict["checkpoint_dir"] = Path(f"./fog_checkpoints/{node_id}")

        return UnifiedConfig(**config_dict)

    def _get_node_type(self, node_id: str) -> str:
        """Get the type of a node (fog_node or edge_device)."""
        if any(n["node_id"] == node_id for n in self.fog_nodes):
            return "fog_node"
        elif any(d["device_id"] == node_id for d in self.edge_devices):
            return "edge_device"
        else:
            return "unknown"


# Integration functions
async def create_fog_compute_pipeline(
    base_config: UnifiedConfig, fog_config: FogComputeConfig | None = None
) -> FogComputeOrchestrator:
    """
    Create a fog compute-enabled Agent Forge pipeline.

    Args:
        base_config: Base Agent Forge configuration
        fog_config: Fog compute configuration

    Returns:
        Configured FogComputeOrchestrator instance
    """
    if fog_config is None:
        fog_config = FogComputeConfig()

    orchestrator = FogComputeOrchestrator(base_config, fog_config)
    return orchestrator


async def run_fog_distributed_agent_forge(
    config_path: str | None = None, fog_config_path: str | None = None, **kwargs
) -> list[PhaseResult]:
    """
    Run Agent Forge pipeline distributed across fog compute infrastructure.

    Args:
        config_path: Path to base Agent Forge configuration
        fog_config_path: Path to fog compute configuration
        **kwargs: Additional configuration overrides

    Returns:
        List of phase results from distributed execution
    """
    # Load base configuration
    if config_path:
        with open(config_path) as f:
            base_config_dict = json.load(f)
        base_config_dict.update(kwargs)
        base_config = UnifiedConfig(**base_config_dict)
    else:
        base_config = UnifiedConfig(**kwargs)

    # Load fog compute configuration
    if fog_config_path:
        with open(fog_config_path) as f:
            fog_config_dict = json.load(f)
        fog_config = FogComputeConfig(**fog_config_dict)
    else:
        fog_config = FogComputeConfig()

    # Create and run fog compute orchestrator
    orchestrator = await create_fog_compute_pipeline(base_config, fog_config)

    # Discover infrastructure
    infrastructure = await orchestrator.discover_fog_infrastructure()

    # Get enabled phases
    phases = []
    if base_config.enable_evomerge:
        phases.append("evomerge")
    if base_config.enable_quietstar:
        phases.append("quietstar")
    if base_config.enable_initial_compression:
        phases.append("bitnet_compression")
    if base_config.enable_training:
        phases.append("forge_training")
    if base_config.enable_tool_baking:
        phases.append("tool_persona_baking")
    if base_config.enable_adas:
        phases.append("adas")
    if base_config.enable_final_compression:
        phases.append("final_compression")

    # Optimize phase distribution
    optimization_plan = await orchestrator.optimize_phase_distribution(phases)

    # Execute distributed pipeline
    results = await orchestrator.execute_distributed_pipeline(optimization_plan)

    return results
