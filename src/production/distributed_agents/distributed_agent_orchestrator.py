"""Distributed Agent Orchestrator.

Orchestrates the deployment of 18 AIVillage agents across distributed devices,
building on Sprint 6's P2P infrastructure and resource management.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.production.communications.p2p.p2p_node import P2PNode

from ...core.resources.resource_monitor import ResourceMonitor
from ..distributed_inference.model_sharding_engine import DeviceProfile, ModelShardingEngine

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the AIVillage system."""

    KING = "king"  # Coordinator agent
    SAGE = "sage"  # Knowledge management
    MAGI = "magi"  # Specialized computation
    AUDITOR = "auditor"  # Quality assurance
    CURATOR = "curator"  # Content management
    ENSEMBLE = "ensemble"  # Model ensemble
    GARDENER = "gardener"  # System maintenance
    LEGAL = "legal"  # Compliance and legal
    MAKER = "maker"  # Content creation
    SHAMAN = "shaman"  # Intuitive reasoning
    MEDIC = "medic"  # System health
    NAVIGATOR = "navigator"  # Path finding and routing
    ORACLE = "oracle"  # Prediction and forecasting
    POLYGLOT = "polyglot"  # Multi-language support
    STRATEGIST = "strategist"  # Strategic planning
    SUSTAINER = "sustainer"  # Resource optimization
    SWORD_SHIELD = "sword_shield"  # Security and defense
    TUTOR = "tutor"  # Educational support


class AgentPriority(Enum):
    """Agent deployment priorities."""

    CRITICAL = "critical"  # Must always be available (King)
    HIGH = "high"  # Important for core functionality (Sage, Magi)
    MEDIUM = "medium"  # Standard agents
    LOW = "low"  # Optional/specialized agents


@dataclass
class AgentSpec:
    """Specification for an agent deployment."""

    agent_type: AgentType
    priority: AgentPriority
    memory_requirement_mb: float
    compute_requirement: float
    specialization: str
    can_migrate: bool = True
    requires_gpu: bool = False
    redundancy_factor: int = 1  # How many instances needed
    dependencies: list[AgentType] = field(default_factory=list)
    preferred_devices: list[str] = field(default_factory=list)
    resource_constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInstance:
    """Running agent instance."""

    instance_id: str
    agent_spec: AgentSpec
    device_id: str
    status: str = "starting"  # starting, running, migrating, stopped, failed
    health_score: float = 1.0
    resource_usage: dict[str, float] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    startup_time: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    migration_count: int = 0


@dataclass
class AgentDeploymentPlan:
    """Plan for deploying agents across devices."""

    plan_id: str
    agent_instances: list[AgentInstance]
    device_assignments: dict[str, list[str]]  # device_id -> instance_ids
    deployment_strategy: str
    estimated_resource_usage: dict[str, dict[str, float]]  # device_id -> resources
    redundancy_coverage: dict[AgentType, int]
    created_at: float = field(default_factory=time.time)


class DistributedAgentOrchestrator:
    """Orchestrates distributed agent deployment across P2P network."""

    def __init__(
        self,
        p2p_node: P2PNode,
        resource_monitor: ResourceMonitor,
        sharding_engine: ModelShardingEngine | None = None,
    ) -> None:
        self.p2p_node = p2p_node
        self.resource_monitor = resource_monitor
        self.sharding_engine = sharding_engine

        # Agent state
        self.current_deployment_plan: AgentDeploymentPlan | None = None
        self.active_agents: dict[str, AgentInstance] = {}  # instance_id -> instance
        self.agent_registry: dict[AgentType, list[str]] = {}  # agent_type -> instance_ids

        # Device assignments
        self.device_agent_assignments: dict[str, list[str]] = {}  # device_id -> instance_ids

        # Performance tracking
        self.deployment_stats = {
            "agents_deployed": 0,
            "agents_migrated": 0,
            "deployment_failures": 0,
            "avg_deployment_time": 0.0,
            "network_coverage": 0.0,
            "redundancy_coverage": 0.0,
        }

        # Agent specifications
        self.agent_specs = self._create_agent_specifications()

        # Event handlers
        self._register_p2p_handlers()

        logger.info("DistributedAgentOrchestrator initialized")

    def _create_agent_specifications(self) -> dict[AgentType, AgentSpec]:
        """Create specifications for all 18 agents."""
        specs = {
            # Critical agents
            AgentType.KING: AgentSpec(
                agent_type=AgentType.KING,
                priority=AgentPriority.CRITICAL,
                memory_requirement_mb=512.0,
                compute_requirement=3.0,
                specialization="coordination",
                can_migrate=False,  # King should be stable
                redundancy_factor=1,
                resource_constraints={"min_reliability": 0.9, "min_battery": 50},
            ),
            # High priority agents
            AgentType.SAGE: AgentSpec(
                agent_type=AgentType.SAGE,
                priority=AgentPriority.HIGH,
                memory_requirement_mb=1024.0,
                compute_requirement=4.0,
                specialization="knowledge_management",
                redundancy_factor=2,  # Distribute for redundancy
                dependencies=[AgentType.KING],
            ),
            AgentType.MAGI: AgentSpec(
                agent_type=AgentType.MAGI,
                priority=AgentPriority.HIGH,
                memory_requirement_mb=2048.0,  # Larger for model inference
                compute_requirement=6.0,
                specialization="advanced_computation",
                requires_gpu=True,
                redundancy_factor=1,
                dependencies=[AgentType.KING],
                resource_constraints={"min_compute_score": 5.0},
            ),
            # Medium priority agents
            AgentType.AUDITOR: AgentSpec(
                agent_type=AgentType.AUDITOR,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=256.0,
                compute_requirement=1.5,
                specialization="quality_assurance",
                dependencies=[AgentType.KING],
            ),
            AgentType.CURATOR: AgentSpec(
                agent_type=AgentType.CURATOR,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=512.0,
                compute_requirement=2.0,
                specialization="content_management",
                dependencies=[AgentType.SAGE],
            ),
            AgentType.ENSEMBLE: AgentSpec(
                agent_type=AgentType.ENSEMBLE,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=1536.0,
                compute_requirement=4.5,
                specialization="model_ensemble",
                requires_gpu=False,
                dependencies=[AgentType.MAGI],
            ),
            AgentType.GARDENER: AgentSpec(
                agent_type=AgentType.GARDENER,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=128.0,
                compute_requirement=1.0,
                specialization="system_maintenance",
            ),
            AgentType.LEGAL: AgentSpec(
                agent_type=AgentType.LEGAL,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=256.0,
                compute_requirement=1.5,
                specialization="compliance",
                dependencies=[AgentType.AUDITOR],
            ),
            AgentType.MAKER: AgentSpec(
                agent_type=AgentType.MAKER,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=768.0,
                compute_requirement=3.0,
                specialization="content_creation",
                dependencies=[AgentType.SAGE, AgentType.CURATOR],
            ),
            AgentType.SHAMAN: AgentSpec(
                agent_type=AgentType.SHAMAN,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=512.0,
                compute_requirement=2.0,
                specialization="intuitive_reasoning",
            ),
            AgentType.MEDIC: AgentSpec(
                agent_type=AgentType.MEDIC,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=256.0,
                compute_requirement=1.5,
                specialization="system_health",
                redundancy_factor=2,  # Health monitoring needs redundancy
            ),
            AgentType.NAVIGATOR: AgentSpec(
                agent_type=AgentType.NAVIGATOR,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=384.0,
                compute_requirement=2.0,
                specialization="routing_pathfinding",
                dependencies=[AgentType.KING],
            ),
            AgentType.ORACLE: AgentSpec(
                agent_type=AgentType.ORACLE,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=512.0,
                compute_requirement=2.5,
                specialization="prediction_forecasting",
                dependencies=[AgentType.SAGE],
            ),
            AgentType.POLYGLOT: AgentSpec(
                agent_type=AgentType.POLYGLOT,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=640.0,
                compute_requirement=2.5,
                specialization="multilingual_processing",
            ),
            AgentType.STRATEGIST: AgentSpec(
                agent_type=AgentType.STRATEGIST,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=384.0,
                compute_requirement=2.0,
                specialization="strategic_planning",
                dependencies=[AgentType.KING, AgentType.ORACLE],
            ),
            AgentType.SUSTAINER: AgentSpec(
                agent_type=AgentType.SUSTAINER,
                priority=AgentPriority.MEDIUM,
                memory_requirement_mb=256.0,
                compute_requirement=1.5,
                specialization="resource_optimization",
                dependencies=[AgentType.MEDIC],
            ),
            AgentType.SWORD_SHIELD: AgentSpec(
                agent_type=AgentType.SWORD_SHIELD,
                priority=AgentPriority.HIGH,  # Security is high priority
                memory_requirement_mb=384.0,
                compute_requirement=2.5,
                specialization="security_defense",
                can_migrate=False,  # Security agents should be stable
                redundancy_factor=2,
                resource_constraints={"min_reliability": 0.8},
            ),
            AgentType.TUTOR: AgentSpec(
                agent_type=AgentType.TUTOR,
                priority=AgentPriority.LOW,
                memory_requirement_mb=512.0,
                compute_requirement=2.0,
                specialization="educational_support",
                dependencies=[AgentType.SAGE, AgentType.POLYGLOT],
            ),
        }

        return specs

    def _register_p2p_handlers(self) -> None:
        """Register P2P event handlers for agent coordination."""
        # These would be registered with the P2P node's message handling system

    async def deploy_agent_constellation(
        self,
        target_devices: list[str] | None = None,
        deployment_strategy: str = "optimal",
    ) -> AgentDeploymentPlan:
        """Deploy all agents optimally across the network."""
        logger.info("Starting agent constellation deployment")
        start_time = time.time()

        try:
            # Get available devices
            device_profiles = await self._get_available_devices(target_devices)

            if not device_profiles:
                msg = "No suitable devices available for agent deployment"
                raise ValueError(msg)

            # Create deployment plan
            deployment_plan = await self._create_deployment_plan(device_profiles, deployment_strategy)

            # Execute deployment
            success = await self._execute_deployment_plan(deployment_plan)

            if success:
                self.current_deployment_plan = deployment_plan
                await self._notify_deployment_complete(deployment_plan)

                # Update statistics
                duration = time.time() - start_time
                self.deployment_stats["avg_deployment_time"] = (
                    self.deployment_stats["avg_deployment_time"] + duration
                ) / 2
                self.deployment_stats["agents_deployed"] = len(deployment_plan.agent_instances)

                logger.info(f"Agent constellation deployed successfully in {duration:.2f}s")
                return deployment_plan
            msg = "Deployment execution failed"
            raise RuntimeError(msg)

        except Exception as e:
            logger.exception(f"Agent constellation deployment failed: {e}")
            self.deployment_stats["deployment_failures"] += 1
            raise

    async def _get_available_devices(self, target_devices: list[str] | None = None) -> list[DeviceProfile]:
        """Get available devices for agent deployment."""
        # Use sharding engine's device discovery if available
        if self.sharding_engine:
            device_profiles = await self.sharding_engine._get_device_profiles(target_devices)
        else:
            # Fallback to P2P peer discovery
            device_profiles = []

            # Get suitable peers
            suitable_peers = self.p2p_node.get_suitable_evolution_peers(min_count=1)

            # Include local device if suitable
            if self.p2p_node.local_capabilities and self.p2p_node.local_capabilities.is_suitable_for_evolution():
                suitable_peers.insert(0, self.p2p_node.local_capabilities)

            # Convert to DeviceProfile objects
            for peer in suitable_peers:
                if target_devices and peer.device_id not in target_devices:
                    continue

                device_profile = DeviceProfile(
                    device_id=peer.device_id,
                    capabilities=peer,
                    available_memory_mb=peer.ram_mb * 0.7,  # Conservative estimate
                    compute_score=(peer.cpu_cores * 2 + peer.ram_mb / 1024) / 10,
                    network_latency_ms=peer.latency_ms,
                    battery_level=peer.battery_percent,
                    reliability_score=peer.trust_score,
                )
                device_profiles.append(device_profile)

        # Filter devices suitable for agent deployment
        suitable_devices = []
        for device in device_profiles:
            if self._is_device_suitable_for_agents(device):
                suitable_devices.append(device)

        logger.info(f"Found {len(suitable_devices)} suitable devices for agent deployment")
        return suitable_devices

    def _is_device_suitable_for_agents(self, device: DeviceProfile) -> bool:
        """Check if device is suitable for running agents."""
        # Minimum requirements
        if device.available_memory_mb < 256:  # At least 256MB available
            return False

        if device.compute_score < 1.0:  # Minimum compute capability
            return False

        if device.reliability_score < 0.6:  # Minimum reliability
            return False

        # Battery constraint for mobile devices
        return not (device.battery_level and device.battery_level < 20)

    async def _create_deployment_plan(
        self, device_profiles: list[DeviceProfile], strategy: str = "optimal"
    ) -> AgentDeploymentPlan:
        """Create deployment plan for agents."""
        logger.info(f"Creating deployment plan with strategy: {strategy}")

        if strategy == "optimal":
            return await self._create_optimal_deployment_plan(device_profiles)
        if strategy == "priority_based":
            return await self._create_priority_based_plan(device_profiles)
        if strategy == "balanced":
            return await self._create_balanced_deployment_plan(device_profiles)
        msg = f"Unknown deployment strategy: {strategy}"
        raise ValueError(msg)

    async def _create_optimal_deployment_plan(self, device_profiles: list[DeviceProfile]) -> AgentDeploymentPlan:
        """Create optimal deployment plan considering all constraints."""
        agent_instances = []
        device_assignments = {device.device_id: [] for device in device_profiles}
        device_resource_usage = {device.device_id: {"memory_mb": 0.0, "compute": 0.0} for device in device_profiles}

        # Sort devices by capability (best first)
        sorted_devices = sorted(
            device_profiles,
            key=lambda d: (d.available_memory_mb * 0.4 + d.compute_score * 0.3 + d.reliability_score * 0.3),
            reverse=True,
        )

        # Sort agents by priority and requirements
        agent_types_to_deploy = []
        for agent_type, spec in self.agent_specs.items():
            for _ in range(spec.redundancy_factor):
                agent_types_to_deploy.append(agent_type)

        # Sort by priority (critical first)
        priority_order = {
            AgentPriority.CRITICAL: 0,
            AgentPriority.HIGH: 1,
            AgentPriority.MEDIUM: 2,
            AgentPriority.LOW: 3,
        }

        agent_types_to_deploy.sort(
            key=lambda agent_type: (
                priority_order[self.agent_specs[agent_type].priority],
                -self.agent_specs[agent_type].memory_requirement_mb,
            )
        )

        # Deploy agents using best-fit algorithm
        for agent_type in agent_types_to_deploy:
            spec = self.agent_specs[agent_type]

            # Find best device for this agent
            best_device = None
            best_score = -1.0

            for device in sorted_devices:
                # Check if device can accommodate this agent
                if not self._can_device_host_agent(device, spec, device_resource_usage[device.device_id]):
                    continue

                # Calculate suitability score
                score = self._calculate_agent_device_suitability(device, spec)

                if score > best_score:
                    best_score = score
                    best_device = device

            if best_device is None:
                logger.warning(f"Could not find suitable device for agent {agent_type.value}")
                continue

            # Create agent instance
            instance = AgentInstance(
                instance_id=f"{agent_type.value}_{uuid.uuid4().hex[:8]}",
                agent_spec=spec,
                device_id=best_device.device_id,
                status="planned",
            )

            agent_instances.append(instance)
            device_assignments[best_device.device_id].append(instance.instance_id)

            # Update resource usage
            device_resource_usage[best_device.device_id]["memory_mb"] += spec.memory_requirement_mb
            device_resource_usage[best_device.device_id]["compute"] += spec.compute_requirement

        # Calculate redundancy coverage
        redundancy_coverage = {}
        for agent_type, spec in self.agent_specs.items():
            deployed_count = sum(1 for instance in agent_instances if instance.agent_spec.agent_type == agent_type)
            redundancy_coverage[agent_type] = deployed_count

        plan = AgentDeploymentPlan(
            plan_id=str(uuid.uuid4()),
            agent_instances=agent_instances,
            device_assignments=device_assignments,
            deployment_strategy="optimal",
            estimated_resource_usage=device_resource_usage,
            redundancy_coverage=redundancy_coverage,
        )

        logger.info(
            f"Created deployment plan with {len(agent_instances)} agent instances across {len([d for d in device_assignments.values() if d])} devices"
        )
        return plan

    async def _create_priority_based_plan(self, device_profiles: list[DeviceProfile]) -> AgentDeploymentPlan:
        """Create deployment plan prioritizing critical agents first."""
        # Simplified implementation - similar to optimal but stricter priority ordering
        return await self._create_optimal_deployment_plan(device_profiles)

    async def _create_balanced_deployment_plan(self, device_profiles: list[DeviceProfile]) -> AgentDeploymentPlan:
        """Create balanced deployment plan distributing load evenly."""
        # Simplified implementation - could implement round-robin assignment
        return await self._create_optimal_deployment_plan(device_profiles)

    def _can_device_host_agent(
        self,
        device: DeviceProfile,
        agent_spec: AgentSpec,
        current_usage: dict[str, float],
    ) -> bool:
        """Check if device can host the agent given current usage."""
        # Memory constraint
        if current_usage["memory_mb"] + agent_spec.memory_requirement_mb > device.available_memory_mb:
            return False

        # Compute constraint (simplified)
        if current_usage["compute"] + agent_spec.compute_requirement > device.compute_score:
            return False

        # GPU requirement
        if agent_spec.requires_gpu:
            # For now, assume high-end devices have GPU access
            if device.compute_score < 8.0:
                return False

        # Resource constraints
        for constraint, value in agent_spec.resource_constraints.items():
            if (constraint == "min_reliability" and device.reliability_score < value) or (
                constraint == "min_battery" and device.battery_level and device.battery_level < value
            ):
                return False
            if constraint == "min_compute_score" and device.compute_score < value:
                return False

        return True

    def _calculate_agent_device_suitability(self, device: DeviceProfile, agent_spec: AgentSpec) -> float:
        """Calculate how suitable a device is for hosting an agent."""
        score = 0.0

        # Memory efficiency (prefer devices with ample memory)
        memory_ratio = device.available_memory_mb / agent_spec.memory_requirement_mb
        memory_score = min(1.0, memory_ratio / 4.0)  # Optimal at 4x requirement
        score += memory_score * 0.3

        # Compute efficiency
        compute_ratio = device.compute_score / agent_spec.compute_requirement
        compute_score = min(1.0, compute_ratio / 2.0)  # Optimal at 2x requirement
        score += compute_score * 0.3

        # Reliability
        score += device.reliability_score * 0.2

        # Network latency (lower is better)
        latency_score = max(0.0, 1.0 - device.network_latency_ms / 100.0)
        score += latency_score * 0.1

        # Battery level (for mobile devices)
        if device.battery_level:
            battery_score = device.battery_level / 100.0
            score += battery_score * 0.1
        else:
            score += 0.1  # Assume non-mobile device (always powered)

        return score

    async def _execute_deployment_plan(self, plan: AgentDeploymentPlan) -> bool:
        """Execute the deployment plan."""
        logger.info(f"Executing deployment plan {plan.plan_id}")

        deployment_results = []

        # Deploy agents in dependency order
        deployed_agents = set()
        remaining_agents = plan.agent_instances.copy()
        max_iterations = len(remaining_agents) + 5  # Prevent infinite loops
        iteration = 0

        while remaining_agents and iteration < max_iterations:
            iteration += 1
            agents_deployed_this_round = []

            for instance in remaining_agents.copy():
                # Check if dependencies are satisfied
                dependencies_met = True
                for dep_type in instance.agent_spec.dependencies:
                    if not any(deployed_agent.agent_spec.agent_type == dep_type for deployed_agent in deployed_agents):
                        dependencies_met = False
                        break

                if dependencies_met:
                    # Deploy this agent
                    success = await self._deploy_agent_instance(instance)
                    if success:
                        deployed_agents.add(instance)
                        agents_deployed_this_round.append(instance)
                        self.active_agents[instance.instance_id] = instance

                        # Update registry
                        agent_type = instance.agent_spec.agent_type
                        if agent_type not in self.agent_registry:
                            self.agent_registry[agent_type] = []
                        self.agent_registry[agent_type].append(instance.instance_id)

                        # Update device assignments
                        device_id = instance.device_id
                        if device_id not in self.device_agent_assignments:
                            self.device_agent_assignments[device_id] = []
                        self.device_agent_assignments[device_id].append(instance.instance_id)

                        deployment_results.append(True)
                    else:
                        deployment_results.append(False)
                        logger.error(f"Failed to deploy agent {instance.instance_id}")

            # Remove successfully deployed agents
            for instance in agents_deployed_this_round:
                remaining_agents.remove(instance)

        if remaining_agents:
            logger.warning(f"Could not deploy {len(remaining_agents)} agents due to dependency issues")

        success_rate = sum(deployment_results) / len(deployment_results) if deployment_results else 0.0
        logger.info(f"Deployment completed: {success_rate:.1%} success rate")

        return success_rate > 0.8  # Consider successful if >80% agents deployed

    async def _deploy_agent_instance(self, instance: AgentInstance) -> bool:
        """Deploy a single agent instance."""
        logger.debug(f"Deploying agent {instance.instance_id} to device {instance.device_id}")

        try:
            instance.status = "starting"

            # Create deployment message
            deployment_message = {
                "type": "DEPLOY_AGENT",
                "instance_id": instance.instance_id,
                "agent_type": instance.agent_spec.agent_type.value,
                "memory_requirement_mb": instance.agent_spec.memory_requirement_mb,
                "compute_requirement": instance.agent_spec.compute_requirement,
                "specialization": instance.agent_spec.specialization,
                "configuration": instance.agent_spec.metadata,
            }

            if instance.device_id == self.p2p_node.node_id:
                # Local deployment
                success = await self._deploy_agent_locally(instance)
            else:
                # Remote deployment
                success = await self.p2p_node.send_to_peer(instance.device_id, deployment_message)

            if success:
                instance.status = "running"
                instance.last_heartbeat = time.time()
                logger.debug(f"Agent {instance.instance_id} deployed successfully")
                return True
            instance.status = "failed"
            return False

        except Exception as e:
            logger.exception(f"Agent deployment failed: {e}")
            instance.status = "failed"
            return False

    async def _deploy_agent_locally(self, instance: AgentInstance) -> bool:
        """Deploy agent on local device."""
        # This would integrate with the actual agent system
        # For now, simulate successful deployment
        logger.debug(f"Locally deploying agent {instance.agent_spec.agent_type.value}")

        # Simulate startup time
        await asyncio.sleep(0.1)

        # Update resource usage tracking
        instance.resource_usage = {
            "memory_mb": instance.agent_spec.memory_requirement_mb,
            "compute": instance.agent_spec.compute_requirement,
            "startup_time": time.time() - instance.startup_time,
        }

        return True

    async def _notify_deployment_complete(self, plan: AgentDeploymentPlan) -> None:
        """Notify network about completed deployment."""
        deployment_summary = {
            "type": "AGENT_DEPLOYMENT_COMPLETE",
            "plan_id": plan.plan_id,
            "agents_deployed": len(plan.agent_instances),
            "devices_used": len([d for d in plan.device_assignments.values() if d]),
            "redundancy_coverage": {agent_type.value: count for agent_type, count in plan.redundancy_coverage.items()},
            "timestamp": time.time(),
        }

        await self.p2p_node.broadcast_to_peers("AGENT_DEPLOYMENT_COMPLETE", deployment_summary)

    async def enable_cross_device_collaboration(self) -> None:
        """Enable agents to work together across device boundaries."""
        logger.info("Enabling cross-device agent collaboration")

        # Create collaboration message routing
        collaboration_config = {
            "type": "ENABLE_AGENT_COLLABORATION",
            "agent_registry": {
                agent_type.value: instance_ids for agent_type, instance_ids in self.agent_registry.items()
            },
            "device_assignments": self.device_agent_assignments,
            "routing_protocol": "p2p_direct",
            "timestamp": time.time(),
        }

        # Broadcast to all devices
        await self.p2p_node.broadcast_to_peers("ENABLE_AGENT_COLLABORATION", collaboration_config)

        # Start collaboration monitoring
        asyncio.create_task(self._monitor_agent_collaboration())

    async def _monitor_agent_collaboration(self) -> None:
        """Monitor agent collaboration across devices."""
        while self.current_deployment_plan:
            try:
                # Check agent health across devices
                for instance_id, instance in self.active_agents.items():
                    if time.time() - instance.last_heartbeat > 60.0:  # 1 minute timeout
                        logger.warning(f"Agent {instance_id} heartbeat timeout")
                        instance.health_score *= 0.9  # Reduce health score

                        if instance.health_score < 0.5:
                            logger.error(f"Agent {instance_id} marked as unhealthy")
                            await self._handle_unhealthy_agent(instance)

                await asyncio.sleep(30.0)  # Check every 30 seconds

            except Exception as e:
                logger.exception(f"Error in collaboration monitoring: {e}")
                await asyncio.sleep(60.0)

    async def _handle_unhealthy_agent(self, instance: AgentInstance) -> None:
        """Handle unhealthy agent instance."""
        logger.info(f"Handling unhealthy agent {instance.instance_id}")

        if instance.agent_spec.can_migrate:
            # Try to migrate agent
            # This would integrate with the migration manager
            logger.info(f"Triggering migration for unhealthy agent {instance.instance_id}")
        else:
            # Try to restart agent on same device
            logger.info(f"Attempting to restart agent {instance.instance_id}")
            instance.status = "restarting"
            success = await self._deploy_agent_instance(instance)
            if not success:
                instance.status = "failed"

    def get_deployment_status(self) -> dict[str, Any]:
        """Get current deployment status."""
        if not self.current_deployment_plan:
            return {
                "deployed": False,
                "plan_id": None,
                "agents_deployed": 0,
                "devices_used": 0,
            }

        # Calculate status metrics
        total_agents = len(self.current_deployment_plan.agent_instances)
        running_agents = sum(1 for agent in self.active_agents.values() if agent.status == "running")
        failed_agents = sum(1 for agent in self.active_agents.values() if agent.status == "failed")

        # Device coverage
        devices_with_agents = len([d for d in self.device_agent_assignments.values() if d])

        # Redundancy coverage
        redundancy_status = {}
        for (
            agent_type,
            target_count,
        ) in self.current_deployment_plan.redundancy_coverage.items():
            actual_count = len(self.agent_registry.get(agent_type, []))
            redundancy_status[agent_type.value] = {
                "target": target_count,
                "actual": actual_count,
                "coverage": actual_count / target_count if target_count > 0 else 0.0,
            }

        return {
            "deployed": True,
            "plan_id": self.current_deployment_plan.plan_id,
            "agents_deployed": total_agents,
            "agents_running": running_agents,
            "agents_failed": failed_agents,
            "devices_used": devices_with_agents,
            "deployment_health": (running_agents / total_agents if total_agents > 0 else 0.0),
            "redundancy_status": redundancy_status,
            "device_assignments": {
                device_id: len(instance_ids)
                for device_id, instance_ids in self.device_agent_assignments.items()
                if instance_ids
            },
            "statistics": self.deployment_stats.copy(),
        }

    async def shutdown_agent_constellation(self) -> None:
        """Gracefully shutdown all deployed agents."""
        logger.info("Shutting down agent constellation")

        if not self.current_deployment_plan:
            logger.info("No active deployment to shutdown")
            return

        # Notify all devices
        shutdown_message = {
            "type": "SHUTDOWN_AGENTS",
            "plan_id": self.current_deployment_plan.plan_id,
            "timestamp": time.time(),
        }

        await self.p2p_node.broadcast_to_peers("SHUTDOWN_AGENTS", shutdown_message)

        # Update local state
        for instance in self.active_agents.values():
            instance.status = "stopped"

        # Clear state
        self.current_deployment_plan = None
        self.active_agents.clear()
        self.agent_registry.clear()
        self.device_agent_assignments.clear()

        logger.info("Agent constellation shutdown completed")

    def get_agent_locations(self) -> dict[str, list[dict[str, Any]]]:
        """Get current locations of all agents."""
        locations = {}

        for agent_type, instance_ids in self.agent_registry.items():
            agent_locations = []

            for instance_id in instance_ids:
                if instance_id in self.active_agents:
                    instance = self.active_agents[instance_id]
                    agent_locations.append(
                        {
                            "instance_id": instance_id,
                            "device_id": instance.device_id,
                            "status": instance.status,
                            "health_score": instance.health_score,
                            "last_heartbeat": instance.last_heartbeat,
                        }
                    )

            locations[agent_type.value] = agent_locations

        return locations
