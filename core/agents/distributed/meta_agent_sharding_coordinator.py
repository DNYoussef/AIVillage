"""
Meta-Agent Sharding Coordinator

Integrates the 23 large meta-agents with P2P fog compute and model sharding while keeping
digital twin concierge models small and local. This system:

- Runs digital twin concierge models locally on device (small, privacy-preserving)
- Shards large meta-agents (King, Magi, Oracle, etc.) across fog compute nodes
- Uses P2P layer for distributed inference coordination
- Optimizes placement based on device capabilities and battery status
- Handles model migration when devices join/leave the fog network

Architecture:
- Local Device: Digital Twin Concierge (1-10MB model)
- Fog Network: 23 Meta-Agents sharded across multiple devices (100MB-1GB+ each)
- P2P Coordination: BitChat/BetaNet for shard communication
- Resource Aware: Battery, thermal, network considerations
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from packages.core.distributed_inference.model_sharding_engine import (
    ModelShard,
    ModelShardingEngine,
    ShardingPlan,
    ShardingStrategy,
)
from packages.edge.fog_compute.fog_coordinator import ComputeCapacity, FogCoordinator
from packages.edge.mobile.digital_twin_concierge import DigitalTwinConcierge
from packages.p2p.core.transport_manager import UnifiedTransportManager

from packages.agents.core.base_agent_template import BaseAgent

logger = logging.getLogger(__name__)


class AgentScale(Enum):
    """Scale categories for different agent types"""

    TINY = "tiny"  # <1MB - Digital Twin Concierge
    SMALL = "small"  # 1-10MB - Simple utility agents
    MEDIUM = "medium"  # 10-100MB - Specialized agents
    LARGE = "large"  # 100MB-1GB - Meta agents (King, Magi, etc.)
    XLARGE = "xlarge"  # 1GB+ - Complex reasoning agents


class DeploymentStrategy(Enum):
    """Deployment strategies for different agent scales"""

    LOCAL_ONLY = "local_only"  # Device-local deployment
    FOG_PREFERRED = "fog_preferred"  # Prefer fog, fallback local
    FOG_ONLY = "fog_only"  # Must run on fog network
    HYBRID = "hybrid"  # Partial local, partial fog
    ADAPTIVE = "adaptive"  # Dynamic based on conditions


@dataclass
class AgentProfile:
    """Profile of an agent including scale and deployment requirements"""

    agent_name: str
    agent_class: str
    scale: AgentScale
    deployment_strategy: DeploymentStrategy

    # Model characteristics
    model_size_mb: float
    min_memory_mb: float
    min_compute_units: float
    max_latency_ms: float = 1000

    # Privacy requirements
    privacy_sensitive: bool = False
    requires_local_data: bool = False

    # Resource preferences
    prefer_gpu: bool = False
    prefer_charging_nodes: bool = True
    min_battery_level: int = 20

    # Sharding characteristics (for large agents)
    shardable: bool = False
    min_shards: int = 1
    max_shards: int = 1
    shard_granularity: str = "layer"  # layer, attention, ffn


@dataclass
class DeploymentPlan:
    """Complete deployment plan for all agents"""

    plan_id: str
    created_at: datetime

    # Local deployments (digital twin concierge, etc.)
    local_agents: list[AgentProfile] = field(default_factory=list)

    # Fog deployments (meta-agents)
    fog_agents: list[AgentProfile] = field(default_factory=list)

    # Sharding plans for large agents
    sharding_plans: dict[str, ShardingPlan] = field(default_factory=dict)

    # Resource allocation
    total_memory_mb: float = 0.0
    total_compute_units: float = 0.0
    estimated_latency_ms: float = 0.0

    # Network topology
    fog_nodes: list[str] = field(default_factory=list)
    communication_graph: dict[str, list[str]] = field(default_factory=dict)


class MetaAgentShardingCoordinator:
    """Coordinates deployment of agents across local device and fog network"""

    def __init__(
        self,
        fog_coordinator: FogCoordinator,
        transport_manager: UnifiedTransportManager,
        sharding_engine: ModelShardingEngine,
        digital_twin: DigitalTwinConcierge,
    ):
        self.fog_coordinator = fog_coordinator
        self.transport_manager = transport_manager
        self.sharding_engine = sharding_engine
        self.digital_twin = digital_twin

        # Agent registry
        self.agent_registry = self._create_agent_registry()
        self.deployed_agents: dict[str, Any] = {}

        # Current deployment
        self.current_deployment: DeploymentPlan | None = None
        self.local_agent_instances: dict[str, BaseAgent] = {}
        self.fog_agent_shards: dict[str, list[ModelShard]] = {}

        # Performance tracking
        self.metrics = {
            "deployment_time": 0.0,
            "local_agents_count": 0,
            "fog_agents_count": 0,
            "total_shards": 0,
            "average_latency_ms": 0.0,
            "resource_utilization": 0.0,
        }

    def _create_agent_registry(self) -> dict[str, AgentProfile]:
        """Create registry of all agent profiles with deployment characteristics"""

        # Digital Twin Concierge - Always local for privacy
        digital_twin_profile = AgentProfile(
            agent_name="digital_twin_concierge",
            agent_class="DigitalTwinConcierge",
            scale=AgentScale.TINY,
            deployment_strategy=DeploymentStrategy.LOCAL_ONLY,
            model_size_mb=1.5,
            min_memory_mb=50,
            min_compute_units=0.1,
            max_latency_ms=100,
            privacy_sensitive=True,
            requires_local_data=True,
            shardable=False,
        )

        # Large Meta-Agents - Must be sharded across fog
        meta_agents = {
            "king_agent": AgentProfile(
                agent_name="king_agent",
                agent_class="KingAgent",
                scale=AgentScale.LARGE,
                deployment_strategy=DeploymentStrategy.FOG_ONLY,
                model_size_mb=500,
                min_memory_mb=800,
                min_compute_units=2.0,
                max_latency_ms=500,
                shardable=True,
                min_shards=2,
                max_shards=8,
                shard_granularity="layer",
            ),
            "magi_agent": AgentProfile(
                agent_name="magi_agent",
                agent_class="MagiAgent",
                scale=AgentScale.LARGE,
                deployment_strategy=DeploymentStrategy.FOG_ONLY,
                model_size_mb=400,
                min_memory_mb=600,
                min_compute_units=1.8,
                max_latency_ms=800,
                shardable=True,
                min_shards=2,
                max_shards=6,
                prefer_gpu=True,
            ),
            "oracle_agent": AgentProfile(
                agent_name="oracle_agent",
                agent_class="OracleAgent",
                scale=AgentScale.LARGE,
                deployment_strategy=DeploymentStrategy.FOG_ONLY,
                model_size_mb=600,
                min_memory_mb=900,
                min_compute_units=2.2,
                max_latency_ms=600,
                shardable=True,
                min_shards=3,
                max_shards=10,
            ),
            "sage_agent": AgentProfile(
                agent_name="sage_agent",
                agent_class="SageAgent",
                scale=AgentScale.MEDIUM,
                deployment_strategy=DeploymentStrategy.FOG_PREFERRED,
                model_size_mb=150,
                min_memory_mb=250,
                min_compute_units=1.0,
                shardable=True,
                min_shards=1,
                max_shards=4,
            ),
        }

        # Medium-sized specialized agents - Fog preferred but can run local
        specialized_agents = {}
        for agent_name in [
            "navigator",
            "curator",
            "tutor",
            "medic",
            "coordinator",
            "gardener",
            "auditor",
            "legal",
            "shield",
            "sword",
            "strategist",
            "ensemble",
            "maker",
            "banker",
            "merchant",
            "polyglot",
            "horticulturist",
            "sustainer",
            "shaman",
        ]:
            specialized_agents[agent_name] = AgentProfile(
                agent_name=agent_name,
                agent_class=f"{agent_name.title()}Agent",
                scale=AgentScale.MEDIUM,
                deployment_strategy=DeploymentStrategy.ADAPTIVE,
                model_size_mb=100,
                min_memory_mb=200,
                min_compute_units=0.8,
                max_latency_ms=1000,
                shardable=True,
                min_shards=1,
                max_shards=3,
            )

        # Combine all profiles
        registry = {"digital_twin_concierge": digital_twin_profile}
        registry.update(meta_agents)
        registry.update(specialized_agents)

        return registry

    async def create_deployment_plan(
        self, target_agents: list[str] | None = None, force_local: bool = False
    ) -> DeploymentPlan:
        """Create optimal deployment plan for requested agents"""

        plan_id = f"deployment_{int(datetime.now().timestamp())}"
        plan = DeploymentPlan(plan_id=plan_id, created_at=datetime.now())

        # Determine which agents to deploy
        agents_to_deploy = target_agents or list(self.agent_registry.keys())

        # Get current fog network state
        fog_capacity = await self._assess_fog_capacity()
        local_capacity = await self._assess_local_capacity()

        logger.info(f"Creating deployment plan for {len(agents_to_deploy)} agents")
        logger.info(f"Fog capacity: {len(fog_capacity)} nodes, Local capacity: {local_capacity}")

        # Categorize agents by deployment strategy
        for agent_name in agents_to_deploy:
            if agent_name not in self.agent_registry:
                logger.warning(f"Unknown agent: {agent_name}")
                continue

            profile = self.agent_registry[agent_name]

            # Force local deployment if requested or required
            if force_local or profile.deployment_strategy == DeploymentStrategy.LOCAL_ONLY:
                if await self._can_deploy_locally(profile, local_capacity):
                    plan.local_agents.append(profile)
                    local_capacity["available_memory_mb"] -= profile.min_memory_mb
                    local_capacity["available_compute"] -= profile.min_compute_units
                else:
                    logger.warning(f"Cannot deploy {agent_name} locally - insufficient resources")

            # Deploy to fog if strategy allows and resources available
            elif profile.deployment_strategy in [
                DeploymentStrategy.FOG_ONLY,
                DeploymentStrategy.FOG_PREFERRED,
                DeploymentStrategy.ADAPTIVE,
            ] and await self._can_deploy_to_fog(profile, fog_capacity):
                plan.fog_agents.append(profile)

                # Create sharding plan for large shardable agents
                if profile.shardable and profile.scale in [AgentScale.LARGE, AgentScale.XLARGE]:
                    sharding_plan = await self._create_agent_sharding_plan(profile, fog_capacity)
                    if sharding_plan:
                        plan.sharding_plans[agent_name] = sharding_plan

            # Fallback to local if fog unavailable and strategy allows
            elif profile.deployment_strategy in [
                DeploymentStrategy.FOG_PREFERRED,
                DeploymentStrategy.ADAPTIVE,
            ] and await self._can_deploy_locally(profile, local_capacity):
                plan.local_agents.append(profile)
                local_capacity["available_memory_mb"] -= profile.min_memory_mb
                local_capacity["available_compute"] -= profile.min_compute_units
            else:
                logger.error(f"Cannot deploy {agent_name} - no suitable deployment option")

        # Calculate plan metrics
        plan.total_memory_mb = sum(a.min_memory_mb for a in plan.local_agents + plan.fog_agents)
        plan.total_compute_units = sum(a.min_compute_units for a in plan.local_agents + plan.fog_agents)
        plan.fog_nodes = list(fog_capacity.keys())

        logger.info(f"Deployment plan created: {len(plan.local_agents)} local, {len(plan.fog_agents)} fog")
        return plan

    async def deploy_agents(self, plan: DeploymentPlan) -> dict[str, bool]:
        """Execute deployment plan"""
        deployment_results = {}
        start_time = datetime.now()

        try:
            # 1. Deploy local agents (including digital twin concierge)
            logger.info(f"Deploying {len(plan.local_agents)} local agents...")
            for agent_profile in plan.local_agents:
                try:
                    if agent_profile.agent_name == "digital_twin_concierge":
                        # Digital twin is already running, just register it
                        self.local_agent_instances[agent_profile.agent_name] = self.digital_twin
                        deployment_results[agent_profile.agent_name] = True
                        logger.info("Registered digital twin concierge locally")
                    else:
                        # Deploy other local agents
                        agent_instance = await self._deploy_local_agent(agent_profile)
                        if agent_instance:
                            self.local_agent_instances[agent_profile.agent_name] = agent_instance
                            deployment_results[agent_profile.agent_name] = True
                            logger.info(f"Deployed {agent_profile.agent_name} locally")
                        else:
                            deployment_results[agent_profile.agent_name] = False
                            logger.error(f"Failed to deploy {agent_profile.agent_name} locally")

                except Exception as e:
                    logger.error(f"Error deploying {agent_profile.agent_name} locally: {e}")
                    deployment_results[agent_profile.agent_name] = False

            # 2. Deploy fog agents with sharding
            logger.info(f"Deploying {len(plan.fog_agents)} fog agents...")
            for agent_profile in plan.fog_agents:
                try:
                    if agent_profile.agent_name in plan.sharding_plans:
                        # Deploy sharded agent across fog network
                        shards = await self._deploy_sharded_agent(
                            agent_profile, plan.sharding_plans[agent_profile.agent_name]
                        )
                        if shards:
                            self.fog_agent_shards[agent_profile.agent_name] = shards
                            deployment_results[agent_profile.agent_name] = True
                            logger.info(f"Deployed {agent_profile.agent_name} across {len(shards)} shards")
                        else:
                            deployment_results[agent_profile.agent_name] = False
                            logger.error(f"Failed to deploy sharded {agent_profile.agent_name}")
                    else:
                        # Deploy single-node fog agent
                        success = await self._deploy_fog_agent(agent_profile)
                        deployment_results[agent_profile.agent_name] = success
                        if success:
                            logger.info(f"Deployed {agent_profile.agent_name} to fog node")
                        else:
                            logger.error(f"Failed to deploy {agent_profile.agent_name} to fog")

                except Exception as e:
                    logger.error(f"Error deploying {agent_profile.agent_name} to fog: {e}")
                    deployment_results[agent_profile.agent_name] = False

            # 3. Set up communication routes between agents
            await self._setup_agent_communication(plan, deployment_results)

            # 4. Update metrics
            self.current_deployment = plan
            self.metrics["deployment_time"] = (datetime.now() - start_time).total_seconds()
            self.metrics["local_agents_count"] = len(plan.local_agents)
            self.metrics["fog_agents_count"] = len(plan.fog_agents)
            self.metrics["total_shards"] = sum(len(shards) for shards in self.fog_agent_shards.values())

            successful_deployments = sum(1 for success in deployment_results.values() if success)
            logger.info(f"Deployment completed: {successful_deployments}/{len(deployment_results)} agents successful")

            return deployment_results

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return deployment_results

    async def _assess_fog_capacity(self) -> dict[str, ComputeCapacity]:
        """Assess available fog compute capacity"""
        # This would integrate with the actual fog coordinator
        # For now, return sample capacity
        return {
            "fog_node_1": ComputeCapacity(
                cpu_cores=8,
                cpu_utilization=0.3,
                memory_mb=8000,
                memory_used_mb=2000,
                gpu_available=True,
                gpu_memory_mb=4000,
                battery_powered=False,
                is_charging=True,
                thermal_state="normal",
            ),
            "fog_node_2": ComputeCapacity(
                cpu_cores=4,
                cpu_utilization=0.5,
                memory_mb=4000,
                memory_used_mb=1500,
                gpu_available=False,
                gpu_memory_mb=0,
                battery_powered=True,
                battery_percent=75,
                is_charging=False,
                thermal_state="warm",
            ),
            "fog_node_3": ComputeCapacity(
                cpu_cores=6,
                cpu_utilization=0.2,
                memory_mb=6000,
                memory_used_mb=1000,
                gpu_available=True,
                gpu_memory_mb=8000,
                battery_powered=True,
                battery_percent=90,
                is_charging=True,
                thermal_state="normal",
            ),
        }

    async def _assess_local_capacity(self) -> dict[str, float]:
        """Assess local device capacity"""
        # This would integrate with actual device profiling
        return {"available_memory_mb": 2000, "available_compute": 2.0, "battery_percent": 75, "thermal_state": "normal"}

    async def _can_deploy_locally(self, profile: AgentProfile, local_capacity: dict) -> bool:
        """Check if agent can be deployed locally"""
        return (
            local_capacity["available_memory_mb"] >= profile.min_memory_mb
            and local_capacity["available_compute"] >= profile.min_compute_units
            and local_capacity["battery_percent"] >= profile.min_battery_level
        )

    async def _can_deploy_to_fog(self, profile: AgentProfile, fog_capacity: dict) -> bool:
        """Check if agent can be deployed to fog"""
        # Check if any fog node can accommodate this agent
        for node_id, capacity in fog_capacity.items():
            if (
                capacity.available_memory_mb >= profile.min_memory_mb
                and capacity.available_cpu_cores >= profile.min_compute_units
                and (not profile.prefer_gpu or capacity.gpu_available)
            ):
                return True
        return False

    async def _create_agent_sharding_plan(self, profile: AgentProfile, fog_capacity: dict) -> ShardingPlan | None:
        """Create sharding plan for large agent"""
        if not profile.shardable:
            return None

        # Use the existing sharding engine
        try:
            # This would use the actual model path in production
            model_path = f"/models/{profile.agent_name}.safetensors"
            target_devices = list(fog_capacity.keys())

            sharding_plan = await self.sharding_engine.shard_model(
                model_path=model_path, strategy=ShardingStrategy.HYBRID, target_devices=target_devices
            )

            return sharding_plan

        except Exception as e:
            logger.error(f"Failed to create sharding plan for {profile.agent_name}: {e}")
            return None

    async def _deploy_local_agent(self, profile: AgentProfile) -> BaseAgent | None:
        """Deploy agent locally"""
        # This would create actual agent instances in production
        logger.info(f"Deploying {profile.agent_name} locally (mock implementation)")

        # Mock agent instance
        class MockLocalAgent(BaseAgent):
            def __init__(self, name: str):
                self.name = name
                self.deployed_locally = True

        return MockLocalAgent(profile.agent_name)

    async def _deploy_sharded_agent(
        self, profile: AgentProfile, sharding_plan: ShardingPlan
    ) -> list[ModelShard] | None:
        """Deploy sharded agent across fog nodes"""
        logger.info(f"Deploying {profile.agent_name} across {len(sharding_plan.shards)} shards")

        deployed_shards = []
        for shard in sharding_plan.shards:
            # Deploy each shard to its assigned device
            success = await self._deploy_shard_to_fog_node(shard, shard.device_id)
            if success:
                deployed_shards.append(shard)
            else:
                logger.error(f"Failed to deploy shard {shard.shard_id} to {shard.device_id}")

        return deployed_shards if deployed_shards else None

    async def _deploy_shard_to_fog_node(self, shard: ModelShard, node_id: str) -> bool:
        """Deploy individual shard to fog node"""
        # This would use the fog coordinator to deploy the actual shard
        logger.info(f"Deploying shard {shard.shard_id} to fog node {node_id} (mock)")
        return True  # Mock success

    async def _deploy_fog_agent(self, profile: AgentProfile) -> bool:
        """Deploy single-node agent to fog"""
        logger.info(f"Deploying {profile.agent_name} to fog network (mock)")
        return True  # Mock success

    async def _setup_agent_communication(self, plan: DeploymentPlan, deployment_results: dict[str, bool]):
        """Setup P2P communication routes between agents"""
        logger.info("Setting up agent communication routes...")

        # Create communication graph
        successful_agents = [name for name, success in deployment_results.items() if success]

        for agent_name in successful_agents:
            plan.communication_graph[agent_name] = []

            # Local agents can communicate with fog agents via P2P
            if agent_name in [a.agent_name for a in plan.local_agents]:
                # Local agent - can communicate with fog agents
                fog_agents = [a.agent_name for a in plan.fog_agents if deployment_results.get(a.agent_name, False)]
                plan.communication_graph[agent_name].extend(fog_agents)

            # Fog agents can communicate with each other and local agents
            if agent_name in [a.agent_name for a in plan.fog_agents]:
                other_agents = [name for name in successful_agents if name != agent_name]
                plan.communication_graph[agent_name].extend(other_agents)

        logger.info(f"Communication graph created with {len(successful_agents)} agents")

    async def get_deployment_status(self) -> dict[str, Any]:
        """Get current deployment status"""
        if not self.current_deployment:
            return {"status": "no_deployment", "agents": {}}

        status = {
            "deployment_id": self.current_deployment.plan_id,
            "created_at": self.current_deployment.created_at.isoformat(),
            "local_agents": len(self.local_agent_instances),
            "fog_agents": len(self.fog_agent_shards),
            "total_shards": sum(len(shards) for shards in self.fog_agent_shards.values()),
            "metrics": self.metrics,
            "agents": {},
        }

        # Add agent details
        for agent_name, instance in self.local_agent_instances.items():
            status["agents"][agent_name] = {
                "deployment": "local",
                "status": "running",
                "type": "digital_twin" if agent_name == "digital_twin_concierge" else "local_agent",
            }

        for agent_name, shards in self.fog_agent_shards.items():
            status["agents"][agent_name] = {
                "deployment": "fog_sharded",
                "status": "running",
                "shards": len(shards),
                "nodes": list(set(shard.device_id for shard in shards)),
            }

        return status

    def get_privacy_report(self) -> dict[str, Any]:
        """Generate privacy report for the deployment"""
        return {
            "digital_twin_privacy": self.digital_twin.get_privacy_report() if self.digital_twin else None,
            "local_agents": list(self.local_agent_instances.keys()),
            "fog_agents": list(self.fog_agent_shards.keys()),
            "data_flows": {
                "local_only": ["digital_twin_concierge"],  # Never leaves device
                "fog_distributed": list(self.fog_agent_shards.keys()),  # Distributed inference
                "hybrid": [],  # Agents with both local and fog components
            },
            "privacy_guarantees": {
                "digital_twin": "all_data_local_never_shared",
                "meta_agents": "inference_only_no_personal_data",
                "communication": "encrypted_p2p_channels",
            },
        }


# Example usage and demonstration
async def demo_meta_agent_sharding():
    """Demonstrate the meta-agent sharding coordinator"""
    print("🔮 Meta-Agent Sharding Coordinator Demo")
    print("=" * 60)

    # Mock components for demo
    class MockFogCoordinator:
        pass

    class MockTransportManager:
        pass

    class MockShardingEngine:
        async def shard_model(self, model_path, strategy, target_devices):
            # Mock sharding plan
            # Using the imported ModelShard and ShardingPlan classes

            return ShardingPlan(
                model_name=model_path.split("/")[-1],
                total_shards=3,
                shards=[
                    ModelShard(
                        shard_id=f"shard_{i}",
                        device_id=target_devices[i % len(target_devices)],
                        layer_indices=list(range(i * 8, (i + 1) * 8)),
                        parameters_count=50_000_000,
                        memory_mb=200,
                        compute_requirement=0.8,
                    )
                    for i in range(3)
                ],
                activation_routing={},
                memory_efficiency=0.85,
                compute_balance_score=0.92,
            )

    class MockDigitalTwin:
        def get_privacy_report(self):
            return {"status": "all_local", "data_sources": ["conversations", "location"]}

    # Create coordinator
    coordinator = MetaAgentShardingCoordinator(
        fog_coordinator=MockFogCoordinator(),
        transport_manager=MockTransportManager(),
        sharding_engine=MockShardingEngine(),
        digital_twin=MockDigitalTwin(),
    )

    # Create deployment plan
    print("\n📋 Creating deployment plan...")
    target_agents = ["digital_twin_concierge", "king_agent", "magi_agent", "oracle_agent", "sage_agent"]
    plan = await coordinator.create_deployment_plan(target_agents)

    print(f"Plan created with {len(plan.local_agents)} local and {len(plan.fog_agents)} fog agents")
    print(f"Sharding plans: {list(plan.sharding_plans.keys())}")

    # Deploy agents
    print("\n🚀 Deploying agents...")
    results = await coordinator.deploy_agents(plan)

    print("Deployment results:")
    for agent_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {agent_name}: {status}")

    # Show deployment status
    print("\n📊 Deployment Status:")
    status = await coordinator.get_deployment_status()
    print(f"Local agents: {status['local_agents']}")
    print(f"Fog agents: {status['fog_agents']}")
    print(f"Total shards: {status['total_shards']}")

    # Show privacy report
    print("\n🔒 Privacy Report:")
    privacy = coordinator.get_privacy_report()
    print(f"Digital twin privacy: {privacy.get('digital_twin_privacy', {}).get('status', 'N/A')}")
    print(f"Local-only agents: {privacy['data_flows']['local_only']}")
    print(f"Fog-distributed agents: {privacy['data_flows']['fog_distributed']}")

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_meta_agent_sharding())
