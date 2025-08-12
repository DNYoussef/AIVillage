"""Infrastructure-Aware Evolution System for Sprint 6."""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any

# Import P2P and resource management components
from ...core.p2p import P2PNode
from ...core.resources import (
    AdaptiveLoader,
    ConstraintManager,
    DeviceProfiler,
    ResourceMonitor,
)

from .base import EvolvableAgent

# Import existing evolution components
from .dual_evolution_system import DualEvolutionSystem

logger = logging.getLogger(__name__)


class EvolutionMode(Enum):
    """Evolution execution modes."""

    LOCAL_ONLY = "local_only"  # Run evolution locally only
    P2P_COORDINATED = "p2p_coordinated"  # Coordinate with P2P peers
    DISTRIBUTED = "distributed"  # Distribute evolution across peers
    HYBRID = "hybrid"  # Mix of local and distributed


@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure-aware evolution."""

    # P2P settings
    enable_p2p: bool = True
    p2p_port: int = 9000
    max_p2p_connections: int = 10

    # Resource management
    enable_resource_monitoring: bool = True
    monitoring_interval: float = 5.0
    resource_safety_margin: float = 0.8

    # Evolution settings
    default_evolution_mode: EvolutionMode = EvolutionMode.LOCAL_ONLY
    enable_adaptive_loading: bool = True
    enable_resource_constraints: bool = True

    # Coordination settings
    consensus_threshold: float = 0.8  # 80% agreement for distributed decisions
    peer_timeout_seconds: float = 30.0
    max_distributed_agents: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_p2p": self.enable_p2p,
            "p2p_port": self.p2p_port,
            "max_p2p_connections": self.max_p2p_connections,
            "enable_resource_monitoring": self.enable_resource_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "resource_safety_margin": self.resource_safety_margin,
            "default_evolution_mode": self.default_evolution_mode.value,
            "enable_adaptive_loading": self.enable_adaptive_loading,
            "enable_resource_constraints": self.enable_resource_constraints,
            "consensus_threshold": self.consensus_threshold,
            "peer_timeout_seconds": self.peer_timeout_seconds,
            "max_distributed_agents": self.max_distributed_agents,
        }


@dataclass
class EvolutionCoordinationPlan:
    """Plan for coordinating evolution across infrastructure."""

    evolution_id: str
    agent_id: str
    evolution_type: str
    mode: EvolutionMode
    local_resources_allocated: dict[str, Any]
    peer_assignments: dict[str, dict[str, Any]] = field(default_factory=dict)
    estimated_duration_minutes: float = 60.0
    quality_target: float = 0.8
    fallback_mode: EvolutionMode | None = None


class InfrastructureAwareEvolution:
    """Evolution system that leverages P2P network and resource management."""

    def __init__(self, config: InfrastructureConfig | None = None) -> None:
        self.config = config or InfrastructureConfig()

        # Core components
        self.device_profiler: DeviceProfiler | None = None
        self.resource_monitor: ResourceMonitor | None = None
        self.constraint_manager: ConstraintManager | None = None
        self.adaptive_loader: AdaptiveLoader | None = None
        self.p2p_node: P2PNode | None = None

        # Evolution system
        self.dual_evolution: DualEvolutionSystem | None = None

        # Infrastructure state
        self.system_initialized = False
        self.infrastructure_status = "not_initialized"

        # Coordination state
        self.active_coordinations: dict[str, EvolutionCoordinationPlan] = {}
        self.peer_evolution_status: dict[str, dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "evolutions_started": 0,
            "evolutions_completed": 0,
            "evolutions_failed": 0,
            "p2p_coordinations": 0,
            "distributed_evolutions": 0,
            "resource_constraint_hits": 0,
            "infrastructure_adaptations": 0,
        }

    async def initialize_infrastructure(self) -> None:
        """Initialize all infrastructure components."""
        try:
            logger.info("Initializing infrastructure-aware evolution system")

            # Initialize device profiler
            self.device_profiler = DeviceProfiler(
                monitoring_interval=self.config.monitoring_interval,
                enable_background_monitoring=self.config.enable_resource_monitoring,
            )

            # Initialize resource monitor
            if self.config.enable_resource_monitoring:
                self.resource_monitor = ResourceMonitor(self.device_profiler)

            # Initialize constraint manager
            if self.config.enable_resource_constraints:
                self.constraint_manager = ConstraintManager(self.device_profiler)

            # Initialize adaptive loader
            if self.config.enable_adaptive_loading and self.constraint_manager:
                self.adaptive_loader = AdaptiveLoader(self.device_profiler, self.constraint_manager)

            # Initialize P2P node
            if self.config.enable_p2p:
                self.p2p_node = P2PNode(listen_port=self.config.p2p_port)
                await self.p2p_node.start(self._get_resource_status)

                # Register P2P event handlers
                self._register_p2p_handlers()

            # Initialize dual evolution system
            self.dual_evolution = DualEvolutionSystem()

            # Start monitoring systems
            if self.device_profiler:
                self.device_profiler.start_monitoring()

            if self.resource_monitor:
                await self.resource_monitor.start_monitoring()

            if self.constraint_manager:
                await self.constraint_manager.start_constraint_monitoring()

            # Start dual evolution system
            await self.dual_evolution.start_system()

            self.system_initialized = True
            self.infrastructure_status = "active"

            logger.info("Infrastructure-aware evolution system initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize infrastructure: {e}")
            self.infrastructure_status = "failed"
            raise

    async def shutdown_infrastructure(self) -> None:
        """Shutdown all infrastructure components."""
        logger.info("Shutting down infrastructure-aware evolution system")

        # Stop dual evolution system
        if self.dual_evolution:
            await self.dual_evolution.stop_system()

        # Stop monitoring systems
        if self.constraint_manager:
            await self.constraint_manager.stop_constraint_monitoring()

        if self.resource_monitor:
            await self.resource_monitor.stop_monitoring()

        if self.device_profiler:
            self.device_profiler.stop_monitoring()

        # Stop P2P node
        if self.p2p_node:
            await self.p2p_node.stop()

        self.system_initialized = False
        self.infrastructure_status = "shutdown"

    def _register_p2p_handlers(self) -> None:
        """Register P2P event handlers for evolution coordination."""
        if not self.p2p_node:
            return

        # Register evolution-specific handlers
        self.p2p_node.register_handler("EVOLUTION_COORDINATION_REQUEST", self._handle_coordination_request)
        self.p2p_node.register_handler("EVOLUTION_COORDINATION_RESPONSE", self._handle_coordination_response)
        self.p2p_node.register_handler("EVOLUTION_PEER_STATUS", self._handle_peer_evolution_status)
        self.p2p_node.register_handler("EVOLUTION_RESOURCE_OFFER", self._handle_resource_offer)
        self.p2p_node.register_handler("EVOLUTION_CONSENSUS_VOTE", self._handle_consensus_vote)

    async def _get_resource_status(self) -> dict[str, Any]:
        """Get current resource status for P2P communication."""
        if not self.device_profiler or not self.device_profiler.current_snapshot:
            return {}

        snapshot = self.device_profiler.current_snapshot
        return {
            "memory_available_mb": snapshot.memory_available / (1024 * 1024),
            "memory_percent": snapshot.memory_percent,
            "cpu_percent": snapshot.cpu_percent,
            "cpu_cores": snapshot.cpu_cores,
            "battery_percent": snapshot.battery_percent,
            "power_plugged": snapshot.power_plugged,
            "thermal_state": snapshot.thermal_state.value,
            "evolution_suitability": snapshot.evolution_suitability_score,
            "device_type": self.device_profiler.profile.device_type.value,
            "performance_tier": self.device_profiler.profile.performance_tier,
        }

    async def evolve_agent_infrastructure_aware(
        self,
        agent: EvolvableAgent,
        evolution_type: str = "nightly",
        preferred_mode: EvolutionMode | None = None,
    ) -> dict[str, Any]:
        """Evolve agent with infrastructure awareness."""
        if not self.system_initialized:
            msg = "Infrastructure not initialized"
            raise RuntimeError(msg)

        evolution_id = f"evo_{agent.agent_id}_{int(time.time())}"

        try:
            self.stats["evolutions_started"] += 1

            # Assess current resources and determine evolution mode
            evolution_mode = await self._determine_evolution_mode(agent, evolution_type, preferred_mode)

            # Create coordination plan
            plan = await self._create_coordination_plan(evolution_id, agent, evolution_type, evolution_mode)

            self.active_coordinations[evolution_id] = plan

            # Execute evolution based on mode
            if evolution_mode == EvolutionMode.LOCAL_ONLY:
                result = await self._execute_local_evolution(plan, agent)
            elif evolution_mode == EvolutionMode.P2P_COORDINATED:
                result = await self._execute_p2p_coordinated_evolution(plan, agent)
            elif evolution_mode == EvolutionMode.DISTRIBUTED:
                result = await self._execute_distributed_evolution(plan, agent)
            elif evolution_mode == EvolutionMode.HYBRID:
                result = await self._execute_hybrid_evolution(plan, agent)
            else:
                msg = f"Unknown evolution mode: {evolution_mode}"
                raise ValueError(msg)

            # Clean up coordination
            if evolution_id in self.active_coordinations:
                del self.active_coordinations[evolution_id]

            if result.get("success", False):
                self.stats["evolutions_completed"] += 1
            else:
                self.stats["evolutions_failed"] += 1

            return result

        except Exception as e:
            logger.exception(f"Error in infrastructure-aware evolution: {e}")
            self.stats["evolutions_failed"] += 1

            # Clean up on error
            if evolution_id in self.active_coordinations:
                del self.active_coordinations[evolution_id]

            return {
                "success": False,
                "error": str(e),
                "evolution_id": evolution_id,
                "mode_used": (evolution_mode.value if "evolution_mode" in locals() else "unknown"),
            }

    async def _determine_evolution_mode(
        self,
        agent: EvolvableAgent,
        evolution_type: str,
        preferred_mode: EvolutionMode | None,
    ) -> EvolutionMode:
        """Determine optimal evolution mode based on current conditions."""
        # Start with preferred mode or default
        mode = preferred_mode or self.config.default_evolution_mode

        # Check if current mode is feasible
        if mode == EvolutionMode.LOCAL_ONLY:
            # Check if we have sufficient local resources
            if not await self._check_local_resources_sufficient(evolution_type):
                # Try to find alternative mode
                if self.config.enable_p2p and self.p2p_node:
                    mode = EvolutionMode.P2P_COORDINATED
                else:
                    # Stay local but use resource constraints
                    self.stats["resource_constraint_hits"] += 1

        elif mode in [EvolutionMode.P2P_COORDINATED, EvolutionMode.DISTRIBUTED]:
            # Check if P2P is available
            if not self.config.enable_p2p or not self.p2p_node:
                mode = EvolutionMode.LOCAL_ONLY
                self.stats["infrastructure_adaptations"] += 1

        elif mode == EvolutionMode.HYBRID:
            # Check if we can support hybrid mode
            if not self.config.enable_p2p or not self.p2p_node:
                mode = EvolutionMode.LOCAL_ONLY
                self.stats["infrastructure_adaptations"] += 1

        logger.info(f"Determined evolution mode: {mode.value} for {evolution_type} evolution")
        return mode

    async def _check_local_resources_sufficient(self, evolution_type: str) -> bool:
        """Check if local resources are sufficient for evolution."""
        if not self.device_profiler or not self.device_profiler.current_snapshot:
            return False

        snapshot = self.device_profiler.current_snapshot

        # Get evolution resource requirements
        if self.constraint_manager:
            template = self.constraint_manager.get_constraint_template(evolution_type)
            if template:
                available_memory_mb = snapshot.memory_available / (1024 * 1024)
                if available_memory_mb < template.max_memory_mb:
                    return False

                available_cpu = 100 - snapshot.cpu_percent
                if available_cpu < template.max_cpu_percent:
                    return False

        # Check evolution suitability score
        return snapshot.evolution_suitability_score > 0.6

    async def _create_coordination_plan(
        self,
        evolution_id: str,
        agent: EvolvableAgent,
        evolution_type: str,
        mode: EvolutionMode,
    ) -> EvolutionCoordinationPlan:
        """Create coordination plan for evolution."""
        # Get resource allocation for local execution
        local_resources = {}
        if self.device_profiler:
            allocation = self.device_profiler.get_evolution_resource_allocation()
            local_resources = allocation

        # Estimate duration based on evolution type and mode
        duration_estimates = {
            "nightly": 30.0,
            "breakthrough": 90.0,
            "emergency": 15.0,
            "lightweight": 20.0,
        }

        base_duration = duration_estimates.get(evolution_type, 60.0)

        # Adjust duration based on mode
        mode_multipliers = {
            EvolutionMode.LOCAL_ONLY: 1.0,
            EvolutionMode.P2P_COORDINATED: 1.2,  # Slight overhead for coordination
            EvolutionMode.DISTRIBUTED: 0.8,  # Faster with distribution
            EvolutionMode.HYBRID: 0.9,  # Balance of both
        }

        estimated_duration = base_duration * mode_multipliers.get(mode, 1.0)

        return EvolutionCoordinationPlan(
            evolution_id=evolution_id,
            agent_id=agent.agent_id,
            evolution_type=evolution_type,
            mode=mode,
            local_resources_allocated=local_resources,
            estimated_duration_minutes=estimated_duration,
            quality_target=0.8,
            fallback_mode=EvolutionMode.LOCAL_ONLY,
        )

    async def _execute_local_evolution(self, plan: EvolutionCoordinationPlan, agent: EvolvableAgent) -> dict[str, Any]:
        """Execute evolution locally with resource constraints."""
        logger.info(f"Executing local evolution for agent {agent.agent_id}")

        # Register task with constraint manager
        if self.constraint_manager:
            registered = self.constraint_manager.register_task(plan.evolution_id, plan.evolution_type)

            if not registered:
                return {
                    "success": False,
                    "error": "Insufficient resources for evolution",
                    "mode_used": "local_only",
                }

        try:
            # Load models adaptively if enabled
            if self.adaptive_loader:
                await self._load_models_for_evolution(agent, plan)

            # Execute evolution using dual evolution system
            if plan.evolution_type == "nightly":
                success = await self.dual_evolution._evolve_agent_nightly(agent)
            elif plan.evolution_type == "breakthrough":
                success = await self.dual_evolution._evolve_agent_breakthrough(agent)
            elif plan.evolution_type == "emergency":
                success = await self.dual_evolution._evolve_agent_emergency(agent)
            else:
                # Default to nightly
                success = await self.dual_evolution._evolve_agent_nightly(agent)

            return {
                "success": success,
                "mode_used": "local_only",
                "evolution_id": plan.evolution_id,
                "duration_minutes": plan.estimated_duration_minutes,
                "resources_used": plan.local_resources_allocated,
            }

        finally:
            # Unregister task
            if self.constraint_manager:
                self.constraint_manager.unregister_task(plan.evolution_id)

    async def _execute_p2p_coordinated_evolution(
        self, plan: EvolutionCoordinationPlan, agent: EvolvableAgent
    ) -> dict[str, Any]:
        """Execute evolution with P2P coordination."""
        logger.info(f"Executing P2P coordinated evolution for agent {agent.agent_id}")

        self.stats["p2p_coordinations"] += 1

        # Broadcast evolution start to peers
        if self.p2p_node:
            await self.p2p_node.broadcast_evolution_event(
                "START",
                {
                    "evolution_id": plan.evolution_id,
                    "agent_id": agent.agent_id,
                    "evolution_type": plan.evolution_type,
                    "estimated_duration": plan.estimated_duration_minutes,
                },
            )

        # Execute locally but with peer coordination
        result = await self._execute_local_evolution(plan, agent)

        # Share results with peers
        if self.p2p_node and result.get("success"):
            await self.p2p_node.broadcast_evolution_event(
                "COMPLETE",
                {
                    "evolution_id": plan.evolution_id,
                    "success": result["success"],
                    "duration": result.get("duration_minutes", 0),
                },
            )

        result["mode_used"] = "p2p_coordinated"
        return result

    async def _execute_distributed_evolution(
        self, plan: EvolutionCoordinationPlan, agent: EvolvableAgent
    ) -> dict[str, Any]:
        """Execute evolution distributed across peers."""
        logger.info(f"Executing distributed evolution for agent {agent.agent_id}")

        self.stats["distributed_evolutions"] += 1

        # For now, fall back to local execution
        # In future sprints, this would coordinate actual distributed processing
        logger.info("Distributed evolution not fully implemented, falling back to local")

        result = await self._execute_local_evolution(plan, agent)
        result["mode_used"] = "distributed_fallback"
        return result

    async def _execute_hybrid_evolution(self, plan: EvolutionCoordinationPlan, agent: EvolvableAgent) -> dict[str, Any]:
        """Execute hybrid evolution (local + peer assistance)."""
        logger.info(f"Executing hybrid evolution for agent {agent.agent_id}")

        # Start with P2P coordinated evolution
        result = await self._execute_p2p_coordinated_evolution(plan, agent)
        result["mode_used"] = "hybrid"
        return result

    async def _load_models_for_evolution(self, agent: EvolvableAgent, plan: EvolutionCoordinationPlan) -> None:
        """Load models adaptively for evolution."""
        if not self.adaptive_loader:
            return

        # Create loading context based on evolution plan
        from ...core.resources.adaptive_loader import LoadingContext

        context = LoadingContext(
            task_type=plan.evolution_type,
            priority_level=3 if plan.evolution_type == "breakthrough" else 2,
            max_loading_time_seconds=300,  # 5 minutes max loading time
            quality_preference=0.8 if plan.evolution_type == "breakthrough" else 0.6,
            resource_constraints=(
                self.constraint_manager.get_constraint_template(plan.evolution_type)
                if self.constraint_manager
                else None
            ),
        )

        # Load base evolution model
        model, loading_info = await self.adaptive_loader.load_model_adaptive("base_evolution_model", context)

        if model:
            logger.info(f"Loaded evolution model using {loading_info.get('strategy', 'unknown')} strategy")
        else:
            logger.warning("Failed to load evolution model")

    # P2P Event Handlers
    async def _handle_coordination_request(self, message: dict, writer) -> None:
        """Handle evolution coordination request from peer."""
        # Placeholder for future implementation

    async def _handle_coordination_response(self, message: dict, writer) -> None:
        """Handle evolution coordination response from peer."""
        # Placeholder for future implementation

    async def _handle_peer_evolution_status(self, message: dict, writer) -> None:
        """Handle evolution status update from peer."""
        sender_id = message.get("sender_id")
        evolution_status = message.get("status", {})

        if sender_id:
            self.peer_evolution_status[sender_id] = evolution_status

    async def _handle_resource_offer(self, message: dict, writer) -> None:
        """Handle resource offer from peer."""
        # Placeholder for future implementation

    async def _handle_consensus_vote(self, message: dict, writer) -> None:
        """Handle consensus vote from peer."""
        # Placeholder for future implementation

    def get_infrastructure_status(self) -> dict[str, Any]:
        """Get comprehensive infrastructure status."""
        status = {
            "system_initialized": self.system_initialized,
            "infrastructure_status": self.infrastructure_status,
            "config": self.config.to_dict(),
            "components": {},
            "active_coordinations": len(self.active_coordinations),
            "peer_count": 0,
            "stats": self.stats.copy(),
        }

        # Component status
        if self.device_profiler:
            status["components"]["device_profiler"] = {
                "active": self.device_profiler.monitoring_active,
                "device_type": self.device_profiler.profile.device_type.value,
                "evolution_capable": self.device_profiler.profile.evolution_capable,
            }

        if self.resource_monitor:
            status["components"]["resource_monitor"] = {
                "active": self.resource_monitor.monitoring_active,
                "monitoring_mode": self.resource_monitor.monitoring_mode.value,
            }

        if self.constraint_manager:
            status["components"]["constraint_manager"] = {
                "active": self.constraint_manager.monitoring_active,
                "active_tasks": len(self.constraint_manager.active_tasks),
            }

        if self.p2p_node:
            p2p_status = self.p2p_node.get_network_status()
            status["components"]["p2p_node"] = p2p_status
            status["peer_count"] = p2p_status.get("connected_peers", 0)

        if self.dual_evolution:
            dual_status = self.dual_evolution.get_system_status()
            status["components"]["dual_evolution"] = dual_status

        return status

    def get_peer_evolution_capabilities(self) -> dict[str, dict[str, Any]]:
        """Get evolution capabilities of connected peers."""
        if not self.p2p_node:
            return {}

        peer_capabilities = {}
        for peer_id, capabilities in self.p2p_node.peer_registry.items():
            peer_capabilities[peer_id] = {
                "device_type": capabilities.device_type,
                "performance_tier": capabilities.performance_tier,
                "evolution_capacity": capabilities.evolution_capacity,
                "available_for_evolution": capabilities.available_for_evolution,
                "current_evolution_load": capabilities.current_evolution_load,
                "evolution_suitability": capabilities.get_evolution_priority(),
            }

        return peer_capabilities
