"""
Specialist Agent Registry + Configs - Prompt 5

Comprehensive registry for AIVillage's 18 specialized agents with their configurations,
capabilities, and coordination metadata. Enables dynamic agent discovery, routing,
and specialized task delegation.

Agent Ecosystem Integration Point: Centralized agent capability and routing system
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Primary roles for agent specialization."""

    LEADERSHIP = "leadership"  # King, Sage
    COMBAT = "combat"  # Sword, Shield, Guardian
    INTELLIGENCE = "intelligence"  # Magi, Navigator, Chronicler
    PRODUCTION = "production"  # Builder, Artificer, Alchemist
    COMMUNICATION = "communication"  # Herald, Emissary, Scribe
    MANAGEMENT = "management"  # Steward, Curator, Logger
    ANALYSIS = "analysis"  # Profiler


class AgentCapability(Enum):
    """Specific capabilities agents can provide."""

    DECISION_MAKING = "decision_making"
    STRATEGIC_PLANNING = "strategic_planning"
    THREAT_DETECTION = "threat_detection"
    DEFENSE_COORDINATION = "defense_coordination"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    PATH_OPTIMIZATION = "path_optimization"
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_ORCHESTRATION = "task_orchestration"
    COMMUNICATION_ROUTING = "communication_routing"
    MESSAGE_ENCODING = "message_encoding"
    DATA_PERSISTENCE = "data_persistence"
    EVENT_LOGGING = "event_logging"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CONTENT_CURATION = "content_curation"
    ASSET_TRANSFORMATION = "asset_transformation"
    SYSTEM_CONSTRUCTION = "system_construction"
    HISTORICAL_ANALYSIS = "historical_analysis"
    DIPLOMATIC_NEGOTIATION = "diplomatic_negotiation"


class AgentStatus(Enum):
    """Current status of agent instances."""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class AgentCapabilityConfig:
    """Configuration for specific agent capability."""

    capability: AgentCapability
    proficiency_level: float  # 0.0 - 1.0
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpecification:
    """Complete specification for a specialized agent type."""

    agent_type: str
    display_name: str
    description: str
    primary_role: AgentRole
    secondary_roles: list[AgentRole] = field(default_factory=list)

    # Capabilities and configuration
    capabilities: list[AgentCapabilityConfig] = field(default_factory=list)
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    coordination_preferences: dict[str, Any] = field(default_factory=dict)

    # Operational parameters
    max_concurrent_tasks: int = 3
    task_timeout_seconds: int = 300
    priority_multiplier: float = 1.0

    # Integration settings
    requires_human_oversight: bool = False
    can_delegate_tasks: bool = True
    collaboration_score: float = 0.8

    # Technical configuration
    implementation_class: str | None = None
    config_template_path: str | None = None
    initialization_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInstance:
    """Runtime instance of a specialized agent."""

    instance_id: str
    agent_type: str
    status: AgentStatus
    device_id: str

    # Runtime state
    current_tasks: list[str] = field(default_factory=list)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)

    # Coordination state
    last_seen: datetime = field(default_factory=datetime.now)
    capabilities_available: list[AgentCapability] = field(default_factory=list)
    resource_usage: dict[str, float] = field(default_factory=dict)

    # Network state
    endpoint: str | None = None
    transport_preferences: list[str] = field(
        default_factory=lambda: ["betanet", "bitchat"]
    )
    latency_ms: float = 0.0


class SpecialistAgentRegistry:
    """
    Central registry for AIVillage's 18 specialized agents.

    Provides agent discovery, capability matching, and coordination services.
    Integration point for Agent Forge, Transport layer, and task orchestration.
    """

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(__file__).parent / "configs"
        self.specifications: dict[str, AgentSpecification] = {}
        self.instances: dict[str, AgentInstance] = {}
        self.capability_index: dict[AgentCapability, list[str]] = {}
        self.role_index: dict[AgentRole, list[str]] = {}

        # Initialize with core agent specifications
        self._initialize_core_agent_specs()
        self._build_capability_indexes()

        logger.info(
            f"SpecialistAgentRegistry initialized with {len(self.specifications)} agent types"
        )

    def _initialize_core_agent_specs(self):
        """Initialize specifications for all 18 core agents."""

        # Leadership Tier (Strategic Decision Making)
        self.specifications["King"] = AgentSpecification(
            agent_type="King",
            display_name="King Agent - Supreme Coordinator",
            description="Supreme decision-making and strategic coordination agent",
            primary_role=AgentRole.LEADERSHIP,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.DECISION_MAKING, 1.0),
                AgentCapabilityConfig(AgentCapability.STRATEGIC_PLANNING, 1.0),
                AgentCapabilityConfig(AgentCapability.RESOURCE_ALLOCATION, 0.9),
            ],
            max_concurrent_tasks=10,
            priority_multiplier=2.0,
            requires_human_oversight=True,
            collaboration_score=1.0,
        )

        self.specifications["Sage"] = AgentSpecification(
            agent_type="Sage",
            display_name="Sage Agent - Wisdom Synthesis",
            description="Knowledge synthesis and strategic advisory agent",
            primary_role=AgentRole.LEADERSHIP,
            secondary_roles=[AgentRole.INTELLIGENCE],
            capabilities=[
                AgentCapabilityConfig(AgentCapability.KNOWLEDGE_SYNTHESIS, 1.0),
                AgentCapabilityConfig(AgentCapability.STRATEGIC_PLANNING, 0.9),
                AgentCapabilityConfig(AgentCapability.HISTORICAL_ANALYSIS, 0.9),
            ],
            max_concurrent_tasks=5,
            priority_multiplier=1.8,
            collaboration_score=0.9,
        )

        # Intelligence Tier (Information & Analysis)
        self.specifications["Magi"] = AgentSpecification(
            agent_type="Magi",
            display_name="Magi Agent - Mystical Intelligence",
            description="Advanced pattern recognition and mystical analysis agent",
            primary_role=AgentRole.INTELLIGENCE,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.KNOWLEDGE_SYNTHESIS, 0.95),
                AgentCapabilityConfig(AgentCapability.THREAT_DETECTION, 0.9),
                AgentCapabilityConfig(AgentCapability.HISTORICAL_ANALYSIS, 0.85),
            ],
            max_concurrent_tasks=4,
            priority_multiplier=1.6,
        )

        self.specifications["Navigator"] = AgentSpecification(
            agent_type="Navigator",
            display_name="Navigator Agent - Path Optimization",
            description="Routing, path optimization, and network navigation agent",
            primary_role=AgentRole.INTELLIGENCE,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.PATH_OPTIMIZATION, 1.0),
                AgentCapabilityConfig(AgentCapability.COMMUNICATION_ROUTING, 0.9),
                AgentCapabilityConfig(AgentCapability.RESOURCE_ALLOCATION, 0.8),
            ],
            max_concurrent_tasks=8,
            coordination_preferences={
                "transport_aware": True,
                "mobile_optimized": True,
            },
        )

        self.specifications["Chronicler"] = AgentSpecification(
            agent_type="Chronicler",
            display_name="Chronicler Agent - Historical Records",
            description="Historical data analysis and pattern recognition agent",
            primary_role=AgentRole.INTELLIGENCE,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.HISTORICAL_ANALYSIS, 1.0),
                AgentCapabilityConfig(AgentCapability.DATA_PERSISTENCE, 0.9),
                AgentCapabilityConfig(AgentCapability.KNOWLEDGE_SYNTHESIS, 0.8),
            ],
            max_concurrent_tasks=6,
        )

        # Combat Tier (Security & Defense)
        self.specifications["Sword"] = AgentSpecification(
            agent_type="Sword",
            display_name="Sword Agent - Offensive Security",
            description="Offensive security and threat engagement agent",
            primary_role=AgentRole.COMBAT,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.THREAT_DETECTION, 1.0),
                AgentCapabilityConfig(AgentCapability.DEFENSE_COORDINATION, 0.8),
            ],
            max_concurrent_tasks=3,
            priority_multiplier=1.4,
            coordination_preferences={"aggressive_stance": True},
        )

        self.specifications["Shield"] = AgentSpecification(
            agent_type="Shield",
            display_name="Shield Agent - Defensive Security",
            description="Defensive security and protection coordination agent",
            primary_role=AgentRole.COMBAT,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.DEFENSE_COORDINATION, 1.0),
                AgentCapabilityConfig(AgentCapability.THREAT_DETECTION, 0.9),
            ],
            max_concurrent_tasks=5,
            priority_multiplier=1.3,
            coordination_preferences={"defensive_stance": True},
        )

        self.specifications["Guardian"] = AgentSpecification(
            agent_type="Guardian",
            display_name="Guardian Agent - Asset Protection",
            description="Asset and resource protection agent",
            primary_role=AgentRole.COMBAT,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.DEFENSE_COORDINATION, 0.95),
                AgentCapabilityConfig(AgentCapability.RESOURCE_ALLOCATION, 0.7),
            ],
            max_concurrent_tasks=4,
            coordination_preferences={"protective_stance": True},
        )

        # Production Tier (Creation & Construction)
        self.specifications["Builder"] = AgentSpecification(
            agent_type="Builder",
            display_name="Builder Agent - System Construction",
            description="System building and infrastructure construction agent",
            primary_role=AgentRole.PRODUCTION,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.SYSTEM_CONSTRUCTION, 1.0),
                AgentCapabilityConfig(AgentCapability.RESOURCE_ALLOCATION, 0.8),
            ],
            max_concurrent_tasks=6,
            resource_requirements={"compute_intensive": True},
        )

        self.specifications["Artificer"] = AgentSpecification(
            agent_type="Artificer",
            display_name="Artificer Agent - Artifact Creation",
            description="Specialized artifact and component creation agent",
            primary_role=AgentRole.PRODUCTION,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.ASSET_TRANSFORMATION, 1.0),
                AgentCapabilityConfig(AgentCapability.SYSTEM_CONSTRUCTION, 0.8),
            ],
            max_concurrent_tasks=4,
        )

        self.specifications["Alchemist"] = AgentSpecification(
            agent_type="Alchemist",
            display_name="Alchemist Agent - Data Transformation",
            description="Data transformation and process optimization agent",
            primary_role=AgentRole.PRODUCTION,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.ASSET_TRANSFORMATION, 0.95),
                AgentCapabilityConfig(AgentCapability.PERFORMANCE_ANALYSIS, 0.8),
            ],
            max_concurrent_tasks=5,
        )

        # Communication Tier (Information Exchange)
        self.specifications["Herald"] = AgentSpecification(
            agent_type="Herald",
            display_name="Herald Agent - Message Broadcasting",
            description="Message broadcasting and announcement agent",
            primary_role=AgentRole.COMMUNICATION,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.COMMUNICATION_ROUTING, 1.0),
                AgentCapabilityConfig(AgentCapability.MESSAGE_ENCODING, 0.9),
            ],
            max_concurrent_tasks=10,
            coordination_preferences={"broadcast_optimized": True},
        )

        self.specifications["Emissary"] = AgentSpecification(
            agent_type="Emissary",
            display_name="Emissary Agent - Diplomatic Communication",
            description="Diplomatic communication and negotiation agent",
            primary_role=AgentRole.COMMUNICATION,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.DIPLOMATIC_NEGOTIATION, 1.0),
                AgentCapabilityConfig(AgentCapability.COMMUNICATION_ROUTING, 0.8),
            ],
            max_concurrent_tasks=3,
            requires_human_oversight=True,
        )

        self.specifications["Scribe"] = AgentSpecification(
            agent_type="Scribe",
            display_name="Scribe Agent - Documentation",
            description="Documentation and record-keeping agent",
            primary_role=AgentRole.COMMUNICATION,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.DATA_PERSISTENCE, 1.0),
                AgentCapabilityConfig(AgentCapability.MESSAGE_ENCODING, 0.9),
            ],
            max_concurrent_tasks=8,
        )

        # Management Tier (Operations & Oversight)
        self.specifications["Steward"] = AgentSpecification(
            agent_type="Steward",
            display_name="Steward Agent - Resource Management",
            description="Resource management and operational oversight agent",
            primary_role=AgentRole.MANAGEMENT,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.RESOURCE_ALLOCATION, 1.0),
                AgentCapabilityConfig(AgentCapability.TASK_ORCHESTRATION, 0.9),
            ],
            max_concurrent_tasks=12,
            collaboration_score=0.9,
        )

        self.specifications["Curator"] = AgentSpecification(
            agent_type="Curator",
            display_name="Curator Agent - Content Management",
            description="Content curation and knowledge organization agent",
            primary_role=AgentRole.MANAGEMENT,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.CONTENT_CURATION, 1.0),
                AgentCapabilityConfig(AgentCapability.KNOWLEDGE_SYNTHESIS, 0.7),
            ],
            max_concurrent_tasks=7,
        )

        self.specifications["Logger"] = AgentSpecification(
            agent_type="Logger",
            display_name="Logger Agent - Event Tracking",
            description="Event logging and audit trail management agent",
            primary_role=AgentRole.MANAGEMENT,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.EVENT_LOGGING, 1.0),
                AgentCapabilityConfig(AgentCapability.DATA_PERSISTENCE, 0.9),
            ],
            max_concurrent_tasks=15,
            coordination_preferences={"high_throughput": True},
        )

        # Analysis Tier (Performance & Optimization)
        self.specifications["Profiler"] = AgentSpecification(
            agent_type="Profiler",
            display_name="Profiler Agent - Performance Analysis",
            description="System performance analysis and optimization agent",
            primary_role=AgentRole.ANALYSIS,
            capabilities=[
                AgentCapabilityConfig(AgentCapability.PERFORMANCE_ANALYSIS, 1.0),
                AgentCapabilityConfig(AgentCapability.RESOURCE_ALLOCATION, 0.8),
            ],
            max_concurrent_tasks=6,
            resource_requirements={"monitoring_intensive": True},
        )

    def _build_capability_indexes(self):
        """Build indexes for fast capability and role-based lookups."""
        for agent_type, spec in self.specifications.items():
            # Index by role
            if spec.primary_role not in self.role_index:
                self.role_index[spec.primary_role] = []
            self.role_index[spec.primary_role].append(agent_type)

            for secondary_role in spec.secondary_roles:
                if secondary_role not in self.role_index:
                    self.role_index[secondary_role] = []
                self.role_index[secondary_role].append(agent_type)

            # Index by capability
            for cap_config in spec.capabilities:
                capability = cap_config.capability
                if capability not in self.capability_index:
                    self.capability_index[capability] = []
                self.capability_index[capability].append(agent_type)

    def register_instance(
        self, agent_type: str, device_id: str, **kwargs
    ) -> AgentInstance:
        """Register a new agent instance."""
        if agent_type not in self.specifications:
            raise ValueError(f"Unknown agent type: {agent_type}")

        instance_id = str(uuid.uuid4())
        spec = self.specifications[agent_type]

        instance = AgentInstance(
            instance_id=instance_id,
            agent_type=agent_type,
            status=AgentStatus.INITIALIZING,
            device_id=device_id,
            capabilities_available=[cap.capability for cap in spec.capabilities],
            **kwargs,
        )

        self.instances[instance_id] = instance
        logger.info(
            f"Registered {agent_type} instance {instance_id} on device {device_id}"
        )

        return instance

    def find_agents_by_capability(self, capability: AgentCapability) -> list[str]:
        """Find agent types that provide a specific capability."""
        return self.capability_index.get(capability, [])

    def find_agents_by_role(self, role: AgentRole) -> list[str]:
        """Find agent types that fulfill a specific role."""
        return self.role_index.get(role, [])

    def get_available_instances(
        self, agent_type: str | None = None, capability: AgentCapability | None = None
    ) -> list[AgentInstance]:
        """Get available agent instances, optionally filtered by type or capability."""
        instances = []

        for instance in self.instances.values():
            if instance.status != AgentStatus.AVAILABLE:
                continue

            if agent_type and instance.agent_type != agent_type:
                continue

            if capability and capability not in instance.capabilities_available:
                continue

            instances.append(instance)

        # Sort by performance and availability
        instances.sort(
            key=lambda x: (
                len(x.current_tasks),
                -x.performance_metrics.get("success_rate", 0.5),
            )
        )

        return instances

    def route_task_to_agent(self, task: dict[str, Any]) -> AgentInstance | None:
        """Route a task to the most suitable available agent."""
        required_capability = task.get("required_capability")
        agent_type_preference = task.get("agent_type_preference")
        task.get("priority", 1.0)

        # Find candidate agents
        candidates = []

        if agent_type_preference:
            candidates = self.get_available_instances(agent_type=agent_type_preference)
        elif required_capability:
            capability_enum = AgentCapability(required_capability)
            agent_types = self.find_agents_by_capability(capability_enum)
            for agent_type in agent_types:
                candidates.extend(self.get_available_instances(agent_type=agent_type))
        else:
            candidates = self.get_available_instances()

        if not candidates:
            logger.warning(
                f"No available agents for task: {task.get('task_id', 'unknown')}"
            )
            return None

        # Score candidates based on suitability
        best_candidate = None
        best_score = -1.0

        for candidate in candidates:
            spec = self.specifications[candidate.agent_type]

            # Base score from specification
            score = spec.collaboration_score * spec.priority_multiplier

            # Adjust for current load
            load_factor = 1.0 - (
                len(candidate.current_tasks) / spec.max_concurrent_tasks
            )
            score *= load_factor

            # Adjust for performance history
            success_rate = candidate.performance_metrics.get("success_rate", 0.5)
            score *= success_rate

            # Adjust for capability match
            if required_capability and required_capability in [
                c.value for c in candidate.capabilities_available
            ]:
                score *= 1.2

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate:
            # Assign task to agent
            task_id = task.get("task_id", str(uuid.uuid4()))
            best_candidate.current_tasks.append(task_id)
            # Only mark as busy if at max capacity, otherwise keep available for more tasks
            spec = self.specifications[best_candidate.agent_type]
            if len(best_candidate.current_tasks) >= spec.max_concurrent_tasks:
                best_candidate.status = AgentStatus.BUSY
            else:
                best_candidate.status = AgentStatus.AVAILABLE

            logger.info(
                f"Routed task {task_id} to {best_candidate.agent_type} instance {best_candidate.instance_id}"
            )

        return best_candidate

    def get_registry_status(self) -> dict[str, Any]:
        """Get comprehensive registry status."""
        status = {
            "total_agent_types": len(self.specifications),
            "total_instances": len(self.instances),
            "instances_by_status": {},
            "instances_by_type": {},
            "capability_coverage": {},
            "role_coverage": {},
        }

        # Count instances by status
        for instance in self.instances.values():
            status_key = instance.status.value
            status["instances_by_status"][status_key] = (
                status["instances_by_status"].get(status_key, 0) + 1
            )

            type_key = instance.agent_type
            status["instances_by_type"][type_key] = (
                status["instances_by_type"].get(type_key, 0) + 1
            )

        # Count capability coverage
        for capability, agent_types in self.capability_index.items():
            available_instances = sum(
                1
                for instance in self.instances.values()
                if instance.agent_type in agent_types
                and instance.status == AgentStatus.AVAILABLE
            )
            status["capability_coverage"][capability.value] = {
                "agent_types": len(agent_types),
                "available_instances": available_instances,
            }

        # Count role coverage
        for role, agent_types in self.role_index.items():
            available_instances = sum(
                1
                for instance in self.instances.values()
                if instance.agent_type in agent_types
                and instance.status == AgentStatus.AVAILABLE
            )
            status["role_coverage"][role.value] = {
                "agent_types": len(agent_types),
                "available_instances": available_instances,
            }

        return status

    def export_specifications(self, output_path: Path) -> None:
        """Export agent specifications to JSON file."""
        export_data = {}

        for agent_type, spec in self.specifications.items():
            export_data[agent_type] = asdict(spec)
            # Convert enums to strings for JSON serialization
            export_data[agent_type]["primary_role"] = spec.primary_role.value
            export_data[agent_type]["secondary_roles"] = [
                r.value for r in spec.secondary_roles
            ]

            # Convert capability configs
            for cap_config in export_data[agent_type]["capabilities"]:
                cap_config["capability"] = cap_config["capability"].value

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported agent specifications to {output_path}")


# Global registry instance
_global_registry = None


def get_specialist_registry() -> SpecialistAgentRegistry:
    """Get global specialist agent registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SpecialistAgentRegistry()
    return _global_registry


# Integration helpers
async def discover_agent_capabilities(
    capability: AgentCapability,
) -> list[dict[str, Any]]:
    """Discover available agents for a specific capability."""
    registry = get_specialist_registry()
    agent_types = registry.find_agents_by_capability(capability)

    results = []
    for agent_type in agent_types:
        spec = registry.specifications[agent_type]
        available_instances = registry.get_available_instances(agent_type=agent_type)

        results.append(
            {
                "agent_type": agent_type,
                "display_name": spec.display_name,
                "description": spec.description,
                "available_instances": len(available_instances),
                "max_concurrent_tasks": spec.max_concurrent_tasks,
                "collaboration_score": spec.collaboration_score,
            }
        )

    return sorted(results, key=lambda x: x["collaboration_score"], reverse=True)


async def delegate_task_to_specialist(task: dict[str, Any]) -> str | None:
    """Delegate a task to the most suitable specialist agent."""
    registry = get_specialist_registry()
    assigned_agent = registry.route_task_to_agent(task)

    if assigned_agent:
        return assigned_agent.instance_id
    else:
        logger.warning(f"Failed to delegate task: {task.get('task_id', 'unknown')}")
        return None
