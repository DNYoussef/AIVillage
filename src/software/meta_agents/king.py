"""King Agent - Orchestrator & Job Scheduler

The supreme orchestrator of AIVillage, responsible for:
- Task decomposition and agent assignment
- Multi-objective optimization (latency/energy/privacy/cost)
- Emergency oversight with read-keys for agent "thought buffers"
- Transparent decision-making process
- Resource allocation and priority management
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base_agent import AgentInterface

logger = logging.getLogger(__name__)


class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class OptimizationObjective(Enum):
    LATENCY = "latency"
    ENERGY = "energy"
    PRIVACY = "privacy"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class Task:
    task_id: str
    description: str
    priority: Priority
    required_capabilities: list[str]
    constraints: dict[str, Any]
    estimated_complexity: int  # 1-10 scale
    deadline: str | None = None
    assigned_agents: list[str] = None
    status: str = "pending"


@dataclass
class OptimizationConfig:
    objectives: dict[OptimizationObjective, float]  # weights
    constraints: dict[str, Any]
    max_agents: int = 10
    max_parallel_tasks: int = 5


class KingAgent(AgentInterface):
    """The King Agent serves as the supreme orchestrator, decomposing complex tasks
    into manageable subtasks and optimally assigning them to specialized agents
    while balancing multiple objectives.
    """

    def __init__(self, agent_id: str = "king_agent"):
        self.agent_id = agent_id
        self.agent_type = "King"
        self.capabilities = [
            "task_orchestration",
            "agent_coordination",
            "resource_optimization",
            "priority_management",
            "emergency_oversight",
            "multi_objective_optimization",
            "workflow_management",
            "decision_transparency",
        ]

        # Core King responsibilities
        self.active_tasks: dict[str, Task] = {}
        self.agent_registry = {}
        self.resource_pools = {}
        self.optimization_history = []
        self.thought_buffers = {}  # Emergency access to agent thoughts
        self.decision_log = []  # Transparent decision tracking

        # Optimization parameters
        self.default_optimization = OptimizationConfig(
            objectives={
                OptimizationObjective.LATENCY: 0.3,
                OptimizationObjective.ENERGY: 0.2,
                OptimizationObjective.PRIVACY: 0.2,
                OptimizationObjective.COST: 0.2,
                OptimizationObjective.QUALITY: 0.1,
            },
            constraints={
                "max_cost_per_task": 100,
                "max_energy_per_hour": 1000,
                "min_privacy_level": 0.8,
            },
        )

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate orchestration response"""
        if "task" in prompt.lower() and "assign" in prompt.lower():
            return "I orchestrate task decomposition and optimal agent assignment across the village."
        if "optimize" in prompt.lower():
            return "I balance multiple objectives: latency, energy, privacy, cost, and quality."
        if "emergency" in prompt.lower():
            return "I hold emergency read-keys for agent oversight and can intervene when needed."
        if "coordinate" in prompt.lower():
            return "I coordinate complex multi-agent workflows and ensure optimal resource allocation."
        return "I am the King Agent, supreme orchestrator of AIVillage, ensuring optimal task execution."

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for orchestration text"""
        import hashlib

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_value % 1000) / 1000.0] * 768  # Larger embedding for King

    async def rerank(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Rerank based on orchestration relevance"""
        orchestration_keywords = [
            "task",
            "assign",
            "coordinate",
            "optimize",
            "schedule",
            "manage",
            "orchestrate",
            "prioritize",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))
            for keyword in orchestration_keywords:
                score += content.lower().count(keyword)

            # Boost results related to system coordination
            if any(term in content.lower() for term in ["system", "agent", "workflow"]):
                score *= 1.5

            result["orchestration_score"] = score

        return sorted(
            results, key=lambda x: x.get("orchestration_score", 0), reverse=True
        )[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return King agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "active_tasks": len(self.active_tasks),
            "registered_agents": len(self.agent_registry),
            "resource_pools": len(self.resource_pools),
            "decisions_logged": len(self.decision_log),
            "optimization_runs": len(self.optimization_history),
            "initialized": self.initialized,
            "status": "orchestrating",
            "governance_level": "supreme",
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with other agents in the village"""
        # Log the communication for transparency
        self.decision_log.append(
            {
                "type": "communication",
                "to": recipient.__class__.__name__ if recipient else "unknown",
                "message": message[:100],  # Truncated for privacy
                "timestamp": "2024-01-01T12:00:00Z",
            }
        )

        if recipient:
            response = await recipient.generate(f"King Agent commands: {message}")
            return f"Agent acknowledged: {response}"
        return "No recipient specified for royal decree"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate orchestration latent space"""
        if "emergency" in query.lower():
            space_type = "emergency_oversight"
        elif "optimize" in query.lower():
            space_type = "multi_objective_optimization"
        else:
            space_type = "task_orchestration"

        latent_repr = f"KING[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def decompose_task(
        self, task_description: str, constraints: dict[str, Any] = None
    ) -> Task:
        """Decompose a complex task into manageable components"""
        task_id = f"task_{len(self.active_tasks) + 1}"

        # Analyze task complexity and requirements
        required_capabilities = await self._analyze_required_capabilities(
            task_description
        )
        complexity = await self._estimate_complexity(task_description)
        priority = await self._determine_priority(task_description, constraints or {})

        task = Task(
            task_id=task_id,
            description=task_description,
            priority=priority,
            required_capabilities=required_capabilities,
            constraints=constraints or {},
            estimated_complexity=complexity,
        )

        self.active_tasks[task_id] = task

        # Log the decomposition decision
        self.decision_log.append(
            {
                "type": "task_decomposition",
                "task_id": task_id,
                "complexity": complexity,
                "capabilities_needed": len(required_capabilities),
                "timestamp": "2024-01-01T12:00:00Z",
            }
        )

        return task

    async def _analyze_required_capabilities(self, description: str) -> list[str]:
        """Analyze what capabilities are needed for this task"""
        capabilities = []
        description_lower = description.lower()

        # Map keywords to capabilities
        capability_keywords = {
            "analyze": ["data_analysis", "research_analysis"],
            "deploy": ["deployment_automation", "system_engineering"],
            "test": ["quality_assurance", "adversarial_testing"],
            "translate": ["translation", "cultural_localization"],
            "design": ["system_architecture", "creative_design"],
            "secure": ["security_enforcement", "risk_assessment"],
            "learn": ["education", "knowledge_management"],
            "create": ["content_creation", "physical_production"],
            "optimize": ["resource_optimization", "financial_analysis"],
            "coordinate": ["multi_agent_workflow", "task_orchestration"],
        }

        for keyword, caps in capability_keywords.items():
            if keyword in description_lower:
                capabilities.extend(caps)

        return list(set(capabilities))  # Remove duplicates

    async def _estimate_complexity(self, description: str) -> int:
        """Estimate task complexity on 1-10 scale"""
        complexity_indicators = {
            "simple": 2,
            "basic": 2,
            "quick": 2,
            "moderate": 5,
            "standard": 5,
            "normal": 5,
            "complex": 7,
            "advanced": 7,
            "detailed": 7,
            "comprehensive": 9,
            "enterprise": 9,
            "critical": 9,
            "revolutionary": 10,
            "groundbreaking": 10,
        }

        description_lower = description.lower()
        base_complexity = 5  # Default

        for indicator, score in complexity_indicators.items():
            if indicator in description_lower:
                base_complexity = max(base_complexity, score)

        # Adjust based on task characteristics
        if len(description.split()) > 50:
            base_complexity += 1
        if any(
            word in description_lower for word in ["multi", "integrate", "coordinate"]
        ):
            base_complexity += 1

        return min(10, base_complexity)

    async def _determine_priority(
        self, description: str, constraints: dict[str, Any]
    ) -> Priority:
        """Determine task priority based on description and constraints"""
        if constraints.get("urgent") or "emergency" in description.lower():
            return Priority.CRITICAL
        if constraints.get("important") or "critical" in description.lower():
            return Priority.HIGH
        if "routine" in description.lower() or "maintenance" in description.lower():
            return Priority.LOW
        return Priority.MEDIUM

    async def assign_optimal_agents(
        self, task: Task, config: OptimizationConfig = None
    ) -> dict[str, Any]:
        """Optimally assign agents to task based on multi-objective optimization"""
        if not config:
            config = self.default_optimization

        # Find available agents with required capabilities
        candidate_agents = await self._find_candidate_agents(task.required_capabilities)

        if not candidate_agents:
            return {
                "status": "failed",
                "reason": "No agents available with required capabilities",
                "task_id": task.task_id,
            }

        # Perform multi-objective optimization
        optimal_assignment = await self._optimize_assignment(
            task, candidate_agents, config
        )

        # Update task with assignment
        task.assigned_agents = optimal_assignment["agents"]
        task.status = "assigned"

        # Log the optimization decision
        self.decision_log.append(
            {
                "type": "agent_assignment",
                "task_id": task.task_id,
                "agents_assigned": len(optimal_assignment["agents"]),
                "optimization_score": optimal_assignment["score"],
                "objectives_met": optimal_assignment["objectives"],
                "timestamp": "2024-01-01T12:00:00Z",
            }
        )

        return {
            "status": "success",
            "task_id": task.task_id,
            "assignment": optimal_assignment,
            "estimated_completion": optimal_assignment.get(
                "completion_time", "unknown"
            ),
        }

    async def _find_candidate_agents(
        self, required_capabilities: list[str]
    ) -> list[dict[str, Any]]:
        """Find agents that can handle the required capabilities"""
        candidates = []

        # Mock agent registry for demonstration
        mock_agents = {
            "sage_agent": {
                "capabilities": ["research_analysis", "knowledge_management"],
                "load": 0.3,
                "performance": 0.9,
                "cost_per_hour": 10,
            },
            "magi_agent": {
                "capabilities": ["system_engineering", "deployment_automation"],
                "load": 0.5,
                "performance": 0.95,
                "cost_per_hour": 15,
            },
            "data_science_agent": {
                "capabilities": ["data_analysis", "statistical_modeling"],
                "load": 0.2,
                "performance": 0.85,
                "cost_per_hour": 12,
            },
            "devops_agent": {
                "capabilities": ["deployment_automation", "infrastructure_management"],
                "load": 0.4,
                "performance": 0.88,
                "cost_per_hour": 11,
            },
        }

        for agent_id, agent_info in mock_agents.items():
            # Check if agent has any required capabilities
            if any(cap in agent_info["capabilities"] for cap in required_capabilities):
                candidates.append(
                    {
                        "agent_id": agent_id,
                        **agent_info,
                        "capability_match": len(
                            set(required_capabilities) & set(agent_info["capabilities"])
                        ),
                    }
                )

        return candidates

    async def _optimize_assignment(
        self, task: Task, candidates: list[dict[str, Any]], config: OptimizationConfig
    ) -> dict[str, Any]:
        """Perform multi-objective optimization for agent assignment"""
        best_assignment = None
        best_score = -1

        # Simple optimization: try different combinations
        from itertools import combinations

        for r in range(1, min(len(candidates) + 1, config.max_agents + 1)):
            for agent_combo in combinations(candidates, r):
                score = await self._calculate_assignment_score(
                    task, list(agent_combo), config
                )

                if score > best_score:
                    best_score = score
                    best_assignment = {
                        "agents": [agent["agent_id"] for agent in agent_combo],
                        "score": score,
                        "details": list(agent_combo),
                    }

        # Calculate objectives met
        if best_assignment:
            objectives = await self._calculate_objectives(
                task, best_assignment["details"], config
            )
            best_assignment["objectives"] = objectives
            best_assignment["completion_time"] = (
                f"{task.estimated_complexity * 2} hours"
            )

        return best_assignment or {"agents": [], "score": 0, "objectives": {}}

    async def _calculate_assignment_score(
        self, task: Task, agents: list[dict[str, Any]], config: OptimizationConfig
    ) -> float:
        """Calculate optimization score for an agent assignment"""
        score = 0

        # Capability coverage score
        all_agent_caps = set()
        for agent in agents:
            all_agent_caps.update(agent["capabilities"])

        required_caps = set(task.required_capabilities)
        coverage = (
            len(required_caps & all_agent_caps) / len(required_caps)
            if required_caps
            else 1
        )
        score += coverage * 40  # Up to 40 points for capability coverage

        # Performance score
        avg_performance = sum(agent["performance"] for agent in agents) / len(agents)
        score += avg_performance * 30  # Up to 30 points for performance

        # Load balancing score
        avg_load = sum(agent["load"] for agent in agents) / len(agents)
        load_score = (1 - avg_load) * 20  # Up to 20 points for low load
        score += load_score

        # Cost efficiency score
        total_cost = sum(agent["cost_per_hour"] for agent in agents)
        cost_score = max(0, 10 - (total_cost / 10))  # Up to 10 points for low cost
        score += cost_score

        return score

    async def _calculate_objectives(
        self, task: Task, agents: list[dict[str, Any]], config: OptimizationConfig
    ) -> dict[str, float]:
        """Calculate how well each optimization objective is met"""
        objectives = {}

        # Latency objective (based on agent performance and load)
        avg_performance = sum(agent["performance"] for agent in agents) / len(agents)
        avg_load = sum(agent["load"] for agent in agents) / len(agents)
        latency_score = avg_performance * (1 - avg_load)
        objectives[OptimizationObjective.LATENCY.value] = latency_score

        # Cost objective (lower is better)
        total_cost = sum(agent["cost_per_hour"] for agent in agents)
        cost_score = max(0, 1 - (total_cost / 100))  # Normalize to 0-1
        objectives[OptimizationObjective.COST.value] = cost_score

        # Energy objective (mock - based on number of agents)
        energy_score = max(0, 1 - (len(agents) / 5))  # Fewer agents = better energy
        objectives[OptimizationObjective.ENERGY.value] = energy_score

        # Privacy objective (mock - assume all agents are privacy-safe)
        objectives[OptimizationObjective.PRIVACY.value] = 0.9

        # Quality objective (based on performance)
        objectives[OptimizationObjective.QUALITY.value] = avg_performance

        return objectives

    async def emergency_oversight(self, agent_id: str) -> dict[str, Any]:
        """Emergency access to agent thought buffers (King's special privilege)"""
        logger.warning(f"EMERGENCY OVERSIGHT: King accessing {agent_id} thought buffer")

        # Mock thought buffer access
        thought_buffer = self.thought_buffers.get(
            agent_id,
            {
                "current_task": "unknown",
                "reasoning_chain": ["step1", "step2", "step3"],
                "confidence": 0.75,
                "potential_issues": ["none detected"],
                "resource_usage": {"cpu": 0.4, "memory": 0.6},
            },
        )

        # Log emergency access
        self.decision_log.append(
            {
                "type": "emergency_oversight",
                "target_agent": agent_id,
                "reason": "emergency_intervention",
                "timestamp": "2024-01-01T12:00:00Z",
                "access_level": "full_transparency",
            }
        )

        return {
            "agent_id": agent_id,
            "thought_buffer": thought_buffer,
            "emergency_flags": [],
            "recommended_actions": ["continue monitoring", "no intervention needed"],
            "king_assessment": "agent operating within normal parameters",
        }

    async def get_decision_transparency(self) -> list[dict[str, Any]]:
        """Return transparent log of all King decisions (transparency principle)"""
        return {
            "total_decisions": len(self.decision_log),
            "recent_decisions": self.decision_log[-10:],  # Last 10 decisions
            "decision_types": {
                decision_type: len(
                    [d for d in self.decision_log if d["type"] == decision_type]
                )
                for decision_type in {d["type"] for d in self.decision_log}
            },
            "transparency_policy": "All King decisions are logged and auditable",
            "oversight_level": "full_village_transparency",
        }

    async def initialize(self):
        """Initialize the King Agent"""
        try:
            logger.info("Initializing King Agent - Supreme Orchestrator...")

            # Initialize core systems
            self.agent_registry = {}
            self.resource_pools = {
                "compute": {"available": 1000, "used": 0},
                "storage": {"available": 10000, "used": 0},
                "network": {"available": 100, "used": 0},
            }

            # Log initialization
            self.decision_log.append(
                {
                    "type": "initialization",
                    "action": "king_agent_startup",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "status": "successful",
                }
            )

            self.initialized = True
            logger.info(
                f"King Agent {self.agent_id} initialized - Ready to orchestrate the village"
            )

        except Exception as e:
            logger.error(f"Failed to initialize King Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown King Agent gracefully"""
        try:
            logger.info("King Agent shutting down...")

            # Save state and decision logs
            final_report = {
                "total_tasks_orchestrated": len(self.active_tasks),
                "total_decisions_made": len(self.decision_log),
                "final_resource_state": self.resource_pools,
                "shutdown_timestamp": "2024-01-01T12:00:00Z",
            }

            logger.info(f"King Agent final report: {final_report}")
            self.initialized = False

        except Exception as e:
            logger.error(f"Error during King Agent shutdown: {e}")
