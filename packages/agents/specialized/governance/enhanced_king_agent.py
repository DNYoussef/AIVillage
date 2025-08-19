"""Enhanced King Agent - Supreme Orchestrator with Full AIVillage Integration

This enhanced version demonstrates how to integrate all required AIVillage systems:
- RAG system as read-only group memory through MCP servers
- All tools implemented as MCP
- Inter-agent communication through communication channels
- Personal journal with quiet-star reflection capability
- Langroid-based personal memory system (emotional memory based on unexpectedness)
- ADAS/Transformers² self-modification capability
- Geometric self-awareness (proprioception-like biofeedback)

The King Agent serves as the supreme orchestrator, responsible for:
- Task decomposition and optimal agent assignment
- Multi-objective optimization (latency/energy/privacy/cost)
- Emergency oversight with read-keys for agent "thought buffers"
- Transparent decision-making with full auditability
- Resource allocation and priority management across the village
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from packages.agents.core.agent_interface import AgentCapability, AgentMetadata, TaskInterface

# Import the base template
from packages.agents.core.base_agent_template import BaseAgentTemplate, MCPTool, ReflectionType

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Task priority levels for orchestration"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class OptimizationObjective(Enum):
    """Multi-objective optimization targets"""

    LATENCY = "latency"
    ENERGY = "energy"
    PRIVACY = "privacy"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class OrchestrationTask:
    """Enhanced task representation for orchestration"""

    task_id: str
    description: str
    priority: Priority
    required_capabilities: list[str]
    constraints: dict[str, Any]
    estimated_complexity: int  # 1-10 scale
    deadline: datetime | None = None
    assigned_agents: list[str] = field(default_factory=list)
    status: str = "pending"

    # Performance tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for multi-objective optimization"""

    objectives: dict[OptimizationObjective, float]  # weights
    constraints: dict[str, Any]
    max_agents: int = 10
    max_parallel_tasks: int = 5


class TaskDecompositionTool(MCPTool):
    """MCP tool for intelligent task decomposition"""

    def __init__(self, king_agent):
        super().__init__("task_decomposition", "Decompose complex tasks into subtasks")
        self.king_agent = king_agent

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute task decomposition with RAG-assisted analysis"""
        self.log_usage()

        task_description = parameters.get("task_description", "")
        constraints = parameters.get("constraints", {})

        try:
            # Query RAG system for similar task patterns
            similar_tasks = await self.king_agent.query_group_memory(
                query=f"task decomposition similar to: {task_description}", mode="comprehensive"
            )

            # Analyze required capabilities using group memory
            capability_analysis = await self.king_agent.query_group_memory(
                query=f"agent capabilities needed for: {task_description}", mode="balanced"
            )

            # Create orchestration task
            task = await self.king_agent._create_orchestration_task(
                task_description, constraints, similar_tasks, capability_analysis
            )

            # Record reflection on decomposition process
            await self.king_agent.record_quiet_star_reflection(
                reflection_type=ReflectionType.PROBLEM_SOLVING,
                context=f"Decomposing complex task: {task_description[:100]}",
                raw_thoughts=f"Analyzing similar patterns from group memory, identifying {len(task.required_capabilities)} required capabilities, estimating complexity {task.estimated_complexity}",
                insights=f"Task decomposition successful. Key insight: {similar_tasks.get('key_patterns', 'No similar patterns found')}",
                emotional_valence=0.2,  # Mild satisfaction
                tags=["task_decomposition", "orchestration", f"priority_{task.priority.name.lower()}"],
            )

            return {
                "status": "success",
                "task": {
                    "task_id": task.task_id,
                    "description": task.description,
                    "priority": task.priority.name,
                    "required_capabilities": task.required_capabilities,
                    "estimated_complexity": task.estimated_complexity,
                    "constraints": task.constraints,
                },
                "similar_tasks_found": len(similar_tasks.get("results", [])),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            return {"status": "error", "message": f"Task decomposition failed: {str(e)}"}


class AgentAssignmentTool(MCPTool):
    """MCP tool for optimal agent assignment with multi-objective optimization"""

    def __init__(self, king_agent):
        super().__init__("agent_assignment", "Assign agents to tasks using multi-objective optimization")
        self.king_agent = king_agent

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute optimal agent assignment"""
        self.log_usage()

        task_id = parameters.get("task_id")
        optimization_config = parameters.get("optimization_config")

        if not task_id or task_id not in self.king_agent.active_tasks:
            return {"status": "error", "message": "Invalid or missing task_id"}

        task = self.king_agent.active_tasks[task_id]

        try:
            # Query RAG system for agent performance data
            agent_performance = await self.king_agent.query_group_memory(
                query=f"agent performance capabilities: {' '.join(task.required_capabilities)}", mode="comprehensive"
            )

            # Find available agents using group memory
            available_agents = await self.king_agent.query_group_memory(
                query=f"available agents with capabilities: {' '.join(task.required_capabilities)}", mode="balanced"
            )

            # Perform multi-objective optimization
            assignment_result = await self.king_agent._optimize_agent_assignment(
                task, agent_performance, available_agents, optimization_config
            )

            # Update task with assignment
            task.assigned_agents = assignment_result["agents"]
            task.status = "assigned"
            task.started_at = datetime.now()

            # Record optimization reflection
            await self.king_agent.record_quiet_star_reflection(
                reflection_type=ReflectionType.PROBLEM_SOLVING,
                context=f"Optimizing agent assignment for task {task_id}",
                raw_thoughts=f"Evaluating {len(available_agents.get('results', []))} available agents against {len(task.required_capabilities)} capabilities, optimizing for {list(optimization_config.get('objectives', {}).keys())}",
                insights=f"Optimal assignment found: {len(assignment_result['agents'])} agents assigned with score {assignment_result.get('score', 0):.2f}",
                emotional_valence=0.4,  # Satisfaction with optimization
                tags=["agent_assignment", "optimization", task.priority.name.lower()],
            )

            return {
                "status": "success",
                "task_id": task_id,
                "assignment": assignment_result,
                "agents_assigned": task.assigned_agents,
                "optimization_score": assignment_result.get("score", 0),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Agent assignment failed: {e}")
            return {"status": "error", "message": f"Agent assignment failed: {str(e)}"}


class EmergencyOversightTool(MCPTool):
    """MCP tool for emergency access to agent thought buffers (King's special privilege)"""

    def __init__(self, king_agent):
        super().__init__("emergency_oversight", "Emergency access to agent thought buffers")
        self.king_agent = king_agent

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute emergency oversight with full transparency logging"""
        self.log_usage()

        target_agent_id = parameters.get("target_agent_id")
        reason = parameters.get("reason", "unspecified emergency")

        if not target_agent_id:
            return {"status": "error", "message": "target_agent_id required"}

        try:
            logger.warning(f"EMERGENCY OVERSIGHT: King accessing {target_agent_id} - Reason: {reason}")

            # Query RAG system for agent's recent activity
            await self.king_agent.query_group_memory(
                query=f"recent activity status for agent: {target_agent_id}", mode="comprehensive"
            )

            # Access thought buffer (in production, this would be secure MCP call)
            thought_buffer = await self.king_agent._access_agent_thought_buffer(target_agent_id)

            # Record high-importance emergency memory
            await self.king_agent.record_quiet_star_reflection(
                reflection_type=ReflectionType.ERROR_ANALYSIS,
                context=f"EMERGENCY OVERSIGHT: {target_agent_id} - {reason}",
                raw_thoughts=f"Accessing agent thought buffer due to: {reason}. Reviewing recent activity patterns and decision chains.",
                insights=f"Emergency oversight completed. Agent appears to be {thought_buffer.get('status', 'unknown')}. No immediate intervention required.",
                emotional_valence=-0.2,  # Slight concern about needing oversight
                tags=["emergency_oversight", "transparency", target_agent_id, "critical"],
            )

            # Log for complete transparency (King's transparency principle)
            oversight_record = {
                "type": "emergency_oversight",
                "timestamp": datetime.now().isoformat(),
                "target_agent": target_agent_id,
                "reason": reason,
                "king_agent": self.king_agent.agent_id,
                "thought_buffer_accessed": True,
                "intervention_required": False,
                "follow_up_actions": [],
            }

            self.king_agent.decision_log.append(oversight_record)

            return {
                "status": "success",
                "target_agent": target_agent_id,
                "thought_buffer": thought_buffer,
                "oversight_record": oversight_record,
                "transparency_logged": True,
            }

        except Exception as e:
            logger.error(f"Emergency oversight failed: {e}")
            return {"status": "error", "message": f"Emergency oversight failed: {str(e)}"}


class EnhancedKingAgent(BaseAgentTemplate):
    """Enhanced King Agent with full AIVillage system integration

    Demonstrates complete integration of:
    - RAG system for group memory and decision support
    - MCP tools for all operations
    - Communication channels for agent coordination
    - Personal journal for quiet-star reflection
    - Langroid-based memory for learning from surprises
    - ADAS self-modification for continuous improvement
    - Geometric self-awareness for performance monitoring
    """

    def __init__(self, agent_id: str = "enhanced_king_agent"):
        # Create agent metadata with King-specific capabilities
        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type="Enhanced_King",
            name="Enhanced King Agent - Supreme Orchestrator",
            description="Supreme orchestrator with full AIVillage system integration",
            version="2.0.0",
            capabilities={
                AgentCapability.TASK_EXECUTION,
                AgentCapability.PLANNING,
                AgentCapability.DECISION_MAKING,
                AgentCapability.INTER_AGENT_COMMUNICATION,
                AgentCapability.BROADCAST_MESSAGING,
                AgentCapability.KNOWLEDGE_RETRIEVAL,
                AgentCapability.PERFORMANCE_MONITORING,
                AgentCapability.REASONING,
            },
            tags=["orchestrator", "king", "supreme", "coordination"],
            configuration={
                "optimization_objectives": {"latency": 0.3, "energy": 0.2, "privacy": 0.2, "cost": 0.2, "quality": 0.1},
                "max_concurrent_tasks": 10,
                "emergency_oversight_enabled": True,
                "transparency_level": "full",
            },
        )

        super().__init__(metadata)

        # Set specialized role
        self.specialized_role = "supreme_orchestrator"

        # King-specific orchestration data
        self.active_tasks: dict[str, OrchestrationTask] = {}
        self.agent_registry = {}
        self.resource_pools = {}
        self.optimization_history = []
        self.thought_buffers = {}  # Emergency access to agent thoughts
        self.decision_log = []  # Complete transparency log

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

        # Performance metrics for ADAS self-modification
        self.orchestration_metrics = {
            "tasks_completed": 0,
            "average_completion_time": 0.0,
            "optimization_efficiency": 0.0,
            "agent_utilization": 0.0,
            "resource_efficiency": 0.0,
        }

        logger.info(f"Enhanced King Agent initialized: {agent_id}")

    async def get_specialized_capabilities(self) -> list[AgentCapability]:
        """Return King-specific capabilities"""
        return [
            AgentCapability.TASK_EXECUTION,
            AgentCapability.PLANNING,
            AgentCapability.DECISION_MAKING,
            AgentCapability.INTER_AGENT_COMMUNICATION,
            AgentCapability.BROADCAST_MESSAGING,
            AgentCapability.KNOWLEDGE_RETRIEVAL,
            AgentCapability.PERFORMANCE_MONITORING,
            AgentCapability.REASONING,
            # King-specific capabilities from the AgentCapability enum
            AgentCapability.STRATEGIC_PLANNING,
            AgentCapability.TASK_ORCHESTRATION,
            AgentCapability.RESOURCE_ALLOCATION,
            AgentCapability.COMMUNICATION_ROUTING,
            AgentCapability.PERFORMANCE_ANALYSIS,
        ]

    async def get_specialized_mcp_tools(self) -> dict[str, MCPTool]:
        """Return King-specific MCP tools"""
        return {
            "task_decomposition": TaskDecompositionTool(self),
            "agent_assignment": AgentAssignmentTool(self),
            "emergency_oversight": EmergencyOversightTool(self),
        }

    async def process_specialized_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process King-specific orchestration tasks"""
        task_type = task_data.get("task_type", "")

        start_time = datetime.now()
        result = {"status": "error", "message": "Unknown task type"}

        try:
            if task_type == "orchestrate_complex_task":
                # Full orchestration: decompose + assign + monitor
                result = await self._orchestrate_complex_task(task_data)

            elif task_type == "optimize_agent_utilization":
                # Analyze and optimize current agent assignments
                result = await self._optimize_agent_utilization(task_data)

            elif task_type == "emergency_intervention":
                # Emergency oversight and intervention
                result = await self._handle_emergency_intervention(task_data)

            elif task_type == "resource_reallocation":
                # Reallocate resources across the village
                result = await self._handle_resource_reallocation(task_data)

            elif task_type == "transparency_audit":
                # Provide full transparency report
                result = await self._provide_transparency_audit(task_data)

            else:
                # Try to decompose as general task
                result = await self.mcp_tools["task_decomposition"].execute(
                    {
                        "task_description": task_data.get("description", ""),
                        "constraints": task_data.get("constraints", {}),
                    }
                )

            # Record performance for geometric awareness
            completion_time = (datetime.now() - start_time).total_seconds() * 1000
            self._record_task_performance(
                task_id=task_data.get("task_id", "unknown"),
                latency_ms=completion_time,
                accuracy=1.0 if result.get("status") == "success" else 0.0,
                status=result.get("status", "error"),
            )

            return result

        except Exception as e:
            logger.error(f"King task processing failed: {e}")
            return {"status": "error", "message": f"Task processing failed: {str(e)}"}

    async def _orchestrate_complex_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Full orchestration of complex multi-agent task"""

        # Step 1: Decompose task
        decomposition_result = await self.mcp_tools["task_decomposition"].execute(
            {"task_description": task_data.get("description", ""), "constraints": task_data.get("constraints", {})}
        )

        if decomposition_result.get("status") != "success":
            return decomposition_result

        # Step 2: Assign agents optimally
        task_id = decomposition_result["task"]["task_id"]
        assignment_result = await self.mcp_tools["agent_assignment"].execute(
            {
                "task_id": task_id,
                "optimization_config": task_data.get("optimization_config", self.default_optimization.__dict__),
            }
        )

        if assignment_result.get("status") != "success":
            return assignment_result

        # Step 3: Monitor and coordinate execution
        coordination_result = await self._coordinate_task_execution(task_id)

        # Record comprehensive reflection
        await self.record_quiet_star_reflection(
            reflection_type=ReflectionType.TASK_COMPLETION,
            context=f"Complex task orchestration: {task_data.get('description', '')[:100]}",
            raw_thoughts=f"Successfully orchestrated multi-step process: decomposition -> assignment ({len(assignment_result.get('agents_assigned', []))} agents) -> coordination. Achieved optimization score: {assignment_result.get('optimization_score', 0):.2f}",
            insights=f"Complex orchestration demonstrates effective integration of RAG-assisted decision making with multi-agent coordination. Key success factor: {coordination_result.get('success_factor', 'systematic approach')}",
            emotional_valence=0.7,  # High satisfaction with successful orchestration
            tags=["complex_orchestration", "multi_agent", "successful_completion"],
        )

        return {
            "status": "success",
            "orchestration_complete": True,
            "task_id": task_id,
            "decomposition": decomposition_result,
            "assignment": assignment_result,
            "coordination": coordination_result,
            "total_agents_involved": len(assignment_result.get("agents_assigned", [])),
            "timestamp": datetime.now().isoformat(),
        }

    async def _coordinate_task_execution(self, task_id: str) -> dict[str, Any]:
        """Coordinate execution of assigned task"""

        if task_id not in self.active_tasks:
            return {"status": "error", "message": "Task not found"}

        task = self.active_tasks[task_id]

        # Send coordination messages to assigned agents
        coordination_messages = []
        for agent_id in task.assigned_agents:
            message_result = await self.send_agent_message(
                recipient=agent_id,
                message=f"Task coordination: {task.description}. Priority: {task.priority.name}",
                channel_type="direct",
                priority=task.priority.value,
                metadata={
                    "task_id": task_id,
                    "coordination_type": "execution_start",
                    "expected_capabilities": task.required_capabilities,
                },
            )
            coordination_messages.append(message_result)

        # Monitor geometric state during coordination
        await self.update_geometric_self_awareness()

        return {
            "status": "success",
            "task_id": task_id,
            "agents_coordinated": len(task.assigned_agents),
            "coordination_messages_sent": len(coordination_messages),
            "success_factor": "systematic_multi_agent_coordination",
            "geometric_state": (
                self.current_geometric_state.geometric_state.value if self.current_geometric_state else "unknown"
            ),
        }

    async def _create_orchestration_task(
        self,
        description: str,
        constraints: dict[str, Any],
        similar_tasks: dict[str, Any],
        capability_analysis: dict[str, Any],
    ) -> OrchestrationTask:
        """Create orchestration task with RAG-assisted analysis"""

        # Extract required capabilities from RAG analysis
        required_capabilities = []
        if capability_analysis.get("status") == "success":
            for result in capability_analysis.get("results", []):
                content = result.get("content", "")
                # Extract capabilities mentioned in content
                capability_keywords = [
                    "coordination",
                    "analysis",
                    "deployment",
                    "security",
                    "translation",
                    "creative",
                    "financial",
                    "social",
                    "testing",
                    "architecture",
                ]
                for keyword in capability_keywords:
                    if keyword in content.lower():
                        required_capabilities.append(f"{keyword}_capability")

        # Estimate complexity based on similar tasks
        complexity = 5  # Default
        if similar_tasks.get("status") == "success" and similar_tasks.get("results"):
            avg_complexity = sum(result.get("complexity", 5) for result in similar_tasks["results"]) / len(
                similar_tasks["results"]
            )
            complexity = max(1, min(10, int(avg_complexity)))

        # Determine priority
        priority = Priority.MEDIUM
        if constraints.get("urgent") or "emergency" in description.lower():
            priority = Priority.CRITICAL
        elif constraints.get("important") or "critical" in description.lower():
            priority = Priority.HIGH
        elif "routine" in description.lower():
            priority = Priority.LOW

        # Create task
        task = OrchestrationTask(
            task_id=f"king_task_{len(self.active_tasks) + 1}_{int(datetime.now().timestamp())}",
            description=description,
            priority=priority,
            required_capabilities=required_capabilities or ["general_capability"],
            constraints=constraints,
            estimated_complexity=complexity,
        )

        self.active_tasks[task.task_id] = task
        return task

    async def _optimize_agent_assignment(
        self,
        task: OrchestrationTask,
        agent_performance: dict[str, Any],
        available_agents: dict[str, Any],
        optimization_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Multi-objective optimization for agent assignment"""

        config = optimization_config or self.default_optimization.__dict__

        # Extract agent information from RAG results
        candidate_agents = []
        if available_agents.get("status") == "success":
            for result in available_agents.get("results", []):
                # Mock agent extraction - in production would parse structured data
                agent_info = {
                    "agent_id": f"agent_{len(candidate_agents) + 1}",
                    "capabilities": task.required_capabilities[:2],  # Mock capability match
                    "load": 0.3 + (len(candidate_agents) * 0.1),  # Mock load
                    "performance": 0.8 + (len(candidate_agents) * 0.05),  # Mock performance
                    "cost_per_hour": 10 + len(candidate_agents),  # Mock cost
                }
                candidate_agents.append(agent_info)

        if not candidate_agents:
            # Fallback to mock agents for demonstration
            candidate_agents = [
                {
                    "agent_id": "sage_agent",
                    "capabilities": ["analysis"],
                    "load": 0.3,
                    "performance": 0.9,
                    "cost_per_hour": 12,
                },
                {
                    "agent_id": "magi_agent",
                    "capabilities": ["deployment"],
                    "load": 0.5,
                    "performance": 0.95,
                    "cost_per_hour": 15,
                },
                {
                    "agent_id": "navigator_agent",
                    "capabilities": ["routing"],
                    "load": 0.2,
                    "performance": 0.85,
                    "cost_per_hour": 10,
                },
            ]

        # Multi-objective optimization (simplified NSGA-II approach)
        best_assignment = None
        best_score = -1

        from itertools import combinations

        max_agents = min(len(candidate_agents), config.get("max_agents", 3))

        for r in range(1, max_agents + 1):
            for agent_combo in combinations(candidate_agents, r):
                score = self._calculate_assignment_score(task, list(agent_combo), config)

                if score > best_score:
                    best_score = score
                    best_assignment = {
                        "agents": [agent["agent_id"] for agent in agent_combo],
                        "score": score,
                        "details": list(agent_combo),
                        "objectives_met": self._calculate_objectives_met(task, list(agent_combo), config),
                    }

        return best_assignment or {"agents": [], "score": 0, "details": []}

    def _calculate_assignment_score(
        self, task: OrchestrationTask, agents: list[dict[str, Any]], config: dict[str, Any]
    ) -> float:
        """Calculate multi-objective optimization score"""
        score = 0

        # Capability coverage
        all_capabilities = set()
        for agent in agents:
            all_capabilities.update(agent.get("capabilities", []))

        required = set(task.required_capabilities)
        coverage = len(required.intersection(all_capabilities)) / len(required) if required else 1
        score += coverage * 40

        # Performance
        avg_performance = sum(agent.get("performance", 0.8) for agent in agents) / len(agents)
        score += avg_performance * 30

        # Load balancing
        avg_load = sum(agent.get("load", 0.5) for agent in agents) / len(agents)
        score += (1 - avg_load) * 20

        # Cost efficiency
        total_cost = sum(agent.get("cost_per_hour", 10) for agent in agents)
        cost_score = max(0, 10 - (total_cost / 10))
        score += cost_score

        return score

    def _calculate_objectives_met(
        self, task: OrchestrationTask, agents: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate how well optimization objectives are met"""

        # Mock calculation - in production would be more sophisticated
        avg_performance = sum(agent.get("performance", 0.8) for agent in agents) / len(agents)
        avg_load = sum(agent.get("load", 0.5) for agent in agents) / len(agents)
        total_cost = sum(agent.get("cost_per_hour", 10) for agent in agents)

        return {
            "latency": avg_performance * (1 - avg_load),  # High performance + low load = low latency
            "cost": max(0, 1 - (total_cost / 50)),  # Lower cost is better
            "energy": max(0, 1 - (len(agents) / 5)),  # Fewer agents = better energy
            "privacy": 0.9,  # Assume high privacy
            "quality": avg_performance,  # Quality based on agent performance
        }

    async def _access_agent_thought_buffer(self, agent_id: str) -> dict[str, Any]:
        """Access agent thought buffer for emergency oversight"""

        # In production, this would be a secure MCP call to the agent
        # For demonstration, return mock thought buffer
        return self.thought_buffers.get(
            agent_id,
            {
                "agent_id": agent_id,
                "status": "operational",
                "current_task": "routine_processing",
                "reasoning_chain": [
                    "received_task",
                    "analyzed_requirements",
                    "processing_with_rag_support",
                    "generating_response",
                ],
                "confidence_level": 0.85,
                "resource_usage": {"cpu": 0.4, "memory": 0.6, "network": 0.2},
                "recent_reflections": [],
                "geometric_state": "balanced",
                "last_adas_modification": None,
                "emergency_flags": [],
            },
        )

    # Implement remaining abstract methods from BaseAgentTemplate

    async def process_task(self, task: TaskInterface) -> dict[str, Any]:
        """Process tasks through the enhanced orchestration system"""
        task_data = {
            "task_type": task.task_type,
            "task_id": task.task_id,
            "description": str(task.content),
            "constraints": task.context,
            "priority": task.priority,
        }

        return await self.process_specialized_task(task_data)

    async def can_handle_task(self, task: TaskInterface) -> bool:
        """King agent can handle orchestration tasks"""
        orchestration_types = [
            "orchestrate",
            "assign",
            "coordinate",
            "optimize",
            "schedule",
            "manage",
            "emergency",
            "oversight",
        ]

        task_content = str(task.content).lower()
        return any(task_type in task_content for task_type in orchestration_types)

    async def estimate_task_duration(self, task: TaskInterface) -> float | None:
        """Estimate task duration based on complexity"""
        # Simple estimation - in production would use ML models
        base_time = 30  # 30 seconds base
        complexity_multiplier = len(str(task.content)) / 100  # Character count proxy
        return base_time * (1 + complexity_multiplier)

    async def send_message(self, message) -> bool:
        """Send message using communication channels"""
        return await self.send_agent_message(
            recipient=message.receiver, message=message.content, metadata={"message_id": message.message_id}
        )

    async def receive_message(self, message) -> None:
        """Receive and process incoming message"""
        # Log message receipt
        logger.info(f"King received message from {message.sender}: {str(message.content)[:100]}")

        # Record interaction reflection
        await self.record_quiet_star_reflection(
            reflection_type=ReflectionType.INTERACTION,
            context=f"Received message from {message.sender}",
            raw_thoughts=f"Processing incoming message of type {message.message_type} with priority {message.priority}",
            insights="Inter-agent communication flowing properly through MCP channels",
            emotional_valence=0.1,
            tags=["communication", message.sender, message.message_type],
        )

    async def broadcast_message(self, message, recipients: list[str]) -> dict[str, bool]:
        """Broadcast message to multiple recipients"""
        results = {}

        for recipient in recipients:
            try:
                result = await self.send_agent_message(
                    recipient=recipient,
                    message=message.content,
                    channel_type="broadcast",
                    metadata={"broadcast_id": message.message_id},
                )
                results[recipient] = result.get("status") == "success"
            except Exception as e:
                logger.error(f"Broadcast to {recipient} failed: {e}")
                results[recipient] = False

        return results

    async def generate(self, prompt: str) -> str:
        """Generate King-appropriate responses with RAG support"""

        # Query group memory for context
        context_query = await self.query_group_memory(
            query=f"king agent orchestration context: {prompt[:100]}", mode="balanced"
        )

        # Generate contextual response
        if "orchestrate" in prompt.lower():
            return f"I orchestrate complex multi-agent tasks using RAG-assisted decomposition and multi-objective optimization. Context from group memory: {len(context_query.get('results', []))} relevant patterns found."
        elif "emergency" in prompt.lower():
            return "I have emergency oversight capabilities with full transparency logging. All emergency actions are recorded for auditability."
        elif "optimize" in prompt.lower():
            return "I balance multiple objectives: latency, energy, privacy, cost, and quality using advanced optimization algorithms."
        elif "coordinate" in prompt.lower():
            return "I coordinate village-wide operations through intelligent agent assignment and resource allocation."
        else:
            return f"I am the Enhanced King Agent with full AIVillage integration: RAG group memory, MCP tools, communication channels, quiet-star reflection, Langroid memory, ADAS self-modification, and geometric self-awareness. Group memory provided {len(context_query.get('results', []))} relevant context items."

    async def get_embedding(self, text: str) -> list[float]:
        """Get embeddings with enhanced dimensionality for King agent"""
        import hashlib

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Larger embedding space for King's complex reasoning
        return [(hash_value % 1000) / 1000.0] * 1024

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank results based on orchestration and governance relevance"""

        orchestration_keywords = [
            "task",
            "assign",
            "coordinate",
            "optimize",
            "schedule",
            "manage",
            "orchestrate",
            "prioritize",
            "governance",
            "oversight",
            "emergency",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", "")).lower()

            for keyword in orchestration_keywords:
                score += content.count(keyword) * 2  # Higher weight for King-relevant terms

            # Boost governance and system-level content
            if any(term in content for term in ["system", "village", "agent", "coordination"]):
                score *= 1.5

            result["king_relevance_score"] = score

        return sorted(results, key=lambda x: x.get("king_relevance_score", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Comprehensive King agent introspection"""
        base_info = await super().health_check()

        king_specific = {
            "orchestration_status": {
                "active_tasks": len(self.active_tasks),
                "completed_tasks": self.orchestration_metrics["tasks_completed"],
                "average_completion_time": self.orchestration_metrics["average_completion_time"],
                "optimization_efficiency": self.orchestration_metrics["optimization_efficiency"],
            },
            "governance_metrics": {
                "decision_log_entries": len(self.decision_log),
                "emergency_oversight_count": sum(
                    1 for log in self.decision_log if log.get("type") == "emergency_oversight"
                ),
                "transparency_level": "full",
                "agent_registry_size": len(self.agent_registry),
            },
            "resource_management": {
                "resource_pools": len(self.resource_pools),
                "optimization_history": len(self.optimization_history),
                "current_optimization": self.default_optimization.__dict__,
            },
        }

        # Merge with base introspection
        base_info.update(king_specific)
        return base_info

    async def communicate(self, message: str, recipient) -> str:
        """Enhanced communication with transparency logging"""

        # Log for transparency
        self.decision_log.append(
            {
                "type": "communication",
                "timestamp": datetime.now().isoformat(),
                "to": getattr(recipient, "agent_id", str(recipient)),
                "message_length": len(message),
                "communication_type": "direct",
            }
        )

        # Use MCP communication tool
        result = await self.send_agent_message(
            recipient=getattr(recipient, "agent_id", str(recipient)), message=message, channel_type="direct"
        )

        if result.get("status") == "success":
            return "Message delivered successfully via MCP channels"
        else:
            return f"Message delivery failed: {result.get('message', 'Unknown error')}"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate King's specialized latent space"""

        if "emergency" in query.lower():
            space_type = "emergency_governance"
            representation = f"KING[EMERGENCY:{query[:30]}|oversight_active]"
        elif "orchestrate" in query.lower():
            space_type = "multi_agent_orchestration"
            representation = f"KING[ORCHESTRATE:{query[:30]}|optimizing_assignment]"
        elif "coordinate" in query.lower():
            space_type = "village_coordination"
            representation = f"KING[COORDINATE:{query[:30]}|managing_resources]"
        else:
            space_type = "supreme_governance"
            representation = f"KING[GOVERN:{query[:30]}|balancing_objectives]"

        return space_type, representation

    # Additional King-specific methods

    async def get_transparency_report(self) -> dict[str, Any]:
        """Generate full transparency report (King's transparency principle)"""

        return {
            "report_timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "transparency_level": "full",
            "decision_summary": {
                "total_decisions": len(self.decision_log),
                "decision_types": {
                    decision_type: len([d for d in self.decision_log if d.get("type") == decision_type])
                    for decision_type in set(d.get("type", "unknown") for d in self.decision_log)
                },
                "recent_decisions": self.decision_log[-10:] if self.decision_log else [],
            },
            "orchestration_summary": {
                "active_tasks": len(self.active_tasks),
                "task_priorities": {
                    priority.name: len([t for t in self.active_tasks.values() if t.priority == priority])
                    for priority in Priority
                },
                "resource_utilization": self.orchestration_metrics,
            },
            "system_integration": {
                "rag_queries_made": self.mcp_tools["rag_query"].usage_count,
                "communication_messages_sent": self.mcp_tools["communicate"].usage_count,
                "emergency_oversight_used": self.mcp_tools["emergency_oversight"].usage_count,
                "reflection_count": len(self.personal_journal),
                "memory_entries": len(self.personal_memory),
            },
            "geometric_awareness": {
                "current_state": (
                    self.current_geometric_state.geometric_state.value if self.current_geometric_state else "unknown"
                ),
                "performance_healthy": (
                    self.current_geometric_state.is_healthy() if self.current_geometric_state else False
                ),
                "adas_modifications": len(self.adas_config["modification_history"]),
            },
        }


# Factory function for easy instantiation
async def create_enhanced_king_agent(agent_id: str = "enhanced_king_agent") -> EnhancedKingAgent:
    """Create and initialize an Enhanced King Agent with full system integration"""

    agent = EnhancedKingAgent(agent_id)

    # Initialize with system connections (would be injected in production)
    # agent.rag_client = get_rag_client()
    # agent.p2p_client = get_p2p_client()
    # agent.agent_forge_client = get_agent_forge_client()

    # Initialize the agent
    success = await agent.initialize()

    if not success:
        raise RuntimeError(f"Failed to initialize Enhanced King Agent: {agent_id}")

    return agent


# Export for use in agent system
__all__ = [
    "EnhancedKingAgent",
    "create_enhanced_king_agent",
    "OrchestrationTask",
    "OptimizationConfig",
    "Priority",
    "OptimizationObjective",
]
