"""
Agent Coordination Domain Service

Handles business logic for agent coordination, task assignment,
and multi-agent collaboration workflows.
"""

from __future__ import annotations

from typing import Protocol

from ..entities.agent_entity import Agent, AgentCapability, AgentId, AgentStatus
from ..entities.task_entity import Task, TaskId, TaskPriority, TaskStatus


class AgentRepository(Protocol):
    """Repository interface for agent persistence"""

    async def save(self, agent: Agent) -> None:
        """Save agent to storage"""
        ...

    async def get_by_id(self, agent_id: AgentId) -> Agent | None:
        """Get agent by ID"""
        ...

    async def get_all_active(self) -> list[Agent]:
        """Get all active agents"""
        ...

    async def get_by_capabilities(self, capabilities: list[AgentCapability]) -> list[Agent]:
        """Get agents with specific capabilities"""
        ...


class TaskRepository(Protocol):
    """Repository interface for task persistence"""

    async def save(self, task: Task) -> None:
        """Save task to storage"""
        ...

    async def get_by_id(self, task_id: TaskId) -> Task | None:
        """Get task by ID"""
        ...

    async def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks"""
        ...


class AgentCoordinationService:
    """
    Domain service for agent coordination and task assignment

    Handles the business logic for:
    - Agent discovery and selection
    - Task assignment optimization
    - Load balancing across agents
    - Multi-agent collaboration
    """

    def __init__(self, agent_repository: AgentRepository, task_repository: TaskRepository):
        self.agent_repository = agent_repository
        self.task_repository = task_repository

    async def assign_task_to_best_agent(
        self, task: Task, exclude_agent_ids: list[AgentId] | None = None
    ) -> AgentId | None:
        """
        Assign task to the most suitable available agent

        Uses business rules for agent selection:
        - Must have required capabilities
        - Must be active and healthy
        - Prefer agents with lower current load
        - Consider agent specialization
        """

        # Get agents with required capabilities
        required_caps = [AgentCapability(cap) for cap in task.required_capabilities]
        if required_caps:
            candidate_agents = await self.agent_repository.get_by_capabilities(required_caps)
        else:
            candidate_agents = await self.agent_repository.get_all_active()

        # Filter out excluded agents
        if exclude_agent_ids:
            candidate_agents = [agent for agent in candidate_agents if agent.id not in exclude_agent_ids]

        # Filter to healthy, active agents
        available_agents = [
            agent for agent in candidate_agents if agent.status == AgentStatus.ACTIVE and agent.is_healthy()
        ]

        if not available_agents:
            return None

        # Score agents for task assignment
        scored_agents = []
        for agent in available_agents:
            score = self._calculate_agent_score_for_task(agent, task)
            scored_agents.append((agent, score))

        # Sort by score (higher is better)
        scored_agents.sort(key=lambda x: x[1], reverse=True)

        # Assign to best agent
        best_agent = scored_agents[0][0]
        task.assign_to_agent(best_agent.id)

        await self.task_repository.save(task)
        return best_agent.id

    def _calculate_agent_score_for_task(self, agent: Agent, task: Task) -> float:
        """
        Calculate suitability score for agent-task pairing

        Considers:
        - Capability match (0.4 weight)
        - Performance metrics (0.3 weight)
        - Current load (0.2 weight)
        - Specialization match (0.1 weight)
        """
        score = 0.0

        # Capability match score
        if task.required_capabilities:
            agent_cap_names = [cap.value for cap in agent.capabilities]
            matched_caps = sum(1 for cap in task.required_capabilities if cap in agent_cap_names)
            capability_score = matched_caps / len(task.required_capabilities)
        else:
            capability_score = 1.0
        score += capability_score * 0.4

        # Performance score (success rate and response time)
        performance_score = agent.success_rate * 0.7
        if agent.average_response_time_ms > 0:
            # Prefer faster agents (normalize to 0-1 range, 10s = 0, 1s = 1)
            time_score = max(0, 1 - (agent.average_response_time_ms / 10000))
            performance_score += time_score * 0.3
        score += performance_score * 0.3

        # Load score (prefer less busy agents)
        # For now, use inverse of tasks completed as proxy for current load
        if agent.tasks_completed == 0:
            load_score = 1.0
        else:
            # Agents with fewer recent tasks get higher score
            load_score = 1.0 / (1 + agent.tasks_completed * 0.01)
        score += load_score * 0.2

        # Specialization match score
        agent_domain = agent.get_specialization_domain()
        task_context = task.metadata.get("domain", "general")
        if agent_domain == task_context:
            specialization_score = 1.0
        elif agent_domain == "general":
            specialization_score = 0.5
        else:
            specialization_score = 0.1
        score += specialization_score * 0.1

        return score

    async def distribute_workload(self, priority_threshold: TaskPriority = TaskPriority.MEDIUM) -> dict[str, int]:
        """
        Distribute pending tasks across available agents

        Returns dictionary of agent_id -> number of tasks assigned
        """

        # Get pending tasks above priority threshold
        pending_tasks = await self.task_repository.get_pending_tasks()
        high_priority_tasks = [task for task in pending_tasks if task.priority.value >= priority_threshold.value]

        # Sort tasks by priority (highest first)
        high_priority_tasks.sort(key=lambda t: t.priority.value, reverse=True)

        assignment_count = {}

        for task in high_priority_tasks:
            assigned_agent_id = await self.assign_task_to_best_agent(task)
            if assigned_agent_id:
                agent_id_str = str(assigned_agent_id)
                assignment_count[agent_id_str] = assignment_count.get(agent_id_str, 0) + 1

        return assignment_count

    async def create_multi_agent_collaboration(
        self, task: Task, required_capabilities: list[AgentCapability], max_agents: int = 3
    ) -> list[AgentId]:
        """
        Create multi-agent team for complex task

        Selects complementary agents based on:
        - Capability coverage
        - Performance metrics
        - Collaboration history
        """

        # Get agents for each required capability
        capability_agents = {}
        for capability in required_capabilities:
            agents = await self.agent_repository.get_by_capabilities([capability])
            active_agents = [a for a in agents if a.status == AgentStatus.ACTIVE and a.is_healthy()]
            capability_agents[capability] = active_agents

        # Select best agent for each capability
        selected_agents = []
        used_agent_ids = set()

        for capability in required_capabilities:
            available_agents = [agent for agent in capability_agents[capability] if agent.id not in used_agent_ids]

            if available_agents:
                # Score agents for this capability
                scored = [(agent, self._calculate_agent_score_for_task(agent, task)) for agent in available_agents]
                scored.sort(key=lambda x: x[1], reverse=True)

                best_agent = scored[0][0]
                selected_agents.append(best_agent.id)
                used_agent_ids.add(best_agent.id)

                if len(selected_agents) >= max_agents:
                    break

        return selected_agents

    async def handle_agent_failure(self, failed_agent_id: AgentId, task_id: TaskId) -> AgentId | None:
        """
        Handle agent failure during task execution

        Reassigns task to another suitable agent
        """

        # Get the failed task
        task = await self.task_repository.get_by_id(task_id)
        if not task:
            return None

        # Mark task as failed and reset for reassignment
        task.fail_with_error(f"Agent {failed_agent_id} failed during execution")
        task.status = TaskStatus.PENDING  # Reset for reassignment
        task.assigned_agent_id = None
        task.assigned_at = None
        task.started_at = None

        # Try to reassign to different agent
        new_agent_id = await self.assign_task_to_best_agent(task, exclude_agent_ids=[failed_agent_id])

        return new_agent_id

    async def get_agent_load_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get load and performance statistics for all agents

        Returns metrics for monitoring and load balancing decisions
        """

        active_agents = await self.agent_repository.get_all_active()
        statistics = {}

        for agent in active_agents:
            agent_stats = {
                "tasks_completed": agent.tasks_completed,
                "success_rate": agent.success_rate,
                "average_response_time_ms": agent.average_response_time_ms,
                "health_score": 1.0 if agent.is_healthy() else 0.0,
                "capabilities_count": len(agent.capabilities),
                "specialization_domain": agent.get_specialization_domain(),
                "status": agent.status.value,
            }
            statistics[str(agent.id)] = agent_stats

        return statistics
