"""King Agent - Task orchestration and job scheduling leader.

The King Agent serves as the central coordinator for task distribution,
resource allocation, and decision-making in the AIVillage ecosystem.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""

    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition for agent coordination."""

    task_id: str
    task_type: str
    priority: TaskPriority
    requirements: dict[str, Any]
    assigned_agent: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def duration(self) -> float | None:
        """Calculate task duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ResourceAllocation:
    """Resource allocation for agents."""

    agent_id: str
    cpu_percent: float
    memory_mb: int
    gpu_percent: float = 0.0
    network_bandwidth_mbps: float = 10.0
    storage_gb: float = 1.0
    allocated_at: float = field(default_factory=time.time)


class KingAgent:
    """Task orchestration and job scheduling leader."""

    def __init__(self, spec=None) -> None:
        """Initialize King Agent."""
        self.spec = spec
        self.name = "King"
        self.role_description = "Task orchestration and job scheduling leader"

        # Task management
        self.tasks: dict[str, Task] = {}
        self.task_queue: list[Task] = []
        self.completed_tasks: list[Task] = []

        # Agent management
        self.registered_agents: dict[str, dict[str, Any]] = {}
        self.agent_workloads: dict[str, int] = {}
        self.resource_allocations: dict[str, ResourceAllocation] = {}

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.kpi_scores: dict[str, float] = {}

        # Decision-making parameters
        self.leadership_style = "collaborative"
        self.decision_speed = "balanced"
        self.delegation_preference = "high"

        # Resource management
        self.total_resources = {
            "cpu": 100.0,
            "memory_gb": 16.0,
            "gpu": 100.0,
            "network_mbps": 1000.0,
            "storage_gb": 1000.0,
        }
        self.available_resources = self.total_resources.copy()

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process orchestration requests."""
        task_type = request.get("task", "unknown")

        if task_type == "ping":
            return {
                "status": "completed",
                "agent": "king",
                "result": "Task orchestration system online",
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif task_type == "create_task":
            return self._create_task(request)
        elif task_type == "assign_task":
            return self._assign_task(request)
        elif task_type == "get_status":
            return self._get_orchestration_status()
        elif task_type == "allocate_resources":
            return self._allocate_resources(request)
        elif task_type == "resolve_conflict":
            return self._resolve_conflict(request)
        else:
            return {
                "status": "completed",
                "agent": "king",
                "result": f"Orchestrated task: {task_type}",
                "orchestration_strategy": self._determine_strategy(request),
            }

    def _create_task(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create a new task for orchestration."""
        task_id = f"task_{int(time.time() * 1000)}"

        task = Task(
            task_id=task_id,
            task_type=request.get("task_type", "generic"),
            priority=TaskPriority(request.get("priority", 3)),
            requirements=request.get("requirements", {}),
        )

        self.tasks[task_id] = task
        self.task_queue.append(task)

        # Sort queue by priority
        self.task_queue.sort(key=lambda t: t.priority.value)

        return {
            "status": "completed",
            "agent": "king",
            "result": f"Task {task_id} created and queued",
            "task_id": task_id,
            "queue_position": len(self.task_queue),
        }

    def _assign_task(self, request: dict[str, Any]) -> dict[str, Any]:
        """Assign task to appropriate agent."""
        task_id = request.get("task_id")
        preferred_agent = request.get("agent")

        if task_id not in self.tasks:
            return {"status": "error", "error": f"Task {task_id} not found"}

        task = self.tasks[task_id]

        # Select best agent for task
        if preferred_agent and preferred_agent in self.registered_agents:
            selected_agent = preferred_agent
        else:
            selected_agent = self._select_best_agent(task)

        if selected_agent:
            task.assigned_agent = selected_agent
            task.status = TaskStatus.ASSIGNED
            self.agent_workloads[selected_agent] = (
                self.agent_workloads.get(selected_agent, 0) + 1
            )

            return {
                "status": "completed",
                "agent": "king",
                "result": f"Task {task_id} assigned to {selected_agent}",
                "assigned_agent": selected_agent,
                "estimated_completion": self._estimate_completion_time(task),
            }
        else:
            return {"status": "error", "error": "No suitable agent available for task"}

    def _get_orchestration_status(self) -> dict[str, Any]:
        """Get current orchestration system status."""
        return {
            "status": "completed",
            "agent": "king",
            "result": {
                "total_tasks": len(self.tasks),
                "queued_tasks": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                ),
                "active_tasks": len(
                    [
                        t
                        for t in self.tasks.values()
                        if t.status == TaskStatus.IN_PROGRESS
                    ]
                ),
                "completed_tasks": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
                ),
                "registered_agents": len(self.registered_agents),
                "resource_utilization": self._calculate_resource_utilization(),
                "average_task_completion_time": self._calculate_average_completion_time(),
            },
        }

    def _allocate_resources(self, request: dict[str, Any]) -> dict[str, Any]:
        """Allocate resources to agents."""
        agent_id = request.get("agent_id")
        requested_resources = request.get("resources", {})

        if not agent_id:
            return {"status": "error", "error": "Agent ID required"}

        # Check resource availability
        can_allocate = True
        allocation_details = {}

        for resource, amount in requested_resources.items():
            if resource in self.available_resources:
                if self.available_resources[resource] >= amount:
                    allocation_details[resource] = amount
                else:
                    can_allocate = False
                    break

        if can_allocate:
            # Allocate resources
            allocation = ResourceAllocation(
                agent_id=agent_id,
                cpu_percent=allocation_details.get("cpu", 10.0),
                memory_mb=int(allocation_details.get("memory_gb", 1.0) * 1024),
                gpu_percent=allocation_details.get("gpu", 0.0),
                network_bandwidth_mbps=allocation_details.get("network_mbps", 10.0),
                storage_gb=allocation_details.get("storage_gb", 1.0),
            )

            self.resource_allocations[agent_id] = allocation

            # Update available resources
            for resource, amount in allocation_details.items():
                self.available_resources[resource] -= amount

            return {
                "status": "completed",
                "agent": "king",
                "result": f"Resources allocated to {agent_id}",
                "allocation": {
                    "cpu_percent": allocation.cpu_percent,
                    "memory_mb": allocation.memory_mb,
                    "gpu_percent": allocation.gpu_percent,
                    "network_mbps": allocation.network_bandwidth_mbps,
                    "storage_gb": allocation.storage_gb,
                },
            }
        else:
            return {
                "status": "error",
                "error": "Insufficient resources available",
                "available": self.available_resources,
                "requested": requested_resources,
            }

    def _resolve_conflict(self, request: dict[str, Any]) -> dict[str, Any]:
        """Resolve conflicts between agents or tasks."""
        conflict_type = request.get("conflict_type", "resource")
        involved_parties = request.get("parties", [])

        resolution_strategy = self._determine_conflict_resolution(
            conflict_type, involved_parties
        )

        return {
            "status": "completed",
            "agent": "king",
            "result": f"Conflict resolved using {resolution_strategy}",
            "resolution": {
                "strategy": resolution_strategy,
                "affected_parties": involved_parties,
                "resolution_time": datetime.utcnow().isoformat(),
            },
        }

    def _determine_strategy(self, request: dict[str, Any]) -> str:
        """Determine orchestration strategy based on request."""
        task_type = request.get("task_type", "")
        complexity = request.get("complexity", "medium")
        urgency = request.get("urgency", "medium")

        if urgency == "high":
            return "direct_assignment"
        elif complexity == "high":
            return "collaborative_delegation"
        elif self.delegation_preference == "high":
            return "distributed_execution"
        else:
            return "balanced_orchestration"

    def _select_best_agent(self, task: Task) -> str | None:
        """Select the best agent for a given task."""
        if not self.registered_agents:
            return None

        # Simple selection based on workload and capabilities
        best_agent = None
        min_workload = float("inf")

        for agent_id, agent_info in self.registered_agents.items():
            workload = self.agent_workloads.get(agent_id, 0)

            # Check if agent has required capabilities
            agent_capabilities = agent_info.get("capabilities", [])
            required_capabilities = task.requirements.get("capabilities", [])

            if all(cap in agent_capabilities for cap in required_capabilities):
                if workload < min_workload:
                    min_workload = workload
                    best_agent = agent_id

        return best_agent

    def _estimate_completion_time(self, task: Task) -> str:
        """Estimate task completion time."""
        base_time = 300  # 5 minutes base
        priority_multiplier = {
            TaskPriority.EMERGENCY: 0.5,
            TaskPriority.HIGH: 0.7,
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.LOW: 1.5,
            TaskPriority.BACKGROUND: 2.0,
        }

        estimated_seconds = base_time * priority_multiplier.get(task.priority, 1.0)
        completion_time = datetime.fromtimestamp(time.time() + estimated_seconds)

        return completion_time.isoformat()

    def _calculate_resource_utilization(self) -> dict[str, float]:
        """Calculate current resource utilization."""
        utilization = {}

        for resource, total in self.total_resources.items():
            used = total - self.available_resources[resource]
            utilization[resource] = (used / total) * 100 if total > 0 else 0

        return utilization

    def _calculate_average_completion_time(self) -> float:
        """Calculate average task completion time."""
        completed = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

        if not completed:
            return 0.0

        total_time = sum(t.duration() or 0 for t in completed)
        return total_time / len(completed)

    def _determine_conflict_resolution(
        self, conflict_type: str, parties: list[str]
    ) -> str:
        """Determine conflict resolution strategy."""
        if self.leadership_style == "collaborative":
            return "mediated_negotiation"
        elif conflict_type == "resource":
            return "priority_based_allocation"
        elif len(parties) > 2:
            return "consensus_building"
        else:
            return "direct_arbitration"

    def register_agent(
        self, agent_id: str, capabilities: list[str], resources: dict[str, Any]
    ) -> None:
        """Register an agent with the orchestration system."""
        self.registered_agents[agent_id] = {
            "capabilities": capabilities,
            "resources": resources,
            "registered_at": time.time(),
        }
        self.agent_workloads[agent_id] = 0

    def update_performance(self, performance_data: dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_history.append({**performance_data, "timestamp": time.time()})

        # Calculate KPIs
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # Last 10 records
            success_rate = sum(
                1 for p in recent_performance if p.get("success", False)
            ) / len(recent_performance)

            self.kpi_scores = {
                "task_completion_rate": success_rate,
                "resource_efficiency": self._calculate_resource_efficiency(),
                "agent_coordination_score": self._calculate_coordination_score(),
                "decision_speed_score": self._calculate_decision_speed(),
            }

    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource utilization efficiency."""
        utilization = self._calculate_resource_utilization()
        avg_utilization = (
            sum(utilization.values()) / len(utilization) if utilization else 0
        )

        # Optimal utilization is around 70-80%
        if 70 <= avg_utilization <= 80:
            return 1.0
        elif avg_utilization < 70:
            return avg_utilization / 70
        else:
            return max(0.5, 100 - avg_utilization) / 20

    def _calculate_coordination_score(self) -> float:
        """Calculate agent coordination effectiveness."""
        if not self.completed_tasks:
            return 0.7  # Default score

        # Factor in task completion times and resource conflicts
        on_time_tasks = sum(
            1 for t in self.completed_tasks if (t.duration() or 0) <= 600
        )  # 10 minutes
        coordination_score = (
            on_time_tasks / len(self.completed_tasks) if self.completed_tasks else 0.7
        )

        return min(1.0, coordination_score)

    def _calculate_decision_speed(self) -> float:
        """Calculate decision-making speed score."""
        if self.decision_speed == "fast":
            return 0.9
        elif self.decision_speed == "balanced":
            return 0.8
        else:  # slow
            return 0.6

    def evaluate_kpi(self) -> dict[str, float]:
        """Evaluate current KPI metrics."""
        if not self.kpi_scores:
            return {
                "task_completion_rate": 0.7,
                "resource_efficiency": 0.6,
                "agent_coordination_score": 0.8,
                "decision_speed_score": 0.8,
                "overall_performance": 0.725,
            }

        overall = sum(self.kpi_scores.values()) / len(self.kpi_scores)
        return {**self.kpi_scores, "overall_performance": overall}
