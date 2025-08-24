"""
Task Domain Entity

Represents work units that agents can execute, with priority,
status tracking, and result management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from .agent_entity import AgentId


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    MEDIUM = 3
    HIGH = 5
    URGENT = 7
    CRITICAL = 9


@dataclass
class TaskId:
    """Task identifier value object"""

    value: str

    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("TaskId must be a non-empty string")

    @classmethod
    def generate(cls) -> TaskId:
        """Generate new unique task ID"""
        return cls(str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass
class Task:
    """
    Core Task domain entity

    Represents work that can be assigned to and executed by agents.
    Contains business logic for task lifecycle and state transitions.
    """

    # Identity
    id: TaskId
    title: str
    description: str

    # Assignment and execution
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_agent_id: AgentId | None = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    deadline: datetime | None = None

    # Task data and results
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error_details: str | None = None

    # Dependencies and context
    required_capabilities: list[str] = field(default_factory=list)
    parent_task_id: TaskId | None = None
    subtask_ids: list[TaskId] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task invariants"""
        if not self.title.strip():
            raise ValueError("Task title cannot be empty")

        if not self.description.strip():
            raise ValueError("Task description cannot be empty")

    def assign_to_agent(self, agent_id: AgentId) -> None:
        """Assign task to an agent"""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot assign task with status {self.status.value}")

        self.assigned_agent_id = agent_id
        self.assigned_at = datetime.now()
        self.status = TaskStatus.ASSIGNED

    def start_execution(self) -> None:
        """Mark task as started"""
        if self.status != TaskStatus.ASSIGNED:
            raise ValueError(f"Cannot start task with status {self.status.value}")

        if not self.assigned_agent_id:
            raise ValueError("Cannot start unassigned task")

        self.started_at = datetime.now()
        self.status = TaskStatus.IN_PROGRESS

    def complete_successfully(self, output_data: dict[str, Any]) -> None:
        """Mark task as completed successfully"""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete task with status {self.status.value}")

        self.output_data = output_data
        self.completed_at = datetime.now()
        self.status = TaskStatus.COMPLETED
        self.error_details = None

    def fail_with_error(self, error_details: str) -> None:
        """Mark task as failed with error details"""
        if self.status not in [TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED]:
            raise ValueError(f"Cannot fail task with status {self.status.value}")

        self.error_details = error_details
        self.completed_at = datetime.now()
        self.status = TaskStatus.FAILED

    def cancel(self) -> None:
        """Cancel task execution"""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel task with status {self.status.value}")

        self.completed_at = datetime.now()
        self.status = TaskStatus.CANCELLED

    def is_overdue(self) -> bool:
        """Check if task is past deadline"""
        if not self.deadline:
            return False

        return self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] and datetime.now() > self.deadline

    def get_execution_duration_ms(self) -> float | None:
        """Get task execution duration in milliseconds"""
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.now()
        duration = end_time - self.started_at
        return duration.total_seconds() * 1000

    def add_subtask(self, subtask_id: TaskId) -> None:
        """Add subtask to this task"""
        if subtask_id not in self.subtask_ids:
            self.subtask_ids.append(subtask_id)

    def remove_subtask(self, subtask_id: TaskId) -> None:
        """Remove subtask from this task"""
        if subtask_id in self.subtask_ids:
            self.subtask_ids.remove(subtask_id)

    def can_be_assigned_to_agent(self, agent_capabilities: list[str]) -> bool:
        """Check if task can be assigned to agent with given capabilities"""
        if not self.required_capabilities:
            return True

        return all(req_cap in agent_capabilities for req_cap in self.required_capabilities)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_agent_id": str(self.assigned_agent_id) if self.assigned_agent_id else None,
            "created_at": self.created_at.isoformat(),
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_details": self.error_details,
            "required_capabilities": self.required_capabilities,
            "parent_task_id": str(self.parent_task_id) if self.parent_task_id else None,
            "subtask_ids": [str(sid) for sid in self.subtask_ids],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Create task from dictionary representation"""
        return cls(
            id=TaskId(data["id"]),
            title=data["title"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            priority=TaskPriority(data["priority"]),
            assigned_agent_id=AgentId(data["assigned_agent_id"]) if data.get("assigned_agent_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            assigned_at=datetime.fromisoformat(data["assigned_at"]) if data.get("assigned_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            error_details=data.get("error_details"),
            required_capabilities=data.get("required_capabilities", []),
            parent_task_id=TaskId(data["parent_task_id"]) if data.get("parent_task_id") else None,
            subtask_ids=[TaskId(sid) for sid in data.get("subtask_ids", [])],
            metadata=data.get("metadata", {}),
        )
