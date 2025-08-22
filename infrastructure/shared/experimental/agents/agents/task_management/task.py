import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    assigned_agents: list[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    deadline: str | None = None
    priority: int = 1
    dependencies: list[str] = field(default_factory=list)

    def update_status(self, new_status: TaskStatus) -> "Task":
        return Task(
            id=self.id,
            description=self.description,
            assigned_agents=self.assigned_agents,
            status=new_status,
            result=self.result,
            deadline=self.deadline,
            priority=self.priority,
            dependencies=self.dependencies,
        )

    def update_result(self, result: Any) -> "Task":
        return Task(
            id=self.id,
            description=self.description,
            assigned_agents=self.assigned_agents,
            status=self.status,
            result=result,
            deadline=self.deadline,
            priority=self.priority,
            dependencies=self.dependencies,
        )
