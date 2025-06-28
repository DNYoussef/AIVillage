from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum
import uuid

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass(frozen=True)
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    assigned_agents: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    deadline: Optional[str] = None
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)

    def update_status(self, new_status: TaskStatus) -> 'Task':
        return Task(
            id=self.id,
            description=self.description,
            assigned_agents=self.assigned_agents,
            status=new_status,
            result=self.result,
            deadline=self.deadline,
            priority=self.priority,
            dependencies=self.dependencies
        )

    def update_result(self, result: Any) -> 'Task':
        return Task(
            id=self.id,
            description=self.description,
            assigned_agents=self.assigned_agents,
            status=self.status,
            result=result,
            deadline=self.deadline,
            priority=self.priority,
            dependencies=self.dependencies
        )
