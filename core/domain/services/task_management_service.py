"""
Task Management Service

Provides task coordination and management capabilities for agents.
This is a reference implementation to resolve import issues after reorganization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Task data structure."""

    task_id: str
    task_type: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TaskManagementService:
    """
    Service for managing agent tasks and workflows.

    This is a reference implementation to resolve import dependencies
    during the reorganization process.
    """

    def __init__(self):
        """Initialize the task management service."""
        self._tasks: dict[str, Task] = {}
        self._active_tasks: list[str] = []

    def create_task(
        self, task_id: str, task_type: str, description: str, priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Task:
        """Create a new task."""
        task = Task(task_id=task_id, task_type=task_type, description=description, priority=priority)
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        task = self.get_task(task_id)
        if task:
            task.status = status
            if status == TaskStatus.IN_PROGRESS and task_id not in self._active_tasks:
                self._active_tasks.append(task_id)
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task_id in self._active_tasks:
                    self._active_tasks.remove(task_id)
            return True
        return False

    def get_active_tasks(self) -> list[Task]:
        """Get all active tasks."""
        return [self._tasks[task_id] for task_id in self._active_tasks if task_id in self._tasks]

    def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get tasks by status."""
        return [task for task in self._tasks.values() if task.status == status]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return self.update_task_status(task_id, TaskStatus.CANCELLED)

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        return self.update_task_status(task_id, TaskStatus.COMPLETED)
