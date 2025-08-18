"""
Fog Node - Individual fog computing node implementation

Represents a single device participating in fog computing.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fog_coordinator import ComputeCapacity, FogTask

logger = logging.getLogger(__name__)


class FogNodeState(Enum):
    """Fog node states"""

    OFFLINE = "offline"
    IDLE = "idle"
    ACTIVE = "active"
    CHARGING = "charging"
    OVERLOADED = "overloaded"
    ERROR = "error"


@dataclass
class TaskExecution:
    """Task execution tracking"""

    task_id: str
    started_at: datetime
    progress: float = 0.0
    status: str = "running"


class FogNode:
    """Individual fog computing node"""

    def __init__(
        self, node_id: str, capacity: "ComputeCapacity", coordinator_id: str, metadata: dict[str, Any] | None = None
    ):
        self.node_id = node_id
        self.capacity = capacity
        self.coordinator_id = coordinator_id
        self.metadata = metadata or {}

        self.state = FogNodeState.IDLE
        self.active_tasks: dict[str, TaskExecution] = {}
        self.completed_tasks: list["FogTask"] = []

        logger.info(f"Fog node {node_id} initialized")

    async def execute_task(self, task: "FogTask") -> None:
        """Execute a fog computing task"""

        execution = TaskExecution(task_id=task.task_id, started_at=datetime.now(UTC))

        self.active_tasks[task.task_id] = execution
        self.state = FogNodeState.ACTIVE

        try:
            # Simulate task execution
            task.status = "running"
            await asyncio.sleep(min(task.estimated_duration_seconds / 10, 2.0))  # Simulate work

            # Mark as completed
            task.status = "completed"
            task.progress = 1.0
            execution.status = "completed"

            # Move to completed tasks
            del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)

            if not self.active_tasks:
                self.state = FogNodeState.IDLE

            logger.info(f"Task {task.task_id} completed on node {self.node_id}")

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            execution.status = "failed"
            logger.error(f"Task {task.task_id} failed on node {self.node_id}: {e}")

    async def get_current_capacity(self) -> "ComputeCapacity":
        """Get current compute capacity"""
        return self.capacity

    async def get_completed_tasks(self) -> list["FogTask"]:
        """Get completed tasks and clear the list"""
        completed = self.completed_tasks.copy()
        self.completed_tasks.clear()
        return completed

    async def shutdown(self) -> None:
        """Shutdown fog node"""
        self.state = FogNodeState.OFFLINE
        self.active_tasks.clear()
        logger.info(f"Fog node {self.node_id} shutdown")
