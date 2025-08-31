"""
Task Creation Service - Handles task creation and validation.
Extracted from UnifiedManagement god class.
"""
import logging
import uuid
from typing import Any

from core.error_handling import AIVillageException

from ..subgoal_generator import SubGoalGenerator
from ..task import Task
from ..interfaces.task_service_interfaces import ITaskAssignmentService, IProjectManagementService

logger = logging.getLogger(__name__)


class TaskCreationService:
    """Service responsible for task creation and validation."""
    
    def __init__(
        self,
        subgoal_generator: SubGoalGenerator,
        assignment_service: ITaskAssignmentService,
        project_service: IProjectManagementService,
    ) -> None:
        """Initialize with dependencies."""
        self._subgoal_generator = subgoal_generator
        self._assignment_service = assignment_service
        self._project_service = project_service
        self._pending_tasks: list[Task] = []
        
    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = 1,
        deadline: str | None = None,
        project_id: str | None = None,
    ) -> Task:
        """Create a single task with validation."""
        try:
            task = Task(
                description=description,
                assigned_agents=[agent],
                priority=priority,
                deadline=deadline,
            )
            self._pending_tasks.append(task)
            logger.info("Created task: %s for agent: %s", task.id, agent)

            if project_id:
                await self._project_service.add_task_to_project(
                    project_id, task.id, {"description": description, "agent": agent}
                )

            return task
        except Exception as e:
            logger.exception("Error creating task: %s", e)
            msg = f"Error creating task: {e!s}"
            raise AIVillageException(msg) from e

    async def create_complex_task(self, description: str, context: dict[str, Any]) -> list[Task]:
        """Create multiple tasks from complex description using subgoal generation."""
        try:
            subgoals = await self._subgoal_generator.generate_subgoals(description, context)
            tasks = []
            
            for subgoal in subgoals:
                agent = await self._assignment_service.select_best_agent_for_task(subgoal)
                task = await self.create_task(subgoal, agent)
                tasks.append(task)
                
            logger.info("Created %d tasks from complex description", len(tasks))
            return tasks
        except Exception as e:
            logger.exception("Error creating complex task: %s", e)
            msg = f"Error creating complex task: {e!s}"
            raise AIVillageException(msg) from e

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return self._pending_tasks.copy()

    def remove_pending_task(self, task_id: str) -> Task | None:
        """Remove and return a pending task by ID."""
        for i, task in enumerate(self._pending_tasks):
            if task.id == task_id:
                return self._pending_tasks.pop(i)
        return None

    def validate_task_data(self, description: str, agent: str) -> bool:
        """Validate task creation data."""
        if not description or not description.strip():
            return False
        if not agent or not agent.strip():
            return False
        return True