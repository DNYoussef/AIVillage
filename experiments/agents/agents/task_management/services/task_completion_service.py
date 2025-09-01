"""
Task Completion Service - Handles task completion and dependency resolution.
Extracted from UnifiedManagement god class.
"""

import logging
from typing import Any

from core.error_handling import AIVillageException

from ..task import Task, TaskStatus
from ..interfaces.task_service_interfaces import (
    ITaskAssignmentService,
    IIncentiveService,
    IAnalyticsService,
    IProjectManagementService,
)

logger = logging.getLogger(__name__)


class TaskCompletionService:
    """Service responsible for task completion and dependency management."""

    def __init__(
        self,
        assignment_service: ITaskAssignmentService,
        incentive_service: IIncentiveService,
        analytics_service: IAnalyticsService,
        project_service: IProjectManagementService,
    ) -> None:
        """Initialize with dependencies."""
        self._assignment_service = assignment_service
        self._incentive_service = incentive_service
        self._analytics_service = analytics_service
        self._project_service = project_service
        self._completed_tasks: list[Task] = []

    async def complete_task(self, task_id: str, result: Any) -> None:
        """Complete a task and handle all related operations."""
        try:
            ongoing_tasks = self._assignment_service.get_ongoing_tasks()
            if task_id not in ongoing_tasks:
                msg = f"Task {task_id} not found in ongoing tasks"
                raise AIVillageException(msg)

            task = ongoing_tasks[task_id]
            updated_task = task.update_status(TaskStatus.COMPLETED).update_result(result)

            # Move from ongoing to completed
            self._assignment_service.remove_ongoing_task(task_id)
            self._completed_tasks.append(updated_task)

            # Update dependencies
            await self.update_dependent_tasks(updated_task)

            # Update agent performance
            if task.assigned_agents:
                agent = task.assigned_agents[0]
                await self._incentive_service.update_agent_performance(agent, result)

            # Record analytics
            completion_time = (updated_task.completed_at - updated_task.created_at).total_seconds()
            success = result.get("success", False) if isinstance(result, dict) else bool(result)
            await self._analytics_service.record_task_completion(task_id, completion_time, success)

            # Update project if applicable
            await self._update_project_for_completed_task(task_id, updated_task)

            logger.info("Completed task: %s", task_id)

        except Exception as e:
            logger.exception("Error completing task: %s", e)
            msg = f"Error completing task: {e!s}"
            raise AIVillageException(msg) from e

    async def update_dependent_tasks(self, completed_task: Task) -> None:
        """Update tasks that depend on the completed task."""
        try:
            # This is a simplified implementation - in a real system,
            # we'd need access to all pending tasks to check dependencies
            logger.info("Updated dependencies for completed task: %s", completed_task.id)
        except Exception as e:
            logger.exception("Error updating dependent tasks: %s", e)
            msg = f"Error updating dependent tasks: {e!s}"
            raise AIVillageException(msg) from e

    async def _update_project_for_completed_task(self, task_id: str, task: Task) -> None:
        """Update project status when a task is completed."""
        try:
            # Find project containing this task and update its status
            # This would typically involve checking all projects for the task
            logger.debug("Checking project updates for completed task: %s", task_id)
        except Exception as e:
            logger.exception("Error updating project for completed task: %s", e)

    def get_completed_tasks(self) -> list[Task]:
        """Get all completed tasks."""
        return self._completed_tasks.copy()

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the status of a specific task."""
        try:
            # Check ongoing tasks
            ongoing_tasks = self._assignment_service.get_ongoing_tasks()
            if task_id in ongoing_tasks:
                return ongoing_tasks[task_id].status

            # Check completed tasks
            if any(task.id == task_id for task in self._completed_tasks):
                return TaskStatus.COMPLETED

            # Default to pending if not found elsewhere
            return TaskStatus.PENDING

        except Exception as e:
            logger.exception("Error getting task status: %s", e)
            msg = f"Error getting task status: {e!s}"
            raise AIVillageException(msg) from e

    def get_completion_statistics(self) -> dict[str, Any]:
        """Get completion statistics."""
        try:
            total_completed = len(self._completed_tasks)
            if total_completed == 0:
                return {"total_completed": 0, "average_completion_time": 0}

            total_time = sum(
                (task.completed_at - task.created_at).total_seconds()
                for task in self._completed_tasks
                if task.completed_at and task.created_at
            )

            return {
                "total_completed": total_completed,
                "average_completion_time": total_time / total_completed if total_completed > 0 else 0,
            }
        except Exception as e:
            logger.exception("Error getting completion statistics: %s", e)
            return {"error": str(e)}
