"""
Project Management Service - Handles project lifecycle and resource management.
Extracted from UnifiedManagement god class.
"""
import logging
import uuid
from typing import Any

from core.error_handling import AIVillageException

from ..task import Task
from ..unified_task_manager import Project

logger = logging.getLogger(__name__)


class ProjectManagementService:
    """Service responsible for project management operations."""
    
    def __init__(self) -> None:
        """Initialize the project management service."""
        self._projects: dict[str, Project] = {}
        
    async def create_project(self, name: str, description: str) -> str:
        """Create a new project."""
        try:
            project_id = str(uuid.uuid4())
            self._projects[project_id] = Project(
                id=project_id, 
                name=name, 
                description=description
            )
            logger.info("Created project: %s (%s)", name, project_id)
            return project_id
        except Exception as e:
            logger.exception("Error creating project: %s", e)
            msg = f"Error creating project: {e!s}"
            raise AIVillageException(msg) from e

    async def get_all_projects(self) -> dict[str, Project]:
        """Get all projects."""
        return self._projects.copy()

    async def get_project(self, project_id: str) -> Project:
        """Get a specific project by ID."""
        project = self._projects.get(project_id)
        if not project:
            msg = f"Project with ID {project_id} not found"
            raise AIVillageException(msg)
        return project

    async def update_project_status(
        self, 
        project_id: str, 
        status: str | None = None, 
        progress: float | None = None
    ) -> None:
        """Update project status and progress."""
        try:
            project = await self.get_project(project_id)
            
            if status:
                project.status = status
            if progress is not None:
                if not 0.0 <= progress <= 1.0:
                    msg = f"Progress must be between 0.0 and 1.0, got {progress}"
                    raise ValueError(msg)
                project.progress = progress
                
            logger.info(
                "Updated project %s - Status: %s, Progress: %s",
                project_id,
                status,
                progress,
            )
        except Exception as e:
            logger.exception("Error updating project status: %s", e)
            msg = f"Error updating project status: {e!s}"
            raise AIVillageException(msg) from e

    async def add_task_to_project(
        self, 
        project_id: str, 
        task_id: str, 
        task_data: dict[str, Any]
    ) -> None:
        """Add a task to a project."""
        try:
            project = await self.get_project(project_id)
            project.tasks[task_id] = Task(id=task_id, **task_data)
            logger.info("Added task %s to project %s", task_id, project_id)
        except Exception as e:
            logger.exception("Error adding task to project: %s", e)
            msg = f"Error adding task to project: {e!s}"
            raise AIVillageException(msg) from e

    async def get_project_tasks(self, project_id: str) -> list[Task]:
        """Get all tasks for a specific project."""
        try:
            project = await self.get_project(project_id)
            return list(project.tasks.values())
        except Exception as e:
            logger.exception("Error getting project tasks: %s", e)
            msg = f"Error getting project tasks: {e!s}"
            raise AIVillageException(msg) from e

    async def add_resources_to_project(
        self, 
        project_id: str, 
        resources: dict[str, Any]
    ) -> None:
        """Add resources to a project."""
        try:
            project = await self.get_project(project_id)
            project.resources.update(resources)
            logger.info("Added resources to project %s: %s", project_id, list(resources.keys()))
        except Exception as e:
            logger.exception("Error adding resources to project: %s", e)
            msg = f"Error adding resources to project: {e!s}"
            raise AIVillageException(msg) from e

    async def get_project_status(self, project_id: str) -> dict[str, Any]:
        """Get comprehensive project status."""
        try:
            project = await self.get_project(project_id)
            
            tasks = [
                {
                    "task_id": task.id,
                    "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                    "description": task.description,
                }
                for task in project.tasks.values()
            ]
            
            return {
                "project_id": project_id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "progress": project.progress,
                "tasks": tasks,
                "resources": project.resources,
                "task_count": len(tasks),
            }
        except Exception as e:
            logger.exception("Error getting project status: %s", e)
            msg = f"Error getting project status: {e!s}"
            raise AIVillageException(msg) from e

    async def delete_project(self, project_id: str) -> None:
        """Delete a project."""
        try:
            if project_id not in self._projects:
                msg = f"Project with ID {project_id} not found"
                raise AIVillageException(msg)
                
            del self._projects[project_id]
            logger.info("Deleted project: %s", project_id)
        except Exception as e:
            logger.exception("Error deleting project: %s", e)
            msg = f"Error deleting project: {e!s}"
            raise AIVillageException(msg) from e

    def calculate_project_progress(self, project: Project) -> float:
        """Calculate project progress based on task completion."""
        try:
            if not project.tasks:
                return 0.0
                
            completed_tasks = sum(
                1 for task in project.tasks.values()
                if hasattr(task, 'status') and str(task.status) == 'TaskStatus.COMPLETED'
            )
            
            return completed_tasks / len(project.tasks)
        except Exception as e:
            logger.exception("Error calculating project progress: %s", e)
            return 0.0

    def get_project_statistics(self) -> dict[str, Any]:
        """Get overall project statistics."""
        try:
            total_projects = len(self._projects)
            if total_projects == 0:
                return {"total_projects": 0, "average_progress": 0.0}
                
            total_progress = sum(project.progress for project in self._projects.values())
            average_progress = total_progress / total_projects
            
            status_counts = {}
            for project in self._projects.values():
                status_counts[project.status] = status_counts.get(project.status, 0) + 1
                
            return {
                "total_projects": total_projects,
                "average_progress": average_progress,
                "status_distribution": status_counts,
            }
        except Exception as e:
            logger.exception("Error getting project statistics: %s", e)
            return {"error": str(e)}