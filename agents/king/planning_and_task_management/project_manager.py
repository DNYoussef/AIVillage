import uuid
from typing import Dict, Any, List
from ...utils.exceptions import AIVillageException
from ...utils.logger import logger

class ProjectManager:
    def __init__(self):
        self.projects: Dict[str, Dict[str, Any]] = {}

    async def create_project(self, project_data: Dict[str, Any]) -> str:
        try:
            project_id = str(uuid.uuid4())
            self.projects[project_id] = {
                "id": project_id,
                "status": "initialized",
                "progress": 0.0,
                **project_data
            }
            logger.info(f"Created project with ID: {project_id}")
            return project_id
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            raise AIVillageException(f"Error creating project: {str(e)}") from e

    async def get_all_projects(self) -> Dict[str, Dict[str, Any]]:
        return self.projects

    async def get_project(self, project_id: str) -> Dict[str, Any]:
        try:
            project = self.projects.get(project_id)
            if not project:
                raise AIVillageException(f"Project with ID {project_id} not found")
            return project
        except Exception as e:
            logger.error(f"Error retrieving project {project_id}: {str(e)}")
            raise AIVillageException(f"Error retrieving project {project_id}: {str(e)}") from e

    async def update_project_status(self, project_id: str, status: str = None, progress: float = None):
        try:
            project = await self.get_project(project_id)
            if status:
                project['status'] = status
            if progress is not None:
                project['progress'] = progress
            logger.info(f"Updated project {project_id} - Status: {status}, Progress: {progress}")
        except Exception as e:
            logger.error(f"Error updating project {project_id}: {str(e)}")
            raise AIVillageException(f"Error updating project {project_id}: {str(e)}") from e

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        try:
            project = await self.get_project(project_id)
            return {
                "status": project['status'],
                "progress": project['progress']
            }
        except Exception as e:
            logger.error(f"Error getting status for project {project_id}: {str(e)}")
            raise AIVillageException(f"Error getting status for project {project_id}: {str(e)}") from e

    async def add_task_to_project(self, project_id: str, task_id: str, task_data: Dict[str, Any]):
        try:
            project = await self.get_project(project_id)
            if 'tasks' not in project:
                project['tasks'] = {}
            project['tasks'][task_id] = task_data
            logger.info(f"Added task {task_id} to project {project_id}")
        except Exception as e:
            logger.error(f"Error adding task to project {project_id}: {str(e)}")
            raise AIVillageException(f"Error adding task to project {project_id}: {str(e)}") from e

    async def update_task_in_project(self, project_id: str, task_id: str, task_data: Dict[str, Any]):
        try:
            project = await self.get_project(project_id)
            if 'tasks' not in project or task_id not in project['tasks']:
                raise AIVillageException(f"Task {task_id} not found in project {project_id}")
            project['tasks'][task_id].update(task_data)
            logger.info(f"Updated task {task_id} in project {project_id}")
        except Exception as e:
            logger.error(f"Error updating task in project {project_id}: {str(e)}")
            raise AIVillageException(f"Error updating task in project {project_id}: {str(e)}") from e

    async def get_project_tasks(self, project_id: str) -> List[Dict[str, Any]]:
        try:
            project = await self.get_project(project_id)
            return list(project.get('tasks', {}).values())
        except Exception as e:
            logger.error(f"Error getting tasks for project {project_id}: {str(e)}")
            raise AIVillageException(f"Error getting tasks for project {project_id}: {str(e)}") from e

    async def delete_project(self, project_id: str):
        try:
            if project_id not in self.projects:
                raise AIVillageException(f"Project with ID {project_id} not found")
            del self.projects[project_id]
            logger.info(f"Deleted project with ID: {project_id}")
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {str(e)}")
            raise AIVillageException(f"Error deleting project {project_id}: {str(e)}") from e