import logging
from typing import List, Dict, Any, Optional
from collections import deque
import asyncio
import uuid
from agents.king.task_management.task import Task, TaskStatus
from agents.king.task_management.workflow import Workflow
from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from langroid.agent.task import Task as LangroidTask
from langroid.agent.chat_agent import ChatAgent
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
from .incentive_model import IncentiveModel
from .subgoal_generator import SubGoalGenerator
from .unified_analytics import UnifiedAnalytics

logger = logging.getLogger(__name__)

class UnifiedTaskManager:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, num_agents: int, num_actions: int):
        self.communication_protocol = communication_protocol
        self.pending_tasks: deque[Task] = deque()
        self.ongoing_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.workflows: Dict[str, Workflow] = {}
        self.incentive_model = IncentiveModel(num_agents, num_actions)
        self.agent_performance: Dict[str, float] = {}
        self.available_agents: List[str] = []
        self.subgoal_generator = SubGoalGenerator()
        self.batch_size = 5  # Default batch size for batch processing
        self.unified_analytics = UnifiedAnalytics()

    @error_handler.handle_error
    async def create_task(self, description: str, agent: str, priority: int = 1, deadline: Optional[str] = None) -> Task:
        task = Task(description=description, assigned_agents=[agent], priority=priority, deadline=deadline)
        self.pending_tasks.append(task)
        logger.info(f"Created task: {task.id} for agent: {agent}")
        return task

    @error_handler.handle_error
    async def create_complex_task(self, description: str, context: Dict[str, Any]) -> List[Task]:
        subgoals = await self.subgoal_generator.generate_subgoals(description, context)
        tasks = []
        for subgoal in subgoals:
            agent = self._select_best_agent_for_task(subgoal)
            task = await self.create_task(subgoal, agent)
            tasks.append(task)
        return tasks

    def _select_best_agent_for_task(self, task_description: str) -> str:
        # This is a placeholder. In a real implementation, you would use more sophisticated logic
        # to select the best agent based on task requirements and agent capabilities
        return self.available_agents[0] if self.available_agents else "default_agent"

    @error_handler.handle_error
    async def get_next_task(self) -> Optional[Task]:
        return self.pending_tasks.popleft() if self.pending_tasks else None

    @error_handler.handle_error
    async def assign_task(self, task: Task):
        self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
        agent = task.assigned_agents[0]  # We now expect only one agent per task
        incentive = self.incentive_model.calculate_incentive({'assigned_agent': agent, 'task_id': task.id}, self.agent_performance)
        await self.notify_agent_with_incentive(agent, task, incentive['incentive'])

    @error_handler.handle_error
    async def notify_agent_with_incentive(self, agent: str, task: Task, incentive: float):
        message = Message(
            type=MessageType.TASK,
            sender="UnifiedTaskManager",
            receiver=agent,
            content={"task_id": task.id, "description": task.description, "incentive": incentive},
            priority=Priority.MEDIUM
        )
        await self.communication_protocol.send_message(message)

    @error_handler.handle_error
    async def complete_task(self, task_id: str, result: Any):
        if task_id not in self.ongoing_tasks:
            raise AIVillageException(f"Task {task_id} not found in ongoing tasks")
        task = self.ongoing_tasks[task_id]
        updated_task = task.update_status(TaskStatus.COMPLETED).update_result(result)
        self.completed_tasks.append(updated_task)
        del self.ongoing_tasks[task_id]
        await self.update_dependent_tasks(updated_task)
        
        agent = task.assigned_agents[0]
        self.incentive_model.update({'assigned_agent': agent, 'task_id': task_id}, result)
        self.update_agent_performance(agent, result)
        
        # Record analytics
        completion_time = (updated_task.completed_at - updated_task.created_at).total_seconds()
        success = result.get('success', False)
        self.unified_analytics.record_task_completion(task_id, completion_time, success)
        
        logger.info(f"Completed task: {task_id}")

    @error_handler.handle_error
    async def update_dependent_tasks(self, completed_task: Task):
        for workflow in self.workflows.values():
            for task in workflow.tasks:
                if completed_task.id in task.dependencies:
                    if all(dep_id in [t.id for t in self.completed_tasks] for dep_id in task.dependencies):
                        await self.assign_task(task)

    def update_agent_performance(self, agent: str, result: Any):
        success = result.get('success', False)
        current_performance = self.agent_performance.get(agent, 1.0)
        if success:
            self.agent_performance[agent] = min(current_performance * 1.1, 2.0)  # Cap at 2.0
        else:
            self.agent_performance[agent] = max(current_performance * 0.9, 0.5)  # Floor at 0.5
        self.unified_analytics.update_performance_history(self.agent_performance[agent])
        logger.info(f"Updated performance for agent {agent}: {self.agent_performance[agent]}")

    @error_handler.handle_error
    async def create_workflow(self, name: str, tasks: List[Task], dependencies: Dict[str, List[str]]) -> Workflow:
        workflow = Workflow(id=str(uuid.uuid4()), name=name, tasks=tasks, dependencies=dependencies)
        self.workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {workflow.id}")
        return workflow

    @error_handler.handle_error
    async def execute_workflow(self, workflow: Workflow):
        for task in workflow.tasks:
            if not workflow.dependencies.get(task.id):
                await self.assign_task(task)
        logger.info(f"Started execution of workflow: {workflow.id}")

    @error_handler.handle_error
    async def get_task_status(self, task_id: str) -> TaskStatus:
        if task_id in self.ongoing_tasks:
            return self.ongoing_tasks[task_id].status
        elif any(task.id == task_id for task in self.completed_tasks):
            return TaskStatus.COMPLETED
        else:
            return TaskStatus.PENDING

    @error_handler.handle_error
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        if workflow_id not in self.workflows:
            raise AIVillageException(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        tasks = [{"task_id": task.id, "status": await self.get_task_status(task.id), "description": task.description} for task in workflow.tasks]
        return {"workflow_id": workflow_id, "name": workflow.name, "tasks": tasks}

    @safe_execute
    async def monitor_tasks(self):
        while True:
            for task_id, task in list(self.ongoing_tasks.items()):
                updated_status = await self.check_task_progress(task)
                if updated_status != task.status:
                    self.ongoing_tasks[task_id] = task.update_status(updated_status)
                    if updated_status == TaskStatus.COMPLETED:
                        await self.complete_task(task_id, task.result)
            
            # Record system efficiency metrics
            self.unified_analytics.record_metric("pending_tasks", len(self.pending_tasks))
            self.unified_analytics.record_metric("ongoing_tasks", len(self.ongoing_tasks))
            self.unified_analytics.record_metric("completed_tasks", len(self.completed_tasks))
            
            await asyncio.sleep(10)  # Check every 10 seconds

    async def check_task_progress(self, task: Task) -> TaskStatus:
        # This is a placeholder. In a real implementation, you might query the assigned agents or an external system
        return task.status

    def to_langroid_task(self, task: Task, agent: ChatAgent) -> LangroidTask:
        return LangroidTask(
            agent,
            name=task.description,
            task_id=task.id,
            priority=task.priority
        )

    @error_handler.handle_error
    async def execute_langroid_task(self, langroid_task: LangroidTask):
        result = await langroid_task.run()
        task_id = langroid_task.task_id
        await self.complete_task(task_id, result)

    def update_agent_list(self, agent_list: List[str]):
        self.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

    @error_handler.handle_error
    async def save_models(self, path: str):
        self.incentive_model.save(f"{path}/incentive_model.pt")
        logger.info(f"Saved models to {path}")

    @error_handler.handle_error
    async def load_models(self, path: str):
        self.incentive_model.load(f"{path}/incentive_model.pt")
        logger.info(f"Loaded models from {path}")

    @error_handler.handle_error
    async def introspect(self) -> Dict[str, Any]:
        return {
            "pending_tasks": len(self.pending_tasks),
            "ongoing_tasks": len(self.ongoing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "workflows": len(self.workflows),
            "available_agents": self.available_agents,
            "agent_performance": self.agent_performance,
            "analytics_report": self.unified_analytics.generate_summary_report()
        }

    @error_handler.handle_error
    async def process_task_batch(self):
        batch = []
        while len(batch) < self.batch_size and self.pending_tasks:
            batch.append(await self.get_next_task())
        
        if not batch:
            return
        
        results = await asyncio.gather(*[self.process_single_task(task) for task in batch])
        
        for task, result in zip(batch, results):
            await self.complete_task(task.id, result)

    @error_handler.handle_error
    async def process_single_task(self, task: Task) -> Any:
        # This is a placeholder. In a real implementation, you would have more sophisticated
        # logic to process the task, possibly involving multiple agents or external systems
        agent = task.assigned_agents[0]
        return await self.communication_protocol.send_and_wait(Message(
            type=MessageType.TASK,
            sender="UnifiedTaskManager",
            receiver=agent,
            content={"task_id": task.id, "description": task.description}
        ))

    @safe_execute
    async def start_batch_processing(self):
        while True:
            await self.process_task_batch()
            await asyncio.sleep(1)  # Adjust this delay as needed

    def set_batch_size(self, size: int):
        self.batch_size = size
        logger.info(f"Set batch size to {size}")

    @safe_execute
    async def run_analytics(self):
        while True:
            report = self.unified_analytics.generate_summary_report()
            logger.info(f"Analytics Report: {report}")
            await asyncio.sleep(3600)  # Run analytics every hour

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
        
        import json
from typing import Dict, Any, List
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from agents.utils.exceptions import AIVillageException
from .incentive_model import IncentiveModel
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    created_at: float = field(default_factory=asyncio.get_event_loop().time)
    completed_at: Optional[float] = None

    def update_status(self, new_status: TaskStatus):
        self.status = new_status
        if new_status == TaskStatus.COMPLETED:
            self.completed_at = asyncio.get_event_loop().time()
        return self

    def update_result(self, result: Any):
        self.result = result
        return self

@dataclass
class Project:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: str = "initialized"
    progress: float = 0.0
    resources: Dict[str, Any] = field(default_factory=dict)

class UnifiedManagement:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, num_agents: int, num_actions: int):
        self.communication_protocol = communication_protocol
        self.pending_tasks: deque[Task] = deque()
        self.ongoing_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.projects: Dict[str, Project] = {}
        self.incentive_model = IncentiveModel(num_agents, num_actions)
        self.agent_performance: Dict[str, float] = {}
        self.available_agents: List[str] = []
        self.batch_size = 5

    async def create_task(self, description: str, agent: str, priority: int = 1, deadline: Optional[str] = None, project_id: Optional[str] = None) -> Task:
        task = Task(description=description, assigned_agents=[agent], priority=priority, deadline=deadline)
        self.pending_tasks.append(task)
        logger.info(f"Created task: {task.id} for agent: {agent}")

        if project_id:
            await self.add_task_to_project(project_id, task.id, {"description": description, "agent": agent})

        return task

    async def assign_task(self, task: Task):
        self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
        agent = task.assigned_agents[0]
        incentive = self.incentive_model.calculate_incentive({'assigned_agent': agent, 'task_id': task.id}, self.agent_performance)
        await self.notify_agent_with_incentive(agent, task, incentive['incentive'])

    async def notify_agent_with_incentive(self, agent: str, task: Task, incentive: float):
        message = Message(
            type=MessageType.TASK,
            sender="UnifiedManagement",
            receiver=agent,
            content={"task_id": task.id, "description": task.description, "incentive": incentive},
            priority=Priority.MEDIUM
        )
        await self.communication_protocol.send_message(message)

    async def complete_task(self, task_id: str, result: Any):
        if task_id not in self.ongoing_tasks:
            raise AIVillageException(f"Task {task_id} not found in ongoing tasks")
        task = self.ongoing_tasks[task_id]
        updated_task = task.update_status(TaskStatus.COMPLETED).update_result(result)
        self.completed_tasks.append(updated_task)
        del self.ongoing_tasks[task_id]
        await self.update_dependent_tasks(updated_task)
        
        agent = task.assigned_agents[0]
        self.incentive_model.update({'assigned_agent': agent, 'task_id': task_id}, result)
        self.update_agent_performance(agent, result)
        
        logger.info(f"Completed task: {task_id}")

        # Update project status if the task belongs to a project
        for project in self.projects.values():
            if task_id in project.tasks:
                project.tasks[task_id] = updated_task
                await self.update_project_status(project.id)
                break

    async def update_dependent_tasks(self, completed_task: Task):
        for task in self.pending_tasks:
            if completed_task.id in task.dependencies:
                task.dependencies.remove(completed_task.id)
                if not task.dependencies:
                    await self.assign_task(task)

    def update_agent_performance(self, agent: str, result: Any):
        success = result.get('success', False)
        current_performance = self.agent_performance.get(agent, 1.0)
        if success:
            self.agent_performance[agent] = min(current_performance * 1.1, 2.0)  # Cap at 2.0
        else:
            self.agent_performance[agent] = max(current_performance * 0.9, 0.5)  # Floor at 0.5
        logger.info(f"Updated performance for agent {agent}: {self.agent_performance[agent]}")

    async def create_project(self, name: str, description: str) -> str:
        project_id = str(uuid.uuid4())
        self.projects[project_id] = Project(id=project_id, name=name, description=description)
        logger.info(f"Created project: {project_id}")
        return project_id

    async def get_all_projects(self) -> Dict[str, Project]:
        return self.projects

    async def get_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if not project:
            raise AIVillageException(f"Project with ID {project_id} not found")
        return project

    async def update_project_status(self, project_id: str, status: str = None, progress: float = None):
        project = await self.get_project(project_id)
        if status:
            project.status = status
        if progress is not None:
            project.progress = progress
        logger.info(f"Updated project {project_id} - Status: {status}, Progress: {progress}")

    async def add_task_to_project(self, project_id: str, task_id: str, task_data: Dict[str, Any]):
        project = await self.get_project(project_id)
        project.tasks[task_id] = Task(id=task_id, **task_data)
        logger.info(f"Added task {task_id} to project {project_id}")

    async def get_project_tasks(self, project_id: str) -> List[Task]:
        project = await self.get_project(project_id)
        return list(project.tasks.values())

    async def add_resources_to_project(self, project_id: str, resources: Dict[str, Any]):
        project = await self.get_project(project_id)
        project.resources.update(resources)
        logger.info(f"Added resources to project {project_id}")

    def update_agent_list(self, agent_list: List[str]):
        self.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

    async def process_task_batch(self):
        batch = []
        while len(batch) < self.batch_size and self.pending_tasks:
            batch.append(self.pending_tasks.popleft())
        
        if not batch:
            return
        
        results = await asyncio.gather(*[self.process_single_task(task) for task in batch])
        
        for task, result in zip(batch, results):
            await self.complete_task(task.id, result)

    async def process_single_task(self, task: Task) -> Any:
        agent = task.assigned_agents[0]
        return await self.communication_protocol.send_and_wait(Message(
            type=MessageType.TASK,
            sender="UnifiedManagement",
            receiver=agent,
            content={"task_id": task.id, "description": task.description}
        ))

    async def start_batch_processing(self):
        while True:
            await self.process_task_batch()
            await asyncio.sleep(1)  # Adjust this delay as needed

    def set_batch_size(self, size: int):
        self.batch_size = size
        logger.info(f"Set batch size to {size}")

    async def get_task_status(self, task_id: str) -> TaskStatus:
        if task_id in self.ongoing_tasks:
            return self.ongoing_tasks[task_id].status
        elif any(task.id == task_id for task in self.completed_tasks):
            return TaskStatus.COMPLETED
        else:
            return TaskStatus.PENDING

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        project = await self.get_project(project_id)
        tasks = [{"task_id": task.id, "status": task.status, "description": task.description} for task in project.tasks.values()]
        return {
            "project_id": project_id,
            "name": project.name,
            "status": project.status,
            "progress": project.progress,
            "tasks": tasks
        }

    async def save_state(self, filename: str):

        state = {
            'tasks': self.tasks,
            'agents': self.agents,
            'projects': self.projects
        }
        with open(filename, 'w') as f:
            json.dump(state, f)
        logger.info(f"Saved state to {filename}")

    async def load_state(self, filename: str):
        with open(filename, 'r') as f:
            state = json.load(f)
        self.tasks = state['tasks']
        self.agents = state['agents']
        self.projects = state['projects']
        logger.info(f"Loaded state from {filename}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "pending_tasks": len(self.pending_tasks),
            "ongoing_tasks": len(self.ongoing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "projects": len(self.projects),
            "available_agents": self.available_agents,
            "agent_performance": self.agent_performance,
            "batch_size": self.batch_size
        }
