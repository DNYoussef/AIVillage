import logging
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import uuid
from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from agents.utils.exceptions import AIVillageException
from .incentive_model import IncentiveModel
from .subgoal_generator import SubGoalGenerator
from ..analytics.unified_analytics import UnifiedAnalytics
# Decision making utilities were moved under the `planning` package.
from ..planning.unified_decision_maker import UnifiedDecisionMaker

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
    def __init__(self, communication_protocol: StandardCommunicationProtocol, decision_maker: UnifiedDecisionMaker, num_agents: int, num_actions: int):
        self.communication_protocol = communication_protocol
        self.decision_maker = decision_maker
        self.pending_tasks: deque[Task] = deque()
        self.ongoing_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.projects: Dict[str, Project] = {}
        self.incentive_model = IncentiveModel(num_agents, num_actions)
        self.agent_performance: Dict[str, float] = {}
        self.available_agents: List[str] = []
        self.subgoal_generator = SubGoalGenerator()
        self.unified_analytics = UnifiedAnalytics()
        self.batch_size = 5

    async def create_task(self, description: str, agent: str, priority: int = 1, deadline: Optional[str] = None, project_id: Optional[str] = None) -> Task:
        try:
            task = Task(description=description, assigned_agents=[agent], priority=priority, deadline=deadline)
            self.pending_tasks.append(task)
            logger.info(f"Created task: {task.id} for agent: {agent}")

            if project_id:
                await self.add_task_to_project(project_id, task.id, {"description": description, "agent": agent})

            return task
        except Exception as e:
            logger.exception(f"Error creating task: {str(e)}")
            raise AIVillageException(f"Error creating task: {str(e)}")

    async def create_complex_task(self, description: str, context: Dict[str, Any]) -> List[Task]:
        try:
            subgoals = await self.subgoal_generator.generate_subgoals(description, context)
            tasks = []
            for subgoal in subgoals:
                agent = await self._select_best_agent_for_task(subgoal)
                task = await self.create_task(subgoal, agent)
                tasks.append(task)
            return tasks
        except Exception as e:
            logger.exception(f"Error creating complex task: {str(e)}")
            raise AIVillageException(f"Error creating complex task: {str(e)}")

    async def _select_best_agent_for_task(self, task_description: str) -> str:
        try:
            # Use the decision maker to select the best agent
            decision = await self.decision_maker.make_decision(f"Select the best agent for task: {task_description}", 0.5)
            return decision.get('best_alternative', self.available_agents[0] if self.available_agents else "default_agent")
        except Exception as e:
            logger.exception(f"Error selecting best agent for task: {str(e)}")
            raise AIVillageException(f"Error selecting best agent for task: {str(e)}")

    async def assign_task(self, task: Task):
        try:
            self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
            agent = task.assigned_agents[0]
            incentive = self.incentive_model.calculate_incentive({'assigned_agent': agent, 'task_id': task.id}, self.agent_performance)
            await self.notify_agent_with_incentive(agent, task, incentive['incentive'])
        except Exception as e:
            logger.exception(f"Error assigning task: {str(e)}")
            raise AIVillageException(f"Error assigning task: {str(e)}")

    async def notify_agent_with_incentive(self, agent: str, task: Task, incentive: float):
        try:
            message = Message(
                type=MessageType.TASK,
                sender="UnifiedManagement",
                receiver=agent,
                content={"task_id": task.id, "description": task.description, "incentive": incentive},
                priority=Priority.MEDIUM
            )
            await self.communication_protocol.send_message(message)
        except Exception as e:
            logger.exception(f"Error notifying agent with incentive: {str(e)}")
            raise AIVillageException(f"Error notifying agent with incentive: {str(e)}")

    async def complete_task(self, task_id: str, result: Any):
        try:
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
            
            completion_time = (updated_task.completed_at - updated_task.created_at)
            success = result.get('success', False)
            self.unified_analytics.record_task_completion(task_id, completion_time, success)
            
            logger.info(f"Completed task: {task_id}")

            # Update project status if the task belongs to a project
            for project in self.projects.values():
                if task_id in project.tasks:
                    project.tasks[task_id] = updated_task
                    await self.update_project_status(project.id)
                    break
        except Exception as e:
            logger.exception(f"Error completing task: {str(e)}")
            raise AIVillageException(f"Error completing task: {str(e)}")

    async def update_dependent_tasks(self, completed_task: Task):
        try:
            for task in list(self.pending_tasks):
                if completed_task.id in task.dependencies:
                    task.dependencies.remove(completed_task.id)
                    if not task.dependencies:
                        self.pending_tasks.remove(task)
                        await self.assign_task(task)
        except Exception as e:
            logger.exception(f"Error updating dependent tasks: {str(e)}")
            raise AIVillageException(f"Error updating dependent tasks: {str(e)}")

    def update_agent_performance(self, agent: str, result: Any):
        try:
            success = result.get('success', False)
            current_performance = self.agent_performance.get(agent, 1.0)
            if success:
                self.agent_performance[agent] = min(current_performance * 1.1, 2.0)  # Cap at 2.0
            else:
                self.agent_performance[agent] = max(current_performance * 0.9, 0.5)  # Floor at 0.5
            self.unified_analytics.update_performance_history(self.agent_performance[agent])
            logger.info(f"Updated performance for agent {agent}: {self.agent_performance[agent]}")
        except Exception as e:
            logger.exception(f"Error updating agent performance: {str(e)}")
            raise AIVillageException(f"Error updating agent performance: {str(e)}")

    async def create_project(self, name: str, description: str) -> str:
        try:
            project_id = str(uuid.uuid4())
            self.projects[project_id] = Project(id=project_id, name=name, description=description)
            logger.info(f"Created project: {project_id}")
            return project_id
        except Exception as e:
            logger.exception(f"Error creating project: {str(e)}")
            raise AIVillageException(f"Error creating project: {str(e)}")

    async def get_all_projects(self) -> Dict[str, Project]:
        return self.projects

    async def get_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if not project:
            raise AIVillageException(f"Project with ID {project_id} not found")
        return project

    async def update_project_status(self, project_id: str, status: str = None, progress: float = None):
        try:
            project = await self.get_project(project_id)
            if status:
                project.status = status
            if progress is not None:
                project.progress = progress
            logger.info(f"Updated project {project_id} - Status: {status}, Progress: {progress}")
        except Exception as e:
            logger.exception(f"Error updating project status: {str(e)}")
            raise AIVillageException(f"Error updating project status: {str(e)}")

    async def add_task_to_project(self, project_id: str, task_id: str, task_data: Dict[str, Any]):
        try:
            project = await self.get_project(project_id)
            project.tasks[task_id] = Task(id=task_id, **task_data)
            logger.info(f"Added task {task_id} to project {project_id}")
        except Exception as e:
            logger.exception(f"Error adding task to project: {str(e)}")
            raise AIVillageException(f"Error adding task to project: {str(e)}")

    async def get_project_tasks(self, project_id: str) -> List[Task]:
        try:
            project = await self.get_project(project_id)
            return list(project.tasks.values())
        except Exception as e:
            logger.exception(f"Error getting project tasks: {str(e)}")
            raise AIVillageException(f"Error getting project tasks: {str(e)}")

    async def add_resources_to_project(self, project_id: str, resources: Dict[str, Any]):
        try:
            project = await self.get_project(project_id)
            project.resources.update(resources)
            logger.info(f"Added resources to project {project_id}")
        except Exception as e:
            logger.exception(f"Error adding resources to project: {str(e)}")
            raise AIVillageException(f"Error adding resources to project: {str(e)}")

    def update_agent_list(self, agent_list: List[str]):
        try:
            self.available_agents = agent_list
            logger.info(f"Updated available agents: {self.available_agents}")
        except Exception as e:
            logger.exception(f"Error updating agent list: {str(e)}")
            raise AIVillageException(f"Error updating agent list: {str(e)}")

    async def process_task_batch(self):
        try:
            batch = []
            while len(batch) < self.batch_size and self.pending_tasks:
                batch.append(self.pending_tasks.popleft())
            
            if not batch:
                return
            
            results = await asyncio.gather(*[self.process_single_task(task) for task in batch])
            
            for task, result in zip(batch, results):
                await self.complete_task(task.id, result)
        except Exception as e:
            logger.exception(f"Error processing task batch: {str(e)}")
            raise AIVillageException(f"Error processing task batch: {str(e)}")

    async def process_single_task(self, task: Task) -> Any:
        try:
            agent = task.assigned_agents[0]
            return await self.communication_protocol.send_and_wait(Message(
                type=MessageType.TASK,
                sender="UnifiedManagement",
                receiver=agent,
                content={"task_id": task.id, "description": task.description}
            ))
        except Exception as e:
            logger.exception(f"Error processing single task: {str(e)}")
            raise AIVillageException(f"Error processing single task: {str(e)}")

    async def start_batch_processing(self):
        try:
            while True:
                await self.process_task_batch()
                await asyncio.sleep(1)  # Adjust this delay as needed
        except Exception as e:
            logger.exception(f"Error in batch processing: {str(e)}")
            raise AIVillageException(f"Error in batch processing: {str(e)}")

    def set_batch_size(self, size: int):
        try:
            self.batch_size = size
            logger.info(f"Set batch size to {size}")
        except Exception as e:
            logger.exception(f"Error setting batch size: {str(e)}")
            raise AIVillageException(f"Error setting batch size: {str(e)}")

    async def get_task_status(self, task_id: str) -> TaskStatus:
        try:
            if task_id in self.ongoing_tasks:
                return self.ongoing_tasks[task_id].status
            elif any(task.id == task_id for task in self.completed_tasks):
                return TaskStatus.COMPLETED
            else:
                return TaskStatus.PENDING
        except Exception as e:
            logger.exception(f"Error getting task status: {str(e)}")
            raise AIVillageException(f"Error getting task status: {str(e)}")

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        try:
            project = await self.get_project(project_id)
            tasks = [{"task_id": task.id, "status": task.status, "description": task.description} for task in project.tasks.values()]
            return {
                "project_id": project_id,
                "name": project.name,
                "status": project.status,
                "progress": project.progress,
                "tasks": tasks
            }
        except Exception as e:
            logger.exception(f"Error getting project status: {str(e)}")
            raise AIVillageException(f"Error getting project status: {str(e)}")

    async def save_state(self, filename: str):
        try:
            state = {
                'tasks': [task.__dict__ for task in self.pending_tasks + list(self.ongoing_tasks.values()) + self.completed_tasks],
                'projects': {pid: project.__dict__ for pid, project in self.projects.items()},
                'agent_performance': self.agent_performance,
                'available_agents': self.available_agents
            }
            with open(filename, 'w') as f:
                json.dump(state, f)
            logger.info(f"Saved state to {filename}")
        except Exception as e:
            logger.exception(f"Error saving state: {str(e)}")
            raise AIVillageException(f"Error saving state: {str(e)}")

    async def load_state(self, filename: str):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            self.pending_tasks = deque(Task(**task) for task in state['tasks'] if task['status'] == TaskStatus.PENDING.value)
            self.ongoing_tasks = {task['id']: Task(**task) for task in state['tasks'] if task['status'] == TaskStatus.IN_PROGRESS.value}
            self.completed_tasks = [Task(**task) for task in state['tasks'] if task['status'] == TaskStatus.COMPLETED.value]
            self.projects = {pid: Project(**project) for pid, project in state['projects'].items()}
            self.agent_performance = state['agent_performance']
            self.available_agents = state['available_agents']
            logger.info(f"Loaded state from {filename}")
        except Exception as e:
            logger.exception(f"Error loading state: {str(e)}")
            raise AIVillageException(f"Error loading state: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        try:
            return {
                "pending_tasks": len(self.pending_tasks),
                "ongoing_tasks": len(self.ongoing_tasks),
                "completed_tasks": len(self.completed_tasks),
                "projects": len(self.projects),
                "available_agents": self.available_agents,
                "agent_performance": self.agent_performance,
                "batch_size": self.batch_size,
                "analytics_report": self.unified_analytics.generate_summary_report()
            }
        except Exception as e:
            logger.exception(f"Error in introspection: {str(e)}")
            raise AIVillageException(f"Error in introspection: {str(e)}")

if __name__ == "__main__":
    raise SystemExit(
        "Run 'agents/orchestration.py' to start the task manager.")
