import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
import logging
from typing import Any
import uuid

from core.error_handling import (
    AIVillageException,
    Message,
    MessageType,
    Priority,
    StandardCommunicationProtocol,
)

from ..analytics.unified_analytics import UnifiedAnalytics

# Decision making utilities were moved under the `planning` package.
from ..planning.unified_decision_maker import UnifiedDecisionMaker
from .incentive_model import IncentiveModel
from .subgoal_generator import SubGoalGenerator
from .task import Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class Project:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: dict[str, Task] = field(default_factory=dict)
    status: str = "initialized"
    progress: float = 0.0
    resources: dict[str, Any] = field(default_factory=dict)


class UnifiedManagement:
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        decision_maker: UnifiedDecisionMaker,
        num_agents: int,
        num_actions: int,
    ):
        self.communication_protocol = communication_protocol
        self.decision_maker = decision_maker
        self.pending_tasks: deque[Task] = deque()
        self.ongoing_tasks: dict[str, Task] = {}
        self.completed_tasks: list[Task] = []
        self.projects: dict[str, Project] = {}
        self.incentive_model = IncentiveModel(num_agents, num_actions)
        self.agent_performance: dict[str, float] = {}
        self.available_agents: list[str] = []
        self.subgoal_generator = SubGoalGenerator()
        self.unified_analytics = UnifiedAnalytics()
        self.batch_size = 5

    async def create_task(
        self,
        description: str,
        agent: str,
        priority: int = 1,
        deadline: str | None = None,
        project_id: str | None = None,
    ) -> Task:
        try:
            task = Task(
                description=description,
                assigned_agents=[agent],
                priority=priority,
                deadline=deadline,
            )
            self.pending_tasks.append(task)
            logger.info(f"Created task: {task.id} for agent: {agent}")

            if project_id:
                await self.add_task_to_project(
                    project_id, task.id, {"description": description, "agent": agent}
                )

            return task
        except Exception as e:
            logger.exception(f"Error creating task: {e!s}")
            raise AIVillageException(f"Error creating task: {e!s}")

    async def create_complex_task(
        self, description: str, context: dict[str, Any]
    ) -> list[Task]:
        try:
            subgoals = await self.subgoal_generator.generate_subgoals(
                description, context
            )
            tasks = []
            for subgoal in subgoals:
                agent = await self._select_best_agent_for_task(subgoal)
                task = await self.create_task(subgoal, agent)
                tasks.append(task)
            return tasks
        except Exception as e:
            logger.exception(f"Error creating complex task: {e!s}")
            raise AIVillageException(f"Error creating complex task: {e!s}")

    async def _select_best_agent_for_task(self, task_description: str) -> str:
        try:
            # Use the decision maker to select the best agent
            decision = await self.decision_maker.make_decision(
                f"Select the best agent for task: {task_description}", 0.5
            )
            return decision.get(
                "best_alternative",
                self.available_agents[0] if self.available_agents else "default_agent",
            )
        except Exception as e:
            logger.exception(f"Error selecting best agent for task: {e!s}")
            raise AIVillageException(f"Error selecting best agent for task: {e!s}")

    async def assign_task(self, task: Task):
        try:
            self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
            agent = task.assigned_agents[0]
            incentive = self.incentive_model.calculate_incentive(
                {"assigned_agent": agent, "task_id": task.id}, self.agent_performance
            )
            await self.notify_agent_with_incentive(agent, task, incentive["incentive"])
        except Exception as e:
            logger.exception(f"Error assigning task: {e!s}")
            raise AIVillageException(f"Error assigning task: {e!s}")

    async def notify_agent_with_incentive(
        self, agent: str, task: Task, incentive: float
    ):
        try:
            message = Message(
                type=MessageType.TASK,
                sender="UnifiedManagement",
                receiver=agent,
                content={
                    "task_id": task.id,
                    "description": task.description,
                    "incentive": incentive,
                },
                priority=Priority.MEDIUM,
            )
            await self.communication_protocol.send_message(message)
        except Exception as e:
            logger.exception(f"Error notifying agent with incentive: {e!s}")
            raise AIVillageException(f"Error notifying agent with incentive: {e!s}")

    async def complete_task(self, task_id: str, result: Any):
        try:
            if task_id not in self.ongoing_tasks:
                raise AIVillageException(f"Task {task_id} not found in ongoing tasks")
            task = self.ongoing_tasks[task_id]
            updated_task = task.update_status(TaskStatus.COMPLETED).update_result(
                result
            )
            self.completed_tasks.append(updated_task)
            del self.ongoing_tasks[task_id]
            await self.update_dependent_tasks(updated_task)

            agent = task.assigned_agents[0]
            self.incentive_model.update(
                {"assigned_agent": agent, "task_id": task_id}, result
            )
            self.incentive_model.update_agent_performance(
                self.agent_performance,
                agent,
                result,
                self.unified_analytics,
            )

            completion_time = updated_task.completed_at - updated_task.created_at
            success = result.get("success", False)
            self.unified_analytics.record_task_completion(
                task_id, completion_time, success
            )

            logger.info(f"Completed task: {task_id}")

            # Update project status if the task belongs to a project
            for project in self.projects.values():
                if task_id in project.tasks:
                    project.tasks[task_id] = updated_task
                    await self.update_project_status(project.id)
                    break
        except Exception as e:
            logger.exception(f"Error completing task: {e!s}")
            raise AIVillageException(f"Error completing task: {e!s}")

    async def update_dependent_tasks(self, completed_task: Task):
        try:
            for task in list(self.pending_tasks):
                if completed_task.id in task.dependencies:
                    task.dependencies.remove(completed_task.id)
                    if not task.dependencies:
                        self.pending_tasks.remove(task)
                        await self.assign_task(task)
        except Exception as e:
            logger.exception(f"Error updating dependent tasks: {e!s}")
            raise AIVillageException(f"Error updating dependent tasks: {e!s}")

    async def create_project(self, name: str, description: str) -> str:
        try:
            project_id = str(uuid.uuid4())
            self.projects[project_id] = Project(
                id=project_id, name=name, description=description
            )
            logger.info(f"Created project: {project_id}")
            return project_id
        except Exception as e:
            logger.exception(f"Error creating project: {e!s}")
            raise AIVillageException(f"Error creating project: {e!s}")

    async def get_all_projects(self) -> dict[str, Project]:
        return self.projects

    async def get_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if not project:
            raise AIVillageException(f"Project with ID {project_id} not found")
        return project

    async def update_project_status(
        self, project_id: str, status: str = None, progress: float = None
    ):
        try:
            project = await self.get_project(project_id)
            if status:
                project.status = status
            if progress is not None:
                project.progress = progress
            logger.info(
                f"Updated project {project_id} - Status: {status}, Progress: {progress}"
            )
        except Exception as e:
            logger.exception(f"Error updating project status: {e!s}")
            raise AIVillageException(f"Error updating project status: {e!s}")

    async def add_task_to_project(
        self, project_id: str, task_id: str, task_data: dict[str, Any]
    ):
        try:
            project = await self.get_project(project_id)
            project.tasks[task_id] = Task(id=task_id, **task_data)
            logger.info(f"Added task {task_id} to project {project_id}")
        except Exception as e:
            logger.exception(f"Error adding task to project: {e!s}")
            raise AIVillageException(f"Error adding task to project: {e!s}")

    async def get_project_tasks(self, project_id: str) -> list[Task]:
        try:
            project = await self.get_project(project_id)
            return list(project.tasks.values())
        except Exception as e:
            logger.exception(f"Error getting project tasks: {e!s}")
            raise AIVillageException(f"Error getting project tasks: {e!s}")

    async def add_resources_to_project(
        self, project_id: str, resources: dict[str, Any]
    ):
        try:
            project = await self.get_project(project_id)
            project.resources.update(resources)
            logger.info(f"Added resources to project {project_id}")
        except Exception as e:
            logger.exception(f"Error adding resources to project: {e!s}")
            raise AIVillageException(f"Error adding resources to project: {e!s}")

    def update_agent_list(self, agent_list: list[str]):
        try:
            self.available_agents = agent_list
            logger.info(f"Updated available agents: {self.available_agents}")
        except Exception as e:
            logger.exception(f"Error updating agent list: {e!s}")
            raise AIVillageException(f"Error updating agent list: {e!s}")

    async def process_task_batch(self):
        try:
            batch = []
            while len(batch) < self.batch_size and self.pending_tasks:
                batch.append(self.pending_tasks.popleft())

            if not batch:
                return

            results = await asyncio.gather(
                *[self.process_single_task(task) for task in batch]
            )

            for task, result in zip(batch, results, strict=False):
                await self.complete_task(task.id, result)
        except Exception as e:
            logger.exception(f"Error processing task batch: {e!s}")
            raise AIVillageException(f"Error processing task batch: {e!s}")

    async def process_single_task(self, task: Task) -> Any:
        try:
            agent = task.assigned_agents[0]
            return await self.communication_protocol.send_and_wait(
                Message(
                    type=MessageType.TASK,
                    sender="UnifiedManagement",
                    receiver=agent,
                    content={"task_id": task.id, "description": task.description},
                )
            )
        except Exception as e:
            logger.exception(f"Error processing single task: {e!s}")
            raise AIVillageException(f"Error processing single task: {e!s}")

    async def start_batch_processing(self):
        try:
            while True:
                await self.process_task_batch()
                await asyncio.sleep(1)  # Adjust this delay as needed
        except Exception as e:
            logger.exception(f"Error in batch processing: {e!s}")
            raise AIVillageException(f"Error in batch processing: {e!s}")

    def set_batch_size(self, size: int):
        try:
            self.batch_size = size
            logger.info(f"Set batch size to {size}")
        except Exception as e:
            logger.exception(f"Error setting batch size: {e!s}")
            raise AIVillageException(f"Error setting batch size: {e!s}")

    async def get_task_status(self, task_id: str) -> TaskStatus:
        try:
            if task_id in self.ongoing_tasks:
                return self.ongoing_tasks[task_id].status
            if any(task.id == task_id for task in self.completed_tasks):
                return TaskStatus.COMPLETED
            return TaskStatus.PENDING
        except Exception as e:
            logger.exception(f"Error getting task status: {e!s}")
            raise AIVillageException(f"Error getting task status: {e!s}")

    async def get_project_status(self, project_id: str) -> dict[str, Any]:
        try:
            project = await self.get_project(project_id)
            tasks = [
                {
                    "task_id": task.id,
                    "status": task.status,
                    "description": task.description,
                }
                for task in project.tasks.values()
            ]
            return {
                "project_id": project_id,
                "name": project.name,
                "status": project.status,
                "progress": project.progress,
                "tasks": tasks,
            }
        except Exception as e:
            logger.exception(f"Error getting project status: {e!s}")
            raise AIVillageException(f"Error getting project status: {e!s}")

    async def save_state(self, filename: str):
        try:
            state = {
                "tasks": [
                    task.__dict__
                    for task in self.pending_tasks
                    + list(self.ongoing_tasks.values())
                    + self.completed_tasks
                ],
                "projects": {
                    pid: project.__dict__ for pid, project in self.projects.items()
                },
                "agent_performance": self.agent_performance,
                "available_agents": self.available_agents,
            }
            with open(filename, "w") as f:
                json.dump(state, f)
            logger.info(f"Saved state to {filename}")
        except Exception as e:
            logger.exception(f"Error saving state: {e!s}")
            raise AIVillageException(f"Error saving state: {e!s}")

    async def load_state(self, filename: str):
        try:
            with open(filename) as f:
                state = json.load(f)
            self.pending_tasks = deque(
                Task(**task)
                for task in state["tasks"]
                if task["status"] == TaskStatus.PENDING.value
            )
            self.ongoing_tasks = {
                task["id"]: Task(**task)
                for task in state["tasks"]
                if task["status"] == TaskStatus.IN_PROGRESS.value
            }
            self.completed_tasks = [
                Task(**task)
                for task in state["tasks"]
                if task["status"] == TaskStatus.COMPLETED.value
            ]
            self.projects = {
                pid: Project(**project) for pid, project in state["projects"].items()
            }
            self.agent_performance = state["agent_performance"]
            self.available_agents = state["available_agents"]
            logger.info(f"Loaded state from {filename}")
        except Exception as e:
            logger.exception(f"Error loading state: {e!s}")
            raise AIVillageException(f"Error loading state: {e!s}")

    async def introspect(self) -> dict[str, Any]:
        try:
            return {
                "pending_tasks": len(self.pending_tasks),
                "ongoing_tasks": len(self.ongoing_tasks),
                "completed_tasks": len(self.completed_tasks),
                "projects": len(self.projects),
                "available_agents": self.available_agents,
                "agent_performance": self.agent_performance,
                "batch_size": self.batch_size,
                "analytics_report": self.unified_analytics.generate_summary_report(),
            }
        except Exception as e:
            logger.exception(f"Error in introspection: {e!s}")
            raise AIVillageException(f"Error in introspection: {e!s}")


if __name__ == "__main__":
    raise SystemExit("Run 'agents/orchestration.py' to start the task manager.")

UnifiedTaskManager = UnifiedManagement
