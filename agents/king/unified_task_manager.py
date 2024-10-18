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
from exceptions import AIVillageException
from .incentive_model import IncentiveModel

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

    async def create_task(self, description: str, agent: str, priority: int = 1, deadline: Optional[str] = None) -> Task:
        task = Task(description=description, assigned_agents=[agent], priority=priority, deadline=deadline)
        self.pending_tasks.append(task)
        logger.info(f"Created task: {task.id} for agent: {agent}")
        return task

    async def get_next_task(self) -> Optional[Task]:
        return self.pending_tasks.popleft() if self.pending_tasks else None

    async def assign_task(self, task: Task):
        try:
            self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
            agent = task.assigned_agents[0]  # We now expect only one agent per task
            incentive = self.incentive_model.calculate_incentive({'assigned_agent': agent, 'task_id': task.id}, self.agent_performance)
            await self.notify_agent_with_incentive(agent, task, incentive['incentive'])
        except Exception as e:
            logger.error(f"Error assigning task {task.id}: {str(e)}")
            raise AIVillageException(f"Error assigning task: {str(e)}")

    async def notify_agent_with_incentive(self, agent: str, task: Task, incentive: float):
        try:
            message = Message(
                type=MessageType.TASK,
                sender="UnifiedTaskManager",
                receiver=agent,
                content={"task_id": task.id, "description": task.description, "incentive": incentive},
                priority=Priority.MEDIUM
            )
            await self.communication_protocol.send_message(message)
        except Exception as e:
            logger.error(f"Error notifying agent {agent} for task {task.id}: {str(e)}")
            raise AIVillageException(f"Error notifying agent: {str(e)}")

    async def complete_task(self, task_id: str, result: Any):
        try:
            if task_id not in self.ongoing_tasks:
                raise AIVillageException(f"Task {task_id} not found in ongoing tasks")
            task = self.ongoing_tasks[task_id]
            updated_task = task.update_status(TaskStatus.COMPLETED).update_result(result)
            self.completed_tasks.append(updated_task)
            del self.ongoing_tasks[task_id]
            await self.update_dependent_tasks(updated_task)
            
            # Update incentive model and agent performance
            agent = task.assigned_agents[0]
            self.incentive_model.update({'assigned_agent': agent, 'task_id': task_id}, result)
            self.update_agent_performance(agent, result)
            logger.info(f"Completed task: {task_id}")
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {str(e)}")
            raise AIVillageException(f"Error completing task: {str(e)}")

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
        logger.info(f"Updated performance for agent {agent}: {self.agent_performance[agent]}")

    async def create_workflow(self, name: str, tasks: List[Task], dependencies: Dict[str, List[str]]) -> Workflow:
        workflow = Workflow(id=str(uuid.uuid4()), name=name, tasks=tasks, dependencies=dependencies)
        self.workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {workflow.id}")
        return workflow

    async def execute_workflow(self, workflow: Workflow):
        for task in workflow.tasks:
            if not workflow.dependencies.get(task.id):
                await self.assign_task(task)
        logger.info(f"Started execution of workflow: {workflow.id}")

    async def get_task_status(self, task_id: str) -> TaskStatus:
        if task_id in self.ongoing_tasks:
            return self.ongoing_tasks[task_id].status
        elif any(task.id == task_id for task in self.completed_tasks):
            return TaskStatus.COMPLETED
        else:
            return TaskStatus.PENDING

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        if workflow_id not in self.workflows:
            raise AIVillageException(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        tasks = [{"task_id": task.id, "status": await self.get_task_status(task.id), "description": task.description} for task in workflow.tasks]
        return {"workflow_id": workflow_id, "name": workflow.name, "tasks": tasks}

    async def monitor_tasks(self):
        while True:
            for task_id, task in list(self.ongoing_tasks.items()):
                updated_status = await self.check_task_progress(task)
                if updated_status != task.status:
                    self.ongoing_tasks[task_id] = task.update_status(updated_status)
                    if updated_status == TaskStatus.COMPLETED:
                        await self.complete_task(task_id, task.result)
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

    async def execute_langroid_task(self, langroid_task: LangroidTask):
        result = await langroid_task.run()
        task_id = langroid_task.task_id
        await self.complete_task(task_id, result)

    def update_agent_list(self, agent_list: List[str]):
        self.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

    async def save_models(self, path: str):
        try:
            self.incentive_model.save(f"{path}/incentive_model.pt")
            logger.info(f"Saved models to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise AIVillageException(f"Error saving models: {str(e)}")

    async def load_models(self, path: str):
        try:
            self.incentive_model.load(f"{path}/incentive_model.pt")
            logger.info(f"Loaded models from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise AIVillageException(f"Error loading models: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "pending_tasks": len(self.pending_tasks),
            "ongoing_tasks": len(self.ongoing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "workflows": len(self.workflows),
            "available_agents": self.available_agents,
            "agent_performance": self.agent_performance
        }
