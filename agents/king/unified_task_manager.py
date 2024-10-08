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
from agents.king.utils.logger import logger

class UnifiedTaskManager:
    def __init__(self, communication_protocol: StandardCommunicationProtocol):
        self.communication_protocol = communication_protocol
        self.pending_tasks: deque[Task] = deque()
        self.ongoing_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.workflows: Dict[str, Workflow] = {}

    async def create_task(self, description: str, agents: List[str], priority: int = 1, deadline: Optional[str] = None) -> Task:
        task = Task(description=description, assigned_agents=agents, priority=priority, deadline=deadline)
        self.pending_tasks.append(task)
        logger.info(f"Created task: {task.id}")
        return task

    async def get_next_task(self) -> Optional[Task]:
        return self.pending_tasks.popleft() if self.pending_tasks else None

    async def assign_task(self, task: Task):
        self.ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
        for agent in task.assigned_agents:
            await self.notify_agent(agent, task)

    async def notify_agent(self, agent: str, task: Task):
        message = Message(
            type=MessageType.TASK,
            sender="UnifiedTaskManager",
            receiver=agent,
            content={"task_id": task.id, "description": task.description},
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

    async def update_dependent_tasks(self, completed_task: Task):
        for workflow in self.workflows.values():
            for task in workflow.tasks:
                if completed_task.id in task.dependencies:
                    if all(dep_id in [t.id for t in self.completed_tasks] for dep_id in task.dependencies):
                        await self.assign_task(task)

    async def create_workflow(self, name: str, tasks: List[Task], dependencies: Dict[str, List[str]]) -> Workflow:
        workflow = Workflow(id=str(uuid.uuid4()), name=name, tasks=tasks, dependencies=dependencies)
        self.workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {workflow.id}")
        return workflow

    async def execute_workflow(self, workflow: Workflow):
        for task in workflow.tasks:
            if not workflow.dependencies.get(task.id):
                await self.assign_task(task)

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
