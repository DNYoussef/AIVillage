"""
Task Assignment Service - Handles task assignment and agent selection.
Extracted from UnifiedManagement god class.
"""

import logging

from AIVillage.experimental.agents.agents.planning.unified_decision_maker import UnifiedDecisionMaker
from core.error_handling import AIVillageException, Message, MessageType, Priority, StandardCommunicationProtocol

from ..task import Task, TaskStatus
from ..interfaces.task_service_interfaces import IIncentiveService

logger = logging.getLogger(__name__)


class TaskAssignmentService:
    """Service responsible for task assignment and agent selection."""

    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        decision_maker: UnifiedDecisionMaker,
        incentive_service: IIncentiveService,
    ) -> None:
        """Initialize with dependencies."""
        self._communication_protocol = communication_protocol
        self._decision_maker = decision_maker
        self._incentive_service = incentive_service
        self._ongoing_tasks: dict[str, Task] = {}
        self._available_agents: list[str] = []

    async def assign_task(self, task: Task) -> None:
        """Assign a task to an agent with incentive calculation."""
        try:
            self._ongoing_tasks[task.id] = task.update_status(TaskStatus.IN_PROGRESS)
            agent = task.assigned_agents[0]

            incentive = await self._incentive_service.calculate_incentive(agent, task)
            await self._notify_agent_with_incentive(agent, task, incentive)

            logger.info("Assigned task %s to agent %s with incentive %f", task.id, agent, incentive)
        except Exception as e:
            logger.exception("Error assigning task: %s", e)
            msg = f"Error assigning task: {e!s}"
            raise AIVillageException(msg) from e

    async def select_best_agent_for_task(self, task_description: str) -> str:
        """Select the optimal agent for a given task."""
        try:
            if not self._available_agents:
                return "default_agent"

            decision = await self._decision_maker.make_decision(
                f"Select the best agent for task: {task_description}", 0.5
            )

            best_agent = decision.get("best_alternative", self._available_agents[0])

            logger.debug("Selected agent %s for task: %s", best_agent, task_description[:50])
            return best_agent
        except Exception as e:
            logger.exception("Error selecting best agent for task: %s", e)
            msg = f"Error selecting best agent for task: {e!s}"
            raise AIVillageException(msg) from e

    async def _notify_agent_with_incentive(self, agent: str, task: Task, incentive: float) -> None:
        """Send task notification to agent with incentive information."""
        try:
            message = Message(
                type=MessageType.TASK,
                sender="TaskAssignmentService",
                receiver=agent,
                content={
                    "task_id": task.id,
                    "description": task.description,
                    "incentive": incentive,
                },
                priority=Priority.MEDIUM,
            )
            await self._communication_protocol.send_message(message)
            logger.debug("Notified agent %s about task %s", agent, task.id)
        except Exception as e:
            logger.exception("Error notifying agent with incentive: %s", e)
            msg = f"Error notifying agent with incentive: {e!s}"
            raise AIVillageException(msg) from e

    def update_agent_list(self, agent_list: list[str]) -> None:
        """Update the list of available agents."""
        try:
            self._available_agents = agent_list.copy()
            logger.info("Updated available agents: %s", self._available_agents)
        except Exception as e:
            logger.exception("Error updating agent list: %s", e)
            msg = f"Error updating agent list: {e!s}"
            raise AIVillageException(msg) from e

    def get_ongoing_tasks(self) -> dict[str, Task]:
        """Get all ongoing tasks."""
        return self._ongoing_tasks.copy()

    def remove_ongoing_task(self, task_id: str) -> Task | None:
        """Remove and return an ongoing task by ID."""
        return self._ongoing_tasks.pop(task_id, None)

    def get_available_agents(self) -> list[str]:
        """Get list of available agents."""
        return self._available_agents.copy()
