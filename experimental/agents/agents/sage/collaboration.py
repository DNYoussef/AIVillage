import logging
from typing import Any

from agents.utils.task import Task as LangroidTask
from core.error_handling import Message, MessageType

logger = logging.getLogger(__name__)


class CollaborationManager:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.collaborating_agents = {}

    async def handle_collaboration_request(self, message: Message) -> None:
        try:
            collaboration_type = message.content.get("collaboration_type")
            if collaboration_type == "knowledge_sharing":
                await self.share_knowledge(message)
            elif collaboration_type == "task_delegation":
                await self.delegate_task(message)
            elif collaboration_type == "joint_reasoning":
                await self.perform_joint_reasoning(message)
            else:
                logger.warning(f"Unknown collaboration type: {collaboration_type}")
        except Exception as e:
            logger.exception(f"Error handling collaboration request: {e!s}")

    async def share_knowledge(self, message: Message) -> None:
        try:
            query = message.content.get("query")
            relevant_knowledge = await self.agent.query_rag(query)
            response = Message(
                type=MessageType.KNOWLEDGE_SHARE,
                sender=self.agent.name,
                receiver=message.sender,
                content=relevant_knowledge,
                parent_id=message.id,
            )
            await self.agent.communication_protocol.send_message(response)
        except Exception as e:
            logger.exception(f"Error sharing knowledge: {e!s}")

    async def delegate_task(self, message: Message) -> None:
        try:
            task_dict = message.content.get("task")
            langroid_task = LangroidTask(
                self.agent,
                task_dict.get("content"),
                task_dict.get("id", ""),
                task_dict.get("priority", 1),
            )
            langroid_task.type = task_dict.get("type", "general")
            result = await self.agent.execute_task(langroid_task)
            response = Message(
                type=MessageType.TASK_RESULT,
                sender=self.agent.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id,
            )
            await self.agent.communication_protocol.send_message(response)
        except Exception as e:
            logger.exception(f"Error delegating task: {e!s}")

    async def perform_joint_reasoning(self, message: Message) -> None:
        try:
            reasoning_context = message.content.get("reasoning_context")
            our_reasoning = await self.agent.apply_advanced_reasoning({"content": reasoning_context})
            response = Message(
                type=MessageType.JOINT_REASONING_RESULT,
                sender=self.agent.name,
                receiver=message.sender,
                content=our_reasoning,
                parent_id=message.id,
            )
            await self.agent.communication_protocol.send_message(response)
        except Exception as e:
            logger.exception(f"Error performing joint reasoning: {e!s}")

    async def request_collaboration(self, agent_name: str, collaboration_type: str, content: Any) -> None:
        try:
            request = Message(
                type=MessageType.COLLABORATION_REQUEST,
                sender=self.agent.name,
                receiver=agent_name,
                content={"collaboration_type": collaboration_type, "content": content},
            )
            await self.agent.communication_protocol.send_message(request)
        except Exception as e:
            logger.exception(f"Error requesting collaboration: {e!s}")

    async def register_collaborating_agent(self, agent_name: str, capabilities: list[str]) -> None:
        self.collaborating_agents[agent_name] = capabilities

    async def find_best_agent_for_task(self, task: dict[str, Any]) -> str:
        best_agent = None
        best_match = 0
        for agent, capabilities in self.collaborating_agents.items():
            match = sum(1 for cap in capabilities if cap in task["content"])
            if match > best_match:
                best_match = match
                best_agent = agent
        return best_agent
