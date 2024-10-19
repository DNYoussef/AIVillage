from typing import List, Dict, Any
from communications.protocol import Message, MessageType
import logging

logger = logging.getLogger(__name__)

class CollaborationManager:
    def __init__(self, agent):
        self.agent = agent
        self.collaborating_agents = {}

    async def handle_collaboration_request(self, message: Message):
        try:
            collaboration_type = message.content.get('collaboration_type')
            if collaboration_type == 'knowledge_sharing':
                await self.share_knowledge(message)
            elif collaboration_type == 'task_delegation':
                await self.delegate_task(message)
            elif collaboration_type == 'joint_reasoning':
                await self.perform_joint_reasoning(message)
            else:
                logger.warning(f"Unknown collaboration type: {collaboration_type}")
        except Exception as e:
            logger.error(f"Error handling collaboration request: {str(e)}")

    async def share_knowledge(self, message: Message):
        try:
            query = message.content.get('query')
            relevant_knowledge = await self.agent.query_rag(query)
            response = Message(
                type=MessageType.KNOWLEDGE_SHARE,
                sender=self.agent.name,
                receiver=message.sender,
                content=relevant_knowledge,
                parent_id=message.id
            )
            await self.agent.communication_protocol.send_message(response)
        except Exception as e:
            logger.error(f"Error sharing knowledge: {str(e)}")

    async def delegate_task(self, message: Message):
        try:
            task = message.content.get('task')
            result = await self.agent.execute_task(task)
            response = Message(
                type=MessageType.TASK_RESULT,
                sender=self.agent.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id
            )
            await self.agent.communication_protocol.send_message(response)
        except Exception as e:
            logger.error(f"Error delegating task: {str(e)}")

    async def perform_joint_reasoning(self, message: Message):
        try:
            reasoning_context = message.content.get('reasoning_context')
            our_reasoning = await self.agent.apply_advanced_reasoning({'content': reasoning_context})
            response = Message(
                type=MessageType.JOINT_REASONING_RESULT,
                sender=self.agent.name,
                receiver=message.sender,
                content=our_reasoning,
                parent_id=message.id
            )
            await self.agent.communication_protocol.send_message(response)
        except Exception as e:
            logger.error(f"Error performing joint reasoning: {str(e)}")

    async def request_collaboration(self, agent_name: str, collaboration_type: str, content: Any):
        try:
            request = Message(
                type=MessageType.COLLABORATION_REQUEST,
                sender=self.agent.name,
                receiver=agent_name,
                content={
                    'collaboration_type': collaboration_type,
                    'content': content
                }
            )
            await self.agent.communication_protocol.send_message(request)
        except Exception as e:
            logger.error(f"Error requesting collaboration: {str(e)}")

    async def register_collaborating_agent(self, agent_name: str, capabilities: List[str]):
        self.collaborating_agents[agent_name] = capabilities

    async def find_best_agent_for_task(self, task: Dict[str, Any]) -> str:
        best_agent = None
        best_match = 0
        for agent, capabilities in self.collaborating_agents.items():
            match = sum(1 for cap in capabilities if cap in task['content'])
            if match > best_match:
                best_match = match
                best_agent = agent
        return best_agent
