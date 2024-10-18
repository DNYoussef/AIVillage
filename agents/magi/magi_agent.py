from typing import List, Dict, Any, Tuple
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.utils.task import Task as LangroidTask
from agents.communication.protocol import StandardCommunicationProtocol, Message, MessageType
from langroid.vector_store.base import VectorStore

class MagiAgentConfig(UnifiedAgentConfig):
    development_capabilities: List[str] = ["coding", "debugging", "code_review"]

class MagiAgent(UnifiedBaseAgent):
    def __init__(self, config: MagiAgentConfig, communication_protocol: StandardCommunicationProtocol, vector_store: VectorStore):
        super().__init__(config, communication_protocol)
        self.development_capabilities = config.development_capabilities
        self.self_evolving_system = SelfEvolvingSystem([self], vector_store)

    async def execute_task(self, task: LangroidTask) -> Dict[str, Any]:
        if task.type == "code":
            return await self.write_code(task)
        elif task.type == "debug":
            return await self.debug_code(task)
        elif task.type == "review":
            return await self.review_code(task)
        else:
            return await super().execute_task(task)

    async def write_code(self, task: LangroidTask) -> Dict[str, Any]:
        code_result = await self.generate(f"Write code for: {task.content}")
        return {"code_result": code_result}

    async def debug_code(self, task: LangroidTask) -> Dict[str, Any]:
        debug_result = await self.generate(f"Debug the following code: {task.content}")
        return {"debug_result": debug_result}

    async def review_code(self, task: LangroidTask) -> Dict[str, Any]:
        review_result = await self.generate(f"Review the following code: {task.content}")
        return {"review_result": review_result}

    async def handle_message(self, message: Message):
        if message.type == MessageType.TASK:
            task = LangroidTask(self, message.content['content'])
            task.type = message.content.get('task_type', 'general')
            result = await self.self_evolving_system.process_task(task)
            response = Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id
            )
            await self.communication_protocol.send_message(response)
        else:
            await super().handle_message(message)

    async def introspect(self) -> Dict[str, Any]:
        base_info = await super().introspect()
        return {
            **base_info,
            "development_capabilities": self.development_capabilities
        }

    async def evolve(self):
        await self.self_evolving_system.evolve()

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()  # Placeholder, implement actual VectorStore
    communication_protocol = StandardCommunicationProtocol()
    
    magi_config = MagiAgentConfig(
        name="MagiAgent",
        description="A development and coding agent",
        capabilities=["coding", "debugging", "code_review"],
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are a Magi agent capable of writing, debugging, and reviewing code."
    )
    
    magi_agent = MagiAgent(magi_config, communication_protocol, vector_store)
    
    # Use the magi_agent to process tasks and evolve
