from typing import List, Dict, Any, Optional
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.vector_store.base import VectorStore

class BaseAgentConfig(ChatAgentConfig):
    name: str
    description: str
    capabilities: List[str]
    vector_store: Optional[VectorStore] = None

class BaseAgent(ChatAgent):
    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities
        self.vector_store = config.vector_store

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        # Implementation will vary based on agent type
        raise NotImplementedError("Subclasses must implement execute_task method")

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        # General message processing logic
        task = Task(self, message['content'])
        return await self.execute_task(task)

    def add_capability(self, capability: str):
        self.capabilities.append(capability)

    def remove_capability(self, capability: str):
        if capability in self.capabilities:
            self.capabilities.remove(capability)

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities
        }
