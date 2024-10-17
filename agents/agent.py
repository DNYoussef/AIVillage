from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.vector_store.base import VectorStore

class AgentConfig(ChatAgentConfig):
    name: str = Field(..., description="The name of the agent")
    description: str = Field(..., description="A brief description of the agent's purpose")
    capabilities: List[str] = Field(default_factory=list, description="List of agent capabilities")
    vector_store: Optional[VectorStore] = Field(None, description="Vector store for the agent")
    model: str = Field(..., description="The language model to be used by the agent")
    instructions: str = Field(..., description="Instructions for the agent's behavior")

class Agent(ChatAgent):
    """
    A comprehensive base agent class that combines features from Agent and BaseAgent.
    """
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities
        self.vector_store = config.vector_store
        self.model = config.model
        self.instructions = config.instructions
        self.tools: List[Callable] = []

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a given task. This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute_task method")

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message by creating a task and executing it.
        """
        task = Task(self, message['content'])
        return await self.execute_task(task)

    def add_capability(self, capability: str):
        """
        Add a new capability to the agent.
        """
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def remove_capability(self, capability: str):
        """
        Remove a capability from the agent.
        """
        if capability in self.capabilities:
            self.capabilities.remove(capability)

    def add_tool(self, tool: Callable):
        """
        Add a new tool to the agent.
        """
        self.tools.append(tool)

    @property
    def info(self) -> Dict[str, Any]:
        """
        Return a dictionary containing information about the agent.
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "model": self.model
        }
