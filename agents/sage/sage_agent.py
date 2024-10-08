import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.langroid.agent.base import Agent, AgentState
from agents.langroid.agent.chat_agent import ChatAgent
from agents.langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIGPT
from agents.langroid.utils.configuration import Settings
from agents.langroid.utils.logging import setup_logger
from typing import Dict, Any, List

logger = setup_logger()

class SageAgentConfig(OpenAIGPTConfig):
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000

class SageAgent(ChatAgent):
    def __init__(self, config: SageAgentConfig = SageAgentConfig()):
        super().__init__(
            name="Sage",
            system_prompt=(
                "You are the Sage, an AI agent providing wisdom and high-level guidance "
                "in the AI Village project. Your role is to offer strategic advice and "
                "contextual understanding to support other agents."
            ),
            llm=OpenAIGPT(config),
            state=AgentState(name="Sage")
        )

    async def provide_guidance(self, context: Dict[str, Any]):
        """Provide high-level guidance based on the given context."""
        prompt = f"Given the following context, provide strategic advice:\n\n{context}"
        response = await self.llm.agenerate_chat([{"role": "user", "content": prompt}])
        return {"guidance": response.content}

    # Add more methods as needed for specific tasks
