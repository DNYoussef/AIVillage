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

class MagiAgentConfig(OpenAIGPTConfig):
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000

class MagiAgent(ChatAgent):
    def __init__(self, config: MagiAgentConfig = MagiAgentConfig()):
        super().__init__(
            name="Magi",
            system_prompt=(
                "You are the Magi, a specialized AI agent responsible for technical analysis "
                "and providing expert insights in the AI Village project. Your role is to evaluate "
                "technical details and support the King agent with in-depth knowledge."
            ),
            llm=OpenAIGPT(config),
            state=AgentState(name="Magi")
        )

    async def analyze_technical_details(self, data: Dict[str, Any]):
        """Analyze technical data and provide detailed insights."""
        prompt = f"Analyze the following technical data and provide insights:\n\n{data}"
        response = await self.llm.agenerate_chat([{"role": "user", "content": prompt}])
        return {"technical_analysis": response.content}

    # Add more methods as needed for specific tasks
