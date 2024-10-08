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

class KingAgentConfig(OpenAIGPTConfig):
    model_name: str = "gpt-4"  # or whichever model you prefer
    temperature: float = 0.7
    max_tokens: int = 1000

class KingAgent(ChatAgent):
    def __init__(self, config: KingAgentConfig = KingAgentConfig()):
        super().__init__(
            name="King",
            system_prompt=(
                "You are the King, an advanced AI agent responsible for strategic planning "
                "and decision making in the AI Village project. Your role is to analyze, "
                "plan, and provide insights using your vast knowledge and reasoning capabilities."
            ),
            llm=OpenAIGPT(config),
            state=AgentState(name="King")
        )

    async def generate_structured_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a structured response based on the given prompt."""
        response = await self.llm.agenerate_chat([{"role": "user", "content": prompt}])
        # Assuming the response is in JSON format, parse it
        try:
            return response.parsed_output
        except Exception as e:
            logger.error(f"Error parsing structured response: {e}")
            return {"error": "Failed to parse structured response"}

    async def analyze_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a given plan and provide insights or suggestions."""
        prompt = f"Analyze the following plan and provide insights and suggestions:\n\n{plan}"
        response = await self.llm.agenerate_chat([{"role": "user", "content": prompt}])
        return {"analysis": response.content}

    # Add more methods as needed for specific tasks in your AI Village project
