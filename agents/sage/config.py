from typing import List
from agents.unified_base_agent import UnifiedAgentConfig

class SageAgentConfig(UnifiedAgentConfig):
    research_capabilities: List[str] = [
        "web_search",
        "data_analysis",
        "information_synthesis",
        "exploration_mode"
    ]
    model: str = "gpt-4"
    instructions: str = (
        "You are a Sage agent capable of conducting web searches, analyzing data, "
        "synthesizing information, and exploring knowledge graphs to discover new relations."
    )
