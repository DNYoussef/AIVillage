
from rag_system.core.config import UnifiedConfig


class SageAgentConfig(UnifiedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.research_capabilities: list[str] = kwargs.get("research_capabilities", [
            "web_search",
            "data_analysis",
            "information_synthesis",
            "exploration_mode"
        ])
        self.model: str = kwargs.get("model", "gpt-4")
        self.instructions: str = kwargs.get("instructions", (
            "You are a Sage agent capable of conducting web searches, analyzing data, "
            "synthesizing information, and exploring knowledge graphs to discover new relations."
        ))

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "research_capabilities": self.research_capabilities,
            "model": self.model,
            "instructions": self.instructions
        })
        return base_dict
