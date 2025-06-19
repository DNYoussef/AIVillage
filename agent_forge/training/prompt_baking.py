from typing import List

from agent_forge.tool_baking import rag_prompt_baker


class PromptBakingManager:
    """Simple wrapper around RAG prompt baking utilities."""

    def __init__(self, model_name: str):
        self.baker = rag_prompt_baker.RAGPromptBaker(model_name)

    def deep_bake(self, prompts: List[str], num_rounds: int = 3) -> None:
        """Iteratively bake prompts into the model."""
        for i in range(num_rounds):
            self.baker.bake_prompts(prompts)

    def save(self, path: str) -> None:
        self.baker.save_model(path)
