from agent_forge.tool_baking import rag_prompt_baker

from .expert_vectors import ExpertVector, ExpertVectorSystem


class PromptBakingManager:
    """Wrapper around RAG prompt baking with optional expert vectors."""

    def __init__(self, model_name: str, expert_vectors: dict[str, ExpertVector] | None = None):
        self.baker = rag_prompt_baker.RAGPromptBaker(model_name)
        self.expert_vectors = expert_vectors or {}
        self._vectors_applied = False

    def _apply_vectors(self) -> None:
        if self._vectors_applied or not self.expert_vectors:
            return
        if self.baker.model is None:
            self.baker.load_model()
        system = ExpertVectorSystem(self.baker.model)
        for vec in self.expert_vectors.values():
            system.apply_expert_vector(vec)
        self._vectors_applied = True

    def deep_bake(self, prompts: list[str], num_rounds: int = 3) -> None:
        """Iteratively bake prompts into the model."""
        self._apply_vectors()
        for _ in range(num_rounds):
            self.baker.bake_prompts(prompts)

    def save(self, path: str) -> None:
        self.baker.save_model(path)
