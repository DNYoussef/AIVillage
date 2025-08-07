"""Wrapper around ``RAGPromptBaker`` for quick prompt freezing."""

from AIVillage.experimental.training.tool_baking.rag_prompt_baker import (
    RAGPromptBaker,
    get_rag_prompts,
)


def freeze_top_prompts(model_name: str, top_k: int = 32) -> None:
    baker = RAGPromptBaker(model_name)
    baker.load_model()
    baker.bake_prompts(get_rag_prompts(), num_iterations=top_k)
