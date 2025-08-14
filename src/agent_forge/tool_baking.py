"""Stub implementation for agent_forge tool baking.
This is a placeholder to fix test infrastructure.
"""

import warnings

warnings.warn(
    "agent_forge.tool_baking is a stub implementation. " "Replace with actual implementation before production use.",
    UserWarning,
    stacklevel=2,
)


class RAGPromptBaker:
    """Placeholder RAGPromptBaker for testing."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self.initialized = True

    def load_model(self):
        """Stub model loading method."""
        return {"status": "model_loaded", "model": self.model_name}

    def bake_prompts(self, prompts):
        """Stub prompt baking method."""
        return {"status": "prompts_baked", "count": len(prompts) if prompts else 0}


def get_rag_prompts():
    """Stub function to get RAG prompts."""
    return [
        "What is the compression ratio?",
        "How does evolution work?",
        "Explain the training process.",
    ]


__all__ = ["RAGPromptBaker", "get_rag_prompts"]
