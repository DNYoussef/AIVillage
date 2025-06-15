from __future__ import annotations

from dataclasses import dataclass
from rag_system.utils.tokenizer import get_cl100k_encoding


@dataclass
class OpenAIGPTConfig:
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000

    def create(self) -> "OpenAIGPT":
        return OpenAIGPT(self)


class OpenAIGPT:
    def __init__(self, config: OpenAIGPTConfig):
        self.config = config
        self.tokenizer = get_cl100k_encoding()

    async def agenerate_chat(self, messages):
        """Placeholder async call to an LLM."""
        response_content = "This is a generated response."
        parsed_output = {}
        return type("Response", (object,), {"content": response_content, "parsed_output": parsed_output})

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using the local tokenizer."""
        return len(self.tokenizer.encode(text))
