from __future__ import annotations

from dataclasses import dataclass
import os

import openai

from rag_system.utils.logging import setup_logger as get_logger
from rag_system.utils.tokenizer import get_cl100k_encoding


@dataclass
class OpenAIGPTConfig:
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000

    def create(self) -> OpenAIGPT:
        return OpenAIGPT(self)


class OpenAIGPT:
    def __init__(self, config: OpenAIGPTConfig) -> None:
        self.config = config
        self.tokenizer = get_cl100k_encoding()
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = get_logger(__name__)

    async def agenerate_chat(self, messages):
        """Call the OpenAI chat completion API."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = resp.choices[0].message.content
            return type(
                "Response",
                (object,),
                {"content": content, "parsed_output": {}},
            )
        except Exception:  # pragma: no cover - network errors
            self.logger.exception("OpenAI completion failed")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using the local tokenizer."""
        return len(self.tokenizer.encode(text))
