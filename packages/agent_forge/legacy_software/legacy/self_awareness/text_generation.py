import random

from langroid import ChatAgent, ChatAgentConfig, Task


class TextGenerationTask(Task):
    def __init__(self, agent: ChatAgent, rag_system) -> None:
        super().__init__(agent)
        self.rag_system = rag_system
        self.NUM_TEXTS_PER_RANGE = 1000  # This can be adjusted as needed

    async def generate_texts(
        self, temp_range: tuple[float, float], complexity: int, curriculum_level: int
    ) -> list[str]:
        texts = []
        for _ in range(self.NUM_TEXTS_PER_RANGE):
            temperature = random.uniform(*temp_range)
            prompt = await self._create_prompt(complexity, curriculum_level)
            prompt = self.rag_system.augment_prompt(prompt)
            response = await self.agent.llm_response(prompt, temperature=temperature)
            texts.append(response.content)
        return texts

    async def _create_prompt(self, complexity: int, curriculum_level: int) -> str:
        base_prompt = await self._get_base_prompt(complexity)
        rag_prompt = await self._get_rag_prompt(curriculum_level)
        interaction_prompt = await self._get_interaction_prompt(curriculum_level)
        metacognition_prompt = await self._get_metacognition_prompt(curriculum_level)

        full_prompt = f"{base_prompt}\n\n{rag_prompt}\n\n{interaction_prompt}\n\n{metacognition_prompt}"
        return full_prompt

    async def _get_base_prompt(self, complexity: int) -> str:
        if complexity <= 100:
            return "Write a short paragraph about a simple topic."
        if complexity <= 500:
            return "Write a detailed explanation of a moderately complex concept."
        return "Provide an in-depth analysis of a complex topic, considering multiple perspectives."

    async def _get_rag_prompt(self, curriculum_level: int) -> str:
        if curriculum_level <= 3:
            return "Include a fact retrieved from your knowledge base."
        if curriculum_level <= 7:
            return "Synthesize information from multiple sources in your knowledge base to support your argument."
        return "Critically evaluate and integrate complex information from your knowledge base, addressing potential conflicts or gaps in the available information."

    async def _get_interaction_prompt(self, curriculum_level: int) -> str:
        if curriculum_level <= 3:
            return "Describe a simple interaction between two AI agents."
        if curriculum_level <= 7:
            return "Simulate a collaborative problem-solving scenario involving multiple AI agents with different specialties."
        return "Design and analyze a complex multi-agent system tackling a real-world problem, considering ethical implications and potential conflicts."

    async def _get_metacognition_prompt(self, curriculum_level: int) -> str:
        if curriculum_level <= 3:
            return "Reflect on the process you used to generate this text."
        if curriculum_level <= 7:
            return "Analyze your own reasoning process, identifying potential biases or limitations in your approach."
        return "Engage in deep self-reflection, evaluating your cognitive strategies, considering alternative approaches, and proposing improvements to your own thought processes."

    async def run(self, temp_range: tuple[float, float], complexity: int, curriculum_level: int) -> list[str]:
        return await self.generate_texts(temp_range, complexity, curriculum_level)


# Usage example
if __name__ == "__main__":
    import asyncio

    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    class MockRAGSystem:
        """Very small RAG wrapper returning canned retrieval snippets."""

        def __init__(self, snippets: list[str] | None = None, n_ctx: int = 3) -> None:
            self.snippets = snippets or [
                "The quick brown fox jumps over the lazy dog.",
                "RAG systems combine retrieval with generation.",
                "Curriculum learning can improve language models.",
            ]
            self.n = n_ctx

        def augment_prompt(self, prompt: str, k: int = 3) -> str:
            docs = self.snippets[:k]
            if not docs:
                return prompt
            ctx = "\n".join(f"[ctx{i}] {d}" for i, d in enumerate(docs[: self.n], 1))
            return f"{ctx}\n\n[question] {prompt}"

    async def main() -> None:
        config = ChatAgentConfig(
            name="TextGenerator",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)
        # In this mock setup we skip a real vector store and return canned
        # retrieval snippets instead.
        rag_system = MockRAGSystem()
        task = TextGenerationTask(agent, rag_system)

        temp_range = (0.7, 0.9)
        complexity = 300
        curriculum_level = 5

        generated_texts = await task.run(temp_range, complexity, curriculum_level)
        print(f"Generated {len(generated_texts)} texts.")
        print("First generated text:", generated_texts[0])

    asyncio.run(main())
