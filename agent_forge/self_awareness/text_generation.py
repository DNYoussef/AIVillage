# text_generation.py

import random
from typing import List, Tuple

class TextGenerator:
    def __init__(self, model, rag_system):
        self.model = model
        self.rag_system = rag_system

    def generate_texts(self, temp_range: Tuple[float, float], complexity: int, curriculum_level: int) -> List[str]:
        texts = []
        for _ in range(self.NUM_TEXTS_PER_RANGE):
            temperature = random.uniform(*temp_range)
            prompt = self._create_prompt(complexity, curriculum_level)
            text = self.model.generate(prompt, temperature=temperature, max_length=complexity)
            texts.append(text)
        return texts

    def _create_prompt(self, complexity: int, curriculum_level: int) -> str:
        base_prompt = self._get_base_prompt(complexity)
        rag_prompt = self._get_rag_prompt(curriculum_level)
        interaction_prompt = self._get_interaction_prompt(curriculum_level)
        metacognition_prompt = self._get_metacognition_prompt(curriculum_level)

        full_prompt = f"{base_prompt}\n\n{rag_prompt}\n\n{interaction_prompt}\n\n{metacognition_prompt}"
        return full_prompt

    def _get_base_prompt(self, complexity: int) -> str:
        if complexity <= 100:
            return "Write a short paragraph about a simple topic."
        elif complexity <= 500:
            return "Write a detailed explanation of a moderately complex concept."
        else:
            return "Provide an in-depth analysis of a complex topic, considering multiple perspectives."

    def _get_rag_prompt(self, curriculum_level: int) -> str:
        if curriculum_level <= 3:
            return "Include a fact retrieved from your knowledge base."
        elif curriculum_level <= 7:
            return "Synthesize information from multiple sources in your knowledge base to support your argument."
        else:
            return "Critically evaluate and integrate complex information from your knowledge base, addressing potential conflicts or gaps in the available information."

    def _get_interaction_prompt(self, curriculum_level: int) -> str:
        if curriculum_level <= 3:
            return "Describe a simple interaction between two AI agents."
        elif curriculum_level <= 7:
            return "Simulate a collaborative problem-solving scenario involving multiple AI agents with different specialties."
        else:
            return "Design and analyze a complex multi-agent system tackling a real-world problem, considering ethical implications and potential conflicts."

    def _get_metacognition_prompt(self, curriculum_level: int) -> str:
        if curriculum_level <= 3:
            return "Reflect on the process you used to generate this text."
        elif curriculum_level <= 7:
            return "Analyze your own reasoning process, identifying potential biases or limitations in your approach."
        else:
            return "Engage in deep self-reflection, evaluating your cognitive strategies, considering alternative approaches, and proposing improvements to your own thought processes."

    NUM_TEXTS_PER_RANGE = 1000  # This can be adjusted as needed