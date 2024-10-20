import logging
from typing import List, Dict, Any
from langroid.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)

class SubGoalGenerator:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    async def generate_subgoals(self, task: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate a list of subgoals for a given task.

        Args:
            task (str): The main task description.
            context (Dict[str, Any]): Additional context or constraints for the task.

        Returns:
            List[str]: A list of generated subgoals.
        """
        prompt = self._create_prompt(task, context)
        response = await self.llm.complete(prompt)
        subgoals = self._parse_response(response.text)
        return subgoals

    def _create_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for the language model to generate subgoals.

        Args:
            task (str): The main task description.
            context (Dict[str, Any]): Additional context or constraints for the task.

        Returns:
            str: The generated prompt.
        """
        prompt = f"""
        Task: {task}

        Context:
        {self._format_context(context)}

        Given the above task and context, generate a list of specific, actionable subgoals that will help accomplish the main task. Each subgoal should be a clear, concise step towards completing the overall task.

        Please format the subgoals as a numbered list, with each subgoal on a new line.

        Subgoals:
        """
        return prompt

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format the context dictionary into a string.

        Args:
            context (Dict[str, Any]): The context dictionary.

        Returns:
            str: A formatted string representation of the context.
        """
        return "\n".join([f"- {key}: {value}" for key, value in context.items()])

    def _parse_response(self, response: str) -> List[str]:
        """
        Parse the response from the language model into a list of subgoals.

        Args:
            response (str): The raw response from the language model.

        Returns:
            List[str]: A list of parsed subgoals.
        """
        lines = response.strip().split("\n")
        subgoals = [line.split(". ", 1)[-1].strip() for line in lines if line.strip()]
        return subgoals

    async def refine_subgoals(self, subgoals: List[str], feedback: str) -> List[str]:
        """
        Refine the generated subgoals based on feedback.

        Args:
            subgoals (List[str]): The initial list of subgoals.
            feedback (str): Feedback on the subgoals.

        Returns:
            List[str]: A refined list of subgoals.
        """
        prompt = f"""
        Original Subgoals:
        {self._format_subgoals(subgoals)}

        Feedback: {feedback}

        Based on the above feedback, please refine and improve the subgoals. Ensure that the refined subgoals address the feedback while maintaining clarity and actionability.

        Refined Subgoals:
        """
        response = await self.llm.complete(prompt)
        refined_subgoals = self._parse_response(response.text)
        return refined_subgoals

    def _format_subgoals(self, subgoals: List[str]) -> str:
        """
        Format a list of subgoals into a numbered string.

        Args:
            subgoals (List[str]): The list of subgoals.

        Returns:
            str: A formatted string of numbered subgoals.
        """
        return "\n".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
