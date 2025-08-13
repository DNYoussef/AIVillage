import json
import logging
from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig
from torch import nn

from core.error_handling import error_handler, safe_execute

logger = logging.getLogger(__name__)


class ResponseGenerationAgent:
    def __init__(self, llm_config: OpenAIGPTConfig) -> None:
        self.llm = llm_config.create()
        self.model = None

    @error_handler.handle_error
    async def generate_response(
        self, input_data: dict[str, Any], user_preferences: dict[str, Any]
    ) -> str:
        """Generate a response based on input data and user preferences.

        Args:
            input_data (Dict[str, Any]): The input data from other agents, including reasoning results.
            user_preferences (Dict[str, Any]): User preferences for response style, tone, etc.

        Returns:
            str: The generated response.
        """
        prompt = self._create_response_prompt(input_data, user_preferences)
        response = await self.llm.complete(prompt)
        return self._post_process_response(response.text, user_preferences)

    def _create_response_prompt(
        self, input_data: dict[str, Any], user_preferences: dict[str, Any]
    ) -> str:
        return f"""
        Given the following input data and user preferences, generate an appropriate response:

        Input Data: {json.dumps(input_data, indent=2)}
        User Preferences: {json.dumps(user_preferences, indent=2)}

        Please generate a response that:
        1. Accurately conveys the information and conclusions from the input data.
        2. Is clear, coherent, and easy to understand.
        3. Adapts to the user's preferred tone and style.
        4. Provides appropriate level of detail based on user preferences.
        5. Includes explanations or additional context where necessary.
        6. Aligns with the system's goals and ethical guidelines.
        7. Is engaging and encourages further interaction if appropriate.

        Consider the following in your response:
        - If the input includes multiple points or steps, structure the response accordingly.
        - If there are uncertainties or alternative viewpoints, present them in a balanced manner.
        - If the user prefers concise responses, focus on the most crucial information.
        - If the user prefers detailed explanations, provide more background and reasoning.
        - Adapt the language complexity to the user's preferences and the topic at hand.
        - Use analogies or examples to illustrate complex concepts if it aids understanding.

        Generate the response in a natural, conversational style that matches the user's preferences.
        """

    def _post_process_response(
        self, response: str, user_preferences: dict[str, Any]
    ) -> str:
        # Implement any post-processing steps here, such as:
        # - Adjusting response length
        # - Adding or removing technical details
        # - Inserting appropriate emojis or formatting
        # - Ensuring the response adheres to specific guidelines

        max_length = user_preferences.get("max_response_length", 1000)
        if len(response) > max_length:
            response = response[:max_length] + "..."

        if user_preferences.get("include_emojis", False):
            response = self._add_emojis(response)

        return response

    def _add_emojis(self, text: str) -> str:
        # Implement emoji addition logic here
        # This is a simple example; you might want to use a more sophisticated approach
        emoji_map = {
            "happy": "😊",
            "sad": "😢",
            "important": "❗",
            "idea": "💡",
            "warning": "⚠️",
        }
        for word, emoji in emoji_map.items():
            text = text.replace(f" {word} ", f" {word} {emoji} ")
        return text

    @error_handler.handle_error
    async def update_model(self, new_model: nn.Module) -> None:
        self.model = new_model
        logger.info("Model updated in ResponseGenerationAgent")

    @error_handler.handle_error
    async def update_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        # Update hyperparameters if needed
        # For example, if using a custom language model:
        # self.llm.update_hyperparameters(hyperparameters)
        logger.info("Hyperparameters updated in ResponseGenerationAgent")

    async def generate_multi_format_response(
        self, input_data: dict[str, Any], user_preferences: dict[str, Any]
    ) -> dict[str, str]:
        """Generate responses in multiple formats based on input data and user preferences.

        Args:
            input_data (Dict[str, Any]): The input data from other agents, including reasoning results.
            user_preferences (Dict[str, Any]): User preferences for response style, tone, etc.

        Returns:
            Dict[str, str]: A dictionary containing responses in different formats.
        """
        formats = user_preferences.get("response_formats", ["text"])
        responses = {}

        for format_type in formats:
            prompt = self._create_multi_format_prompt(
                input_data, user_preferences, format_type
            )
            response = await self.llm.complete(prompt)
            responses[format_type] = self._post_process_response(
                response.text, user_preferences
            )

        return responses

    def _create_multi_format_prompt(
        self,
        input_data: dict[str, Any],
        user_preferences: dict[str, Any],
        format_type: str,
    ) -> str:
        base_prompt = self._create_response_prompt(input_data, user_preferences)
        format_specific_instructions = {
            "text": "Generate a standard text response.",
            "bullet_points": "Generate a response in the form of concise bullet points.",
            "step_by_step": "Generate a response as a series of step-by-step instructions.",
            "eli5": "Generate a response explaining the concept as you would to a 5-year-old.",
            "technical": "Generate a more technical and detailed response for an expert audience.",
        }

        return f"""
        {base_prompt}

        Specific instructions for this response:
        {format_specific_instructions.get(format_type, "Generate a standard response.")}
        """

    @safe_execute
    async def process_and_respond(
        self, reasoning_result: dict[str, Any], user_preferences: dict[str, Any]
    ) -> dict[str, Any]:
        """Process the reasoning result and generate appropriate responses.

        Args:
            reasoning_result (Dict[str, Any]): The result from the Reasoning Agent.
            user_preferences (Dict[str, Any]): User preferences for response style, tone, etc.

        Returns:
            Dict[str, Any]: A dictionary containing the generated responses and any additional information.
        """
        standard_response = await self.generate_response(
            reasoning_result, user_preferences
        )
        multi_format_responses = await self.generate_multi_format_response(
            reasoning_result, user_preferences
        )

        return {
            "standard_response": standard_response,
            "multi_format_responses": multi_format_responses,
            "input_summary": self._summarize_input(reasoning_result),
            "response_metadata": self._generate_response_metadata(
                reasoning_result, user_preferences
            ),
        }

    def _summarize_input(self, reasoning_result: dict[str, Any]) -> str:
        # Implement logic to create a brief summary of the input reasoning result
        return f"Input based on reasoning about: {reasoning_result.get('main_topic', 'Unspecified topic')}"

    def _generate_response_metadata(
        self, reasoning_result: dict[str, Any], user_preferences: dict[str, Any]
    ) -> dict[str, Any]:
        # Generate metadata about the response, which could be useful for analytics or further processing
        return {
            "confidence_level": reasoning_result.get("confidence", 0),
            "response_tone": user_preferences.get("tone", "neutral"),
            "complexity_level": user_preferences.get("complexity", "medium"),
            "source_count": len(reasoning_result.get("sources", [])),
            "generated_at": self._get_current_timestamp(),
        }

    def _get_current_timestamp(self) -> str:
        from datetime import datetime

        return datetime.now().isoformat()


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        response_agent = ResponseGenerationAgent(llm_config)

        reasoning_result = {
            "main_topic": "Climate Change Mitigation",
            "conclusions": [
                "Reducing carbon emissions is crucial",
                "Renewable energy adoption should be accelerated",
                "Individual actions can contribute significantly",
            ],
            "confidence": 0.85,
            "sources": [
                "IPCC Report",
                "Energy Industry Analysis",
                "Behavioral Studies",
            ],
            "uncertainties": [
                "Exact timeline for critical thresholds",
                "Political willingness to act",
            ],
        }

        user_preferences = {
            "tone": "informative",
            "complexity": "medium",
            "max_response_length": 500,
            "include_emojis": True,
            "response_formats": ["text", "bullet_points", "eli5"],
        }

        result = await response_agent.process_and_respond(
            reasoning_result, user_preferences
        )

        print("Standard Response:")
        print(result["standard_response"])
        print("\nMulti-format Responses:")
        for format_type, response in result["multi_format_responses"].items():
            print(f"\n{format_type.upper()}:")
            print(response)
        print("\nResponse Metadata:")
        print(json.dumps(result["response_metadata"], indent=2))

    asyncio.run(main())
