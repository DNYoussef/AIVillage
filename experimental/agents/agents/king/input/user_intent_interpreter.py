import logging
from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig

from core.error_handling import AIVillageException, error_handler, safe_execute

logger = logging.getLogger(__name__)


class UserIntentInterpreter:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    @error_handler.handle_error
    async def interpret_intent(self, user_input: str) -> dict[str, Any]:
        """Interpret the user's intent from their input.

        Args:
            user_input (str): The raw input from the user.

        Returns:
            Dict[str, Any]: A dictionary containing the interpreted intent and relevant information.
        """
        prompt = self._create_intent_interpretation_prompt(user_input)
        response = await self.llm.complete(prompt)
        return self._parse_intent_response(response.text)

    def _create_intent_interpretation_prompt(self, user_input: str) -> str:
        return f"""
        Analyze the following user input and determine the user's intent:

        User Input: "{user_input}"

        Please provide a detailed interpretation of the user's intent, including:
        1. Primary Intent: The main goal or purpose of the user's input.
        2. Secondary Intents: Any additional or implied intentions.
        3. Key Entities: Important entities (e.g., names, places, concepts) mentioned in the input.
        4. Sentiment: The overall sentiment or emotion expressed in the input.
        5. Urgency: The level of urgency or importance of the user's request.
        6. Context: Any relevant context that might be important for understanding the intent.

        Provide your analysis in a structured JSON format.
        """

    def _parse_intent_response(self, response: str) -> dict[str, Any]:
        # In a real implementation, you would parse the JSON response
        # For simplicity, we'll assume the response is already in the correct format
        import json

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse intent response: {response}")
            raise AIVillageException("Failed to parse intent response")

    @error_handler.handle_error
    async def extract_key_concepts(self, interpreted_intent: dict[str, Any]) -> list[str]:
        """Extract key concepts from the interpreted intent.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted intent from interpret_intent method.

        Returns:
            List[str]: A list of key concepts extracted from the intent.
        """
        prompt = self._create_key_concepts_prompt(interpreted_intent)
        response = await self.llm.complete(prompt)
        return self._parse_key_concepts_response(response.text)

    def _create_key_concepts_prompt(self, interpreted_intent: dict[str, Any]) -> str:
        return f"""
        Based on the following interpreted user intent, extract the key concepts that are most relevant for further processing:

        Interpreted Intent: {interpreted_intent}

        Please provide a list of key concepts, considering:
        1. The primary and secondary intents
        2. Key entities mentioned
        3. Any important context

        Provide your list of key concepts in a JSON array format.
        """

    def _parse_key_concepts_response(self, response: str) -> list[str]:
        import json

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse key concepts response: {response}")
            raise AIVillageException("Failed to parse key concepts response")

    @safe_execute
    async def process_user_input(self, user_input: str) -> dict[str, Any]:
        """Process the user input by interpreting the intent and extracting key concepts.

        Args:
            user_input (str): The raw input from the user.

        Returns:
            Dict[str, Any]: A dictionary containing the interpreted intent and extracted key concepts.
        """
        interpreted_intent = await self.interpret_intent(user_input)
        key_concepts = await self.extract_key_concepts(interpreted_intent)

        return {"interpreted_intent": interpreted_intent, "key_concepts": key_concepts}


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        interpreter = UserIntentInterpreter(llm_config)

        user_input = "I need help organizing my team's project deadlines for the next quarter."
        result = await interpreter.process_user_input(user_input)

        print("Interpreted Intent:")
        print(result["interpreted_intent"])
        print("\nKey Concepts:")
        print(result["key_concepts"])

    asyncio.run(main())
