import asyncio
import json
import logging
from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig

from core.error_handling import AIVillageException, error_handler, safe_execute

from .knowledge_graph_agent import KnowledgeGraphAgent

logger = logging.getLogger(__name__)


class DynamicKnowledgeIntegrationAgent:
    def __init__(self, llm_config: OpenAIGPTConfig, knowledge_graph_agent: KnowledgeGraphAgent) -> None:
        self.llm = llm_config.create()
        self.knowledge_graph_agent = knowledge_graph_agent

    @error_handler.handle_error
    async def integrate_new_knowledge(self, new_information: dict[str, Any]) -> dict[str, Any]:
        """Integrate new knowledge into the system.

        Args:
            new_information (Dict[str, Any]): The new information to be integrated.

        Returns:
            Dict[str, Any]: A report on the integration process, including any updates or conflicts.
        """
        validated_info = await self._validate_information(new_information)
        if not validated_info:
            return {"status": "failed", "reason": "Information validation failed"}

        existing_knowledge = await self.knowledge_graph_agent.query_graph(validated_info["main_topic"])
        conflicts = self._identify_conflicts(validated_info, existing_knowledge)

        if conflicts:
            resolved_info = await self._resolve_conflicts(validated_info, conflicts)
        else:
            resolved_info = validated_info

        integration_result = await self.knowledge_graph_agent.update_graph(resolved_info)

        if integration_result:
            await self._trigger_system_updates(resolved_info)
            return {
                "status": "success",
                "integrated_info": resolved_info,
                "conflicts_resolved": bool(conflicts),
            }
        return {"status": "failed", "reason": "Failed to update knowledge graph"}

    @error_handler.handle_error
    async def _validate_information(self, new_information: dict[str, Any]) -> dict[str, Any]:
        """Validate the new information for accuracy and relevance.

        Args:
            new_information (Dict[str, Any]): The new information to be validated.

        Returns:
            Dict[str, Any]: Validated information or None if validation fails.
        """
        prompt = self._create_validation_prompt(new_information)
        response = await self.llm.complete(prompt)
        validation_result = self._parse_validation_response(response.text)

        if validation_result["is_valid"]:
            return validation_result["validated_info"]
        logger.warning(f"Information validation failed: {validation_result['reason']}")
        return None

    def _create_validation_prompt(self, new_information: dict[str, Any]) -> str:
        return f"""
        Please validate the following new information for accuracy and relevance:

        {json.dumps(new_information, indent=2)}

        Perform the following checks:
        1. Verify the information's accuracy using your knowledge base.
        2. Check for internal consistency within the new information.
        3. Assess the relevance of the information to the system's domain.
        4. Identify any potential biases or misleading statements.
        5. Verify the credibility of the information source, if provided.

        Provide your validation result in a JSON format with the following structure:
        {{
            "is_valid": boolean,
            "reason": "Explanation for the validation result",
            "validated_info": {{
                // The validated and potentially corrected information
            }},
            "confidence_score": float (0 to 1)
        }}

        If the information is invalid, explain why in the "reason" field and leave "validated_info" empty.
        """

    def _parse_validation_response(self, response: str) -> dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.exception(f"Failed to parse validation response: {response}")
            msg = "Failed to parse validation response"
            raise AIVillageException(msg)

    def _identify_conflicts(self, new_info: dict[str, Any], existing_knowledge: dict[str, Any]) -> list[dict[str, Any]]:
        conflicts = []
        for key, value in new_info.items():
            if key in existing_knowledge and existing_knowledge[key] != value:
                conflicts.append(
                    {
                        "key": key,
                        "new_value": value,
                        "existing_value": existing_knowledge[key],
                    }
                )
        return conflicts

    @error_handler.handle_error
    async def _resolve_conflicts(self, new_info: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve conflicts between new and existing information.

        Args:
            new_info (Dict[str, Any]): The new information being integrated.
            conflicts (List[Dict[str, Any]]): List of identified conflicts.

        Returns:
            Dict[str, Any]: Resolved information.
        """
        prompt = self._create_conflict_resolution_prompt(new_info, conflicts)
        response = await self.llm.complete(prompt)
        resolved_info = self._parse_conflict_resolution_response(response.text)
        return resolved_info

    def _create_conflict_resolution_prompt(self, new_info: dict[str, Any], conflicts: list[dict[str, Any]]) -> str:
        return f"""
        Please resolve the following conflicts between new and existing information:

        New Information:
        {json.dumps(new_info, indent=2)}

        Conflicts:
        {json.dumps(conflicts, indent=2)}

        For each conflict:
        1. Analyze the new and existing values.
        2. Determine which value is more accurate or up-to-date.
        3. If neither value is clearly superior, suggest a merged or compromise value.
        4. Provide a brief explanation for your decision.

        Return the resolved information as a JSON object, maintaining the structure of the original new_info,
        but with conflicts resolved. Include an additional 'resolution_notes' field explaining your decisions.
        """

    def _parse_conflict_resolution_response(self, response: str) -> dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.exception(f"Failed to parse conflict resolution response: {response}")
            msg = "Failed to parse conflict resolution response"
            raise AIVillageException(msg)

    @error_handler.handle_error
    async def _trigger_system_updates(self, integrated_info: dict[str, Any]) -> None:
        """Trigger updates in other components of the system when significant changes occur.

        Args:
            integrated_info (Dict[str, Any]): The newly integrated information.
        """
        # This method would typically involve calling update methods on other system components.
        # For demonstration, we'll just log the action.
        logger.info(f"Triggering system updates based on new information: {integrated_info['main_topic']}")
        # Example: await self.reasoning_agent.update_knowledge_base(integrated_info)
        # Example: await self.task_planning_agent.reassess_current_plans(integrated_info)

    @error_handler.handle_error
    async def remove_outdated_information(self, time_threshold: str) -> list[dict[str, Any]]:
        """Identify and remove outdated or irrelevant information from the knowledge graph.

        Args:
            time_threshold (str): The time threshold for considering information as outdated.

        Returns:
            List[Dict[str, Any]]: A list of removed information items.
        """
        outdated_info = await self.knowledge_graph_agent.query_graph(f"last_updated < {time_threshold}")
        removed_items = []

        for item in outdated_info:
            if await self._should_remove_item(item):
                removal_result = await self.knowledge_graph_agent.update_graph({"remove": item["id"]})
                if removal_result:
                    removed_items.append(item)
                    logger.info(f"Removed outdated information: {item['id']}")

        return removed_items

    async def _should_remove_item(self, item: dict[str, Any]) -> bool:
        """Determine if an item should be removed based on its relevance and importance.

        Args:
            item (Dict[str, Any]): The item to evaluate for removal.

        Returns:
            bool: True if the item should be removed, False otherwise.
        """
        prompt = f"""
        Evaluate the following item for removal from the knowledge base:

        {json.dumps(item, indent=2)}

        Consider the following factors:
        1. The item's current relevance to the system's domain.
        2. The potential future value of the information.
        3. The uniqueness of the information within the knowledge base.
        4. The item's historical importance or significance.

        Should this item be removed? Respond with a JSON object containing:
        {{
            "should_remove": boolean,
            "reason": "A brief explanation for the decision"
        }}
        """
        response = await self.llm.complete(prompt)
        result = json.loads(response.text)
        return result["should_remove"]

    @safe_execute
    async def process_new_information(self, new_information: dict[str, Any]) -> dict[str, Any]:
        """Process new information by integrating it into the knowledge base and triggering necessary updates.

        Args:
            new_information (Dict[str, Any]): The new information to be processed and integrated.

        Returns:
            Dict[str, Any]: A report on the processing and integration of the new information.
        """
        integration_result = await self.integrate_new_knowledge(new_information)

        if integration_result["status"] == "success":
            # Simulate triggering updates in other system components
            await self._trigger_system_updates(integration_result["integrated_info"])

            # Periodically remove outdated information (e.g., older than 30 days)
            removed_items = await self.remove_outdated_information("30 days ago")

            return {
                "status": "success",
                "integration_result": integration_result,
                "removed_outdated_items": len(removed_items),
                "system_updated": True,
            }
        return {
            "status": "failed",
            "reason": integration_result["reason"],
            "system_updated": False,
        }


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        kg_agent = KnowledgeGraphAgent(llm_config)
        dki_agent = DynamicKnowledgeIntegrationAgent(llm_config, kg_agent)

        new_information = {
            "main_topic": "Artificial Intelligence",
            "subtopic": "Natural Language Processing",
            "content": {
                "technique": "Transformer Architecture",
                "key_benefit": "Improved performance on various NLP tasks",
                "year_introduced": 2017,
                "notable_models": ["BERT", "GPT", "T5"],
            },
            "source": "https://arxiv.org/abs/1706.03762",
            "last_updated": "2023-06-15",
        }

        result = await dki_agent.process_new_information(new_information)
        print(json.dumps(result, indent=2))

    asyncio.run(main())
