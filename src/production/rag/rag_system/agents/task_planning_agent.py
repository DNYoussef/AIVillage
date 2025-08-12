from typing import Any

from rag_system.agents.key_concept_extractor import KeyConceptExtractorAgent
from rag_system.agents.user_intent_interpreter import UserIntentInterpreterAgent
from rag_system.core.agent_interface import AgentInterface


class TaskPlanningAgent(AgentInterface):
    """Agent responsible for planning tasks based on user intent and extracted concepts."""

    def __init__(self) -> None:
        super().__init__()
        self.intent_interpreter = UserIntentInterpreterAgent()
        self.concept_extractor = KeyConceptExtractorAgent()

    def plan_tasks(self, query: str) -> dict[str, Any]:
        """Plan tasks based on the user's query.

        Args:
            query (str): The user's input query.

        Returns:
            Dict[str, Any]: A dictionary containing the planned tasks and any relevant metadata.
        """
        # Interpret user intent
        intent = self.intent_interpreter.interpret_intent(query)

        # Extract key concepts
        concepts = self.concept_extractor.extract_key_concepts(query)

        # Plan tasks based on intent and concepts
        tasks = self._generate_task_plan(intent, concepts)

        return {"intent": intent, "concepts": concepts, "tasks": tasks}

    def _generate_task_plan(
        self, intent: dict[str, Any], concepts: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a task plan based on interpreted intent and extracted concepts.

        Args:
            intent (Dict[str, Any]): The interpreted user intent.
            concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A structured task plan.
        """
        # Collate search terms from both keywords and entity texts
        keywords = concepts.get("keywords", [])
        entities = [
            e.get("text") for e in concepts.get("entities", []) if isinstance(e, dict)
        ]
        search_terms = [t for t in keywords + entities if t]

        # Fall back to intent information if no explicit concepts were found
        if not search_terms:
            topic = (
                intent.get("topic")
                or intent.get("primary_intent")
                or intent.get("type")
            )
            if topic:
                search_terms.append(topic)

        # Determine the analysis action based on the primary intent
        primary_intent = intent.get("primary_intent") or intent.get("type", "")
        action = "analyze_information"
        lower_intent = primary_intent.lower() if isinstance(primary_intent, str) else ""
        if any(word in lower_intent for word in ["summarize", "describe"]):
            action = "summarize_information"
        elif any(word in lower_intent for word in ["create", "generate", "write"]):
            action = "create_content"

        task_plan = {
            "analysis": {
                "primary_intent": primary_intent,
                "search_terms": search_terms,
            },
            "steps": [
                {"action": "retrieve_information", "parameters": search_terms},
                {"action": action, "parameters": intent},
                {"action": "generate_response", "parameters": intent},
            ],
        }
        return task_plan
