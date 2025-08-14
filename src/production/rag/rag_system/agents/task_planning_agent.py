from typing import Any

from ..core.agent_interface import AgentInterface
from ..utils.embedding import BERTEmbeddingModel
from .key_concept_extractor import KeyConceptExtractorAgent


# Simple intent interpreter fallback
class UserIntentInterpreterAgent:
    """Simple intent interpreter for basic task categorization."""

    def interpret_intent(self, query: str) -> dict[str, Any]:
        """Simple intent classification."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["search", "find", "look for", "query"]):
            return {"intent": "search", "confidence": 0.8}
        elif any(word in query_lower for word in ["analyze", "analysis", "examine"]):
            return {"intent": "analyze", "confidence": 0.8}
        elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return {"intent": "summarize", "confidence": 0.8}
        elif any(word in query_lower for word in ["explain", "describe", "what is"]):
            return {"intent": "explain", "confidence": 0.8}
        else:
            return {"intent": "general", "confidence": 0.5}


class TaskPlanningAgent(AgentInterface):
    """Agent responsible for planning tasks based on user intent and extracted concepts."""

    def __init__(self) -> None:
        super().__init__()
        self.intent_interpreter = UserIntentInterpreterAgent()
        self.concept_extractor = KeyConceptExtractorAgent()
        self.embedding_model = BERTEmbeddingModel()

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

    async def generate(self, prompt: str) -> str:
        """Generate response with task planning context."""
        task_plan = self.plan_tasks(prompt)
        intent = task_plan.get("intent", {}).get("intent", "general")

        response = f"Task planning analysis for {intent} intent: {prompt[:50]}..."

        steps = task_plan.get("tasks", {}).get("steps", [])
        if steps:
            response += f" Planned {len(steps)} steps: "
            response += " -> ".join(
                [step.get("action", "unknown") for step in steps[:3]]
            )

        return response

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        try:
            _, embeddings = self.embedding_model.encode(text)
            mean_embedding = embeddings.mean(dim=0).detach().cpu().tolist()
            return [float(x) for x in mean_embedding]
        except Exception:
            # Fallback to deterministic random embedding
            import hashlib
            import random

            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            rng = random.Random(seed)
            return [rng.random() for _ in range(self.embedding_model.hidden_size)]

    async def rerank(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Rerank results based on task relevance."""
        task_plan = self.plan_tasks(query)
        search_terms = (
            task_plan.get("tasks", {}).get("analysis", {}).get("search_terms", [])
        )

        for result in results:
            content = result.get("content", "").lower()
            relevance_boost = 0.0

            # Boost based on search term matches
            for term in search_terms:
                if term.lower() in content:
                    relevance_boost += 0.1

            result["score"] = result.get("score", 0.0) + relevance_boost

        return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return introspection information."""
        return {
            "type": "TaskPlanningAgent",
            "embedding_model": "BERTEmbeddingModel",
            "intent_interpreter": "SimplePatternBasedInterpreter",
            "concept_extractor": "KeyConceptExtractorAgent",
            "capabilities": [
                "task_planning",
                "intent_interpretation",
                "concept_extraction",
                "multi_step_workflows",
            ],
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with another agent."""
        task_plan = self.plan_tasks(message)
        enhanced_message = f"Task plan for '{message}': {task_plan}"
        response = await recipient.generate(enhanced_message)
        return f"Sent task plan: {task_plan.get('tasks', {}).get('steps', [])}, Received: {response}"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space for task decomposition."""
        task_plan = self.plan_tasks(query)
        intent = task_plan.get("intent", {})

        background = f"Task planning analysis: Detected {intent.get('intent', 'unknown')} intent "
        background += f"with {intent.get('confidence', 0.0):.1f} confidence. "

        steps = task_plan.get("tasks", {}).get("steps", [])
        background += f"Generated {len(steps)}-step execution plan."

        refined_query = f"Task-enhanced query: {query} [Plan: {len(steps)} steps]"
        return background, refined_query
