import json
import logging
from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig

from core.error_handling import AIVillageException, error_handler, safe_execute

from .knowledge_graph_agent import KnowledgeGraphAgent

logger = logging.getLogger(__name__)


class ReasoningAgent:
    def __init__(
        self, llm_config: OpenAIGPTConfig, knowledge_graph_agent: KnowledgeGraphAgent
    ):
        self.llm = llm_config.create()
        self.knowledge_graph_agent = knowledge_graph_agent

    @error_handler.handle_error
    async def perform_reasoning(
        self, context: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Perform reasoning based on the given context and query.

        Args:
            context (Dict[str, Any]): The context for the reasoning task.
            query (str): The specific query or problem to reason about.

        Returns:
            Dict[str, Any]: The results of the reasoning process.
        """
        kg_query_result = await self.knowledge_graph_agent.query_graph(query)
        prompt = self._create_reasoning_prompt(context, query, kg_query_result)
        response = await self.llm.complete(prompt)
        return self._parse_reasoning_response(response.text)

    def _create_reasoning_prompt(
        self, context: dict[str, Any], query: str, kg_query_result: dict[str, Any]
    ) -> str:
        return f"""
        Given the following context, query, and knowledge graph query result:

        Context: {json.dumps(context, indent=2)}
        Query: {query}
        Knowledge Graph Query Result: {json.dumps(kg_query_result, indent=2)}

        Please perform advanced reasoning to address the query. Consider the following:
        1. Deductive reasoning: Draw logical conclusions based on the given information.
        2. Inductive reasoning: Identify patterns and make generalizations.
        3. Abductive reasoning: Form the most likely explanation for the observations.
        4. Probabilistic reasoning: Consider uncertainties and likelihoods.
        5. Analogical reasoning: Draw parallels between similar situations or concepts.

        In your reasoning process:
        - Integrate information from the context and knowledge graph query result.
        - Identify and resolve any conflicts or inconsistencies in the information.
        - Consider multiple perspectives and alternative explanations.
        - Provide a step-by-step explanation of your reasoning process.
        - Assign confidence levels to your conclusions (0-100%).
        - Identify any assumptions made and their potential impact on the conclusions.

        Present your reasoning and conclusions in a structured JSON format, including:
        - Main conclusions
        - Supporting evidence
        - Reasoning steps
        - Confidence levels
        - Potential alternative explanations
        - Identified conflicts or inconsistencies
        - Assumptions made
        """

    def _parse_reasoning_response(self, response: str) -> dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse reasoning response: {response}")
            raise AIVillageException("Failed to parse reasoning response")

    @error_handler.handle_error
    async def resolve_conflicts(
        self, conflicting_info: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Resolve conflicts between different pieces of information.

        Args:
            conflicting_info (List[Dict[str, Any]]): List of conflicting information.

        Returns:
            Dict[str, Any]: The resolved information and explanation.
        """
        prompt = self._create_conflict_resolution_prompt(conflicting_info)
        response = await self.llm.complete(prompt)
        return self._parse_conflict_resolution_response(response.text)

    def _create_conflict_resolution_prompt(
        self, conflicting_info: list[dict[str, Any]]
    ) -> str:
        return f"""
        Given the following conflicting pieces of information:

        {json.dumps(conflicting_info, indent=2)}

        Please analyze the conflicts and propose a resolution. Consider the following:
        1. Evaluate the reliability and credibility of each source of information.
        2. Identify potential reasons for the discrepancies (e.g., outdated information, different contexts, errors).
        3. Determine if the conflicts can be reconciled or if they are mutually exclusive.
        4. Propose a resolution that best fits the available evidence and explains the discrepancies.
        5. If a definitive resolution is not possible, provide a nuanced explanation of the situation.

        Present your analysis and proposed resolution in a structured JSON format, including:
        - Summary of the conflicts
        - Analysis of each conflicting piece of information
        - Proposed resolution
        - Confidence level in the resolution (0-100%)
        - Explanation of the reasoning behind the resolution
        - Any remaining uncertainties or areas that require further investigation
        """

    def _parse_conflict_resolution_response(self, response: str) -> dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse conflict resolution response: {response}")
            raise AIVillageException("Failed to parse conflict resolution response")

    @error_handler.handle_error
    async def generate_explanation(self, reasoning_result: dict[str, Any]) -> str:
        """Generate a natural language explanation of the reasoning process and conclusions.

        Args:
            reasoning_result (Dict[str, Any]): The result of the reasoning process.

        Returns:
            str: A natural language explanation of the reasoning and conclusions.
        """
        prompt = self._create_explanation_prompt(reasoning_result)
        response = await self.llm.complete(prompt)
        return response.text

    def _create_explanation_prompt(self, reasoning_result: dict[str, Any]) -> str:
        return f"""
        Given the following reasoning result:

        {json.dumps(reasoning_result, indent=2)}

        Please generate a clear and concise natural language explanation of the reasoning process and conclusions. The explanation should:
        1. Summarize the main conclusions and their confidence levels.
        2. Explain the key steps in the reasoning process.
        3. Highlight the most important evidence supporting the conclusions.
        4. Address any uncertainties, alternative explanations, or assumptions.
        5. Explain how conflicts or inconsistencies were resolved (if applicable).
        6. Use analogies or examples to illustrate complex concepts (if appropriate).
        7. Be understandable to a non-expert audience while maintaining accuracy.

        Your explanation should be well-structured, using paragraphs to separate main ideas, and should flow logically from the problem statement to the conclusions.
        """

    @safe_execute
    async def process_query(
        self, context: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Process a query by performing reasoning, resolving conflicts, and generating an explanation.

        Args:
            context (Dict[str, Any]): The context for the reasoning task.
            query (str): The specific query or problem to reason about.

        Returns:
            Dict[str, Any]: A dictionary containing the reasoning results, resolved conflicts, and explanation.
        """
        reasoning_result = await self.perform_reasoning(context, query)

        # Check for conflicts in the reasoning result
        if reasoning_result.get("conflicts"):
            conflict_resolution = await self.resolve_conflicts(
                reasoning_result["conflicts"]
            )
            reasoning_result["conflict_resolution"] = conflict_resolution

        explanation = await self.generate_explanation(reasoning_result)

        return {"reasoning_result": reasoning_result, "explanation": explanation}


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        kg_agent = KnowledgeGraphAgent(llm_config)
        reasoning_agent = ReasoningAgent(llm_config, kg_agent)

        # Initialize the knowledge graph with some sample data
        kg_agent.graph.add_node("COVID-19", type="Disease")
        kg_agent.graph.add_node("Vaccine", type="Medical")
        kg_agent.graph.add_node("Mask Wearing", type="Preventive Measure")
        kg_agent.graph.add_edge("Vaccine", "COVID-19", relationship="prevents")
        kg_agent.graph.add_edge(
            "Mask Wearing", "COVID-19", relationship="reduces spread"
        )

        context = {
            "current_situation": "Global pandemic",
            "available_data": {"vaccine_efficacy": 0.95, "mask_effectiveness": 0.7},
        }
        query = "What are the most effective ways to combat the spread of COVID-19?"

        result = await reasoning_agent.process_query(context, query)

        print("Reasoning Result:")
        print(json.dumps(result["reasoning_result"], indent=2))
        print("\nExplanation:")
        print(result["explanation"])

    asyncio.run(main())
