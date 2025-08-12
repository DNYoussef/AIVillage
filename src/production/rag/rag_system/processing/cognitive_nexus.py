# rag_system/processing/cognitive_nexus.py

from typing import Any

from ..core.agent_interface import AgentInterface
from ..core.interface import ReasoningEngine
from .self_referential_query_processor import (
    SelfReferentialQueryProcessor,
)


class CognitiveNexus:
    def __init__(
        self,
        reasoning_engine: ReasoningEngine,
        self_ref_processor: SelfReferentialQueryProcessor,
    ) -> None:
        self.reasoning_engine = reasoning_engine
        self.self_ref_processor = self_ref_processor

    async def process(self, query: str, context: dict[str, Any]):
        if self._is_self_referential(query):
            return await self.self_ref_processor.process_self_query(query)
        # Use the reasoning engine for non-self-referential queries
        return await self.reasoning_engine.reason(query, context)

    async def integrate(
        self,
        query: str,
        constructed_knowledge: dict[str, Any],
        final_plan: dict[str, Any],
        retrieval_history: list[dict[str, Any]],
        agent: AgentInterface,
    ) -> str:
        prompt = f"""
        Cognitive Integration Task:

        Original Query: {query}

        Final Retrieval Plan:
        {self._format_plan(final_plan)}

        Constructed Knowledge:
        {self._format_knowledge(constructed_knowledge)}

        Retrieval History:
        {self._format_retrieval_history(retrieval_history)}

        Your task is to:
        1. Analyze the relationship between the original query, the final retrieval plan, and the constructed knowledge.
        2. Identify any gaps or inconsistencies in the information.
        3. Synthesize a comprehensive understanding that addresses the original query.
        4. Provide reasoning for your synthesis, referencing specific parts of the constructed knowledge and retrieval plan.
        5. If there are any uncertainties or areas requiring further investigation, explicitly state them.

        Produce a final answer that:
        - Directly addresses the original query
        - Incorporates insights from the constructed knowledge
        - Reflects the strategy outlined in the final retrieval plan
        - Acknowledges any limitations or uncertainties in the available information

        Format your response as follows:

        Synthesis:
        [Your synthesized understanding here]

        Reasoning:
        [Your step-by-step reasoning process here]

        Final Answer:
        [Your concise final answer to the original query]

        Confidence: [High/Medium/Low]

        Areas for Further Investigation:
        [List any areas that require additional research or clarification]
        """

        return await agent.generate(prompt)

    def _is_self_referential(self, query: str) -> bool:
        # Implement logic to detect self-referential queries
        return query.strip().upper().startswith("SELF:")

    def _format_plan(self, plan: dict[str, Any]) -> str:
        # Convert the plan dictionary into a formatted string
        return "\n".join([f"- {key}: {value}" for key, value in plan.items()])

    def _format_knowledge(self, knowledge: dict[str, Any]) -> str:
        # Convert the constructed knowledge into a formatted string
        formatted = []
        for key, value in knowledge.items():
            formatted.append(f"{key}:")
            if isinstance(value, list):
                formatted.extend([f"  - {item}" for item in value])
            else:
                formatted.append(f"  {value}")
        return "\n".join(formatted)

    def _format_retrieval_history(self, history: list[dict[str, Any]]) -> str:
        # Format the retrieval history, showing how the search evolved
        formatted = []
        for i, step in enumerate(history, 1):
            formatted.append(f"Step {i}:")
            formatted.append(f"  Plan: {step['plan']}")
            formatted.append("  Top Results:")
            for result in step["top_results"][:3]:  # Show top 3 results for brevity
                formatted.append(f"    - {result['id']}: {result['title']}")
        return "\n".join(formatted)
