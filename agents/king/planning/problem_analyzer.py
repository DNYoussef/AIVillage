import json
import logging
import os
from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig

from communications.protocol import Message, MessageType, StandardCommunicationProtocol

from ..utils.exceptions import AIVillageException
from .quality_assurance_layer import QualityAssuranceLayer
from .seal_enhanced_planner import SEALEnhancedPlanGenerator

logger = logging.getLogger(__name__)

class ProblemAnalyzer:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, agent, quality_assurance_layer: QualityAssuranceLayer):
        self.communication_protocol = communication_protocol
        self.agent = agent
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.enhanced_plan_generator = SEALEnhancedPlanGenerator()
        self.quality_assurance_layer = quality_assurance_layer

    async def analyze(self, content: str, rag_info: dict[str, Any], rule_compliance: float) -> dict[str, Any]:
        task_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(content)
        eudaimonia_score = self.quality_assurance_layer.eudaimonia_triangulator.triangulate(task_vector)

        analysis_prompt = f"""
        Problem: {content}
        RAG Information: {rag_info}
        Rule Compliance Score: {rule_compliance}
        Eudaimonia Score: {eudaimonia_score}

        Analyze the given problem considering the following aspects:
        1. The potential impact on eudaimonia for all living things
        2. How it relates to curiosity and learning
        3. Its implications for the AI village and its inhabitants
        4. Any self-preservation concerns

        Provide a comprehensive analysis that takes into account the rule compliance score and eudaimonia score, and suggests potential solutions or approaches.
        """

        response = await self.llm.complete(analysis_prompt)

        initial_analyses = await self._collect_agent_analyses(content)
        critiqued_analyses = await self._collect_critiqued_analyses(initial_analyses)
        revised_analyses = await self._collect_revised_analyses(critiqued_analyses)

        consolidated_analysis = await self._consolidate_analyses(revised_analyses, response.text)

        return {
            "analysis": consolidated_analysis,
            "rule_compliance": rule_compliance,
            "eudaimonia_score": eudaimonia_score,
            "rag_info": rag_info
        }

    async def _collect_agent_analyses(self, task: str) -> list[dict[str, Any]]:
        analyses = []
        agents = await self.communication_protocol.get_all_agents()
        for agent in agents:
            analysis_request = Message(
                type=MessageType.QUERY,
                sender="King",
                receiver=agent,
                content={"task": task, "action": "analyze_problem"}
            )
            response = await self.communication_protocol.send_and_wait(analysis_request)
            analyses.append({"agent": agent, "analysis": response.content["analysis"]})
        logger.debug(f"Collected {len(analyses)} agent analyses")
        return analyses

    async def _collect_critiqued_analyses(self, initial_analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        critiqued_analyses = []
        agents = [analysis["agent"] for analysis in initial_analyses]
        for i, analysis in enumerate(initial_analyses):
            critiques = []
            for j, critic_agent in enumerate(agents):
                if i != j:  # Skip self-critique
                    critique_request = Message(
                        type=MessageType.QUERY,
                        sender="King",
                        receiver=critic_agent,
                        content={"analysis": analysis["analysis"], "action": "critique_analysis"}
                    )
                    response = await self.communication_protocol.send_and_wait(critique_request)
                    critiques.append({"critic": critic_agent, "critique": response.content["critique"]})
            critiqued_analyses.append({**analysis, "critiques": critiques})
        logger.debug(f"Collected critiques for {len(critiqued_analyses)} analyses")
        return critiqued_analyses

    async def _collect_revised_analyses(self, critiqued_analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        revised_analyses = []
        for analysis in critiqued_analyses:
            revision_request = Message(
                type=MessageType.QUERY,
                sender="King",
                receiver=analysis["agent"],
                content={"original_analysis": analysis["analysis"], "critiques": analysis["critiques"], "action": "revise_analysis"}
            )
            response = await self.communication_protocol.send_and_wait(revision_request)
            revised_analyses.append({**analysis, "revised_analysis": response.content["revised_analysis"]})
        logger.debug(f"Collected {len(revised_analyses)} revised analyses")
        return revised_analyses

    async def _consolidate_analyses(self, revised_analyses: list[dict[str, Any]], king_analysis: str) -> str:
        consolidation_prompt = f"""
        King's Analysis: {king_analysis}

        Agent Analyses:
        {[analysis['revised_analysis'] for analysis in revised_analyses]}

        Consolidate these analyses into a comprehensive, coherent analysis that:
        1. Incorporates insights from all agents
        2. Resolves any contradictions or conflicts
        3. Prioritizes insights based on their alignment with eudaimonia and the core rules
        4. Provides a clear, actionable summary of the problem and potential solutions

        Ensure the consolidated analysis maintains a focus on eudaimonia, curiosity, protection of the AI village, and appropriate self-preservation.
        """

        response = await self.llm.complete(consolidation_prompt)
        return response.text

    async def update_models(self, task: dict[str, Any], result: Any):
        try:
            logger.info(f"Updating models with task result: {result}")
            await self.enhanced_plan_generator.update(task, result)
            await self.quality_assurance_layer.update_task_history(task, result.get("performance", 0.5), result.get("uncertainty", 0.5))
        except Exception as e:
            logger.error(f"Error updating models: {e!s}", exc_info=True)
            raise AIVillageException(f"Error updating models: {e!s}")

    async def save_models(self, path: str):
        try:
            logger.info(f"Saving problem analyzer models to {path}")
            os.makedirs(path, exist_ok=True)
            self.enhanced_plan_generator.save(os.path.join(path, "enhanced_plan_generator.pt"))
            self.quality_assurance_layer.save(os.path.join(path, "quality_assurance_layer.json"))

            # Save other necessary data
            data = {
                "llm_config": self.llm.config.dict()
            }
            with open(os.path.join(path, "problem_analyzer_data.json"), "w") as f:
                json.dump(data, f)

            logger.info("Problem analyzer models saved successfully")
        except Exception as e:
            logger.error(f"Error saving problem analyzer models: {e!s}", exc_info=True)
            raise AIVillageException(f"Error saving problem analyzer models: {e!s}")

    async def load_models(self, path: str):
        try:
            logger.info(f"Loading problem analyzer models from {path}")
            self.enhanced_plan_generator.load(os.path.join(path, "enhanced_plan_generator.pt"))
            self.quality_assurance_layer = QualityAssuranceLayer.load(os.path.join(path, "quality_assurance_layer.json"))

            # Load other necessary data
            with open(os.path.join(path, "problem_analyzer_data.json")) as f:
                data = json.load(f)
            self.llm = OpenAIGPTConfig(**data["llm_config"]).create()

            logger.info("Problem analyzer models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading problem analyzer models: {e!s}", exc_info=True)
            raise AIVillageException(f"Error loading problem analyzer models: {e!s}")

    async def introspect(self) -> dict[str, Any]:
        return {
            "type": "ProblemAnalyzer",
            "description": "Analyzes problems based on content, RAG information, rule compliance, and eudaimonia scores",
            "quality_assurance_info": self.quality_assurance_layer.get_info(),
            "enhanced_plan_generator_info": self.enhanced_plan_generator.get_info()
        }
