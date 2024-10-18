import logging
from typing import List, Dict, Any
from ..communication.protocol import StandardCommunicationProtocol, Message, MessageType
from ..utils.exceptions import AIVillageException
from .seal_enhanced_planner import SEALEnhancedPlanGenerator

logger = logging.getLogger(__name__)

class ProblemAnalyzer:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, king_agent):
        self.communication_protocol = communication_protocol
        self.king_agent = king_agent
        self.enhanced_plan_generator = SEALEnhancedPlanGenerator()

    async def analyze(self, task: str, rag_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Starting problem analysis for task: {task}")
            agent_analyses = await self._collect_agent_analyses(task)
            critiqued_analyses = await self._collect_critiqued_analyses(agent_analyses)
            revised_analyses = await self._collect_revised_analyses(critiqued_analyses)
            
            # Generate enhanced plan
            enhanced_plan = await self.enhanced_plan_generator.generate_enhanced_plan(task, rag_info)
            logger.debug(f"Enhanced plan generated: {enhanced_plan}")
            
            final_analysis = await self._create_final_analysis(revised_analyses, rag_info, enhanced_plan)
            logger.info("Problem analysis completed successfully")
            return final_analysis
        except Exception as e:
            logger.error(f"Error in problem analysis: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error in problem analysis: {str(e)}")

    async def _collect_agent_analyses(self, task: str) -> List[Dict[str, Any]]:
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

    async def _collect_critiqued_analyses(self, initial_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    async def _collect_revised_analyses(self, critiqued_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        revised_analyses = []
        for analysis in critiqued_analyses:
            revision_request = Message(
                type=MessageType.QUERY,
                sender="King",
                receiver=analysis["agent"],
                content={"original_analysis": analysis["analysis"], "critiques": analysis["critiques"], "action": "revise_analysis"}
            )
            response = await self.communication_protocol.send_and_wait(revision_request)
            revised_analyses.append({"agent": analysis["agent"], "revised_analysis": response.content["revised_analysis"]})
        logger.debug(f"Collected {len(revised_analyses)} revised analyses")
        return revised_analyses
    
    async def _create_final_analysis(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any], enhanced_plan: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Create a final analysis based on the following information:
            Revised Analyses: {revised_analyses}
            RAG Info: {rag_info}
            Enhanced Plan: {enhanced_plan}

            Synthesize this information into a comprehensive final analysis.
            """
            final_analysis = await self.king_agent.generate(prompt)
            final_analysis_dict = {
                'final_analysis': final_analysis,
                'revised_analyses': revised_analyses,
                'rag_info': rag_info,
                'enhanced_plan': enhanced_plan
            }
            logger.info("Final analysis created successfully")
            return final_analysis_dict
        except Exception as e:
            logger.error(f"Error in creating final analysis: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error in creating final analysis: {str(e)}")

    async def update_models(self, task: Dict[str, Any], result: Dict[str, Any]):
        try:
            logger.info(f"Updating models with task result: {result}")
            await self.enhanced_plan_generator.update(task, result)
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating models: {str(e)}")

    async def save_models(self, path: str):
        try:
            logger.info(f"Saving models to {path}")
            self.enhanced_plan_generator.save(path)
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error saving models: {str(e)}")

    async def load_models(self, path: str):
        try:
            logger.info(f"Loading models from {path}")
            self.enhanced_plan_generator.load(path)
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error loading models: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "enhanced_plan_generator_info": self.enhanced_plan_generator.introspect()
        }
