from typing import List, Dict, Any
from ..communication.protocol import StandardCommunicationProtocol, Message, MessageType
from ..utils.exceptions import AIVillageException

class ProblemAnalyzer:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, king_coordinator):
        self.communication_protocol = communication_protocol
        self.king_coordinator = king_coordinator

    async def analyze(self, task: str, rag_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_analyses = await self._collect_agent_analyses(task)
            critiqued_analyses = await self._collect_critiqued_analyses(agent_analyses)
            revised_analyses = await self._collect_revised_analyses(critiqued_analyses)
            final_analysis = await self._create_final_analysis(revised_analyses, rag_info)
            return final_analysis
        except Exception as e:
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
        return revised_analyses
    
    async def _create_final_analysis(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            final_analysis = await self.king_coordinator.create_final_analysis(revised_analyses, rag_info)
            return final_analysis
        except Exception as e:
            raise AIVillageException(f"Error in creating final analysis: {str(e)}")