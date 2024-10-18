import logging
from typing import Dict, Any, List
from ..sage.sage_agent import SageAgent
from ..magi.magi_agent import MagiAgent
from .route_llm import AgentRouter
from ..communication.protocol import StandardCommunicationProtocol, Message, MessageType
from ..utils.exceptions import AIVillageException
from rag_system.core.pipeline import EnhancedRAGPipeline as RAGSystem
from .decision_maker import DecisionMaker
from .unified_task_manager import UnifiedTaskManager
from .problem_analyzer import ProblemAnalyzer

logger = logging.getLogger(__name__)

class KingCoordinator:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: RAGSystem, king_agent):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.king_agent = king_agent
        self.agents = {
            'sage': SageAgent(communication_protocol),
            'magi': MagiAgent(communication_protocol)
        }
        self.router = AgentRouter()
        self.decision_maker = DecisionMaker(communication_protocol, rag_system, king_agent)
        self.task_manager = UnifiedTaskManager(communication_protocol, len(self.agents), 10)
        self.problem_analyzer = ProblemAnalyzer(communication_protocol, self)
        self.update_agent_list()

    async def handle_task_message(self, message: Message):
        try:
            routing_decisions = await self.router.route([message.content['description']])
            routing_decision, confidence = routing_decisions[0]
            
            if routing_decision == 'undecided' or confidence < self.router.confidence_threshold:
                decision_result = await self.decision_maker.make_decision(message.content['description'])
                await self._implement_decision(decision_result)
            else:
                await self.assign_task_to_agent(routing_decision, message)
        except Exception as e:
            logger.error(f"Error in handling task message: {str(e)}")
            raise AIVillageException(f"Error in handling task message: {str(e)}")

    async def assign_task_to_agent(self, agent_name: str, message: Message):
        if agent_name not in self.agents:
            logger.warning(f"Unknown agent {agent_name}. Falling back to decision maker.")
            decision_result = await self.decision_maker.make_decision(message.content['description'])
            await self._implement_decision(decision_result)
        else:
            task = await self.task_manager.create_task(
                message.content['description'],
                agent_name
            )
            await self.task_manager.assign_task(task)

    async def _implement_decision(self, decision_result: Dict[str, Any]):
        try:
            chosen_alternative = decision_result['chosen_alternative']
            plan = decision_result['plan']
            suggested_agent = decision_result['suggested_agent']

            task = await self.task_manager.create_task(
                description=chosen_alternative,
                agent=suggested_agent
            )
            await self.task_manager.assign_task(task)

            # Implement the plan
            for step in plan:
                subtask = await self.task_manager.create_task(
                    description=step['description'],
                    agent=step.get('agent', suggested_agent)
                )
                await self.task_manager.assign_task(subtask)

        except Exception as e:
            logger.error(f"Error implementing decision: {str(e)}")
            raise AIVillageException(f"Error implementing decision: {str(e)}")

    async def process_task_completion(self, task: Dict[str, Any], result: Any):
        try:
            # Update router
            await self.router.train_model([{'task': task['description'], 'assigned_agent': task['assigned_agents'][0]}])
            
            # Update task manager
            await self.task_manager.complete_task(task['id'], result)
            
            # Update decision maker
            await self.decision_maker.update_model(task, result)
            
            # Update problem analyzer (which includes SEALEnhancedPlanGenerator)
            await self.problem_analyzer.update_models(task, result)
            
            # Update MCTS in decision maker
            await self.decision_maker.update_mcts(task, result)

            # Update the King agent
            await self.king_agent.update(task, result)

        except Exception as e:
            logger.error(f"Error processing task completion: {str(e)}")
            raise AIVillageException(f"Error processing task completion: {str(e)}")

    async def save_models(self, path: str):
        try:
            self.router.save(f"{path}/agent_router.pt")
            await self.decision_maker.save_models(f"{path}/decision_maker")
            await self.task_manager.save_models(f"{path}/task_manager")
            await self.problem_analyzer.save_models(f"{path}/problem_analyzer")
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise AIVillageException(f"Error saving models: {str(e)}")

    async def load_models(self, path: str):
        try:
            self.router.load(f"{path}/agent_router.pt")
            await self.decision_maker.load_models(f"{path}/decision_maker")
            await self.task_manager.load_models(f"{path}/task_manager")
            await self.problem_analyzer.load_models(f"{path}/problem_analyzer")
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise AIVillageException(f"Error loading models: {str(e)}")

    async def create_final_analysis(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            combined_analysis = {
                "agent_analyses": revised_analyses,
                "rag_info": rag_info
            }
            final_analysis = await self.king_agent.generate(f"Create a final analysis based on the following information: {combined_analysis}")
            return {"final_analysis": final_analysis}
        except Exception as e:
            logger.error(f"Error creating final analysis: {str(e)}")
            raise AIVillageException(f"Error creating final analysis: {str(e)}")

    def update_agent_list(self):
        agent_list = list(self.agents.keys())
        self.router.update_agent_list(agent_list)
        logger.info(f"Updated agent list: {agent_list}")

    async def add_agent(self, agent_name: str, agent_instance):
        self.agents[agent_name] = agent_instance
        self.update_agent_list()
        logger.info(f"Added new agent: {agent_name}")

    async def remove_agent(self, agent_name: str):
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.update_agent_list()
            logger.info(f"Removed agent: {agent_name}")
        else:
            logger.warning(f"Attempted to remove non-existent agent: {agent_name}")

    async def introspect(self) -> Dict[str, Any]:
        return {
            "agents": list(self.agents.keys()),
            "router_info": self.router.introspect(),
            "decision_maker_info": await self.decision_maker.introspect(),
            "task_manager_info": await self.task_manager.introspect(),
            "problem_analyzer_info": await self.problem_analyzer.introspect()
        }
