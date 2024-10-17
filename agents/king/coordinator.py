import logging
from ..sage.sage_agent import SageAgent
from ..magi.magi_agent import MagiAgent
from .route_llm import AgentRouter
from ..communication.protocol import StandardCommunicationProtocol, Message, MessageType
from ..utils.exceptions import AIVillageException
from ..rag_system import RAGSystem
from .decision_maker import DecisionMaker
from .unified_task_manager import UnifiedTaskManager
from .problem_analyzer import ProblemAnalyzer
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class KingCoordinator:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: RAGSystem, ai_provider):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.agents = {
            'sage': SageAgent(communication_protocol),
            'magi': MagiAgent(communication_protocol)
        }
        self.router = AgentRouter()
        self.decision_maker = DecisionMaker(communication_protocol, rag_system, ai_provider)
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
                assigned_agents=[agent_name]
            )
            await self.task_manager.assign_task(task)

    async def _implement_decision(self, decision_result: Dict[str, Any]):
        # Implement the decision made by the DecisionMaker
        # This method should be implemented based on the structure of decision_result
        pass

    async def process_task_completion(self, task: Dict[str, Any], result: Any):
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

    async def save_models(self, path: str):
        self.router.save(f"{path}/agent_router.pt")
        self.decision_maker.save_models(f"{path}/decision_maker")
        self.task_manager.save_models(f"{path}/task_manager")
        self.problem_analyzer.save_models(f"{path}/problem_analyzer")

    async def load_models(self, path: str):
        self.router.load(f"{path}/agent_router.pt")
        self.decision_maker.load_models(f"{path}/decision_maker")
        self.task_manager.load_models(f"{path}/task_manager")
        self.problem_analyzer.load_models(f"{path}/problem_analyzer")

    async def create_final_analysis(self, revised_analyses: List[Dict[str, Any]], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        # This method should be implemented to create the final analysis
        # It should combine the revised analyses from different agents and the RAG information
        pass

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

    # Add more methods as needed for the KingCoordinator's functionality


