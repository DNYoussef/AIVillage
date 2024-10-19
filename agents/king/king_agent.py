from typing import List, Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.unified_config import unified_config
from rag_system.retrieval.vector_store import VectorStore
from .coordinator import KingCoordinator
from .decision_maker import DecisionMaker
from .problem_analyzer import ProblemAnalyzer
from .unified_task_manager import UnifiedTaskManager
from .quality_assurance_layer import QualityAssuranceLayer
from .continuous_learner import ContinuousLearningLayer
from .subgoal_generator import SubGoalGenerator
from .unified_analytics import UnifiedAnalytics
from .evolution_manager import EvolutionManager
from .user_intent_interpreter import UserIntentInterpreter
from .key_concept_extractor import KeyConceptExtractor
from .task_planning_agent import TaskPlanningAgent
from .knowledge_graph_agent import KnowledgeGraphAgent
from .reasoning_agent import ReasoningAgent
from .response_generation_agent import ResponseGenerationAgent
from .dynamic_knowledge_integration_agent import DynamicKnowledgeIntegrationAgent
from agents.sage.self_evolving_system import SelfEvolvingSystem
import logging

logger = logging.getLogger(__name__)

class KingAgent(UnifiedBaseAgent):
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore
    ):
        super().__init__(unified_config, communication_protocol)
        self.rag_system = EnhancedRAGPipeline()
        self.vector_store = vector_store
        self.coordinator = KingCoordinator(communication_protocol, self.rag_system, self)
        self.decision_maker = DecisionMaker(communication_protocol, self.rag_system, self)
        self.problem_analyzer = ProblemAnalyzer(communication_protocol, self)
        self.task_manager = UnifiedTaskManager(communication_protocol)
        self.quality_assurance_layer = QualityAssuranceLayer()
        self.continuous_learning_layer = ContinuousLearningLayer(self.quality_assurance_layer)
        self.subgoal_generator = SubGoalGenerator()
        self.unified_analytics = UnifiedAnalytics()
        self.evolution_manager = EvolutionManager()
        self.user_intent_interpreter = UserIntentInterpreter()
        self.key_concept_extractor = KeyConceptExtractor()
        self.task_planning_agent = TaskPlanningAgent()
        self.knowledge_graph_agent = KnowledgeGraphAgent()
        self.reasoning_agent = ReasoningAgent()
        self.response_generation_agent = ResponseGenerationAgent()
        self.dynamic_knowledge_integration_agent = DynamicKnowledgeIntegrationAgent()
        self.self_evolving_system = SelfEvolvingSystem(self)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Executing task: {task}")
            start_time = self.unified_analytics.get_current_time()
            result = await self.coordinator.coordinate_task(task)
            end_time = self.unified_analytics.get_current_time()
            execution_time = end_time - start_time

            self.unified_analytics.record_task_completion(task['id'], execution_time, result.get('success', False))
            self.unified_analytics.update_performance_history(result.get('performance', 0.5))

            await self.continuous_learning_layer.update(task, result)
            return result
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            return {"error": str(e)}

    async def handle_message(self, message: Message):
        try:
            if message.type == MessageType.TASK:
                result = await self.execute_task(message.content)
                response = Message(
                    type=MessageType.RESPONSE,
                    sender=self.name,
                    receiver=message.sender,
                    content=result,
                    parent_id=message.id
                )
                await self.communication_protocol.send_message(response)
            else:
                await self.coordinator.handle_message(message)
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")

    async def evolve(self):
        try:
            await self.self_evolving_system.evolve()
            await self.evolution_manager.evolve()
            await self.continuous_learning_layer.evolve()
            await self.coordinator.evolve()
            await self.decision_maker.evolve()
            await self.problem_analyzer.evolve()
            logger.info("KingAgent evolved")
        except Exception as e:
            logger.error(f"Error during evolution: {str(e)}")

    async def introspect(self) -> Dict[str, Any]:
        try:
            return {
                "name": self.name,
                "type": "KingAgent",
                "components": {
                    "coordinator": await self.coordinator.introspect(),
                    "decision_maker": await self.decision_maker.introspect(),
                    "problem_analyzer": await self.problem_analyzer.introspect(),
                    "task_manager": await self.task_manager.introspect(),
                    "quality_assurance_layer": self.quality_assurance_layer.get_info(),
                    "continuous_learning_layer": self.continuous_learning_layer.get_info(),
                    "unified_analytics": self.unified_analytics.get_info(),
                    "evolution_manager": self.evolution_manager.get_info(),
                    "self_evolving_system": {
                        "current_architecture": self.self_evolving_system.current_architecture,
                        "evolution_rate": self.self_evolving_system.evolution_rate,
                        "mutation_rate": self.self_evolving_system.mutation_rate,
                        "learning_rate": self.self_evolving_system.learning_rate,
                    }
                },
                "analytics": self.unified_analytics.generate_summary_report()
            }
        except Exception as e:
            logger.error(f"Error during introspection: {str(e)}")
            return {"error": str(e)}

    async def save_state(self, path: str):
        try:
            # Implement state saving logic
            pass
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")

    async def load_state(self, path: str):
        try:
            # Implement state loading logic
            pass
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
