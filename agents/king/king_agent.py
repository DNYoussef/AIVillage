from typing import Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.retrieval.vector_store import VectorStore
from .coordinator import KingCoordinator
from .planning.reasoning.decision_maker import DecisionMaker
from .problem_analyzer import ProblemAnalyzer
from .unified_task_manager import UnifiedTaskManager
from .analytics.unified_analytics import UnifiedAnalytics
from .evolution_manager import EvolutionManager
from .user_intent_interpreter import UserIntentInterpreter
from .key_concept_extractor import KeyConceptExtractor
from .task_planning_agent import TaskPlanningAgent
from .knowledge_graph_agent import KnowledgeGraphAgent
from .reasoning_agent import ReasoningAgent
from .response_generation_agent import ResponseGenerationAgent
from .dynamic_knowledge_integration_agent import DynamicKnowledgeIntegrationAgent
from rag_system.utils.error_handling import log_and_handle_errors

class KingAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore
    ):
        super().__init__(config, communication_protocol)
        self.vector_store = vector_store
        self.coordinator = KingCoordinator(communication_protocol, self.rag_pipeline, self)
        self.decision_maker = DecisionMaker(communication_protocol, self.rag_pipeline, self)
        self.problem_analyzer = ProblemAnalyzer(communication_protocol, self)
        self.task_manager = UnifiedTaskManager(communication_protocol)
        self.unified_analytics = UnifiedAnalytics()
        self.evolution_manager = EvolutionManager()
        self.user_intent_interpreter = UserIntentInterpreter()
        self.key_concept_extractor = KeyConceptExtractor()
        self.task_planning_agent = TaskPlanningAgent()
        self.knowledge_graph_agent = KnowledgeGraphAgent()
        self.reasoning_agent = ReasoningAgent()
        self.response_generation_agent = ResponseGenerationAgent()
        self.dynamic_knowledge_integration_agent = DynamicKnowledgeIntegrationAgent()

        # Add tools
        self.add_tool("coordinate_task", self.coordinator.coordinate_task)
        self.add_tool("interpret_user_intent", self.user_intent_interpreter.interpret)
        self.add_tool("extract_key_concepts", self.key_concept_extractor.extract)
        self.add_tool("plan_task", self.task_planning_agent.plan)
        self.add_tool("query_knowledge_graph", self.knowledge_graph_agent.query)
        self.add_tool("reason", self.reasoning_agent.reason)
        self.add_tool("generate_response", self.response_generation_agent.generate)
        self.add_tool("integrate_knowledge", self.dynamic_knowledge_integration_agent.integrate)

    @log_and_handle_errors
    async def process_message(self, message: Message) -> Any:
        if message.type == MessageType.TASK:
            return await self.execute_task(message.content)
        else:
            return await self.coordinator.handle_message(message)

    @log_and_handle_errors
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Executing task: {task}")
        start_time = self.unified_analytics.get_current_time()
        result = await self.get_tool("coordinate_task")(task)
        end_time = self.unified_analytics.get_current_time()
        execution_time = end_time - start_time

        self.unified_analytics.record_task_completion(task['id'], execution_time, result.get('success', False))
        self.unified_analytics.update_performance_history(result.get('performance', 0.5))

        await self.continuous_learning_layer.update(task, result)
        return result

    @log_and_handle_errors
    async def evolve(self):
        await super().evolve()
        await self.evolution_manager.evolve()
        await self.coordinator.evolve()
        await self.decision_maker.evolve()
        await self.problem_analyzer.evolve()
        self.logger.info("KingAgent evolved")

    @log_and_handle_errors
    async def get_status(self) -> Dict[str, Any]:
        base_status = await super().get_status()
        king_status = {
            "components": {
                "coordinator": await self.coordinator.get_status(),
                "decision_maker": await self.decision_maker.get_status(),
                "problem_analyzer": await self.problem_analyzer.get_status(),
                "task_manager": await self.task_manager.get_status(),
                "unified_analytics": self.unified_analytics.get_info(),
                "evolution_manager": self.evolution_manager.get_info(),
            },
            "analytics": self.unified_analytics.generate_summary_report()
        }
        return {**base_status, **king_status}

    @log_and_handle_errors
    async def update_config(self, new_config: Dict[str, Any]) -> None:
        await super().update_config(new_config)
        # Update configurations of other components as needed
        await self.coordinator.update_config(new_config)
        await self.decision_maker.update_config(new_config)
        await self.problem_analyzer.update_config(new_config)

    @log_and_handle_errors
    async def shutdown(self) -> None:
        await super().shutdown()
        # Perform cleanup for other components as needed
        await self.coordinator.shutdown()
        await self.decision_maker.shutdown()
        await self.problem_analyzer.shutdown()
        self.logger.info("KingAgent shut down")
