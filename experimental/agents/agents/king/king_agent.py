from typing import Any

from langroid.language_models.openai_gpt import OpenAIGPTConfig
from torch import nn

from agents.unified_base_agent import UnifiedAgentConfig, UnifiedBaseAgent
from agents.utils.task import Task as LangroidTask
from core.error_handling import Message, MessageType, StandardCommunicationProtocol
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.retrieval.vector_store import VectorStore
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.utils.error_handling import log_and_handle_errors

from .analytics.unified_analytics import UnifiedAnalytics
from .coordinator import KingCoordinator
from .dynamic_knowledge_integration_agent import DynamicKnowledgeIntegrationAgent
from .evolution_manager import EvolutionManager, run_evolution_and_optimization
from .key_concept_extractor import KeyConceptExtractor
from .knowledge_graph_agent import KnowledgeGraphAgent

# `planning_and_task_management` was split into separate `planning` and
# `task_management` packages.  The unified planning/management class now
# resides in `planning.unified_planning`.
from .planning.unified_planning import UnifiedPlanningAndManagement
from .reasoning_agent import ReasoningAgent
from .response_generation_agent import ResponseGenerationAgent
from .task_planning_agent import TaskPlanningAgent
from .user_intent_interpreter import UserIntentInterpreter

# Backwards compatibility alias
KingAgentConfig = UnifiedAgentConfig


class KingAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore,
        knowledge_tracker: UnifiedKnowledgeTracker | None = None,
    ) -> None:
        super().__init__(config, communication_protocol, knowledge_tracker)
        self.vector_store = vector_store
        self.rag_pipeline = EnhancedRAGPipeline(
            config.rag_config, knowledge_tracker
        )  # Initialize the RAG pipeline
        self.coordinator = KingCoordinator(config, communication_protocol)
        self.unified_planning_and_management = UnifiedPlanningAndManagement(
            communication_protocol, self.rag_pipeline, self
        )
        self.unified_analytics = UnifiedAnalytics()
        self.evolution_manager = EvolutionManager()
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        self.response_generation_agent = ResponseGenerationAgent(llm_config)

        # Add tools
        self.add_tool("coordinate_task", self.coordinator.coordinate_task)
        self.add_tool("plan_task", self.unified_planning_and_management.make_decision)
        self.add_tool(
            "generate_response", self.response_generation_agent.generate_response
        )

    async def _maybe_async(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        if hasattr(result, "__await") or hasattr(result, "__await__"):
            return await result
        return result

    @log_and_handle_errors
    async def process_user_input(self, user_input: str) -> dict[str, Any]:
        interpreter = UserIntentInterpreter(OpenAIGPTConfig(chat_model="gpt-4"))
        extractor = KeyConceptExtractor(OpenAIGPTConfig(chat_model="gpt-4"))
        planner = TaskPlanningAgent()
        kg_agent = KnowledgeGraphAgent(OpenAIGPTConfig(chat_model="gpt-4"))
        reasoner = ReasoningAgent(OpenAIGPTConfig(chat_model="gpt-4"), kg_agent)
        dki_agent = DynamicKnowledgeIntegrationAgent(None)

        interpreted_intent = await self._maybe_async(
            interpreter.interpret_intent, user_input
        )
        key_concepts = await self._maybe_async(
            extractor.extract_key_concepts, user_input
        )
        task_plan = await self._maybe_async(planner.generate_task_plan, key_concepts)
        graph_data = await self._maybe_async(kg_agent.query_graph, user_input)
        reasoning_result = await self._maybe_async(
            reasoner.perform_reasoning, graph_data
        )
        response = await self._maybe_async(
            self.response_generation_agent.generate_response, reasoning_result
        )
        await self._maybe_async(dki_agent.integrate_new_knowledge, reasoning_result)

        return {
            "interpreted_intent": interpreted_intent,
            "key_concepts": key_concepts,
            "task_plan": task_plan,
            "reasoning_result": reasoning_result,
            "response": response,
        }

    @log_and_handle_errors
    async def process_message(self, message: Message) -> Any:
        if message.type == MessageType.TASK:
            task_dict = message.content
            task = LangroidTask(
                self,
                task_dict.get("content"),
                task_dict.get("id", ""),
                task_dict.get("priority", 1),
            )
            task.type = task_dict.get("type", "general")
            return await self.execute_task(task)
        return await self.coordinator.handle_message(message)

    @log_and_handle_errors
    async def execute_task(self, task: LangroidTask) -> dict[str, Any]:
        self.logger.info(f"Executing task: {task}")
        start_time = self.unified_analytics.get_current_time()
        result = await self.unified_planning_and_management.manage_task(
            {
                "type": getattr(task, "type", "general"),
                "content": task.name if hasattr(task, "name") else task.content,
            }
        )
        end_time = self.unified_analytics.get_current_time()
        execution_time = end_time - start_time

        self.unified_analytics.record_task_completion(
            getattr(task, "task_id", "unknown"),
            execution_time,
            result.get("success", False),
        )
        self.unified_analytics.update_performance_history(
            result.get("performance", 0.5)
        )

        await self.continuous_learning_layer.update(task, result)
        return result

    @log_and_handle_errors
    async def evolve(self) -> None:
        await super().evolve()
        await run_evolution_and_optimization(self)
        await self.coordinator.evolve()
        self.logger.info("KingAgent evolved")

    @log_and_handle_errors
    async def get_status(self) -> dict[str, Any]:
        base_status = await super().get_status()
        king_status = {
            "components": {
                "coordinator": await self.coordinator.introspect(),
                "unified_planning_and_management": await self.unified_planning_and_management.introspect(),
                "unified_analytics": self.unified_analytics.get_info(),
                "evolution_manager": self.evolution_manager.get_info(),
            },
            "analytics": self.unified_analytics.generate_summary_report(),
        }
        return {**base_status, **king_status}

    @log_and_handle_errors
    async def update_config(self, new_config: dict[str, Any]) -> None:
        await super().update_config(new_config)
        # Update configurations of other components as needed
        await self.coordinator.update_config(new_config)
        # Add other component config updates if necessary

    @log_and_handle_errors
    async def shutdown(self) -> None:
        await super().shutdown()
        # Perform cleanup for other components as needed
        await self.coordinator.save_models("coordinator_shutdown_backup")
        self.logger.info("KingAgent shut down")

    def create_model_from_architecture(self, architecture: dict[str, Any]) -> nn.Module:
        layers = []
        in_features = architecture["input_size"]

        for i in range(architecture["num_layers"]):
            out_features = architecture["hidden_sizes"][i]
            layers.append(nn.Linear(in_features, out_features))

            if architecture["activation"] == "relu":
                layers.append(nn.ReLU())
            elif architecture["activation"] == "tanh":
                layers.append(nn.Tanh())
            elif architecture["activation"] == "sigmoid":
                layers.append(nn.Sigmoid())

            if (
                i < architecture["num_layers"] - 1
            ):  # Don't add dropout after the last layer
                layers.append(nn.Dropout(architecture["dropout_rate"]))

            in_features = out_features

        layers.append(nn.Linear(in_features, architecture["output_size"]))

        return nn.Sequential(*layers)

    @log_and_handle_errors
    async def update_model_architecture(self, architecture: dict[str, Any]) -> None:
        self.logger.info(f"Updating model architecture: {architecture}")
        try:
            new_model = self.create_model_from_architecture(architecture)
            await self.unified_planning_and_management.update_model(new_model)
            await self.response_generation_agent.update_model(new_model)
            self.current_architecture = architecture
            self.logger.info("Model architecture updated successfully")
        except Exception as e:
            self.logger.exception(f"Failed to update model architecture: {e!s}")
            raise

    @log_and_handle_errors
    async def update_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        self.logger.info(f"Updating hyperparameters: {hyperparameters}")
        try:
            await self.unified_planning_and_management.update_hyperparameters(
                hyperparameters
            )
            await self.response_generation_agent.update_hyperparameters(hyperparameters)
            self.current_hyperparameters = hyperparameters
            self.logger.info("Hyperparameters updated successfully")
        except Exception as e:
            self.logger.exception(f"Failed to update hyperparameters: {e!s}")
            raise

    # ... (rest of the code remains the same)
