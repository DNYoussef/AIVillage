import torch
import torch.nn as nn
from typing import Dict, Any
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.retrieval.vector_store import VectorStore
from .coordinator import KingCoordinator
from .planning_and_task_management.unified_planning_and_management import UnifiedPlanningAndManagement
from .analytics.unified_analytics import UnifiedAnalytics
from .evolution_manager import EvolutionManager, run_evolution_and_optimization
from .response_generation_agent import ResponseGenerationAgent
from rag_system.utils.error_handling import log_and_handle_errors
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.core.pipeline import EnhancedRAGPipeline

class KingAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore
    ):
        super().__init__(config, communication_protocol)
        self.vector_store = vector_store
        self.rag_pipeline = EnhancedRAGPipeline()  # Initialize the RAG pipeline
        self.coordinator = KingCoordinator(config, communication_protocol)
        self.unified_planning_and_management = UnifiedPlanningAndManagement(communication_protocol, self.rag_pipeline, self)
        self.unified_analytics = UnifiedAnalytics()
        self.evolution_manager = EvolutionManager()
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        self.response_generation_agent = ResponseGenerationAgent(llm_config)

        # Add tools
        self.add_tool("coordinate_task", self.coordinator.coordinate_task)
        self.add_tool("plan_task", self.unified_planning_and_management.make_decision)
        self.add_tool("generate_response", self.response_generation_agent.generate_response)

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
        result = await self.unified_planning_and_management.manage_task(task)
        end_time = self.unified_analytics.get_current_time()
        execution_time = end_time - start_time

        self.unified_analytics.record_task_completion(task.get('id', 'unknown'), execution_time, result.get('success', False))
        self.unified_analytics.update_performance_history(result.get('performance', 0.5))

        await self.continuous_learning_layer.update(task, result)
        return result

    @log_and_handle_errors
    async def evolve(self):
        await super().evolve()
        await run_evolution_and_optimization(self)
        await self.coordinator.evolve()
        self.logger.info("KingAgent evolved")

    @log_and_handle_errors
    async def get_status(self) -> Dict[str, Any]:
        base_status = await super().get_status()
        king_status = {
            "components": {
                "coordinator": await self.coordinator.introspect(),
                "unified_planning_and_management": await self.unified_planning_and_management.introspect(),
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
        # Add other component config updates if necessary

    @log_and_handle_errors
    async def shutdown(self) -> None:
        await super().shutdown()
        # Perform cleanup for other components as needed
        await self.coordinator.save_models("coordinator_shutdown_backup")
        self.logger.info("KingAgent shut down")

    def create_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        layers = []
        in_features = architecture['input_size']

        for i in range(architecture['num_layers']):
            out_features = architecture['hidden_sizes'][i]
            layers.append(nn.Linear(in_features, out_features))
            
            if architecture['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif architecture['activation'] == 'tanh':
                layers.append(nn.Tanh())
            elif architecture['activation'] == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if i < architecture['num_layers'] - 1:  # Don't add dropout after the last layer
                layers.append(nn.Dropout(architecture['dropout_rate']))
            
            in_features = out_features

        layers.append(nn.Linear(in_features, architecture['output_size']))
        
        return nn.Sequential(*layers)

    @log_and_handle_errors
    async def update_model_architecture(self, architecture: Dict[str, Any]):
        self.logger.info(f"Updating model architecture: {architecture}")
        try:
            new_model = self.create_model_from_architecture(architecture)
            await self.unified_planning_and_management.update_model(new_model)
            await self.response_generation_agent.update_model(new_model)
            self.current_architecture = architecture
            self.logger.info("Model architecture updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update model architecture: {str(e)}")
            raise

    @log_and_handle_errors
    async def update_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self.logger.info(f"Updating hyperparameters: {hyperparameters}")
        try:
            await self.unified_planning_and_management.update_hyperparameters(hyperparameters)
            await self.response_generation_agent.update_hyperparameters(hyperparameters)
            self.current_hyperparameters = hyperparameters
            self.logger.info("Hyperparameters updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update hyperparameters: {str(e)}")
            raise

    # ... (rest of the code remains the same)
