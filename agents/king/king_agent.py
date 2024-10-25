import torch
import torch.nn as nn
from typing import Dict, Any, List
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.retrieval.vector_store import VectorStore
from rag_system.core.pipeline import EnhancedRAGPipeline
from .coordinator import KingCoordinator
from .planning.unified_planning_and_decision import UnifiedPlanningAndDecision
from .analytics.unified_analytics import UnifiedAnalytics
from .evolution_manager import EvolutionManager, run_evolution_and_optimization
from .response_generation_agent import ResponseGenerationAgent
from rag_system.utils.error_handling import log_and_handle_errors
from langroid.language_models.openai_gpt import OpenAIGPTConfig

class KingAgent(UnifiedBaseAgent):
    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        vector_store: VectorStore
    ):
        super().__init__(config, communication_protocol)
        self.vector_store = vector_store
        self.rag_pipeline = EnhancedRAGPipeline()
        self.coordinator = KingCoordinator(config, communication_protocol)
        self.unified_planning_and_decision = UnifiedPlanningAndDecision(communication_protocol, self.rag_pipeline, self)
        self.unified_analytics = UnifiedAnalytics()
        self.evolution_manager = EvolutionManager()
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        self.response_generation_agent = ResponseGenerationAgent(llm_config)

        # Add tools
        self.add_tool("coordinate_task", self.coordinator.coordinate_task)
        self.add_tool("plan_task", self.unified_planning_and_decision.make_decision)
        self.add_tool("generate_response", self.response_generation_agent.generate_response)

    async def integrate_rag_results(self, query: str, rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate RAG results into agent's decision making.

        :param query: Original query string.
        :param rag_results: Results from RAG system.
        :return: Processed and integrated results.
        """
        # Process RAG results through planning system
        processed_results = await self.unified_planning_and_management.process_rag_results(rag_results)
        
        # Extract key insights
        insights = await self._extract_key_insights(processed_results)
        
        # Update knowledge base
        await self._update_knowledge_base(insights)
        
        # Incorporate into decision making
        enhanced_results = await self.coordinator.incorporate_knowledge(processed_results)
        
        # Track analytics
        self.unified_analytics.track_rag_integration(query, rag_results, enhanced_results)
        
        return enhanced_results

    async def _extract_key_insights(self, processed_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key insights from processed RAG results."""
        insights = []
        for result in processed_results.get('results', []):
            insight = {
                'content': result.get('content'),
                'relevance': result.get('relevance_score'),
                'confidence': result.get('confidence_score'),
                'source': result.get('source'),
                'timestamp': result.get('timestamp')
            }
            insights.append(insight)
        return insights

    async def _update_knowledge_base(self, insights: List[Dict[str, Any]]):
        """Update knowledge base with new insights."""
        for insight in insights:
            await self.vector_store.add_document(
                content=insight['content'],
                metadata={
                    'relevance': insight['relevance'],
                    'confidence': insight['confidence'],
                    'source': insight['source'],
                    'timestamp': insight['timestamp']
                }
            )

    async def evolve_capabilities(self):
        """
        Evolve agent capabilities based on performance metrics.
        """
        # Get performance metrics
        performance_data = self.unified_analytics.get_performance_metrics()
        
        # Analyze areas for improvement
        improvement_areas = await self._analyze_improvement_areas(performance_data)
        
        # Evolve based on metrics
        await self.evolution_manager.evolve_based_on_metrics(performance_data)
        
        # Update model architecture
        new_architecture = self.evolution_manager.get_optimal_architecture()
        await self.update_model_architecture(new_architecture)
        
        # Evolve planning strategies
        await self.unified_planning_and_management.evolve_strategies(improvement_areas)
        
        # Update analytics tracking
        self.unified_analytics.track_evolution(improvement_areas)

    async def _analyze_improvement_areas(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze areas needing improvement based on performance data."""
        improvement_areas = {}
        
        # Analyze task completion rate
        if performance_data.get('task_completion_rate', 0) < 0.8:
            improvement_areas['task_planning'] = 0.8 - performance_data['task_completion_rate']
            
        # Analyze decision accuracy
        if performance_data.get('decision_accuracy', 0) < 0.9:
            improvement_areas['decision_making'] = 0.9 - performance_data['decision_accuracy']
            
        # Analyze response quality
        if performance_data.get('response_quality', 0) < 0.85:
            improvement_areas['response_generation'] = 0.85 - performance_data['response_quality']
            
        return improvement_areas

    @log_and_handle_errors
    async def process_message(self, message: Message) -> Any:
        """Process incoming messages."""
        if message.type == MessageType.TASK:
            return await self.execute_task(message.content)
        else:
            return await self.coordinator.handle_message(message)

    @log_and_handle_errors
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with enhanced RAG integration."""
        self.logger.info(f"Executing task: {task}")
        start_time = self.unified_analytics.get_current_time()
        
        # Get RAG results for task context
        rag_results = await self.rag_pipeline.process_query(task.get('content', ''))
        
        # Integrate RAG results
        enhanced_task = await self.integrate_rag_results(task.get('content', ''), rag_results)
        
        # Execute enhanced task
        result = await self.unified_planning_and_management.manage_task(enhanced_task)
        
        # Record analytics
        end_time = self.unified_analytics.get_current_time()
        execution_time = end_time - start_time
        self.unified_analytics.record_task_completion(
            task.get('id', 'unknown'),
            execution_time,
            result.get('success', False)
        )
        self.unified_analytics.update_performance_history(result.get('performance', 0.5))
        
        # Update continuous learning
        await self.continuous_learning_layer.update(task, result)
        
        return result

    @log_and_handle_errors
    async def evolve(self):
        """Evolve the agent's capabilities."""
        await super().evolve()
        await self.evolve_capabilities()
        await run_evolution_and_optimization(self)
        await self.coordinator.evolve()
        self.logger.info("KingAgent evolved")

    @log_and_handle_errors
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        base_status = await super().get_status()
        king_status = {
            "components": {
                "coordinator": await self.coordinator.introspect(),
                "unified_planning_and_management": await self.unified_planning_and_management.introspect(),
                "unified_analytics": self.unified_analytics.get_info(),
                "evolution_manager": self.evolution_manager.get_info(),
                "rag_integration": {
                    "active": True,
                    "pipeline_status": await self.rag_pipeline.get_status(),
                    "recent_integrations": self.unified_analytics.get_recent_rag_integrations()
                }
            },
            "analytics": self.unified_analytics.generate_summary_report(),
            "evolution_status": {
                "current_generation": self.evolution_manager.current_generation,
                "improvement_areas": await self._analyze_improvement_areas(
                    self.unified_analytics.get_performance_metrics()
                ),
                "recent_evolutions": self.evolution_manager.get_recent_evolutions()
            }
        }
        return {**base_status, **king_status}

    def create_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        """Create a neural network model from architecture specification."""
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
            
            if i < architecture['num_layers'] - 1:
                layers.append(nn.Dropout(architecture['dropout_rate']))
            
            in_features = out_features

        layers.append(nn.Linear(in_features, architecture['output_size']))
        
        return nn.Sequential(*layers)

    @log_and_handle_errors
    async def update_model_architecture(self, architecture: Dict[str, Any]):
        """Update the model architecture."""
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
        """Update model hyperparameters."""
        self.logger.info(f"Updating hyperparameters: {hyperparameters}")
        try:
            await self.unified_planning_and_management.update_hyperparameters(hyperparameters)
            await self.response_generation_agent.update_hyperparameters(hyperparameters)
            self.current_hyperparameters = hyperparameters
            self.logger.info("Hyperparameters updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update hyperparameters: {str(e)}")
            raise
