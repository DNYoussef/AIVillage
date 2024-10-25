"""King Agent implementation."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.retrieval.vector_store import VectorStore
from rag_system.core.pipeline import EnhancedRAGPipeline
from agents.utils.task import Task as LangroidTask

# King-specific imports
from .coordinator import KingCoordinator
from .planning.unified_planning_and_decision import UnifiedPlanningAndDecision
from .analytics.unified_analytics import UnifiedAnalytics
from .evolution_manager import EvolutionManager, run_evolution_and_optimization
from .response_generation_agent import ResponseGenerationAgent
from rag_system.utils.error_handling import log_and_handle_errors
from langroid.language_models.openai_gpt import OpenAIGPTConfig

class KingAgent(UnifiedBaseAgent):
    """
    KingAgent specializes in decision-making and task delegation.
    Inherits from UnifiedBaseAgent to ensure standardized implementation.
    """
    
    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol
    ):
        super().__init__(config, communication_protocol)
        
        # Initialize King-specific components
        self.coordinator = KingCoordinator(config, communication_protocol)
        self.unified_planning_and_decision = UnifiedPlanningAndDecision(
            communication_protocol, 
            self.rag_pipeline,
            self
        )
        self.unified_analytics = UnifiedAnalytics()
        self.evolution_manager = EvolutionManager()
        self.response_generation_agent = ResponseGenerationAgent(
            OpenAIGPTConfig(chat_model=self.model)
        )

        # Add King-specific tools
        self.add_tool(
            "coordinate_task",
            self.coordinator.coordinate_task,
            "Coordinate and delegate tasks to other agents"
        )
        self.add_tool(
            "plan_task",
            self.unified_planning_and_decision.make_decision,
            "Make strategic decisions and plans"
        )
        self.add_tool(
            "generate_response",
            self.response_generation_agent.generate_response,
            "Generate sophisticated responses"
        )

    async def _process_task(self, task: LangroidTask) -> Dict[str, Any]:
        """
        Implement core task processing logic specific to KingAgent.
        """
        try:
            # Get enhanced context from base RAG integration
            rag_context = await self.query_rag(task.content)
            
            # Process through King-specific components
            coordination_result = await self.coordinator.coordinate_task(task)
            planning_result = await self.unified_planning_and_decision.make_decision(task)
            
            # Generate response using both base and King-specific capabilities
            response = await self.response_generation_agent.generate_response({
                "task": task,
                "rag_context": rag_context,
                "coordination": coordination_result,
                "planning": planning_result
            })
            
            # Track analytics
            self.unified_analytics.track_task_processing(task, response)
            
            return {
                "status": "success",
                "response": response,
                "coordination": coordination_result,
                "planning": planning_result,
                "rag_context": rag_context
            }

        except Exception as e:
            self.state.error_count += 1
            return {
                "error": str(e),
                "status": "failed"
            }

    async def integrate_rag_results(self, query: str, rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced RAG integration that combines base and King-specific processing.
        """
        # Use base RAG capabilities
        base_results = await super().query_rag(query)
        
        # Enhance with King-specific processing
        processed_results = await self.unified_planning_and_decision.process_rag_results(
            rag_results
        )
        
        # Extract and incorporate insights
        insights = await self._extract_key_insights(processed_results)
        await self._update_knowledge_base(insights)
        
        # Combine results
        enhanced_results = await self.coordinator.incorporate_knowledge(
            {**base_results, **processed_results}
        )
        
        # Track analytics
        self.unified_analytics.track_rag_integration(query, enhanced_results)
        
        return enhanced_results

    async def evolve_capabilities(self):
        """
        Enhanced evolution that combines base and King-specific evolution.
        """
        # Evolve base capabilities
        await super().evolve()
        
        # Get performance metrics
        performance_data = self.unified_analytics.get_performance_metrics()
        
        # Analyze areas for improvement
        improvement_areas = await self._analyze_improvement_areas(performance_data)
        
        # Evolve King-specific components
        await self.evolution_manager.evolve_based_on_metrics(performance_data)
        new_architecture = self.evolution_manager.get_optimal_architecture()
        await self.update_model_architecture(new_architecture)
        await self.unified_planning_and_decision.evolve_strategies(improvement_areas)
        
        # Track evolution
        self.unified_analytics.track_evolution(improvement_areas)

    async def _analyze_improvement_areas(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze areas needing improvement based on performance data."""
        improvement_areas = {}
        
        # Analyze various metrics
        if performance_data.get('task_completion_rate', 0) < 0.8:
            improvement_areas['task_planning'] = 0.8 - performance_data['task_completion_rate']
        
        if performance_data.get('decision_accuracy', 0) < 0.9:
            improvement_areas['decision_making'] = 0.9 - performance_data['decision_accuracy']
        
        if performance_data.get('response_quality', 0) < 0.85:
            improvement_areas['response_generation'] = 0.85 - performance_data['response_quality']
        
        return improvement_areas

    async def get_status(self) -> Dict[str, Any]:
        """Get enhanced status information combining base and King-specific status."""
        base_status = await super().info
        king_status = {
            "components": {
                "coordinator": await self.coordinator.introspect(),
                "unified_planning_and_decision": await self.unified_planning_and_decision.introspect(),
                "unified_analytics": self.unified_analytics.get_info(),
                "evolution_manager": self.evolution_manager.get_info()
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
        try:
            new_model = self.create_model_from_architecture(architecture)
            await self.unified_planning_and_decision.update_model(new_model)
            await self.response_generation_agent.update_model(new_model)
            self.current_architecture = architecture
        except Exception as e:
            self.state.error_count += 1
            raise

    @log_and_handle_errors
    async def update_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Update model hyperparameters."""
        try:
            await self.unified_planning_and_decision.update_hyperparameters(hyperparameters)
            await self.response_generation_agent.update_hyperparameters(hyperparameters)
            self.current_hyperparameters = hyperparameters
        except Exception as e:
            self.state.error_count += 1
            raise
