from typing import Dict, Any
from rag_system.core.unified_config import unified_config
from rag_system.core.base_component import BaseComponent
from rag_system.core.latent_space_activation import LatentSpaceActivation
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.utils.error_handling import log_and_handle_errors

class EnhancedRAGPipeline(BaseComponent):
    def __init__(self):
        self.config = unified_config
        self.latent_space_activation = LatentSpaceActivation()
        self.hybrid_retriever = HybridRetriever(self.config)
        self.reasoning_engine = UncertaintyAwareReasoningEngine(self.config)
        self.cognitive_nexus = CognitiveNexus()

    @log_and_handle_errors
    async def initialize(self) -> None:
        await self.latent_space_activation.initialize()
        await self.hybrid_retriever.initialize()
        await self.reasoning_engine.initialize()
        await self.cognitive_nexus.initialize()

    @log_and_handle_errors
    async def process(self, query: str) -> Dict[str, Any]:
        # Latent space activation
        activated_knowledge = await self.latent_space_activation.activate(query)

        # Retrieval
        retrieved_info = await self.hybrid_retriever.retrieve(query, activated_knowledge)

        # Reasoning
        reasoning_result = await self.reasoning_engine.reason(query, retrieved_info, activated_knowledge)

        # Cognitive integration
        integrated_result = await self.cognitive_nexus.integrate(query, reasoning_result, activated_knowledge)

        return {
            "query": query,
            "activated_knowledge": activated_knowledge,
            "retrieved_info": retrieved_info,
            "reasoning_result": reasoning_result,
            "integrated_result": integrated_result
        }

    @log_and_handle_errors
    async def shutdown(self) -> None:
        await self.latent_space_activation.shutdown()
        await self.hybrid_retriever.shutdown()
        await self.reasoning_engine.shutdown()
        await self.cognitive_nexus.shutdown()

    @log_and_handle_errors
    async def get_status(self) -> Dict[str, Any]:
        return {
            "latent_space_activation": await self.latent_space_activation.get_status(),
            "hybrid_retriever": await self.hybrid_retriever.get_status(),
            "reasoning_engine": await self.reasoning_engine.get_status(),
            "cognitive_nexus": await self.cognitive_nexus.get_status(),
        }

    @log_and_handle_errors
    async def update_config(self, config: Dict[str, Any]) -> None:
        self.config.update(config)
        await self.hybrid_retriever.update_config(config)
        await self.reasoning_engine.update_config(config)
        # Update other components as needed
