from typing import Dict, Any
from rag_system.core.config import UnifiedConfig
from rag_system.core.latent_space_activation import LatentSpaceActivation
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.core.cognitive_nexus import CognitiveNexus

class EnhancedRAGPipeline:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.latent_space_activation = LatentSpaceActivation()
        self.hybrid_retriever = HybridRetriever(config)
        self.reasoning_engine = UncertaintyAwareReasoningEngine(config)
        self.cognitive_nexus = CognitiveNexus()

    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
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
