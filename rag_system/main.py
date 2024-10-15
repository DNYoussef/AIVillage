# rag_system/main.py

from core.config import RAGConfig
from core.pipeline import EnhancedRAGPipeline
from core.agent_interface import AgentInterface
from tracking.knowledge_tracker import KnowledgeTracker
from tracking.knowledge_evolution_tracker import KnowledgeEvolutionTracker
from utils.logging import setup_logger
from utils.embedding import DefaultEmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from processing.reasoning_engine import UncertaintyAwareReasoningEngine
from processing.cognitive_nexus import CognitiveNexus

logger = setup_logger(__name__)

async def main():
    config = RAGConfig()
    embedding_model = DefaultEmbeddingModel(config)
    hybrid_retriever = HybridRetriever(config, agent=AgentInterface())
    reasoning_engine = UncertaintyAwareReasoningEngine(config)
    cognitive_nexus = CognitiveNexus()
    
    pipeline = EnhancedRAGPipeline(config)
    agent = AgentInterface()
    knowledge_tracker = KnowledgeTracker()
    knowledge_evolution_tracker = KnowledgeEvolutionTracker(hybrid_retriever.vector_store, hybrid_retriever.graph_store)
    
    # Example usage
    query = "What are the key features of the RAG system?"
    try:
        result = await pipeline.process_query(query, agent)
        logger.info(f"Query result: {result}")

        # Knowledge tracking example
        knowledge_tracker.record_change(KnowledgeTracker.KnowledgeChange(
            entity="RAG system",
            relation="has_feature",
            old_value="",
            new_value="uncertainty-aware reasoning",
            timestamp=datetime.now(),
            source="query_result"
        ))

        # Uncertainty analysis example
        uncertainty_analysis = await pipeline.analyze_uncertainty(query, agent)
        logger.info(f"Uncertainty analysis: {uncertainty_analysis}")

        # Knowledge evolution tracking
        knowledge_evolution_tracker.track_changes(result, datetime.now())

        # Cognitive integration example
        integrated_result = await cognitive_nexus.integrate(query, result, uncertainty_analysis, agent)
        logger.info(f"Integrated result: {integrated_result}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    asyncio.run(main())
