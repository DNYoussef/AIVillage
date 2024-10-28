import unittest
import asyncio
from config.unified_config import UnifiedConfig
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.core.structures import RetrievalResult
import pytest

class MockVectorStore:
    async def retrieve(self, query_vector, k=5, timestamp=None):
        return [RetrievalResult(id="1", content="Mock vector result", score=0.9)]

class MockGraphStore:
    async def retrieve(self, query, k=5, timestamp=None):
        return [RetrievalResult(id="2", content="Mock graph result", score=0.8)]

@pytest.mark.asyncio
async def test_rag_system_integration():
    """Test RAG system integration."""
    # Initialize configuration with required attributes
    config = UnifiedConfig()
    config.vector_dimension = 768  # Add required attribute
    
    # Initialize components
    components = {
        "vector_store": MockVectorStore(),
        "graph_store": MockGraphStore(),
        "hybrid_retriever": HybridRetriever(config),
        "rag_pipeline": EnhancedRAGPipeline(config)
    }
    
    # Set up mock stores
    components["hybrid_retriever"].vector_store = components["vector_store"]
    components["hybrid_retriever"].graph_store = components["graph_store"]
    
    # Process a sample query
    sample_query = "What are the key features of the RAG system?"
    result = await process_user_query(components, sample_query)
    
    # Verify results
    assert result is not None
    assert "query" in result
    assert "integrated_result" in result
    assert len(result["integrated_result"]) > 0

async def process_user_query(components, query):
    """Process a user query through the RAG system."""
    # Use the RAG pipeline to process the query
    result = await components["rag_pipeline"].process_query(query)
    
    return {
        "query": query,
        "integrated_result": result
    }

if __name__ == '__main__':
    pytest.main([__file__])
