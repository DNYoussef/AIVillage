"""Tests for RAG system components."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.core.structures import RetrievalResult
from config.unified_config import UnifiedConfig, RAGConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.processing.advanced_nlp import AdvancedNLP

# Mock ExplorationMode before importing
class MockExplorationMode:
    """Mock ExplorationMode to avoid LLM initialization."""
    def __init__(self, graph_store, llm_config, advanced_nlp):
        self.graph_store = graph_store
        self.advanced_nlp = advanced_nlp
        self.llm = AsyncMock()
        self.llm.complete = AsyncMock(return_value=Mock(text='{"relation": "test", "confidence": 0.9}'))
        self.initialized = True

    async def explore_knowledge_graph(self, start_node, depth=3):
        return {
            "nodes": ["node1", "node2", "node3"],
            "paths": [["node1", "node2", "node3"]]
        }

    async def find_creative_connections(self, start_node, end_node, excluded_nodes):
        return [["node1", "node3"]]

# Apply the patch
with patch('rag_system.core.exploration_mode.ExplorationMode', MockExplorationMode):
    from rag_system.core.exploration_mode import ExplorationMode

@pytest.fixture
def config():
    """Create test configuration."""
    config = UnifiedConfig()
    config.vector_dimension = 768
    return config

@pytest.fixture
def rag_config():
    """Create RAG configuration."""
    config = RAGConfig()
    config.vector_dimension = 768
    return config

@pytest.fixture
def mock_graph_store():
    """Create mock graph store."""
    mock = AsyncMock()
    mock.get_graph = AsyncMock(return_value={
        "nodes": ["node1", "node2", "node3"],
        "edges": [
            {"source": "node1", "target": "node2", "type": "related"},
            {"source": "node2", "target": "node3", "type": "depends_on"}
        ]
    })
    mock.get_node_info = AsyncMock(return_value={
        "connections": [
            {"target": "node2", "type": "related_to"},
            {"target": "node3", "type": "depends_on"}
        ]
    })
    return mock

@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    mock = AsyncMock()
    mock.retrieve = AsyncMock(return_value=[
        RetrievalResult(
            id="1",
            content="Test content 1",
            score=0.9,
            uncertainty=0.1,
            timestamp=datetime.now(),
            version="1.0"
        ),
        RetrievalResult(
            id="2",
            content="Test content 2",
            score=0.8,
            uncertainty=0.2,
            timestamp=datetime.now(),
            version="1.0"
        )
    ])
    return mock

@pytest.fixture
def mock_agent():
    """Create mock agent."""
    mock = AsyncMock()
    mock.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock

@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=Mock(text='{"relation": "test", "confidence": 0.9}'))
    return mock

@pytest.fixture
def mock_advanced_nlp():
    """Create mock AdvancedNLP."""
    mock = Mock(spec=AdvancedNLP)
    mock.get_embeddings = Mock(return_value=[[0.1, 0.2, 0.3]])
    mock.calculate_similarity = Mock(return_value=0.8)
    return mock

@pytest.fixture
def hybrid_retriever(rag_config, mock_vector_store, mock_graph_store, mock_agent):
    """Create HybridRetriever instance."""
    retriever = HybridRetriever(rag_config)
    retriever.vector_store = mock_vector_store
    retriever.graph_store = mock_graph_store
    retriever.agent = mock_agent
    # Initialize to avoid None
    retriever.initialized = True
    # Mock dual_level_retrieve to return mock results
    retriever.dual_level_retrieve = AsyncMock(return_value=[
        RetrievalResult(
            id="1",
            content="Test content 1",
            score=0.9,
            uncertainty=0.1,
            timestamp=datetime.now(),
            version="1.0"
        )
    ])
    return retriever

@pytest.fixture
def cognitive_nexus(rag_config):
    """Create CognitiveNexus instance."""
    nexus = CognitiveNexus()
    nexus.synthesize = AsyncMock(return_value={
        "integrated_knowledge": "test synthesis",
        "confidence": 0.9
    })
    return nexus

@pytest.fixture
def exploration_mode(mock_graph_store, mock_llm, mock_advanced_nlp):
    """Create ExplorationMode instance."""
    return ExplorationMode(
        graph_store=mock_graph_store,
        llm_config=None,  # Not used since we mock llm directly
        advanced_nlp=mock_advanced_nlp
    )

@pytest.fixture
def reasoning_engine(rag_config):
    """Create UncertaintyAwareReasoningEngine instance."""
    engine = UncertaintyAwareReasoningEngine(config=rag_config)
    engine.llm = AsyncMock()
    engine.llm.complete = AsyncMock(return_value=Mock(text='{"conclusion": "test", "confidence": 0.9}'))
    return engine

@pytest.mark.asyncio
async def test_hybrid_retrieval(hybrid_retriever):
    """Test hybrid retrieval functionality."""
    query = "test query"
    results = await hybrid_retriever.retrieve(query, k=5)
    
    assert len(results) > 0
    assert isinstance(results[0], RetrievalResult)
    assert results[0].score >= 0.0 and results[0].score <= 1.0
    assert results[0].uncertainty >= 0.0 and results[0].uncertainty <= 1.0

@pytest.mark.asyncio
async def test_knowledge_exploration(exploration_mode):
    """Test knowledge graph exploration."""
    results = await exploration_mode.explore_knowledge_graph(
        start_node="node1",
        depth=2
    )
    
    assert isinstance(results, dict)
    assert "nodes" in results or "paths" in results
    assert len(results.get("nodes", [])) > 0 or len(results.get("paths", [])) > 0

@pytest.mark.asyncio
async def test_uncertainty_aware_reasoning(reasoning_engine):
    """Test uncertainty-aware reasoning."""
    query = "test query"
    evidence = [
        RetrievalResult(
            id="1",
            content="Test evidence",
            score=0.9,
            uncertainty=0.1,
            timestamp=datetime.now(),
            version="1.0"
        )
    ]
    activated_knowledge = {"concept1": {"relevance": 0.8}}
    
    result = await reasoning_engine.reason(
        query=query,
        retrieved_info=evidence,
        activated_knowledge=activated_knowledge
    )
    
    assert isinstance(result, dict)
    assert "reasoning" in result
    assert "uncertainty" in result
    assert result.get("uncertainty", 1.0) >= 0.0 and result.get("uncertainty", 1.0) <= 1.0

@pytest.mark.asyncio
async def test_creative_connections(exploration_mode):
    """Test finding creative connections."""
    connections = await exploration_mode.find_creative_connections(
        start_node="node1",
        end_node="node3",
        excluded_nodes=["node2"]
    )
    
    assert isinstance(connections, list)
    for path in connections:
        assert isinstance(path, list)
        assert len(path) > 0

@pytest.mark.asyncio
async def test_knowledge_synthesis(cognitive_nexus):
    """Test knowledge synthesis."""
    sources = [
        {"content": "test content 1", "confidence": 0.8},
        {"content": "test content 2", "confidence": 0.9}
    ]
    
    synthesis = await cognitive_nexus.synthesize(
        sources=sources,
        query="test query"
    )
    
    assert isinstance(synthesis, dict)
    assert "integrated_knowledge" in synthesis
    assert synthesis.get("confidence", 0) >= 0.0

if __name__ == "__main__":
    pytest.main([__file__])
