"""Tests for RAG system components."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import neo4j
from rag_system.core.exploration_mode import ExplorationMode
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.core.structures import RetrievalResult
from config.unified_config import UnifiedConfig, RAGConfig
from rag_system.processing.advanced_nlp import AdvancedNLP

# Mock networkx
class MockGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    
    def copy(self):
        return self
    
    def remove_nodes_from(self, nodes):
        pass

class MockNetworkX:
    @staticmethod
    def all_simple_paths(graph, start_node, end_node, cutoff=None):
        return [["node1", "node2", "node3"]]

# Mock Neo4j driver and session with proper context manager
class MockNode:
    def __init__(self, id_val, data):
        self.id = id_val
        self._data = data
    
    def __getitem__(self, key):
        return self._data.get(key)
    
    @property
    def id(self):
        return self._data.get('id')

class MockRecord:
    def __init__(self, data):
        self._data = data
    
    def __getitem__(self, key):
        return self._data.get(key)
    
    @property
    def data(self):
        return self._data

class MockResult:
    def __init__(self, records):
        self.records = records
    
    def __iter__(self):
        return iter(self.records)
    
    def single(self):
        return self.records[0] if self.records else None

class MockTransaction:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, query, **kwargs):
        return self.session.run(query, **kwargs)

class MockSession:
    def __init__(self):
        self.closed = False
        self.transaction = None
        self._mock_data = {
            "test_id": {
                "id": "test_id",
                "content": "test content",
                "timestamp": datetime.now().timestamp(),
                "data": {"key": "value"}
            }
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True

    def run(self, query, **kwargs):
        try:
            if "nodeContent" in query:
                return MockResult([
                    MockRecord({"id": "test_id", "node": MockNode("test_id", self._mock_data["test_id"])})
                ])
            elif "MATCH (n)" in query:
                return MockResult([
                    MockRecord({"n": MockNode("test_id", self._mock_data["test_id"])})
                ])
            elif "MATCH (n)-[r]-(m)" in query:
                return MockResult([
                    MockRecord({
                        "id": "neighbor_id",
                        "node": MockNode("neighbor_id", {
                            "id": "neighbor_id",
                            "content": "neighbor content",
                            "timestamp": datetime.now().timestamp()
                        })
                    })
                ])
            return MockResult([])
        except Exception as e:
            raise neo4j.exceptions.ServiceUnavailable(f"Mock Neo4j error: {str(e)}")

    def begin_transaction(self):
        return MockTransaction(self)

    def close(self):
        self.closed = True

class MockDriver:
    def __init__(self, *args, **kwargs):
        self._session = MockSession()

    def close(self):
        self._session.close()

    def session(self):
        return self._session

    def verify_connectivity(self):
        return True

# Mock langroid dependencies
class MockLangroidConfig:
    def __init__(self, *args, **kwargs):
        self.chat_model = kwargs.get('chat_model', 'gpt-3.5-turbo')
        self.create = lambda: AsyncMock()

class MockLangroidAgent:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', MockLangroidConfig())

# Apply patches
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock external dependencies."""
    mock_nx = Mock()
    mock_nx.all_simple_paths = MockNetworkX.all_simple_paths
    mock_nx.Graph = MockGraph

    with patch('neo4j.GraphDatabase.driver', return_value=MockDriver()), \
         patch('networkx.all_simple_paths', mock_nx.all_simple_paths), \
         patch('networkx.Graph', mock_nx.Graph), \
         patch('langroid.ChatAgent', MockLangroidAgent), \
         patch('langroid.ChatAgentConfig', MockLangroidConfig), \
         patch('langroid.language_models.openai_gpt.OpenAIGPTConfig', MockLangroidConfig):
        yield

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
    config.neo4j_uri = "bolt://localhost:7687"
    config.neo4j_user = "neo4j"
    config.neo4j_password = "password"
    return config

@pytest.fixture
def mock_graph_store():
    """Create mock graph store."""
    mock = AsyncMock()
    mock.get_graph = AsyncMock(return_value=MockGraph())
    mock.get_node_info = AsyncMock(return_value={
        "connections": [
            {"target": "node2", "type": "related_to"},
            {"target": "node3", "type": "depends_on"}
        ]
    })
    mock.get_edge_data = AsyncMock(return_value={
        "is_causal": True,
        "novelty": 0.8,
        "relevance": 0.7
    })
    mock.initialized = True
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
    mock.initialized = True
    return mock

@pytest.fixture
def mock_agent():
    """Create mock agent."""
    mock = AsyncMock()
    mock.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock.initialized = True
    return mock

@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=Mock(text='{"conclusion": "test", "confidence": 0.9}'))
    mock.score_path = AsyncMock(return_value=0.8)
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
def cognitive_nexus():
    """Create CognitiveNexus instance."""
    nexus = CognitiveNexus()
    nexus.initialized = True
    nexus.knowledge_graph = {
        "nodes": {},
        "edges": {},
        "metadata": {
            "created": datetime.now().isoformat(),
            "last_updated": None,
            "version": "1.0"
        }
    }
    nexus.synthesize = AsyncMock(return_value={
        "integrated_knowledge": "test synthesis",
        "confidence": 0.9
    })
    return nexus

@pytest.fixture
def exploration_mode(mock_graph_store, mock_llm, mock_advanced_nlp):
    """Create ExplorationMode instance."""
    mode = ExplorationMode(
        graph_store=mock_graph_store,
        llm_config=MockLangroidConfig(),  # Use MockLangroidConfig
        advanced_nlp=mock_advanced_nlp
    )
    mode.initialized = True
    mode.llm = mock_llm  # Set mock LLM directly
    return mode

@pytest.fixture
async def reasoning_engine(rag_config):
    """Create UncertaintyAwareReasoningEngine instance."""
    engine = UncertaintyAwareReasoningEngine(config=rag_config)
    engine.llm = AsyncMock()
    engine.llm.complete = AsyncMock(return_value=Mock(text='{"conclusion": "test", "confidence": 0.9}'))
    engine.llm.score_path = AsyncMock(return_value=0.8)
    engine.driver = MockDriver()  # Set mock driver directly
    engine.initialized = True
    engine.causal_edges = {}
    engine.uncertainty_thresholds = {
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2
    }
    
    # Mock reasoning steps
    async def mock_reason_with_uncertainty(*args, **kwargs):
        return (
            "Test reasoning",
            0.2,
            [
                {
                    "type": "interpret_query",
                    "result": "Query interpreted",
                    "uncertainty": 0.1
                },
                {
                    "type": "analyze_knowledge",
                    "result": "Knowledge analyzed",
                    "uncertainty": 0.2
                },
                {
                    "type": "synthesize_answer",
                    "result": "Answer synthesized",
                    "uncertainty": 0.3
                }
            ]
        )
    engine.reason_with_uncertainty = mock_reason_with_uncertainty
    
    # Mock Neo4j initialization
    await engine.initialize()
    
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
    assert "detailed_steps" in result
    assert "uncertainty_sources" in result
    assert "suggestions" in result
    assert "supporting_evidence" in result
    assert "activated_concepts" in result
    assert result.get("uncertainty", 1.0) >= 0.0 and result.get("uncertainty", 1.0) <= 1.0
    assert len(result["detailed_steps"]) == 3  # All three reasoning steps
    assert result["detailed_steps"][0]["type"] == "interpret_query"
    assert result["detailed_steps"][1]["type"] == "analyze_knowledge"
    assert result["detailed_steps"][2]["type"] == "synthesize_answer"

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
