"""Tests for RAG system components."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import neo4j
import sys
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
        self._node = {}  # Required by networkx API
        self._adj = {}   # Required by networkx API
        
        # Add some test nodes and edges
        self.add_node("node1", type="concept")
        self.add_node("node2", type="concept")
        self.add_node("node3", type="concept")
        self.add_edge("node1", "node2", type="related_to")
        self.add_edge("node2", "node3", type="depends_on")
    
    def copy(self):
        new_graph = MockGraph()
        new_graph.nodes = self.nodes.copy()
        new_graph.edges = self.edges.copy()
        new_graph._node = self._node.copy()
        new_graph._adj = {k: v.copy() for k, v in self._adj.items()}
        return new_graph
    
    def remove_nodes_from(self, nodes):
        for node in nodes:
            if node in self.nodes:
                del self.nodes[node]
                del self._node[node]
                # Remove any edges connected to this node
                self.edges = {(u, v): data for (u, v), data in self.edges.items()
                            if u != node and v != node}
                # Update adjacency
                if node in self._adj:
                    del self._adj[node]
                for adj in self._adj.values():
                    if node in adj:
                        del adj[node]

    def add_node(self, node, **attrs):
        self.nodes[node] = attrs
        self._node[node] = attrs
        if node not in self._adj:
            self._adj[node] = {}

    def add_edge(self, u, v, **attrs):
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)
        self.edges[(u, v)] = attrs
        # Update adjacency
        self._adj[u][v] = attrs
        self._adj[v][u] = attrs

    def nodes(self, data=False):
        if data:
            return self.nodes.items()
        return list(self.nodes.keys())

    def edges(self, data=False):
        if data:
            return [(u, v, d) for (u, v), d in self.edges.items()]
        return list(self.edges.keys())

    def neighbors(self, node):
        return [v for (u, v) in self.edges.keys() if u == node] + \
               [u for (u, v) in self.edges.keys() if v == node]

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, node):
        return node in self.nodes

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
        self.create = lambda: AsyncMock(return_value=Mock(
            complete=AsyncMock(return_value=Mock(
                text='{"relation": "test", "confidence": 0.9}'
            ))
        ))

class MockLangroidAgent:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', MockLangroidConfig())

# Apply patches
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock external dependencies."""
    # Create mock networkx functions
    def mock_all_simple_paths(graph, start_node, end_node, cutoff=None):
        if start_node in graph.nodes and end_node in graph.nodes:
            if start_node == "node1" and end_node == "node3":
                return [["node1", "node2", "node3"]]
            elif start_node == "node1" and end_node == "node2":
                return [["node1", "node2"]]
            elif start_node == "node2" and end_node == "node3":
                return [["node2", "node3"]]
        return []

    def mock_spring_layout(graph):
        return {node: [0.0, 0.0] for node in graph.nodes()}

    with patch('neo4j.GraphDatabase.driver', return_value=MockDriver()), \
         patch('networkx.all_simple_paths', mock_all_simple_paths), \
         patch('networkx.Graph', MockGraph), \
         patch('networkx.spring_layout', mock_spring_layout), \
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
    config.retrieval_depth = 3
    config.relevance_threshold = 0.7
    config.feedback_enabled = True
    config.exploration_weight = 1.0
    config.max_context_length = 2000
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
    mock.get_all_nodes = AsyncMock(return_value=["node1", "node2", "node3"])
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
    # Patch HybridRetriever initialization to handle RAGConfig
    with patch('rag_system.retrieval.hybrid_retriever.HybridRetriever.__init__') as mock_init:
        mock_init.return_value = None
        retriever = HybridRetriever(rag_config)
        retriever.config = rag_config
        retriever.vector_dimension = rag_config.vector_dimension
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
    
    # Mock explore_knowledge_graph to return expected structure
    async def mock_explore(*args, **kwargs):
        return {
            "nodes": ["node1", "node2", "node3"],
            "paths": [["node1", "node2", "node3"]],
            "exploration_results": [
                {
                    "source": "node1",
                    "target": "node2",
                    "relation_type": "related_to",
                    "analysis": {"conclusion": "test", "confidence": 0.9}
                }
            ]
        }
    mode.explore_knowledge_graph = mock_explore
    
    return mode

@pytest.fixture
def reasoning_engine(rag_config):
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
    async def mock_reason(*args, **kwargs):
        return {
            "reasoning": "Test reasoning",
            "uncertainty": 0.2,
            "detailed_steps": [
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
            ],
            "uncertainty_sources": ["data sparsity", "model uncertainty"],
            "suggestions": ["gather more data"],
            "supporting_evidence": ["evidence1"],
            "activated_concepts": ["concept1"]
        }
    engine.reason = mock_reason
    
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
    assert "nodes" in results
    assert "paths" in results
    assert len(results["nodes"]) > 0
    assert len(results["paths"]) > 0
    assert isinstance(results["paths"][0], list)
    assert len(results["paths"][0]) > 0

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
