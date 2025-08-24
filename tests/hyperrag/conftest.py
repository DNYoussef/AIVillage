"""
Shared test fixtures and configuration for HyperRAG tests.

Provides common fixtures for testing the consolidated HyperRAG system:
- Mock configurations
- Test data fixtures
- Common test utilities
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

# Configure test logging
logging.getLogger("core.hyperrag").setLevel(logging.WARNING)


@pytest.fixture
def hyperrag_config():
    """Standard HyperRAG configuration for testing."""
    try:
        from core.hyperrag import HyperRAGConfig

        return HyperRAGConfig(
            max_results=5,
            min_confidence=0.1,
            vector_dimensions=384,
            enable_caching=False,  # Disable for predictable tests
            timeout_seconds=10.0,
            fallback_enabled=True,
        )
    except ImportError:
        # Fallback mock config
        class MockConfig:
            max_results = 5
            min_confidence = 0.1
            vector_dimensions = 384
            enable_caching = False
            timeout_seconds = 10.0
            fallback_enabled = True

        return MockConfig()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "content": "Python is a high-level programming language with dynamic typing and garbage collection.",
            "metadata": {"source": "programming", "type": "definition"},
        },
        {
            "id": "doc_2",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "metadata": {"source": "ai", "type": "definition"},
        },
        {
            "id": "doc_3",
            "content": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
            "metadata": {"source": "ai", "type": "explanation"},
        },
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {"query": "What is Python?", "expected_doc_id": "doc_1", "mode": "FAST"},
        {"query": "Tell me about machine learning", "expected_doc_id": "doc_2", "mode": "BALANCED"},
        {"query": "How do neural networks work?", "expected_doc_id": "doc_3", "mode": "COMPREHENSIVE"},
    ]


@pytest.fixture
def mock_hyperrag():
    """Mock HyperRAG system for testing."""
    mock_system = MagicMock()

    # Mock async methods
    mock_system.initialize = AsyncMock(return_value=True)
    mock_system.shutdown = AsyncMock()
    mock_system.process_query_async = AsyncMock()

    # Mock sync methods
    mock_system.add_document = MagicMock(return_value="test_doc_id")
    mock_system.process_query = MagicMock()
    mock_system.get_stats = MagicMock(
        return_value={"queries_processed": 0, "documents_indexed": 0, "average_response_time": 0.0}
    )
    mock_system.health_check = MagicMock(
        return_value={"status": "healthy", "components": {"vector_store": "operational"}}
    )

    return mock_system


@pytest.fixture
def mock_synthesized_answer():
    """Mock synthesized answer for testing."""
    try:
        from core.hyperrag import SynthesizedAnswer

        return SynthesizedAnswer(
            answer="This is a test answer synthesized from multiple sources.",
            confidence=0.85,
            supporting_sources=["doc_1", "doc_2"],
            synthesis_method="test_synthesis",
            processing_time=0.5,
            query_mode="balanced",
        )
    except ImportError:
        # Fallback mock
        class MockAnswer:
            answer = "This is a test answer synthesized from multiple sources."
            confidence = 0.85
            supporting_sources = ["doc_1", "doc_2"]
            synthesis_method = "test_synthesis"
            processing_time = 0.5
            query_mode = "balanced"

        return MockAnswer()


@pytest.fixture
def mock_retrieved_info():
    """Mock retrieved information for testing."""
    try:
        from core.hyperrag import RetrievedInformation

        return [
            RetrievedInformation(
                id="info_1",
                content="First piece of retrieved information",
                source="vector_store",
                relevance_score=0.9,
                retrieval_confidence=0.85,
            ),
            RetrievedInformation(
                id="info_2",
                content="Second piece of retrieved information",
                source="graph_store",
                relevance_score=0.75,
                retrieval_confidence=0.8,
            ),
        ]
    except ImportError:
        # Fallback mock
        class MockInfo:
            def __init__(self, id, content, source, relevance_score, retrieval_confidence):
                self.id = id
                self.content = content
                self.source = source
                self.relevance_score = relevance_score
                self.retrieval_confidence = retrieval_confidence

        return [
            MockInfo("info_1", "First piece of retrieved information", "vector_store", 0.9, 0.85),
            MockInfo("info_2", "Second piece of retrieved information", "graph_store", 0.75, 0.8),
        ]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Helper functions for tests
def assert_valid_answer(answer):
    """Assert that an answer object is valid."""
    assert hasattr(answer, "answer"), "Answer must have 'answer' attribute"
    assert hasattr(answer, "confidence"), "Answer must have 'confidence' attribute"
    assert hasattr(answer, "supporting_sources"), "Answer must have 'supporting_sources' attribute"
    assert 0 <= answer.confidence <= 1, "Confidence must be between 0 and 1"
    assert isinstance(answer.supporting_sources, list), "Supporting sources must be a list"


def assert_performance_acceptable(processing_time, max_time=5.0):
    """Assert that processing time is acceptable."""
    assert processing_time < max_time, f"Processing time {processing_time}s exceeds maximum {max_time}s"


def assert_health_check_valid(health_status):
    """Assert that health check response is valid."""
    assert "status" in health_status, "Health check must include status"
    assert "components" in health_status, "Health check must include components"
    assert health_status["status"] in ["healthy", "degraded", "unhealthy"], "Status must be valid"
