# ruff: noqa: S101  # Use of assert detected - Expected in test files
"""Test configuration and fixtures for agent testing.

TDD London School methodology with comprehensive mock infrastructure.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, Mock, patch
import pytest
import pytest_asyncio
import logging

from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from agents.king.analytics.base_analytics import BaseAnalytics
from agents.interfaces.processing_interface import (
    ProcessingInterface, 
    ProcessingMetrics,
    ProcessorStatus,
    ProcessorCapability
)
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore
from core.communication import StandardCommunicationProtocol
from core.error_handling import get_component_logger
from agents.utils.task import Task as LangroidTask


# Configure test logging
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock = Mock(spec=VectorStore)
    mock.add_texts = AsyncMock(return_value=["doc1"])
    mock.similarity_search = AsyncMock(return_value=[
        {"content": "test content", "metadata": {"source": "test"}}
    ])
    mock.similarity_search_with_score = AsyncMock(return_value=[
        ({"content": "test content", "metadata": {"source": "test"}}, 0.95)
    ])
    return mock


@pytest.fixture
def mock_communication_protocol():
    """Mock communication protocol for testing."""
    mock = Mock(spec=StandardCommunicationProtocol)
    mock.subscribe = Mock()
    mock.send_message = AsyncMock()
    mock.query = AsyncMock(return_value={"result": "test_response"})
    return mock


@pytest.fixture
def mock_rag_config():
    """Mock RAG configuration for testing."""
    return UnifiedConfig(
        model_name="gpt-4",
        max_tokens=1000,
        temperature=0.1
    )


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline for testing."""
    mock = Mock()
    mock.process_query = AsyncMock(return_value={"answer": "test answer"})
    mock.get_embedding = AsyncMock(return_value=[0.1] * 384)
    mock.rerank = AsyncMock(return_value=[{"content": "ranked"}])
    mock.add_document = AsyncMock()
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.complete = AsyncMock()
    mock.complete.return_value = Mock(text="Generated response")
    return mock


@pytest.fixture
def sample_agent_config(mock_rag_config, mock_vector_store):
    """Sample agent configuration for testing."""
    return UnifiedAgentConfig(
        name="TestAgent",
        description="Agent for testing",
        capabilities=["test_capability", "analysis"],
        rag_config=mock_rag_config,
        vector_store=mock_vector_store,
        model="gpt-4",
        instructions="Test instructions"
    )


@pytest.fixture
def sample_langroid_task():
    """Sample Langroid task for testing."""
    task = Mock(spec=LangroidTask)
    task.content = "Test task content"
    task.type = "test_task"
    return task


class MockAnalytics(BaseAnalytics):
    """Mock analytics implementation for testing."""
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        return {
            "total_metrics": len(self.metrics),
            "metric_names": list(self.metrics.keys()),
            "sample_values": {k: v[-1] if v else 0.0 for k, v in self.metrics.items()},
            "timestamp": datetime.now().isoformat()
        }


@pytest.fixture
def mock_analytics():
    """Mock analytics instance for testing."""
    return MockAnalytics()


class MockProcessingInterface(ProcessingInterface[str, str]):
    """Mock processing interface implementation for testing."""
    
    def __init__(self, processor_id: str = "test_processor", config=None):
        super().__init__(processor_id, config)
        self.add_capability(ProcessorCapability.TEXT_PROCESSING)
        self.add_capability(ProcessorCapability.BATCH_PROCESSING)
        self.add_capability(ProcessorCapability.CACHING)
        self.initialize_called = False
        self.shutdown_called = False
        self.processed_inputs = []
    
    async def initialize(self) -> bool:
        self.initialize_called = True
        self.set_status(ProcessorStatus.IDLE)
        return True
    
    async def shutdown(self) -> bool:
        self.shutdown_called = True
        self.set_status(ProcessorStatus.SHUTTING_DOWN)
        return True
    
    async def process(self, input_data: str, **kwargs) -> Any:
        from agents.base import ProcessResult, ProcessStatus
        
        self.processed_inputs.append(input_data)
        
        if input_data == "error_input":
            return ProcessResult(
                status=ProcessStatus.FAILED,
                error="Simulated processing error"
            )
        
        result = f"processed_{input_data}"
        return ProcessResult(
            status=ProcessStatus.COMPLETED,
            data=result
        )
    
    async def validate_input(self, input_data: str) -> bool:
        return isinstance(input_data, str) and len(input_data) > 0
    
    async def estimate_processing_time(self, input_data: str) -> float:
        return len(input_data) * 0.1  # 0.1 seconds per character


@pytest.fixture
def mock_processing_interface():
    """Mock processing interface for testing."""
    return MockProcessingInterface()


@pytest.fixture
def chaos_scenarios():
    """Chaos engineering scenarios for resilience testing."""
    return {
        "network_partition": {
            "description": "Simulate network partition between agents",
            "implementation": lambda: patch('core.communication.StandardCommunicationProtocol.send_message', 
                                          side_effect=ConnectionError("Network partition"))
        },
        "memory_pressure": {
            "description": "Simulate memory pressure",
            "implementation": lambda: patch('psutil.virtual_memory', 
                                          return_value=Mock(percent=95))
        },
        "llm_timeout": {
            "description": "Simulate LLM timeout",
            "implementation": lambda: patch('agents.language_models.openai_gpt.OpenAIGPTConfig.create',
                                          side_effect=TimeoutError("LLM timeout"))
        },
        "vector_store_failure": {
            "description": "Simulate vector store failure",
            "implementation": lambda: patch.object(VectorStore, 'similarity_search',
                                                  side_effect=ConnectionError("Vector store down"))
        }
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        "task_execution_ms": 1000,
        "memory_usage_mb": 100,
        "agent_initialization_ms": 500,
        "communication_latency_ms": 200,
        "batch_processing_throughput": 10  # items per second
    }


@pytest.fixture
def test_data_generator():
    """Generator for test data."""
    class TestDataGenerator:
        @staticmethod
        def generate_task_batch(size: int = 10) -> List[LangroidTask]:
            tasks = []
            for i in range(size):
                task = Mock(spec=LangroidTask)
                task.content = f"Test task {i}"
                task.type = "batch_test_task"
                tasks.append(task)
            return tasks
        
        @staticmethod
        def generate_processing_data(size: int = 100) -> List[str]:
            return [f"data_item_{i}" for i in range(size)]
        
        @staticmethod
        def generate_metrics_data() -> Dict[str, List[float]]:
            import random
            return {
                "response_time": [random.uniform(0.1, 2.0) for _ in range(100)],
                "accuracy": [random.uniform(0.8, 1.0) for _ in range(100)],
                "memory_usage": [random.uniform(50, 200) for _ in range(100)]
            }
    
    return TestDataGenerator()


@pytest.fixture
def isolation_manager():
    """Test isolation manager for cleanup."""
    class IsolationManager:
        def __init__(self):
            self.created_files = []
            self.temp_dirs = []
            self.mock_patches = []
        
        def add_temp_file(self, file_path: str):
            self.created_files.append(file_path)
        
        def add_temp_dir(self, dir_path: str):
            self.temp_dirs.append(dir_path)
        
        def add_patch(self, patch_obj):
            self.mock_patches.append(patch_obj)
        
        def cleanup(self):
            # Clean up files
            for file_path in self.created_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            
            # Clean up directories
            for dir_path in self.temp_dirs:
                try:
                    import shutil
                    shutil.rmtree(dir_path, ignore_errors=True)
                except Exception:
                    pass
            
            # Stop patches
            for patch_obj in self.mock_patches:
                try:
                    patch_obj.stop()
                except Exception:
                    pass
    
    manager = IsolationManager()
    yield manager
    manager.cleanup()


# Utility functions for tests
def assert_performance_threshold(duration_ms: float, threshold_ms: float, operation: str):
    """Assert performance threshold is met."""
    assert duration_ms <= threshold_ms, (
        f"{operation} took {duration_ms:.2f}ms, "
        f"exceeded threshold of {threshold_ms}ms"
    )


def assert_memory_usage(memory_mb: float, threshold_mb: float, operation: str):
    """Assert memory usage is within threshold."""
    assert memory_mb <= threshold_mb, (
        f"{operation} used {memory_mb:.2f}MB memory, "
        f"exceeded threshold of {threshold_mb}MB"
    )


async def assert_async_timeout(coro, timeout_seconds: float, operation: str):
    """Assert async operation completes within timeout."""
    try:
        await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        pytest.fail(f"{operation} timed out after {timeout_seconds} seconds")


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.chaos = pytest.mark.chaos
pytest.mark.behavior = pytest.mark.behavior