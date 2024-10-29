"""Tests for AI Village core functionality."""

import pytest
from pytest_asyncio import fixture as async_fixture
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from config.unified_config import UnifiedConfig, AgentConfig, ModelConfig, AgentType, ModelType
from agent_forge.main import AIVillage
from agent_forge.agents.openrouter_agent import AgentInteraction
from agent_forge.data.data_collector import DataCollector
from agent_forge.data.complexity_evaluator import ComplexityEvaluator

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def config():
    """Create test configuration."""
    with patch('config.unified_config.UnifiedConfig._load_configs'):
        config = UnifiedConfig()
        config.config = {
            'openrouter_api_key': 'test_key',
            'model_name': 'test-model',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        # Add agent configurations
        config.agents = {
            'king': AgentConfig(
                type=AgentType.KING,
                frontier_model=ModelConfig(
                    name="test-frontier-model",
                    type=ModelType.FRONTIER,
                    temperature=0.7,
                    max_tokens=1000
                ),
                local_model=ModelConfig(
                    name="test-local-model",
                    type=ModelType.LOCAL,
                    temperature=0.7,
                    max_tokens=1000
                ),
                description="Strategic decision making agent",
                capabilities=["task_management", "coordination"],
                performance_threshold=0.7,
                complexity_threshold=0.6,
                evolution_rate=0.1
            ),
            'magi': AgentConfig(
                type=AgentType.MAGI,
                frontier_model=ModelConfig(
                    name="test-frontier-model",
                    type=ModelType.FRONTIER,
                    temperature=0.7,
                    max_tokens=1000
                ),
                local_model=ModelConfig(
                    name="test-local-model",
                    type=ModelType.LOCAL,
                    temperature=0.7,
                    max_tokens=1000
                ),
                description="Code generation agent",
                capabilities=["code_generation", "optimization"],
                performance_threshold=0.7,
                complexity_threshold=0.6,
                evolution_rate=0.1
            ),
            'sage': AgentConfig(
                type=AgentType.SAGE,
                frontier_model=ModelConfig(
                    name="test-frontier-model",
                    type=ModelType.FRONTIER,
                    temperature=0.7,
                    max_tokens=1000
                ),
                local_model=ModelConfig(
                    name="test-local-model",
                    type=ModelType.LOCAL,
                    temperature=0.7,
                    max_tokens=1000
                ),
                description="Research and analysis agent",
                capabilities=["research", "analysis"],
                performance_threshold=0.7,
                complexity_threshold=0.6,
                evolution_rate=0.1
            )
        }
        return config

@pytest.fixture
def mock_env():
    """Mock environment variables."""
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'}):
        yield

@pytest.fixture
def mock_agent_manager():
    """Create mock agent manager."""
    manager = AsyncMock()
    
    # Create a mock agent
    mock_agent = AsyncMock()
    mock_agent.get_performance_metrics = AsyncMock(return_value={
        "task_success_rate": 0.9,
        "local_model_performance": 0.85
    })
    
    # Make get_agent return the mock agent
    async def mock_get_agent(*args, **kwargs):
        return mock_agent
    manager.get_agent = mock_get_agent
    
    # Mock process_task to return AgentInteraction
    async def mock_process_task(*args, **kwargs):
        return AgentInteraction(
            prompt="Test task",
            response="Test response",
            model="test-model",
            timestamp=datetime.now().timestamp(),
            metadata={"test": "data"}
        )
    manager.process_task = mock_process_task
    
    return manager

@pytest.fixture
def mock_complexity_evaluator():
    """Create mock complexity evaluator."""
    evaluator = Mock()
    evaluator.evaluate_complexity = Mock(return_value={
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    })
    evaluator.get_threshold = Mock(return_value=0.7)
    evaluator.adjust_thresholds = Mock(return_value=0.7)
    return evaluator

@pytest.fixture
def mock_data_collector():
    """Create mock data collector."""
    collector = AsyncMock()
    collector.store_interaction = AsyncMock()
    collector.store_training_example = AsyncMock()
    collector.get_training_data = AsyncMock(return_value=[])
    collector.get_performance_history = AsyncMock(return_value=[])
    return collector

@pytest.fixture
def mock_create_task():
    """Mock asyncio.create_task."""
    async def mock_coro(*args, **kwargs):
        return None
    
    def mock_task(*args, **kwargs):
        return asyncio.create_task(mock_coro())
    
    with patch('asyncio.create_task', side_effect=mock_task) as mock:
        yield mock

@pytest.fixture
async def ai_village(config, mock_env, mock_create_task):
    """Create AIVillage instance for testing."""
    with patch('agent_forge.main.asyncio.create_task', side_effect=mock_create_task):
        village = AIVillage(config)
        
        # Create a mock agent
        mock_agent = AsyncMock()
        async def mock_get_metrics(*args, **kwargs):
            return {
                "task_success_rate": 0.9,
                "local_model_performance": 0.85
            }
        mock_agent.get_performance_metrics = mock_get_metrics
        
        # Mock components
        village.agent_manager = AsyncMock()
        
        # Mock process_task to return AgentInteraction
        async def mock_process_task(*args, **kwargs):
            return AgentInteraction(
                prompt="Test task",
                response="Test response",
                model="test-model",
                timestamp=datetime.now().timestamp(),
                metadata={"test": "data"}
            )
        village.agent_manager.process_task = mock_process_task
        
        # Make get_agent return the mock agent
        async def mock_get_agent(*args, **kwargs):
            return mock_agent
        village.agent_manager.get_agent = mock_get_agent
        
        village.complexity_evaluator = Mock()
        village.complexity_evaluator.evaluate_complexity = Mock(return_value={
            "is_complex": True,
            "complexity_score": 0.8,
            "confidence": 0.9
        })
        village.complexity_evaluator.get_threshold = Mock(return_value=0.7)
        
        # Mock data collector
        village.data_collector = AsyncMock()
        async def mock_get_training_data(*args, **kwargs):
            return []
        village.data_collector.get_training_data = mock_get_training_data
        village.data_collector.get_performance_history = AsyncMock(return_value=[])
        
        village.task_queue = asyncio.Queue()
        return village

@pytest.mark.asyncio
async def test_process_task_with_specified_agent(ai_village, mock_agent_manager):
    """Test processing a task with a specified agent."""
    village = await ai_village  # Await the fixture
    village.agent_manager = mock_agent_manager

    # Mock response data
    mock_response = AgentInteraction(
        prompt="Test task",
        response="Test response",
        model="test_model",
        timestamp=datetime.now().timestamp(),
        metadata={"test": "data"}
    )

    # Set up mocks
    async def mock_process_task(*args, **kwargs):
        return mock_response
    mock_agent_manager.process_task = mock_process_task
    
    village.complexity_evaluator.evaluate_complexity = Mock(return_value={
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    })

    # Process task
    result = await village.process_task(
        task="Test task",
        agent_type="king"
    )

    # Verify results
    assert isinstance(result, AgentInteraction)
    assert result.response == "Test response"
    assert result.model == "test_model"

@pytest.mark.asyncio
async def test_agent_type_determination(ai_village):
    """Test automatic agent type determination."""
    village = await ai_village  # Await the fixture

    # Mock complexity evaluator
    village.complexity_evaluator.evaluate_complexity = Mock(return_value={
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    })

    # Test code-related task
    code_task = "Write a Python function to sort a list"
    assert village._determine_agent_type(code_task) == "magi"

    # Test research-related task
    research_task = "Analyze the impact of AI on healthcare"
    assert village._determine_agent_type(research_task) == "sage"

    # Test general task
    general_task = "Create a marketing strategy"
    assert village._determine_agent_type(general_task) == "king"

    # Test complex tasks
    complex_task = "Design and implement a distributed system architecture"
    complexity_result = village.complexity_evaluator.evaluate_complexity(
        agent_type="king",
        task=complex_task
    )
    assert complexity_result["is_complex"]

@pytest.mark.asyncio
async def test_task_queue_processing(ai_village):
    """Test task queue processing."""
    village = await ai_village  # Await the fixture
    
    # Add tasks to queue
    await village.task_queue.put(("Test task 1", None))
    await village.task_queue.put(("Test task 2", "king"))
    
    # Verify queue size
    assert village.task_queue.qsize() == 2

@pytest.mark.asyncio
async def test_complexity_threshold_updates(ai_village, mock_complexity_evaluator):
    """Test complexity threshold adjustment."""
    village = await ai_village  # Await the fixture
    village.complexity_evaluator = mock_complexity_evaluator

    # Mock performance data
    mock_performance = {
        "local_model_performance": 0.85,
        "task_success_rate": 0.9,
        "complexity_handling": 0.8
    }

    # Mock complexity history
    mock_history = [
        {"is_complex": True, "success": True, "confidence": 0.9},
        {"is_complex": False, "success": True, "confidence": 0.8}
    ]

    # Set up mocks
    async def mock_get_metrics(*args, **kwargs):
        return mock_performance
    village.agent_manager.get_performance_metrics = mock_get_metrics
    
    async def mock_get_history(*args, **kwargs):
        return mock_history
    village.data_collector.get_performance_history = mock_get_history

    # Update thresholds
    await village._update_complexity_thresholds()

    # Verify complexity evaluator was called with correct data
    mock_complexity_evaluator.adjust_thresholds.assert_called_with(
        agent_type="king",
        performance_metrics=mock_performance,
        complexity_history=mock_history
    )

@pytest.mark.asyncio
async def test_data_collection(ai_village, mock_data_collector):
    """Test data collection during task processing."""
    village = await ai_village  # Await the fixture
    village.data_collector = mock_data_collector

    # Mock interaction data
    mock_interaction = AgentInteraction(
        prompt="Test task",
        response="Test response",
        model="test_model",
        timestamp=123456789,
        metadata={"test": "data"}
    )

    # Mock complexity evaluation
    mock_complexity = {
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    }

    # Process task
    async def mock_process_task(*args, **kwargs):
        return mock_interaction
    with patch.object(village.agent_manager, 'process_task', new=mock_process_task), \
         patch.object(village.complexity_evaluator, 'evaluate_complexity', return_value=mock_complexity):
        await village.process_task(
            task="Test task",
            agent_type="king"
        )

    # Verify data collector methods were called with correct data
    mock_data_collector.store_interaction.assert_called_once()

@pytest.mark.asyncio
async def test_system_status(ai_village):
    """Test system status reporting."""
    village = await ai_village  # Await the fixture

    # Create a mock agent
    mock_agent = AsyncMock()
    async def mock_get_metrics(*args, **kwargs):
        return {
            "task_success_rate": 0.9,
            "local_model_performance": 0.85
        }
    mock_agent.get_performance_metrics = mock_get_metrics
    
    # Make get_agent return the mock agent
    async def mock_get_agent(*args, **kwargs):
        return mock_agent
    village.agent_manager.get_agent = mock_get_agent
    
    # Mock data collector methods
    async def mock_get_training_data(*args, **kwargs):
        return []
    village.data_collector.get_training_data = mock_get_training_data
    village.complexity_evaluator.get_threshold = Mock(return_value=0.7)

    # Get system status
    status = await village.get_system_status()

    # Verify status contains required information
    assert "queue_size" in status
    assert "agent_metrics" in status
    assert "complexity_thresholds" in status
    assert "training_data_counts" in status
    assert "system_health" in status

@pytest.mark.asyncio
async def test_error_handling(ai_village, mock_agent_manager):
    """Test error handling during task processing."""
    village = await ai_village  # Await the fixture
    village.agent_manager = mock_agent_manager

    # Mock an error in processing
    async def mock_error_task(*args, **kwargs):
        raise Exception("Test error")
    village.agent_manager.process_task = mock_error_task

    # Process should not raise exception but return error info
    result = await village.process_task("Test task")
    assert isinstance(result, dict)
    assert "error" in result
    assert result["status"] == "failed"
    assert "Test error" in str(result["error"])

@pytest.mark.asyncio
async def test_model_selection(ai_village, mock_complexity_evaluator):
    """Test model selection based on complexity."""
    village = await ai_village  # Await the fixture
    village.complexity_evaluator = mock_complexity_evaluator

    # Test with simple task
    mock_complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": False,
        "complexity_score": 0.3,
        "confidence": 0.9
    }

    with patch.object(village.agent_manager, 'process_task', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = AgentInteraction(
            prompt="Simple task",
            response="Test response",
            model="local_model",
            timestamp=datetime.now().timestamp(),
            metadata={"model_used": "local_model"}
        )
        result = await village.process_task("Simple task")
        assert isinstance(result, AgentInteraction)
        assert "local" in result.model.lower()
    
    # Test with complex task
    mock_complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    }
    
    with patch.object(village.agent_manager, 'process_task', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = AgentInteraction(
            prompt="Complex task",
            response="Test response",
            model="frontier_model",
            timestamp=123456789,
            metadata={"model_used": "frontier_model"}
        )
        result = await village.process_task("Complex task")
        assert isinstance(result, AgentInteraction)
        assert "frontier" in result.model.lower()

@pytest.mark.asyncio
async def test_training_data_collection(ai_village, mock_data_collector):
    """Test training data collection from frontier model responses."""
    village = await ai_village  # Await the fixture
    village.data_collector = mock_data_collector

    # Mock interaction
    mock_interaction = AgentInteraction(
        prompt="Test task",
        response="Test response",
        model="frontier_model",
        timestamp=123456789,
        metadata={"quality_score": 0.9}
    )

    # Mock agent config
    mock_config = {
        "frontier_model": ModelConfig(
            name="frontier_model",
            type=ModelType.FRONTIER,
            temperature=0.7
        ),
        "local_model": ModelConfig(
            name="local_model",
            type=ModelType.LOCAL,
            temperature=0.5
        )
    }

    # Process task
    async def mock_process_task(*args, **kwargs):
        return mock_interaction
    with patch.object(village.agent_manager, 'process_task', new=mock_process_task), \
         patch.object(village.agent_manager, 'get_agent_config', return_value=mock_config):
        await village.process_task("Test task")

    # Verify training example was stored with correct data
    mock_data_collector.store_training_example.assert_called_once()

@pytest.mark.asyncio
async def test_unified_config_integration(ai_village, config):
    """Test integration with unified configuration system."""
    village = await ai_village  # Await the fixture

    # Verify config is properly integrated
    assert village.config == config

    # Test agent configuration access
    agent_config = village.config.get_agent_config("king")
    assert isinstance(agent_config, AgentConfig)
