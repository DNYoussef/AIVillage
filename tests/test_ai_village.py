"""Tests for AI Village core functionality."""

import pytest
from pytest_asyncio import fixture
import asyncio
import os
import signal
import sys
import atexit
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager

from config.unified_config import UnifiedConfig, AgentConfig, ModelConfig, AgentType, ModelType
from agent_forge.main import AIVillage
from agent_forge.agents.openrouter_agent import AgentInteraction
from agent_forge.data.data_collector import DataCollector
from agent_forge.data.complexity_evaluator import ComplexityEvaluator

def force_close_connections():
    """Force close any open SQLite connections."""
    try:
        # Close all SQLite connections
        for conn in sqlite3.connect(':memory:').__dict__.get('_connections', []):
            try:
                conn.close()
            except:
                pass
    except:
        pass

# Global cleanup handler
def cleanup_handler():
    """Clean up any remaining processes."""
    try:
        # Get the event loop
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = None

        # Close database connections
        force_close_connections()

        # Cancel all tasks if loop exists
        if loop and not loop.is_closed():
            try:
                # Cancel all tasks
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                
                # Run loop until tasks are cancelled
                loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
                
                # Stop and close loop
                loop.stop()
                loop.close()
            except:
                pass

        # Force exit
        if sys.platform == 'win32':
            os._exit(0)
        else:
            os.kill(os.getpid(), signal.SIGKILL)
    except:
        os._exit(0)

# Register cleanup handler
atexit.register(cleanup_handler)

@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Session-wide cleanup."""
    yield
    cleanup_handler()

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
    
    # Mock process_task
    async def mock_process(*args, **kwargs):
        return {
            "response": "Test response",
            "model_used": "test-model",
            "metadata": {
                "quality_score": 0.9,
                "usage": {"total_tokens": 100},
                "validation_results": {
                    "passes_syntax": True,
                    "meets_requirements": {"test": True},
                    "test_results": {"coverage": 100.0},
                    "metrics": {"maintainability": 0.9, "performance": 0.9}
                }
            }
        }
    manager.process_task.side_effect = mock_process
    
    # Mock get_agent
    mock_agent = Mock()
    mock_agent.get_performance_metrics = Mock(return_value={
        "task_success_rate": 0.9,
        "local_model_performance": 0.85
    })
    manager.get_agent = Mock(return_value=mock_agent)
    
    # Mock _start_monitoring to do nothing
    manager._start_monitoring = Mock()
    
    return manager

@pytest.fixture
def mock_complexity_evaluator():
    """Create mock complexity evaluator."""
    evaluator = Mock()
    evaluator.evaluate_complexity = Mock(return_value={
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9,
        "threshold_used": 0.7,
        "components": {
            "token_complexity": 0.7,
            "indicator_complexity": 0.8,
            "semantic_complexity": 0.9,
            "structural_complexity": 0.8
        }
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
    collector.get_performance_history = AsyncMock(return_value=[
        {
            "complexity_score": 0.8,
            "is_complex": True,
            "threshold_used": 0.7,
            "performance": {"success_rate": 0.9}
        }
    ])
    collector.close = AsyncMock()
    return collector

@fixture
async def ai_village(config, mock_env, mock_agent_manager, mock_complexity_evaluator, mock_data_collector):
    """Create AIVillage instance for testing."""
    # Patch AgentManager and LocalAgent
    with patch('agent_forge.agents.agent_manager.AgentManager', return_value=mock_agent_manager), \
         patch('agent_forge.agents.local_agent.LocalAgent._load_and_bake_model', AsyncMock()):
        village = AIVillage(config)
        village.agent_manager = mock_agent_manager
        village.complexity_evaluator = mock_complexity_evaluator
        village.data_collector = mock_data_collector
        village.task_queue = asyncio.Queue()
        return village

@pytest.mark.asyncio
async def test_process_task_with_specified_agent(ai_village):
    """Test processing a task with a specified agent."""
    result = await ai_village.process_task(
        task="Test task",
        agent_type="king"
    )

    assert isinstance(result, dict)
    assert "response" in result
    assert "model_used" in result
    assert "status" in result
    assert result["status"] == "success"
    assert result["response"] == "Test response"

@pytest.mark.asyncio
async def test_agent_type_determination(ai_village):
    """Test automatic agent type determination."""
    # Test code-related task
    code_task = "Write a Python function to sort a list"
    assert ai_village._determine_agent_type(code_task) == "magi"

    # Test research-related task
    research_task = "Analyze the impact of AI on healthcare"
    assert ai_village._determine_agent_type(research_task) == "sage"

    # Test general task
    general_task = "Create a marketing strategy"
    assert ai_village._determine_agent_type(general_task) == "king"

@pytest.mark.asyncio
async def test_task_queue_processing(ai_village):
    """Test task queue processing."""
    # Add tasks to queue
    await ai_village.task_queue.put({
        "task": "Test task 1",
        "agent_type": None
    })
    await ai_village.task_queue.put({
        "task": "Test task 2",
        "agent_type": "king"
    })
    
    assert ai_village.task_queue.qsize() == 2

@pytest.mark.asyncio
async def test_complexity_threshold_updates(ai_village):
    """Test complexity threshold adjustment."""
    await ai_village._update_complexity_thresholds()
    ai_village.complexity_evaluator.adjust_thresholds.assert_called()

@pytest.mark.asyncio
async def test_data_collection(ai_village):
    """Test data collection during task processing."""
    result = await ai_village.process_task(
        task="Test task",
        agent_type="king"
    )
    ai_village.data_collector.store_interaction.assert_called_once()

@pytest.mark.asyncio
async def test_system_status(ai_village):
    """Test system status reporting."""
    status = await ai_village.get_system_status()
    assert "queue_size" in status
    assert "agent_metrics" in status
    assert "complexity_thresholds" in status
    assert "training_data_counts" in status
    assert "system_health" in status

@pytest.mark.asyncio
async def test_model_selection(ai_village):
    """Test model selection based on complexity."""
    # Test with simple task
    ai_village.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": False,
        "complexity_score": 0.3,
        "confidence": 0.9,
        "threshold_used": 0.7,
        "components": {
            "token_complexity": 0.2,
            "indicator_complexity": 0.3,
            "semantic_complexity": 0.4,
            "structural_complexity": 0.3
        }
    }

    result = await ai_village.process_task("Simple task")
    assert isinstance(result, dict)
    assert "model_used" in result

@pytest.mark.asyncio
async def test_training_data_collection(ai_village):
    """Test training data collection."""
    # Mock frontier model response
    async def mock_process(*args, **kwargs):
        return {
            "response": "Test response",
            "model_used": "test-frontier-model",  # Use frontier model
            "metadata": {
                "quality_score": 0.9,  # Add quality score
                "usage": {"total_tokens": 100},
                "validation_results": {
                    "passes_syntax": True,
                    "meets_requirements": {"test": True},
                    "test_results": {"coverage": 100.0},
                    "metrics": {"maintainability": 0.9, "performance": 0.9}
                }
            }
        }
    ai_village.agent_manager.process_task.side_effect = mock_process

    result = await ai_village.process_task("Test task")
    ai_village.data_collector.store_training_example.assert_called_once()

@pytest.mark.asyncio
async def test_unified_config_integration(ai_village, config):
    """Test integration with unified configuration system."""
    assert ai_village.config == config
    agent_config = ai_village.config.get_agent_config("king")
    assert isinstance(agent_config, AgentConfig)

if __name__ == "__main__":
    try:
        pytest.main([__file__])
    finally:
        force_close_connections()
        cleanup_handler()
