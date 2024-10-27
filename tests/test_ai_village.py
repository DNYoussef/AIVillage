"""Tests for AI Village system."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import json
from pathlib import Path
from typing import Dict, Any, List

from config.unified_config import UnifiedConfig, AgentConfig, ModelConfig
from agent_forge.main import AIVillage
from agent_forge.agents.agent_manager import AgentManager
from agent_forge.data.data_collector import DataCollector
from agent_forge.data.complexity_evaluator import ComplexityEvaluator
from agent_forge.agents.openrouter_agent import OpenRouterAgent, AgentInteraction

# Mock environment variables
os.environ["OPENROUTER_API_KEY"] = "test_key"

@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig()

@pytest.fixture
def ai_village(config):
    """Create AIVillage instance for testing."""
    return AIVillage(config=config)

@pytest.fixture
def mock_agent_manager():
    """Create mock AgentManager."""
    manager = Mock(spec=AgentManager)
    manager.process_task = AsyncMock()
    return manager

@pytest.fixture
def mock_data_collector():
    """Create mock DataCollector."""
    collector = Mock(spec=DataCollector)
    collector.store_interaction = AsyncMock()
    collector.store_performance_metrics = AsyncMock()
    collector.store_training_example = AsyncMock()
    return collector

@pytest.fixture
def mock_complexity_evaluator():
    """Create mock ComplexityEvaluator."""
    evaluator = Mock(spec=ComplexityEvaluator)
    evaluator.evaluate_complexity = AsyncMock()
    evaluator.adjust_thresholds = AsyncMock()
    return evaluator

@pytest.mark.asyncio
async def test_process_task_with_specified_agent(ai_village, mock_agent_manager):
    """Test processing a task with a specified agent."""
    ai_village.agent_manager = mock_agent_manager
    
    # Mock response data
    mock_response = AgentInteraction(
        prompt="Test task",
        response="Test response",
        model="test_model",
        timestamp=123456789,
        metadata={"test": "data"}
    )
    
    # Set up mocks
    mock_agent_manager.process_task.return_value = mock_response
    ai_village.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    }
    
    # Process task
    result = await ai_village.process_task(
        task="Test task",
        agent_type="king"
    )
    
    # Verify results
    assert result["response"] == "Test response"
    assert result["model_used"] == "test_model"
    assert "complexity_analysis" in result
    assert "performance_metrics" in result
    assert result["complexity_analysis"]["confidence"] == 0.9

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
    
    # Test complex tasks
    complex_task = "Design and implement a distributed system architecture"
    complexity_result = await ai_village.complexity_evaluator.evaluate_complexity(
        agent_type="king",
        task=complex_task
    )
    assert complexity_result["is_complex"]

@pytest.mark.asyncio
async def test_task_queue_processing(ai_village):
    """Test task queue processing."""
    # Add test tasks
    test_tasks = [
        "Task 1",
        "Task 2",
        "Task 3"
    ]
    
    for task in test_tasks:
        await ai_village.add_task(task)
    
    # Verify queue size
    assert ai_village.task_queue.qsize() == len(test_tasks)
    
    # Process one task
    with patch.object(ai_village, 'process_task') as mock_process:
        mock_process.return_value = {"status": "success"}
        task = await ai_village.task_queue.get()
        await mock_process(task)
        ai_village.task_queue.task_done()
    
    # Verify queue size decreased
    assert ai_village.task_queue.qsize() == len(test_tasks) - 1

@pytest.mark.asyncio
async def test_complexity_threshold_updates(ai_village, mock_complexity_evaluator):
    """Test complexity threshold adjustment."""
    ai_village.complexity_evaluator = mock_complexity_evaluator
    
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
    ai_village.agent_manager.get_performance_metrics.return_value = mock_performance
    ai_village.data_collector.get_performance_history.return_value = mock_history
    
    # Update thresholds
    await ai_village._update_complexity_thresholds()
    
    # Verify complexity evaluator was called with correct data
    mock_complexity_evaluator.adjust_thresholds.assert_called_with(
        agent_type="king",
        performance_metrics=mock_performance,
        complexity_history=mock_history
    )

@pytest.mark.asyncio
async def test_data_collection(ai_village, mock_data_collector):
    """Test data collection during task processing."""
    ai_village.data_collector = mock_data_collector
    
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
    with patch.object(ai_village.agent_manager, 'process_task', return_value=mock_interaction), \
         patch.object(ai_village.complexity_evaluator, 'evaluate_complexity', return_value=mock_complexity):
        await ai_village.process_task(
            task="Test task",
            agent_type="king"
        )
    
    # Verify data collector methods were called with correct data
    mock_data_collector.store_interaction.assert_called_once()
    mock_data_collector.store_performance_metrics.assert_called_once()

def test_system_status(ai_village):
    """Test system status reporting."""
    status = ai_village.get_system_status()
    
    # Verify all required fields
    assert "queue_size" in status
    assert "agent_metrics" in status
    assert "complexity_thresholds" in status
    assert "training_data_counts" in status
    assert "system_health" in status
    
    # Verify agent metrics structure
    for agent_type in ["king", "sage", "magi"]:
        assert agent_type in status["agent_metrics"]
        agent_metrics = status["agent_metrics"][agent_type]
        assert "task_success_rate" in agent_metrics
        assert "local_model_performance" in agent_metrics

@pytest.mark.asyncio
async def test_error_handling(ai_village, mock_agent_manager):
    """Test error handling during task processing."""
    ai_village.agent_manager = mock_agent_manager
    
    # Mock an error in processing
    mock_agent_manager.process_task.side_effect = Exception("Test error")
    
    # Process should not raise exception but return error info
    result = await ai_village.process_task("Test task")
    assert "error" in result
    assert result["status"] == "failed"
    assert "Test error" in result["error"]

@pytest.mark.asyncio
async def test_model_selection(ai_village, mock_complexity_evaluator):
    """Test model selection based on complexity."""
    ai_village.complexity_evaluator = mock_complexity_evaluator
    
    # Test with simple task
    mock_complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": False,
        "complexity_score": 0.3,
        "confidence": 0.9
    }
    
    with patch.object(ai_village.agent_manager, 'process_task') as mock_process:
        mock_process.return_value = AgentInteraction(
            prompt="Simple task",
            response="Test response",
            model="local_model",
            timestamp=123456789,
            metadata={}
        )
        result = await ai_village.process_task("Simple task")
        assert "local" in result["model_used"].lower()
    
    # Test with complex task
    mock_complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    }
    
    with patch.object(ai_village.agent_manager, 'process_task') as mock_process:
        mock_process.return_value = AgentInteraction(
            prompt="Complex task",
            response="Test response",
            model="frontier_model",
            timestamp=123456789,
            metadata={}
        )
        result = await ai_village.process_task("Complex task")
        assert "frontier" in result["model_used"].lower()

@pytest.mark.asyncio
async def test_training_data_collection(ai_village, mock_data_collector):
    """Test training data collection from frontier model responses."""
    ai_village.data_collector = mock_data_collector
    
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
            type="frontier",
            temperature=0.7
        ),
        "local_model": ModelConfig(
            name="local_model",
            type="local",
            temperature=0.5
        )
    }
    
    # Process task
    with patch.object(ai_village.agent_manager, 'process_task', return_value=mock_interaction), \
         patch.object(ai_village.agent_manager, 'get_agent_config', return_value=mock_config):
        await ai_village.process_task("Test task")
    
    # Verify training example was stored with correct data
    mock_data_collector.store_training_example.assert_called_once()
    call_args = mock_data_collector.store_training_example.call_args[1]
    assert call_args["frontier_model"] == "frontier_model"
    assert call_args["local_model"] == "local_model"
    assert call_args["quality_score"] == 0.9

@pytest.mark.asyncio
async def test_unified_config_integration(ai_village, config):
    """Test integration with unified configuration system."""
    # Verify config is properly integrated
    assert ai_village.config == config
    
    # Test agent configuration access
    agent_config = ai_village.config.get_agent_config("king")
    assert isinstance(agent_config, AgentConfig)
    assert agent_config.type == "king"
    
    # Test model configuration access
    assert isinstance(agent_config.frontier_model, ModelConfig)
    assert isinstance(agent_config.local_model, ModelConfig)

if __name__ == '__main__':
    pytest.main([__file__])
