import pytest
import asyncio
from unittest.mock import Mock, patch
import os
import json
from pathlib import Path

from agent_forge.main import AIVillage
from agent_forge.agents.agent_manager import AgentManager
from agent_forge.data.data_collector import DataCollector
from agent_forge.data.complexity_evaluator import ComplexityEvaluator

# Mock environment variables
os.environ["OPENROUTER_API_KEY"] = "test_key"

@pytest.fixture
def ai_village():
    """Create AIVillage instance for testing."""
    return AIVillage()

@pytest.fixture
def mock_agent_manager():
    """Create mock AgentManager."""
    with patch('agent_forge.main.AgentManager') as mock:
        yield mock

@pytest.fixture
def mock_data_collector():
    """Create mock DataCollector."""
    with patch('agent_forge.main.DataCollector') as mock:
        yield mock

@pytest.fixture
def mock_complexity_evaluator():
    """Create mock ComplexityEvaluator."""
    with patch('agent_forge.main.ComplexityEvaluator') as mock:
        yield mock

@pytest.mark.asyncio
async def test_process_task_with_specified_agent(ai_village, mock_agent_manager):
    """Test processing a task with a specified agent."""
    # Mock response data
    mock_response = {
        "response": "Test response",
        "model": "test_model",
        "metadata": {"test": "data"}
    }
    
    # Set up mock
    ai_village.agent_manager.process_task.return_value = mock_response
    ai_village.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.8
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

@pytest.mark.asyncio
async def test_agent_type_determination(ai_village):
    """Test automatic agent type determination."""
    # Test code-related task
    assert ai_village._determine_agent_type(
        "Write a Python function to sort a list"
    ) == "magi"
    
    # Test research-related task
    assert ai_village._determine_agent_type(
        "Analyze the impact of AI on healthcare"
    ) == "sage"
    
    # Test general task
    assert ai_village._determine_agent_type(
        "Create a marketing strategy"
    ) == "king"

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
        await ai_village.task_queue.get()
        await mock_process()
        ai_village.task_queue.task_done()
    
    # Verify queue size decreased
    assert ai_village.task_queue.qsize() == len(test_tasks) - 1

@pytest.mark.asyncio
async def test_complexity_threshold_updates(ai_village, mock_complexity_evaluator):
    """Test complexity threshold adjustment."""
    # Mock performance data
    mock_performance = {
        "local_model_performance": 0.85,
        "task_success_rate": 0.9
    }
    
    # Mock complexity history
    mock_history = [
        {"is_complex": True, "success": True},
        {"is_complex": False, "success": True}
    ]
    
    # Set up mocks
    ai_village.agent_manager.get_agent().get_performance_metrics.return_value = mock_performance
    ai_village.data_collector.get_performance_history.return_value = mock_history
    
    # Update thresholds
    await ai_village._update_complexity_thresholds()
    
    # Verify complexity evaluator was called
    ai_village.complexity_evaluator.adjust_thresholds.assert_called()

@pytest.mark.asyncio
async def test_data_collection(ai_village, mock_data_collector):
    """Test data collection during task processing."""
    # Process a task
    await ai_village.process_task(
        task="Test task",
        agent_type="king"
    )
    
    # Verify data collector methods were called
    ai_village.data_collector.store_interaction.assert_called_once()
    ai_village.data_collector.store_performance_metrics.assert_called_once()

def test_system_status(ai_village):
    """Test system status reporting."""
    status = ai_village.get_system_status()
    
    assert "queue_size" in status
    assert "agent_metrics" in status
    assert "complexity_thresholds" in status
    assert "training_data_counts" in status

@pytest.mark.asyncio
async def test_error_handling(ai_village):
    """Test error handling during task processing."""
    # Mock an error in processing
    ai_village.agent_manager.process_task.side_effect = Exception("Test error")
    
    # Process should not raise exception
    await ai_village.process_task("Test task")
    
    # Verify error was logged
    # Note: Would need to set up logging capture to verify

@pytest.mark.asyncio
async def test_model_selection(ai_village):
    """Test model selection based on complexity."""
    # Test with simple task
    ai_village.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": False,
        "complexity_score": 0.3
    }
    
    result = await ai_village.process_task("Simple task")
    assert "local" in result["model_used"].lower()
    
    # Test with complex task
    ai_village.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.8
    }
    
    result = await ai_village.process_task("Complex task")
    assert "frontier" in result["model_used"].lower()

@pytest.mark.asyncio
async def test_training_data_collection(ai_village, mock_data_collector):
    """Test training data collection from frontier model responses."""
    # Process task with frontier model
    mock_response = {
        "response": "Test response",
        "model": "frontier_model",
        "metadata": {}
    }
    
    ai_village.agent_manager.process_task.return_value = mock_response
    ai_village.agent_manager.get_agent_config.return_value = {
        "frontier_model": "frontier_model",
        "local_model": "local_model"
    }
    
    await ai_village.process_task("Test task")
    
    # Verify training example was stored
    ai_village.data_collector.store_training_example.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__])
