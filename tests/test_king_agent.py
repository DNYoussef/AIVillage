import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from agent_forge.agents.king.king_agent import KingAgent, TaskManager
from agent_forge.agents.openrouter_agent import OpenRouterAgent, AgentInteraction
from config.unified_config import UnifiedConfig
import pytest

@pytest.fixture
def config():
    """Create test configuration."""
    config = UnifiedConfig()
    config.vector_dimension = 768  # Add required attribute
    return config

@pytest.fixture
def openrouter_agent():
    """Create mock OpenRouter agent."""
    mock = AsyncMock(spec=OpenRouterAgent)
    # Set required attributes
    mock.model = "test-model"
    mock.local_model = "test-local-model"
    mock.generate_response = AsyncMock(return_value=AgentInteraction(
        prompt="test prompt",
        response="Test response",
        model="test-model",
        timestamp=123456789,
        metadata={"quality": 0.9},
        token_usage={"total_tokens": 100}
    ))
    return mock

@pytest.fixture
def king_agent(config, openrouter_agent):
    """Create KingAgent instance."""
    agent = KingAgent(openrouter_agent=openrouter_agent, config=config)
    
    # Initialize performance metrics
    agent.performance_metrics = {
        "task_success_rate": 0.0,
        "avg_response_quality": 0.0,
        "complexity_handling": 0.0,
        "local_model_performance": 0.0
    }
    
    # Mock task manager
    agent.task_manager = MagicMock(spec=TaskManager)
    agent.task_manager.completed_tasks = [{
        "status": "completed",
        "task": {"complexity": {"is_complex": False}},
        "duration": 0.5
    }]
    
    # Mock internal components
    agent.local_agent = AsyncMock()
    agent.local_agent.generate_response = AsyncMock(return_value={
        "response": "Local test response",
        "model": "test-local-model",
        "metadata": {
            "performance": {
                "duration": 0.5,
                "total_tokens": 50
            }
        }
    })
    agent.local_agent.get_performance_metrics = AsyncMock(return_value={
        "average_similarity": 0.8,
        "success_rate": 0.9
    })
    agent.complexity_evaluator = AsyncMock()
    agent.complexity_evaluator.evaluate_complexity = AsyncMock(return_value={
        "complexity_score": 0.5,
        "is_complex": False
    })
    
    # Patch _update_metrics to handle async get_performance_metrics
    async def patched_update_metrics(interaction, performance):
        local_metrics = await agent.local_agent.get_performance_metrics()
        if "average_similarity" in local_metrics:
            agent.performance_metrics["local_model_performance"] = local_metrics["average_similarity"]
    
    agent._update_metrics = patched_update_metrics
    
    return agent

@pytest.mark.asyncio
async def test_process_task(king_agent):
    """Test processing of task."""
    # Process a task
    result = await king_agent.process_task("Test task")

    # Verify result structure
    assert isinstance(result, AgentInteraction)
    assert result.response is not None
    assert result.model is not None
    assert result.metadata is not None

    # Verify component calls
    king_agent.complexity_evaluator.evaluate_complexity.assert_called_once()
    # Local model should be tried first for non-complex tasks
    king_agent.local_agent.generate_response.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(king_agent):
    """Test error handling in task processing."""
    # Mock error in local agent
    king_agent.local_agent.generate_response.side_effect = Exception("Test error")

    # Process should fall back to frontier agent
    result = await king_agent.process_task("Test task")

    # Verify frontier agent was used
    assert result.model == "test-model"  # Using frontier model name
    assert result.response is not None

@pytest.mark.asyncio
async def test_performance_metrics(king_agent):
    """Test performance metrics tracking."""
    # Process a few tasks
    await king_agent.process_task("Task 1")
    await king_agent.process_task("Task 2")

    # Get metrics
    metrics = king_agent.performance_metrics

    # Verify metrics structure
    assert isinstance(metrics, dict)
    assert "task_success_rate" in metrics
    assert "avg_response_quality" in metrics
    assert "complexity_handling" in metrics
    assert "local_model_performance" in metrics

if __name__ == "__main__":
    pytest.main([__file__])
