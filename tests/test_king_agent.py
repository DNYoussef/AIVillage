"""Tests for King agent functionality."""

import pytest
from pytest_asyncio import fixture as async_fixture
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from config.unified_config import UnifiedConfig, AgentConfig, ModelConfig, AgentType, ModelType
from agent_forge.agents.king.king_agent import KingAgent
from agent_forge.agents.openrouter_agent import AgentInteraction

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
def mock_create_task():
    """Mock asyncio.create_task."""
    async def mock_coro(*args, **kwargs):
        return None
    
    def mock_task(*args, **kwargs):
        return asyncio.create_task(mock_coro())
    
    with patch('asyncio.create_task', side_effect=mock_task) as mock:
        yield mock

@pytest.fixture
async def king_agent(config, mock_create_task):
    """Create KingAgent instance for testing."""
    with patch('agent_forge.agents.king.king_agent.asyncio.create_task', side_effect=mock_create_task):
        agent = KingAgent(config)
        agent.frontier_agent = AsyncMock()
        agent.local_agent = AsyncMock()
        agent.local_agent.generate_response = AsyncMock(return_value=AgentInteraction(
            prompt="Test task",
            response="Local test response",
            model="test-local-model",
            timestamp=datetime.now().timestamp(),
            metadata={
                "performance": {
                    "duration": 0.5,
                    "total_tokens": 50
                }
            }
        ))
        agent.local_agent.get_performance_metrics = AsyncMock(return_value={
            "average_similarity": 0.8,
            "success_rate": 0.9
        })
        agent.complexity_evaluator = AsyncMock()
        agent.complexity_evaluator.evaluate_complexity = AsyncMock(return_value={
            "complexity_score": 0.5,
            "is_complex": False
        })
        
        # Mock process_task method
        async def mock_process_task(task, system_prompt=None):
            return AgentInteraction(
                prompt="Test task",
                response="Test response",
                model="test-model",
                timestamp=datetime.now().timestamp(),
                metadata={"test": "data"}
            )
        agent.frontier_agent.process_task = mock_process_task
        agent.local_agent.process_task = mock_process_task
        
        return agent

@pytest.mark.asyncio
async def test_process_task(king_agent):
    """Test task processing."""
    task = "Test task"
    result = await king_agent.process_task(task)
    
    assert isinstance(result, AgentInteraction)
    assert result.response == "Test response"
    assert result.model == "test-model"

@pytest.mark.asyncio
async def test_error_handling(king_agent):
    """Test error handling during task processing."""
    task = "Invalid task that should raise an error"
    
    # Mock process_task to raise an error
    async def mock_error_task(*args, **kwargs):
        raise Exception("Test error")
    king_agent.frontier_agent.process_task = mock_error_task
    king_agent.local_agent.process_task = mock_error_task
    
    # Process should not raise exception but return error info
    result = await king_agent.process_task(task)
    assert isinstance(result, dict)
    assert "error" in result
    assert result["status"] == "failed"
    assert "Test error" in str(result["error"])

@pytest.mark.asyncio
async def test_performance_metrics(king_agent):
    """Test performance metrics collection."""
    # Process some tasks
    task1 = "Test task 1"
    task2 = "Test task 2"
    
    await king_agent.process_task(task1)
    await king_agent.process_task(task2)
    
    # Get metrics
    metrics = await king_agent.get_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert "task_success_rate" in metrics
    assert "local_model_performance" in metrics
    assert 0 <= metrics["task_success_rate"] <= 1
    assert 0 <= metrics["local_model_performance"] <= 1

if __name__ == "__main__":
    pytest.main([__file__])
