"""Tests for King agent functionality."""

import pytest
from pytest_asyncio import fixture as async_fixture
import asyncio
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime

from config.unified_config import UnifiedConfig, AgentConfig, ModelConfig, AgentType, ModelType
from agent_forge.agents.king.king_agent import KingAgent, TaskManager, ResourceAllocator
from agent_forge.agents.openrouter_agent import AgentInteraction, OpenRouterAgent

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

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    # Clean up pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    # Run loop until tasks are cancelled
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    force_close_connections()

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
def mock_openrouter_agent(config):
    """Create mock OpenRouter agent."""
    agent = AsyncMock(spec=OpenRouterAgent)
    agent.model = "test-frontier-model"
    agent.local_model = "test-local-model"
    
    # Create complexity evaluation
    complexity_evaluation = {
        "complexity_score": 0.5,
        "is_complex": False,
        "confidence": 0.9,
        "threshold_used": 0.7,
        "components": {
            "token_complexity": 0.4,
            "indicator_complexity": 0.5,
            "semantic_complexity": 0.6,
            "structural_complexity": 0.5
        },
        "analysis": {
            "explanation": "Task is considered manageable for king agent",
            "primary_factors": ["analyze", "evaluate"],
            "score_breakdown": {
                "raw_score": 0.5,
                "threshold": 0.7,
                "margin": 0.2
            }
        }
    }
    
    async def mock_generate_response(*args, **kwargs):
        current_time = datetime.now().timestamp()
        return AgentInteraction(
            prompt=args[0] if args else kwargs.get('prompt', ''),
            response="Test response",
            model="test-model",
            timestamp=current_time,
            metadata={
                "test": "data",
                "quality_score": 0.9,
                "performance": {
                    "duration": 0.5,
                    "total_tokens": 100
                },
                "token_usage": {
                    "total_tokens": 100,
                    "prompt_tokens": 50,
                    "completion_tokens": 50
                },
                "complexity_evaluation": complexity_evaluation,
                "resources_allocated": {
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "timeout": 45
                }
            }
        )
    agent.generate_response = mock_generate_response
    agent.get_training_data = Mock(return_value=[])
    return agent

@pytest.fixture
def king_agent(config, mock_openrouter_agent):
    """Create KingAgent instance for testing."""
    with patch('agent_forge.agents.local_agent.LocalAgent._load_and_bake_model'), \
         patch('sqlite3.connect', sqlite3.connect), \
         patch('aiosqlite.connect', AsyncMock()):
        agent = KingAgent(openrouter_agent=mock_openrouter_agent, config=config)
        
        # Mock complexity evaluator with AsyncMock
        complexity_evaluation = {
            "complexity_score": 0.5,
            "is_complex": False,
            "confidence": 0.9,
            "threshold_used": 0.7,
            "components": {
                "token_complexity": 0.4,
                "indicator_complexity": 0.5,
                "semantic_complexity": 0.6,
                "structural_complexity": 0.5
            },
            "analysis": {
                "explanation": "Task is considered manageable for king agent",
                "primary_factors": ["analyze", "evaluate"],
                "score_breakdown": {
                    "raw_score": 0.5,
                    "threshold": 0.7,
                    "margin": 0.2
                }
            }
        }
        agent.complexity_evaluator = Mock()
        agent.complexity_evaluator.evaluate_complexity = AsyncMock(return_value=complexity_evaluation)
        agent.complexity_evaluator.record_performance = Mock()
        agent.complexity_evaluator.adjust_thresholds = Mock(return_value=0.7)
        agent.complexity_evaluator.get_threshold = Mock(return_value=0.7)
        agent.complexity_evaluator.get_threshold_analysis = Mock(return_value={
            "current_threshold": 0.7,
            "min_threshold": 0.5,
            "max_threshold": 0.9,
            "complex_task_ratio": 0.3,
            "performance_by_complexity": {
                "complex": 0.85,
                "simple": 0.9
            }
        })
        
        # Mock local agent responses
        async def mock_local_response(*args, **kwargs):
            current_time = datetime.now().timestamp()
            return AgentInteraction(
                prompt=args[0] if args else kwargs.get('prompt', ''),
                response="Local test response",
                model="test-local-model",
                timestamp=current_time,
                metadata={
                    "performance": {
                        "duration": 0.5,
                        "total_tokens": 50,
                        "tokens_per_second": 100,
                        "complexity_score": complexity_evaluation["complexity_score"]
                    },
                    "quality_score": 0.85,
                    "device": "cpu",
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7),
                    "system_prompt_used": bool(kwargs.get("system_prompt")),
                    "stream_mode": False,
                    "token_usage": {
                        "total_tokens": 50,
                        "prompt_tokens": 25,
                        "completion_tokens": 25
                    },
                    "complexity_evaluation": complexity_evaluation,
                    "resources_allocated": {
                        "max_tokens": kwargs.get("max_tokens", 1000),
                        "temperature": kwargs.get("temperature", 0.7),
                        "timeout": 45
                    }
                }
            )
        agent.local_agent.generate_response = AsyncMock(side_effect=mock_local_response)
        agent.local_agent.get_performance_metrics = Mock(return_value={
            "average_similarity": 0.8,
            "success_rate": 0.9,
            "local_model_performance": 0.85,
            "average_duration": 0.5,
            "tokens_per_second": 100,
            "average_total_tokens": 50
        })
        
        # Initialize task manager metrics with non-zero duration
        agent.task_manager.task_metrics = {
            "success_rate": 1.0,
            "average_duration": 0.5,  # Set non-zero duration
            "resource_efficiency": 1.0
        }
        
        # Mock task manager methods
        async def mock_track_task(*args, **kwargs):
            task_id = "test_task_id"
            agent.task_manager.active_tasks[task_id] = {
                "status": "in_progress",
                "start_time": datetime.now().timestamp(),
                "duration": 0.5,  # Set non-zero duration
                "task": args[0] if args else kwargs.get("task"),
                "complexity": complexity_evaluation
            }
            return task_id
            
        async def mock_complete_task(*args, **kwargs):
            task_id = args[0] if args else kwargs.get("task_id")
            task_data = agent.task_manager.active_tasks.pop(task_id)
            task_data["status"] = "completed"
            task_data["end_time"] = datetime.now().timestamp()
            task_data["duration"] = 0.5  # Set non-zero duration
            
            # Include interaction in result
            result = kwargs.get("result", {})
            interaction = result.get("interaction")
            if interaction:
                task_data["result"] = {
                    "interaction": interaction,
                    "performance": result.get("performance", {
                        "duration": 0.5,
                        "complexity_score": complexity_evaluation["complexity_score"],
                        "quality_score": 0.9
                    })
                }
            else:
                # Create mock interaction if none provided
                task_data["result"] = {
                    "interaction": AgentInteraction(
                        prompt=task_data.get("task", "Test task"),  # Use stored task or default
                        response="Test response",
                        model="test-model",
                        timestamp=datetime.now().timestamp(),
                        metadata={
                            "performance": {
                                "duration": 0.5,
                                "complexity_score": complexity_evaluation["complexity_score"]
                            }
                        }
                    ),
                    "performance": {
                        "duration": 0.5,
                        "complexity_score": complexity_evaluation["complexity_score"],
                        "quality_score": 0.9
                    }
                }
            
            agent.task_manager.completed_tasks.append(task_data)
            
            # Update metrics with non-zero duration
            agent.task_manager.task_metrics["average_duration"] = 0.5
            
            return task_data
            
        agent.task_manager.track_task = AsyncMock(side_effect=mock_track_task)
        agent.task_manager.complete_task = AsyncMock(side_effect=mock_complete_task)
        
        # Mock resource allocator
        async def mock_allocate_resources(*args, **kwargs):
            return {
                "max_tokens": 1000,
                "temperature": 0.7,
                "timeout": 45
            }
        agent.resource_allocator = Mock()
        agent.resource_allocator.allocate_resources = AsyncMock(side_effect=mock_allocate_resources)
        
        # Mock task manager add_task
        async def mock_add_task(*args, **kwargs):
            task_id = args[0] if args else kwargs.get("task_id")
            task_data = kwargs.get("task", {})
            agent.task_manager.active_tasks[task_id] = {
                "status": "in_progress",
                "start_time": datetime.now().timestamp(),
                "duration": 0.5,
                "task": task_data.get("task", task_data),  # Handle both dict and direct task string
                "complexity": task_data.get("complexity", complexity_evaluation)
            }
            return task_id
        agent.task_manager.add_task = AsyncMock(side_effect=mock_add_task)
        
        # Initialize performance metrics
        agent.performance_metrics = {
            "task_success_rate": 1.0,
            "avg_response_quality": 0.9,
            "complexity_handling": 0.9,
            "local_model_performance": 0.85
        }
        
        # Mock task manager
        agent.task_manager.active_tasks = {}
        agent.task_manager.completed_tasks = []
        
        # Mock frontier agent
        agent.frontier_agent = mock_openrouter_agent
        
        return agent

@pytest.mark.asyncio
async def test_process_task(king_agent):
    """Test task processing."""
    task = "Test task"
    result = await king_agent.process_task(task)
    
    assert isinstance(result, AgentInteraction)
    assert result.response in ["Test response", "Local test response"]
    assert result.model in ["test-model", "test-local-model"]
    assert "quality_score" in result.metadata
    assert "performance" in result.metadata
    assert "token_usage" in result.metadata
    assert "complexity_evaluation" in result.metadata
    assert "resources_allocated" in result.metadata

@pytest.mark.asyncio
async def test_error_handling(king_agent):
    """Test error handling during task processing."""
    task = "Invalid task that should raise an error"
    
    # Mock process_task to raise an error
    king_agent.frontier_agent.generate_response = AsyncMock(side_effect=Exception("Test error"))
    king_agent.local_agent.generate_response = AsyncMock(side_effect=Exception("Test error"))
    
    # Process should raise the exception
    with pytest.raises(Exception) as exc_info:
        await king_agent.process_task(task)
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_performance_metrics(king_agent):
    """Test performance metrics collection."""
    # Process some tasks
    task1 = "Test task 1"
    task2 = "Test task 2"
    
    await king_agent.process_task(task1)
    await king_agent.process_task(task2)
    
    # Get metrics
    metrics = king_agent.get_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert "task_success_rate" in metrics
    assert "local_model_performance" in metrics
    assert 0 <= metrics["task_success_rate"] <= 1
    assert 0 <= metrics["local_model_performance"] <= 1

@pytest.mark.asyncio
async def test_complexity_evaluation(king_agent):
    """Test complexity evaluation during task processing."""
    task = "Test strategic task requiring analysis"
    
    # Process task
    result = await king_agent.process_task(task)
    
    # Verify complexity evaluator was called correctly
    king_agent.complexity_evaluator.evaluate_complexity.assert_called_once_with(
        agent_type="king",
        task=task
    )
    
    # Get complexity evaluation result
    complexity_eval = king_agent.complexity_evaluator.evaluate_complexity.return_value
    
    # Verify task was properly tracked
    assert len(king_agent.task_manager.completed_tasks) > 0
    completed_task = king_agent.task_manager.completed_tasks[-1]
    assert completed_task["status"] == "completed"
    assert "interaction" in completed_task["result"]
    assert "performance" in completed_task["result"]
    assert completed_task["result"]["performance"]["complexity_score"] == complexity_eval["complexity_score"]
    
    # Verify task metrics were updated
    assert king_agent.task_manager.task_metrics["success_rate"] == 1.0
    assert king_agent.task_manager.task_metrics["average_duration"] > 0
    assert king_agent.task_manager.task_metrics["resource_efficiency"] == 1.0
    
    # Verify performance metrics were updated
    assert king_agent.performance_metrics["task_success_rate"] == 1.0
    assert king_agent.performance_metrics["avg_response_quality"] > 0
    assert king_agent.performance_metrics["complexity_handling"] >= 0
    assert king_agent.performance_metrics["local_model_performance"] > 0
    
    # Verify model selection based on complexity
    if complexity_eval["is_complex"]:
        # Complex tasks should use frontier model
        assert result.model == king_agent.frontier_agent.model
        # Verify frontier agent was called
        king_agent.frontier_agent.generate_response.assert_called_once()
        # Local agent should not be called
        king_agent.local_agent.generate_response.assert_not_called()
    else:
        # Simple tasks should try local model first
        assert result.model == king_agent.local_agent.local_model
        # Verify local agent was called
        king_agent.local_agent.generate_response.assert_called_once()
        # Frontier agent should not be called
        king_agent.frontier_agent.generate_response.assert_not_called()
    
    # Verify result includes complexity information
    assert isinstance(result, AgentInteraction)
    assert "complexity_evaluation" in result.metadata
    assert result.metadata["complexity_evaluation"] == complexity_eval
    assert "resources_allocated" in result.metadata
    
    # Verify resource allocation based on complexity
    resources = result.metadata["resources_allocated"]
    assert isinstance(resources, dict)
    assert "max_tokens" in resources
    assert "temperature" in resources
    assert resources["max_tokens"] >= 1000  # Base allocation
    assert 0.3 <= resources["temperature"] <= 0.9  # Temperature range
    
    # Verify performance tracking
    if not complexity_eval["is_complex"]:
        # For simple tasks using local model, verify performance comparison
        local_metrics = king_agent.local_agent.get_performance_metrics()
        assert "average_similarity" in local_metrics
        assert "success_rate" in local_metrics
        assert "local_model_performance" in local_metrics
        assert local_metrics["average_similarity"] >= 0.0
        assert local_metrics["success_rate"] >= 0.0
        assert local_metrics["local_model_performance"] >= 0.0
    
    # Verify complexity evaluator recorded performance
    king_agent.complexity_evaluator.record_performance.assert_called_once()
    record_call = king_agent.complexity_evaluator.record_performance.call_args[1]
    assert record_call["agent_type"] == "king"
    assert record_call["task_complexity"] == complexity_eval
    assert "performance_metrics" in record_call

if __name__ == "__main__":
    pytest.main([__file__])
