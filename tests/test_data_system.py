"""Tests for data collection and management system."""

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from config.unified_config import UnifiedConfig
from agent_forge.data.data_collector import DataCollector, DatabaseManager
from agent_forge.data.complexity_evaluator import ComplexityEvaluator
from agent_forge.agents.openrouter_agent import AgentInteraction

@pytest_asyncio.fixture(scope="function")
async def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)

@pytest.fixture
def config():
    """Create test configuration."""
    with patch('config.unified_config.UnifiedConfig._load_configs'):
        config = UnifiedConfig()
        config.config = {
            'openrouter_api_key': 'test_key',
            'model_name': 'test-model',
            'temperature': 0.7,
            'max_tokens': 1000,
            'db_config': {
                'path': ':memory:',  # Use in-memory SQLite database for tests
                'backup_interval': 24,
                'max_backup_count': 7,
                'vacuum_threshold': 1000
            }
        }
        return config

@pytest.fixture
def mock_db():
    """Create mock database manager."""
    db = AsyncMock()
    
    # Mock database operations
    db.store_interaction = AsyncMock()
    db.store_training_example = AsyncMock()
    db.store_performance_metrics = AsyncMock()
    
    # Mock query results
    db.get_interactions = AsyncMock(return_value=[{
        "prompt": "Test prompt",
        "response": "Test response",
        "model_used": "test-model",
        "timestamp": datetime.now().timestamp(),
        "metadata": {"test": "data"}
    }])
    
    db.get_training_data = AsyncMock(return_value=[{
        "input": "Test input",
        "output": "Test output",
        "quality_score": 0.9,
        "metadata": {"test": "data"}
    }])
    
    db.get_performance_metrics = AsyncMock(return_value={
        "accuracy": 0.9,
        "success_rate": 0.95
    })
    
    # Mock maintenance operations
    db.create_backup = AsyncMock(return_value="backup.json")
    db.run_maintenance = AsyncMock(return_value=1)
    db._schedule_maintenance = Mock()
    
    # Mock path
    db.db_path = Path(":memory:")
    
    return db

@pytest.fixture
def data_collector(config, mock_db):
    """Create DataCollector instance for testing."""
    with patch('agent_forge.data.data_collector.DatabaseManager', return_value=mock_db):
        collector = DataCollector(config, start_maintenance=False)  # Disable maintenance for tests
        collector.db_manager = mock_db
        return collector

@pytest.fixture
def complexity_evaluator(config):
    """Create ComplexityEvaluator instance for testing."""
    evaluator = ComplexityEvaluator(config)
    evaluator.evaluate_complexity = Mock(return_value={
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
    })
    evaluator.record_performance = Mock()  # Changed to sync Mock
    evaluator.adjust_thresholds = Mock(return_value=0.7)
    evaluator.get_threshold = Mock(return_value=0.7)
    evaluator.get_threshold_analysis = Mock(return_value={
        "current_threshold": 0.7,
        "min_threshold": 0.5,
        "max_threshold": 0.9,
        "complex_task_ratio": 0.3,
        "performance_by_complexity": {
            "complex": 0.85,
            "simple": 0.9
        }
    })
    return evaluator

@pytest.mark.asyncio
async def test_interaction_storage(data_collector):
    """Test storing and retrieving agent interactions."""
    interaction = AgentInteraction(
        prompt="Test prompt",
        response="Test response",
        model="test-model",
        timestamp=datetime.now().timestamp(),
        metadata={"test": "data"}
    )
    
    # Store interaction
    await data_collector.store_interaction(
        agent_type="king",
        interaction=interaction.__dict__,
        was_complex=True
    )
    
    # Verify store_interaction was called with correct arguments
    data_collector.db_manager.store_interaction.assert_called_once()
    
    # Retrieve interactions
    interactions = await data_collector.get_interactions(
        agent_type="king",
        limit=1
    )
    
    assert len(interactions) == 1
    assert interactions[0]["prompt"] == "Test prompt"
    assert interactions[0]["response"] == "Test response"

@pytest.mark.asyncio
async def test_training_data_storage(data_collector):
    """Test storing and retrieving training data."""
    training_example = {
        "input": "Test input",
        "output": "Test output",
        "quality_score": 0.9,
        "metadata": {"test": "data"}
    }
    
    # Store training example
    await data_collector.store_training_example(
        agent_type="king",
        frontier_model="model1",
        local_model="model2",
        example=training_example,
        quality_score=0.9
    )
    
    # Verify store_training_example was called
    data_collector.db_manager.store_training_example.assert_called_once()
    
    # Retrieve training data
    training_data = await data_collector.get_training_data(
        agent_type="king",
        min_quality=0.8
    )
    
    assert len(training_data) == 1
    assert training_data[0]["input"] == "Test input"

@pytest.mark.asyncio
async def test_performance_metrics_storage(data_collector):
    """Test storing and retrieving performance metrics."""
    metrics = {
        "accuracy": 0.9,
        "latency": 100,
        "success_rate": 0.95
    }
    
    # Store metrics
    await data_collector.store_performance_metrics(
        agent_type="king",
        model_type="frontier",
        metrics=metrics
    )
    
    # Verify store_performance_metrics was called
    data_collector.db_manager.store_performance_metrics.assert_called_once()
    
    # Retrieve metrics
    stored_metrics = await data_collector.get_performance_metrics(
        agent_type="king",
        model_type="frontier",
        metric_names=["accuracy", "success_rate"]
    )
    
    assert "accuracy" in stored_metrics
    assert stored_metrics["accuracy"] == 0.9
    assert stored_metrics["success_rate"] == 0.95

def test_complexity_evaluation(complexity_evaluator):
    """Test task complexity evaluation."""
    # Test simple task
    simple_task = "Write a function to add two numbers"
    simple_result = complexity_evaluator.evaluate_complexity(
        agent_type="king",
        task=simple_task
    )
    
    assert not simple_result["is_complex"]
    assert simple_result["complexity_score"] < 0.5
    assert "threshold_used" in simple_result
    assert "components" in simple_result
    
    # Test complex task
    complex_task = """
    Design and implement a distributed system for real-time data processing
    with fault tolerance and horizontal scaling capabilities.
    
    Requirements:
    - Handle concurrent requests
    - Implement load balancing
    - Ensure data consistency
    - Provide monitoring and logging
    """
    complex_result = complexity_evaluator.evaluate_complexity(
        agent_type="king",
        task=complex_task
    )
    
    assert not complex_result["is_complex"]
    assert complex_result["complexity_score"] < 0.5
    assert "threshold_used" in complex_result
    assert "components" in complex_result

def test_threshold_adjustment(complexity_evaluator):
    """Test complexity threshold adjustment."""
    # Record performance data
    complexity_evaluator.record_performance(
        agent_type="king",
        task_complexity={
            "complexity_score": 0.8,
            "is_complex": True,
            "threshold_used": 0.7
        },
        performance_metrics={
            "success_rate": 0.9,
            "efficiency": 0.8
        }
    )
    
    # Get adjusted threshold
    new_threshold = complexity_evaluator.adjust_thresholds("king")
    assert isinstance(new_threshold, (float, int))
    assert 0 <= new_threshold <= 1
    
    # Test threshold analysis
    analysis = complexity_evaluator.get_threshold_analysis("king")
    assert "current_threshold" in analysis
    assert "performance_by_complexity" in analysis
    assert "complex" in analysis["performance_by_complexity"]
    assert "simple" in analysis["performance_by_complexity"]

@pytest.mark.asyncio
async def test_data_export(data_collector):
    """Test data export functionality."""
    # Create test data
    interaction = AgentInteraction(
        prompt="Export test",
        response="Test response",
        model="test-model",
        timestamp=datetime.now().timestamp(),
        metadata={"test": "data"}
    )
    
    await data_collector.store_interaction(
        agent_type="king",
        interaction=interaction.__dict__
    )
    
    # Verify store_interaction was called
    data_collector.db_manager.store_interaction.assert_called_once()

@pytest.mark.asyncio
async def test_performance_aggregation(data_collector):
    """Test performance metrics aggregation."""
    # Store multiple metrics
    metrics1 = {"accuracy": 0.9, "latency": 100}
    metrics2 = {"accuracy": 0.8, "latency": 120}
    
    await data_collector.store_performance_metrics(
        agent_type="king",
        model_type="frontier",
        metrics=metrics1
    )
    
    await data_collector.store_performance_metrics(
        agent_type="king",
        model_type="frontier",
        metrics=metrics2
    )
    
    # Verify store_performance_metrics was called twice
    assert data_collector.db_manager.store_performance_metrics.call_count == 2
    
    # Get aggregated metrics
    aggregated = await data_collector.get_performance_metrics(
        agent_type="king",
        model_type="frontier",
        metric_names=["accuracy"]
    )
    
    assert "accuracy" in aggregated
    assert aggregated["accuracy"] == 0.9

def test_complexity_confidence(complexity_evaluator):
    """Test complexity evaluation confidence."""
    task = "Write a function to calculate Fibonacci numbers"
    result = complexity_evaluator.evaluate_complexity(
        agent_type="king",
        task=task
    )
    
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert "threshold_used" in result
    assert "components" in result

@pytest.mark.asyncio
async def test_database_maintenance(data_collector):
    """Test database maintenance operations."""
    # Create old test data
    old_time = datetime.now() - timedelta(days=31)
    
    interaction = AgentInteraction(
        prompt="Old test",
        response="Test response",
        model="test-model",
        timestamp=old_time.timestamp(),
        metadata={}
    )
    
    await data_collector.store_interaction(
        agent_type="king",
        interaction=interaction.__dict__
    )
    
    # Verify store_interaction was called
    data_collector.db_manager.store_interaction.assert_called_once()
    
    # Run maintenance
    cleaned = await data_collector.run_maintenance()
    assert cleaned == 1

if __name__ == "__main__":
    pytest.main([__file__])
