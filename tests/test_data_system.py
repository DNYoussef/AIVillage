"""Tests for data collection and complexity evaluation system."""

import pytest
import sqlite3
import json
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import shutil

from config.unified_config import UnifiedConfig, DatabaseConfig
from agent_forge.data.data_collector import DataCollector, DatabaseManager
from agent_forge.data.complexity_evaluator import ComplexityEvaluator

@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig()

@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary database path."""
    return tmp_path / "test.db"

@pytest.fixture
def db_manager(test_db_path):
    """Create DatabaseManager instance with test database."""
    config = DatabaseConfig(
        path=str(test_db_path),
        backup_interval=1,  # 1 hour for testing
        max_backup_count=2,
        vacuum_threshold=10
    )
    return DatabaseManager(config)

@pytest.fixture
def data_collector(config, test_db_path):
    """Create DataCollector instance with test configuration."""
    config.config['database']['path'] = str(test_db_path)
    return DataCollector(config)

@pytest.fixture
def complexity_evaluator(config):
    """Create ComplexityEvaluator instance."""
    return ComplexityEvaluator(config)

@pytest.mark.asyncio
async def test_interaction_storage(data_collector):
    """Test storing and retrieving agent interactions."""
    # Store interaction
    interaction_data = {
        "agent_type": "king",
        "model": "test_model",
        "timestamp": datetime.now().timestamp(),
        "prompt": "test prompt",
        "response": "test response",
        "was_complex": True,
        "performance_metrics": {
            "duration": 1.0,
            "tokens": 100
        },
        "metadata": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    await data_collector.store_interaction(
        agent_type=interaction_data["agent_type"],
        interaction=interaction_data,
        was_complex=interaction_data["was_complex"],
        performance_metrics=interaction_data["performance_metrics"]
    )
    
    # Verify storage
    with sqlite3.connect(data_collector.db_manager.db_path) as conn:
        cursor = conn.execute("SELECT * FROM interactions WHERE agent_type = ?", 
                            (interaction_data["agent_type"],))
        stored = cursor.fetchone()
        
        assert stored is not None
        assert json.loads(stored[7]) == interaction_data["performance_metrics"]
        assert json.loads(stored[8]) == interaction_data["metadata"]

@pytest.mark.asyncio
async def test_training_data_storage(data_collector):
    """Test storing and retrieving training data."""
    # Store training example
    example_data = {
        "agent_type": "king",
        "frontier_model": "model1",
        "local_model": "model2",
        "prompt": "test prompt",
        "response": "test response",
        "quality_score": 0.9,
        "performance_metrics": {
            "accuracy": 0.95,
            "latency": 0.1
        }
    }
    
    await data_collector.store_training_example(**example_data)
    
    # Retrieve training data
    training_data = await data_collector.get_training_data(
        agent_type="king",
        min_quality=0.8
    )
    
    # Verify retrieval
    assert len(training_data) == 1
    assert training_data[0]["prompt"] == example_data["prompt"]
    assert training_data[0]["quality_score"] == example_data["quality_score"]

@pytest.mark.asyncio
async def test_performance_metrics_storage(data_collector):
    """Test storing and retrieving performance metrics."""
    # Store metrics
    metrics_data = {
        "response_time": 0.5,
        "success_rate": 0.95,
        "token_efficiency": 0.8
    }
    
    await data_collector.store_performance_metrics(
        agent_type="king",
        model_type="frontier",
        metrics=metrics_data,
        context={"batch_size": 32},
        aggregation_period="hour"
    )
    
    # Retrieve performance history
    history = await data_collector.get_performance_history(
        agent_type="king",
        model_type="frontier",
        metric_name="success_rate",
        days=1
    )
    
    # Verify retrieval
    assert len(history) > 0
    assert history[0]["avg_value"] == 0.95

@pytest.mark.asyncio
async def test_database_backup(db_manager):
    """Test database backup functionality."""
    # Create some test data
    with sqlite3.connect(db_manager.db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
    
    # Create backup
    await db_manager.create_backup()
    
    # Verify backup was created
    backup_dir = db_manager.db_path.parent / "backups"
    assert backup_dir.exists()
    assert len(list(backup_dir.glob("*.db"))) > 0

@pytest.mark.asyncio
async def test_complexity_evaluation(complexity_evaluator):
    """Test task complexity evaluation."""
    # Test simple task
    simple_task = "Write a function to add two numbers"
    simple_result = await complexity_evaluator.evaluate_complexity(
        agent_type="magi",
        task=simple_task
    )
    
    assert not simple_result["is_complex"]
    assert simple_result["complexity_score"] < 0.5
    
    # Test complex task
    complex_task = """
    Design and implement a distributed system for real-time data processing
    with fault tolerance and horizontal scaling capabilities.
    Consider performance optimization and concurrent request handling.
    """
    complex_result = await complexity_evaluator.evaluate_complexity(
        agent_type="magi",
        task=complex_task
    )
    
    assert complex_result["is_complex"]
    assert complex_result["complexity_score"] > 0.7

@pytest.mark.asyncio
async def test_threshold_adjustment(complexity_evaluator):
    """Test complexity threshold adjustment."""
    # Record some performance data
    await complexity_evaluator.record_performance(
        agent_type="king",
        task_complexity={"complexity_score": 0.8, "is_complex": True},
        performance_metrics={"success_rate": 0.9, "efficiency": 0.8}
    )
    
    # Adjust thresholds
    await complexity_evaluator.adjust_thresholds("king")
    
    # Get threshold analysis
    analysis = complexity_evaluator.get_threshold_analysis("king")
    
    assert "current_threshold" in analysis
    assert "threshold_range" in analysis
    assert "performance_by_complexity" in analysis

@pytest.mark.asyncio
async def test_data_export(data_collector):
    """Test data export functionality."""
    # Store some test data
    await data_collector.store_interaction(
        agent_type="king",
        interaction={
            "prompt": "test",
            "response": "test",
            "model": "test_model",
            "timestamp": datetime.now().timestamp(),
            "metadata": {}
        },
        was_complex=False
    )
    
    # Export data
    export_files = await data_collector.export_data(format="json")
    
    # Verify export
    assert "interactions" in export_files
    assert os.path.exists(export_files["interactions"])
    
    # Check exported data
    with open(export_files["interactions"], 'r') as f:
        exported_data = json.load(f)
        assert len(exported_data) > 0

@pytest.mark.asyncio
async def test_performance_aggregation(data_collector):
    """Test performance metrics aggregation."""
    # Store metrics at different times
    base_time = datetime.now()
    
    for i in range(5):
        timestamp = base_time + timedelta(minutes=i*15)
        await data_collector.store_performance_metrics(
            agent_type="king",
            model_type="frontier",
            metrics={"accuracy": 0.8 + i*0.05},
            timestamp=timestamp.timestamp(),
            aggregation_period="hour"
        )
    
    # Get aggregated metrics
    history = await data_collector.get_performance_history(
        agent_type="king",
        model_type="frontier",
        metric_name="accuracy",
        days=1,
        aggregation="hour"
    )
    
    # Verify aggregation
    assert len(history) == 1  # All metrics within same hour
    assert "avg_value" in history[0]
    assert "min_value" in history[0]
    assert "max_value" in history[0]

@pytest.mark.asyncio
async def test_complexity_confidence(complexity_evaluator):
    """Test complexity evaluation confidence scoring."""
    task = "Implement a web server with authentication"
    
    result = await complexity_evaluator.evaluate_complexity(
        agent_type="magi",
        task=task,
        context={"requirements": ["security", "performance"]}
    )
    
    # Verify confidence score
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    
    # Verify analysis
    assert "analysis" in result
    assert "explanation" in result["analysis"]
    assert "primary_factors" in result["analysis"]

@pytest.mark.asyncio
async def test_database_maintenance(db_manager):
    """Test database maintenance operations."""
    # Add test data
    with sqlite3.connect(db_manager.db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)")
        for i in range(20):  # Exceed vacuum threshold
            conn.execute("INSERT INTO test (value) VALUES (?)", (f"value_{i}",))
    
    # Run maintenance
    await db_manager.vacuum_if_needed()
    
    # Clean old backups
    backup_dir = db_manager.db_path.parent / "backups"
    if backup_dir.exists():
        for i in range(5):  # Create more than max_backup_count
            backup_path = backup_dir / f"backup_{i}.db"
            shutil.copy2(str(db_manager.db_path), str(backup_path))
    
    await db_manager.clean_old_backups()
    
    # Verify backup cleanup
    remaining_backups = list(backup_dir.glob("*.db"))
    assert len(remaining_backups) <= db_manager.max_backup_count

if __name__ == "__main__":
    pytest.main([__file__])
