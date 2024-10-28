"""Tests for data collection and management system."""

import pytest
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from config.unified_config import UnifiedConfig, DatabaseConfig
from agent_forge.data.data_collector import DataCollector
from agent_forge.data.complexity_evaluator import ComplexityEvaluator

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def test_db_path():
    """Create temporary database path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test.db"

@pytest.fixture
def db_manager(test_db_path):
    """Create DatabaseManager instance with test database."""
    config = DatabaseConfig(
        path=str(test_db_path),
        backup_interval=1,  # 1 hour for testing
        max_backup_count=2,
        vacuum_threshold=10
    )
    with patch('agent_forge.data.data_collector.DatabaseManager._schedule_maintenance'):
        return DatabaseManager(config)

@pytest.fixture
def config():
    """Create test configuration."""
    config = UnifiedConfig()
    return config

@pytest.fixture
def data_collector(config, test_db_path):
    """Create DataCollector instance with test configuration."""
    config.config['database']['path'] = str(test_db_path)
    with patch('agent_forge.data.data_collector.DatabaseManager._schedule_maintenance'):
        return DataCollector(config)

@pytest.fixture
def complexity_evaluator():
    """Create ComplexityEvaluator instance."""
    return ComplexityEvaluator()

@pytest.mark.asyncio
async def test_interaction_storage(data_collector):
    """Test interaction data storage."""
    interaction = {
        "agent_type": "king",
        "task": "test task",
        "result": "test result",
        "timestamp": datetime.now().isoformat()
    }
    
    await data_collector.store_interaction(interaction)
    interactions = await data_collector.get_interactions()
    
    assert len(interactions) > 0
    assert interactions[0]["task"] == "test task"

@pytest.mark.asyncio
async def test_training_data_storage(data_collector):
    """Test training data storage."""
    training_data = {
        "input": "test input",
        "output": "test output",
        "agent_type": "king",
        "timestamp": datetime.now().isoformat()
    }
    
    await data_collector.store_training_data(training_data)
    data = await data_collector.get_training_data("king")
    
    assert len(data) > 0
    assert data[0]["input"] == "test input"

@pytest.mark.asyncio
async def test_performance_metrics_storage(data_collector):
    """Test performance metrics storage."""
    metrics = {
        "agent_type": "king",
        "success_rate": 0.8,
        "response_time": 1.2,
        "timestamp": datetime.now().isoformat()
    }
    
    await data_collector.store_performance_metrics(metrics)
    stored_metrics = await data_collector.get_performance_metrics("king")
    
    assert len(stored_metrics) > 0
    assert stored_metrics[0]["success_rate"] == 0.8

@pytest.mark.asyncio
async def test_database_backup(db_manager):
    """Test database backup functionality."""
    # Add some test data
    await db_manager.execute(
        "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)"
    )
    await db_manager.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
    
    # Trigger backup
    await db_manager.backup()
    
    # Verify backup exists
    backup_path = Path(db_manager.config.path).parent / "backups"
    assert backup_path.exists()
    assert any(backup_path.glob("*.db"))

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
    with fault tolerance and horizontal scaling capabilities:
    - Handle concurrent requests
    - Implement load balancing
    - Ensure data consistency
    - Provide monitoring and logging
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
        task_complexity={
            "complexity_score": 0.8,
            "is_complex": True,
            "threshold_used": 0.7
        },
        performance_metrics={"success_rate": 0.9, "efficiency": 0.8}
    )
    
    # Get adjusted threshold
    new_threshold = await complexity_evaluator.get_adjusted_threshold("king")
    assert isinstance(new_threshold, float)
    assert 0.0 <= new_threshold <= 1.0

@pytest.mark.asyncio
async def test_data_export(data_collector):
    """Test data export functionality."""
    # Add test data
    interaction = {
        "agent_type": "king",
        "task": "test task",
        "result": "test result",
        "timestamp": datetime.now().isoformat()
    }
    await data_collector.store_interaction(interaction)
    
    # Export data
    export_path = Path(tempfile.mkdtemp()) / "export.json"
    await data_collector.export_data(str(export_path))
    
    # Verify export
    assert export_path.exists()
    assert export_path.stat().st_size > 0

@pytest.mark.asyncio
async def test_performance_aggregation(data_collector):
    """Test performance metrics aggregation."""
    # Add test metrics
    metrics = [
        {
            "agent_type": "king",
            "success_rate": 0.8,
            "response_time": 1.2,
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat()
        }
        for i in range(5)
    ]
    
    for m in metrics:
        await data_collector.store_performance_metrics(m)
    
    # Get aggregated metrics
    aggregated = await data_collector.get_aggregated_metrics("king", hours=24)
    assert isinstance(aggregated, dict)
    assert "avg_success_rate" in aggregated
    assert "avg_response_time" in aggregated

@pytest.mark.asyncio
async def test_complexity_confidence(complexity_evaluator):
    """Test complexity evaluation confidence."""
    task = "Write a function to calculate Fibonacci numbers"
    result = await complexity_evaluator.evaluate_complexity(
        agent_type="magi",
        task=task
    )
    
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_database_maintenance(db_manager):
    """Test database maintenance operations."""
    # Add test data
    await db_manager.execute(
        "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)"
    )
    for i in range(20):
        await db_manager.execute("INSERT INTO test (value) VALUES (?)", (f"value_{i}",))
    
    # Run maintenance
    await db_manager.run_maintenance()
    
    # Verify database is still accessible
    result = await db_manager.fetch_one("SELECT COUNT(*) FROM test")
    assert result[0] == 20
    
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
