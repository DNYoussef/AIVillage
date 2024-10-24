"""Unit tests for MAGI integration systems."""

import pytest
import asyncio
import aiohttp
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from ....magi.integrations.database import DatabaseManager
from ....magi.integrations.api import APIManager
from ....magi.core.exceptions import DatabaseError, APIError

@pytest.fixture
def db_manager():
    """Create database manager instance with test database."""
    return DatabaseManager(db_path=":memory:")

@pytest.fixture
async def api_manager():
    """Create API manager instance."""
    manager = APIManager(
        config_path=None,
        rate_limit=10,
        timeout=5.0
    )
    return manager

@pytest.fixture
def sample_tool_data():
    """Create sample tool data."""
    return {
        "name": "test_tool",
        "description": "Test tool",
        "code": "def test(): pass",
        "parameters": {"param1": "str"},
        "version": "1.0.0",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "usage_count": 0,
        "success_rate": 0.0
    }

@pytest.fixture
def sample_technique_data():
    """Create sample technique data."""
    return {
        "name": "test_technique",
        "description": "Test technique",
        "parameters": {"param1": "str"},
        "version": "1.0.0",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "usage_count": 0,
        "success_rate": 0.0
    }

# Database Tests
@pytest.mark.asyncio
async def test_database_initialization(db_manager):
    """Test database initialization."""
    # Check tables exist
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table'
        """)
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {
            'tools',
            'tool_executions',
            'techniques',
            'technique_executions',
            'performance_metrics',
            'system_state',
            'analytics_data'
        }
        
        assert required_tables.issubset(tables)

@pytest.mark.asyncio
async def test_tool_storage(db_manager, sample_tool_data):
    """Test tool data storage and retrieval."""
    # Store tool
    await db_manager.store_tool(sample_tool_data)
    
    # Retrieve tool
    tool = await db_manager.get_tool(sample_tool_data["name"])
    
    assert tool is not None
    assert tool["name"] == sample_tool_data["name"]
    assert tool["version"] == sample_tool_data["version"]
    assert tool["parameters"] == sample_tool_data["parameters"]

@pytest.mark.asyncio
async def test_execution_history(db_manager, sample_tool_data):
    """Test execution history storage and retrieval."""
    # Store execution data
    execution_data = {
        "tool_name": sample_tool_data["name"],
        "parameters": {"input": "test"},
        "result": {"output": "success"},
        "success": True,
        "execution_time": 0.1,
        "timestamp": datetime.now()
    }
    
    await db_manager.store_tool_execution(execution_data)
    
    # Retrieve history
    history = await db_manager.get_tool_history(sample_tool_data["name"])
    
    assert len(history) == 1
    assert history[0]["tool_name"] == execution_data["tool_name"]
    assert history[0]["success"] == execution_data["success"]

@pytest.mark.asyncio
async def test_performance_metrics(db_manager):
    """Test performance metrics storage and retrieval."""
    # Store metrics
    await db_manager.store_metric(
        metric_type="response_time",
        target_type="technique",
        target_name="test_technique",
        value=0.5
    )
    
    # Retrieve metrics
    metrics = await db_manager.get_metrics(
        metric_type="response_time",
        target_type="technique"
    )
    
    assert len(metrics) == 1
    assert metrics[0]["metric_type"] == "response_time"
    assert metrics[0]["value"] == 0.5

@pytest.mark.asyncio
async def test_system_state(db_manager):
    """Test system state storage and retrieval."""
    state = {
        "active_techniques": ["technique1", "technique2"],
        "memory_usage": 1024,
        "uptime": 3600
    }
    
    await db_manager.store_system_state("test_component", state)
    
    # Verify state storage
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT state FROM system_state WHERE component = ?",
            ("test_component",)
        )
        stored_state = cursor.fetchone()
        assert stored_state is not None

# API Tests
@pytest.mark.asyncio
async def test_api_initialization(api_manager):
    """Test API manager initialization."""
    assert api_manager.rate_limit == 10
    assert api_manager.timeout == 5.0
    assert api_manager.session is None

@pytest.mark.asyncio
async def test_rate_limiting(api_manager):
    """Test API rate limiting."""
    async with api_manager:
        start_time = datetime.now()
        
        # Make multiple requests quickly
        for _ in range(5):
            await api_manager._check_rate_limit()
        
        # Verify rate limiting
        elapsed = (datetime.now() - start_time).total_seconds()
        assert elapsed < 1.0  # Should be quick since under limit
        
        # Reset request times to simulate limit
        api_manager.request_times = [
            datetime.now() for _ in range(api_manager.rate_limit)
        ]
        
        # Next request should wait
        start_time = datetime.now()
        await api_manager._check_rate_limit()
        elapsed = (datetime.now() - start_time).total_seconds()
        assert elapsed > 0.0  # Should have waited

@pytest.mark.asyncio
async def test_request_methods(api_manager):
    """Test different HTTP request methods."""
    async with api_manager:
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        
        with patch.object(api_manager.session, 'get', return_value=mock_response):
            result = await api_manager.get("test_api", "endpoint")
            assert result["status"] == "success"
        
        with patch.object(api_manager.session, 'post', return_value=mock_response):
            result = await api_manager.post("test_api", "endpoint", {"data": "test"})
            assert result["status"] == "success"
        
        with patch.object(api_manager.session, 'put', return_value=mock_response):
            result = await api_manager.put("test_api", "endpoint", {"data": "test"})
            assert result["status"] == "success"
        
        with patch.object(api_manager.session, 'delete', return_value=mock_response):
            result = await api_manager.delete("test_api", "endpoint")
            assert result["status"] == "success"

@pytest.mark.asyncio
async def test_error_handling(api_manager):
    """Test API error handling."""
    async with api_manager:
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        
        with patch.object(api_manager.session, 'get', return_value=mock_response):
            with pytest.raises(APIError) as exc_info:
                await api_manager.get("test_api", "endpoint")
            assert "404" in str(exc_info.value)

@pytest.mark.asyncio
async def test_streaming(api_manager):
    """Test streaming API responses."""
    async with api_manager:
        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content.iter_any = AsyncMock(
            return_value=["data1", "data2", "data3"]
        )
        
        with patch.object(api_manager.session, 'get', return_value=mock_response):
            stream = await api_manager.stream("test_api", "stream_endpoint")
            data = []
            async for chunk in stream.content:
                data.append(chunk)
            
            assert len(data) == 3

@pytest.mark.asyncio
async def test_timeout_handling(api_manager):
    """Test request timeout handling."""
    async with api_manager:
        # Mock timeout
        async def timeout_effect(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError()
        
        with patch.object(api_manager.session, 'get', side_effect=timeout_effect):
            with pytest.raises(APIError) as exc_info:
                await api_manager.get("test_api", "endpoint")
            assert "timeout" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_concurrent_requests(api_manager):
    """Test handling of concurrent API requests."""
    async with api_manager:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        
        with patch.object(api_manager.session, 'get', return_value=mock_response):
            # Make concurrent requests
            tasks = [
                api_manager.get("test_api", f"endpoint{i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert all(r["status"] == "success" for r in results)

@pytest.mark.asyncio
async def test_session_management(api_manager):
    """Test API session management."""
    # Session should start as None
    assert api_manager.session is None
    
    # Session should be created in context
    async with api_manager:
        assert api_manager.session is not None
        assert not api_manager.session.closed
    
    # Session should be closed after context
    assert api_manager.session.closed
