"""Unit tests for MAGI core components."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.core.config import MagiAgentConfig
from ....magi.core.constants import DEFAULT_TIMEOUT, MAX_RETRIES
from ....magi.core.exceptions import (
    ConfigurationError,
    ExecutionError,
    ValidationError,
    ToolError
)
from ....magi.core.magi_agent import MagiAgent

@pytest.fixture
def config():
    """Create test configuration."""
    return MagiAgentConfig(
        name="test_magi",
        description="Test MAGI instance",
        capabilities=["coding", "debugging", "analysis"],
        development_capabilities=["refactoring", "optimization"],
        model="test-model"
    )

@pytest.fixture
def mock_llm():
    """Create mock language model."""
    mock = AsyncMock()
    mock.complete = AsyncMock()
    return mock

@pytest.fixture
async def magi_agent(config, mock_llm):
    """Create MAGI agent instance."""
    agent = MagiAgent(config)
    agent.llm = mock_llm
    return agent

@pytest.mark.asyncio
async def test_config_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = MagiAgentConfig(
        name="test",
        description="test",
        capabilities=["test"],
        model="test"
    )
    assert valid_config is not None
    
    # Invalid configs
    with pytest.raises(ConfigurationError):
        MagiAgentConfig(
            name="",  # Empty name
            description="test",
            capabilities=["test"],
            model="test"
        )
    
    with pytest.raises(ConfigurationError):
        MagiAgentConfig(
            name="test",
            description="test",
            capabilities=[],  # Empty capabilities
            model="test"
        )

@pytest.mark.asyncio
async def test_agent_initialization(config):
    """Test agent initialization."""
    agent = MagiAgent(config)
    
    assert agent.name == config.name
    assert agent.description == config.description
    assert set(agent.capabilities) == set(config.capabilities)
    assert agent.model == config.model
    assert len(agent.techniques) > 0
    assert agent.task_history == []

@pytest.mark.asyncio
async def test_timeout_handling(magi_agent, mock_llm):
    """Test handling of execution timeouts."""
    # Mock timeout
    async def timeout_effect(*args, **kwargs):
        await asyncio.sleep(0.1)
        raise asyncio.TimeoutError()
    
    mock_llm.complete.side_effect = timeout_effect
    
    with pytest.raises(ExecutionError) as exc_info:
        await magi_agent.execute_task(
            "Test task",
            timeout=0.05  # Short timeout
        )
    assert "timeout" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_retry_mechanism(magi_agent, mock_llm):
    """Test retry mechanism for failed operations."""
    # Mock temporary failures then success
    mock_llm.complete.side_effect = [
        Exception("Temporary error"),
        Exception("Temporary error"),
        Mock(text="Success\nConfidence: 0.9")
    ]
    
    result = await magi_agent.execute_task("Test task")
    
    assert result is not None
    assert "success" in result.result.lower()
    assert mock_llm.complete.call_count == 3

@pytest.mark.asyncio
async def test_capability_validation(magi_agent):
    """Test validation of agent capabilities."""
    # Test with supported capability
    assert await magi_agent.validate_capability("coding")
    
    # Test with unsupported capability
    assert not await magi_agent.validate_capability("unsupported")
    
    # Test with invalid input
    with pytest.raises(ValidationError):
        await magi_agent.validate_capability("")

@pytest.mark.asyncio
async def test_error_handling(magi_agent, mock_llm):
    """Test error handling and propagation."""
    # Mock various error types
    errors = [
        ConfigurationError("Config error"),
        ExecutionError("Execution error"),
        ValidationError("Validation error"),
        ToolError("Tool error"),
        Exception("Generic error")
    ]
    
    for error in errors:
        mock_llm.complete.side_effect = error
        
        try:
            await magi_agent.execute_task("Test task")
            assert False, f"Should have raised {type(error)}"
        except Exception as e:
            assert isinstance(e, type(error))
            assert str(error) in str(e)

@pytest.mark.asyncio
async def test_task_history(magi_agent, mock_llm):
    """Test maintenance of task execution history."""
    mock_llm.complete.side_effect = [
        Mock(text="Result 1\nConfidence: 0.8"),
        Mock(text="Result 2\nConfidence: 0.9")
    ]
    
    # Execute multiple tasks
    await magi_agent.execute_task("Task 1")
    await magi_agent.execute_task("Task 2")
    
    assert len(magi_agent.task_history) == 2
    assert magi_agent.task_history[0].task == "Task 1"
    assert magi_agent.task_history[1].task == "Task 2"

@pytest.mark.asyncio
async def test_state_management(magi_agent):
    """Test agent state management."""
    # Get initial state
    state = magi_agent.get_state()
    assert state is not None
    assert "capabilities" in state
    assert "task_history" in state
    
    # Modify state
    new_state = state.copy()
    new_state["test_key"] = "test_value"
    
    # Load modified state
    magi_agent.load_state(new_state)
    assert magi_agent.get_state()["test_key"] == "test_value"

@pytest.mark.asyncio
async def test_concurrent_execution(magi_agent, mock_llm):
    """Test handling of concurrent task execution."""
    mock_llm.complete.side_effect = [
        Mock(text="Result 1\nConfidence: 0.8"),
        Mock(text="Result 2\nConfidence: 0.9"),
        Mock(text="Result 3\nConfidence: 0.7")
    ]
    
    # Execute tasks concurrently
    tasks = [
        magi_agent.execute_task(f"Task {i}")
        for i in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    assert all(result is not None for result in results)
    assert len(magi_agent.task_history) == 3

@pytest.mark.asyncio
async def test_resource_management(magi_agent):
    """Test resource management and cleanup."""
    # Track initial resources
    initial_resources = magi_agent.get_resource_usage()
    
    # Execute resource-intensive task
    await magi_agent.execute_task("Heavy task")
    
    # Check resource usage
    peak_resources = magi_agent.get_resource_usage()
    assert peak_resources.memory >= initial_resources.memory
    
    # Cleanup
    await magi_agent.cleanup()
    final_resources = magi_agent.get_resource_usage()
    assert final_resources.memory <= peak_resources.memory

@pytest.mark.asyncio
async def test_configuration_updates(magi_agent):
    """Test dynamic configuration updates."""
    # Update configuration
    new_config = MagiAgentConfig(
        name=magi_agent.name,
        description="Updated description",
        capabilities=magi_agent.capabilities + ["new_capability"],
        model=magi_agent.model
    )
    
    await magi_agent.update_config(new_config)
    
    assert magi_agent.description == new_config.description
    assert "new_capability" in magi_agent.capabilities

@pytest.mark.asyncio
async def test_constants_validation():
    """Test validation of system constants."""
    assert DEFAULT_TIMEOUT > 0
    assert MAX_RETRIES > 0
    assert isinstance(DEFAULT_TIMEOUT, (int, float))
    assert isinstance(MAX_RETRIES, int)

@pytest.mark.asyncio
async def test_exception_hierarchy():
    """Test exception class hierarchy."""
    # Verify base exception
    assert issubclass(ConfigurationError, Exception)
    assert issubclass(ExecutionError, Exception)
    assert issubclass(ValidationError, Exception)
    assert issubclass(ToolError, Exception)
    
    # Test exception creation
    error = ConfigurationError("test")
    assert str(error) == "test"
    assert isinstance(error, Exception)
