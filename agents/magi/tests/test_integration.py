"""Integration tests for MAGI system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ...magi.magi_agent import MagiAgent
from ...magi.core.config import MagiAgentConfig
from ...magi.techniques.registry import TechniqueRegistry
from ...magi.feedback.analysis import FeedbackAnalyzer
from ...magi.feedback.improvement import ImprovementManager
from ...magi.integrations.database import DatabaseManager
from ...magi.integrations.api import APIManager

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
async def test_technique_integration(magi_agent, mock_llm):
    """Test integration of multiple reasoning techniques."""
    mock_llm.complete.side_effect = [
        # Chain of Thought response
        Mock(text="""
        Thought: Break down the problem
        Reasoning: Need to understand components
        Next: Analyze each part
        Confidence: 0.8
        """),
        # Tree of Thoughts response
        Mock(text="""
        Branch 1:
        Approach: Use caching
        Score: 0.9
        
        Branch 2:
        Approach: Use indexing
        Score: 0.7
        """),
        # Final synthesis
        Mock(text="""
        Solution: Implement caching with indexing
        Reasoning: Combined best approaches
        Confidence: 0.85
        """)
    ]
    
    result = await magi_agent.execute_task("Optimize database queries")
    
    assert result is not None
    assert "caching" in result.result.lower()
    assert "indexing" in result.result.lower()
    assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_feedback_integration(magi_agent):
    """Test integration of feedback and improvement systems."""
    analyzer = FeedbackAnalyzer()
    improvement_manager = ImprovementManager(analyzer)
    
    # Execute task and collect feedback
    task_result = await magi_agent.execute_task("Implement feature")
    analyzer.record_execution(task_result)
    
    # Generate and apply improvements
    improvements = await improvement_manager.generate_improvement_plans()
    improvement_results = await improvement_manager.implement_improvements(improvements)
    
    assert len(improvement_results) > 0
    assert all(result.success for result in improvement_results)

@pytest.mark.asyncio
async def test_database_integration(magi_agent):
    """Test integration with database system."""
    db_manager = DatabaseManager()
    
    # Store and retrieve tool data
    tool_data = {
        "name": "test_tool",
        "description": "Test tool",
        "code": "def test(): pass",
        "parameters": {"param1": "str"},
        "version": "1.0.0"
    }
    
    await db_manager.store_tool(tool_data)
    retrieved_tool = await db_manager.get_tool("test_tool")
    
    assert retrieved_tool is not None
    assert retrieved_tool["name"] == tool_data["name"]
    assert retrieved_tool["version"] == tool_data["version"]

@pytest.mark.asyncio
async def test_api_integration(magi_agent):
    """Test integration with API system."""
    api_manager = APIManager()
    
    async with api_manager:
        # Test API request
        try:
            response = await api_manager.get(
                "test_api",
                "test_endpoint",
                params={"test": "value"}
            )
            assert response is not None
        except Exception as e:
            assert "No base URL configured" in str(e)

@pytest.mark.asyncio
async def test_technique_chaining(magi_agent, mock_llm):
    """Test chaining of multiple techniques."""
    mock_llm.complete.side_effect = [
        # Self Ask response
        Mock(text="""
        Question: What are the requirements?
        Answer: Need fast data access
        Confidence: 0.9
        """),
        # Least to Most response
        Mock(text="""
        Step 1: Implement basic structure
        Step 2: Add optimization
        Confidence: 0.85
        """),
        # Program of Thoughts response
        Mock(text="""
        Code:
        def optimize():
            implement_cache()
        Confidence: 0.8
        """),
        # Final synthesis
        Mock(text="""
        Solution: Optimized implementation complete
        Confidence: 0.85
        """)
    ]
    
    result = await magi_agent.execute_task("Design and implement feature")
    
    assert result is not None
    assert result.confidence > 0.8
    assert len(result.technique_results) > 1

@pytest.mark.asyncio
async def test_error_propagation(magi_agent, mock_llm):
    """Test error handling across components."""
    mock_llm.complete.side_effect = Exception("Test error")
    
    try:
        await magi_agent.execute_task("Test task")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Test error" in str(e)
        # Verify error was logged
        assert len(magi_agent.error_history) > 0

@pytest.mark.asyncio
async def test_concurrent_execution(magi_agent, mock_llm):
    """Test concurrent execution of tasks."""
    mock_llm.complete.side_effect = [
        Mock(text="Result 1\nConfidence: 0.8"),
        Mock(text="Result 2\nConfidence: 0.85"),
        Mock(text="Result 3\nConfidence: 0.9")
    ]
    
    tasks = [
        magi_agent.execute_task(f"Task {i}")
        for i in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    assert all(result is not None for result in results)
    assert all(result.confidence > 0.7 for result in results)

@pytest.mark.asyncio
async def test_state_persistence(magi_agent, mock_llm):
    """Test persistence of agent state across executions."""
    db_manager = DatabaseManager()
    
    # Execute task and store state
    result1 = await magi_agent.execute_task("First task")
    await db_manager.store_system_state("magi", magi_agent.get_state())
    
    # Create new agent instance
    new_agent = MagiAgent(config)
    new_agent.llm = mock_llm
    
    # Load state and execute another task
    state = await db_manager.get_system_state("magi")
    new_agent.load_state(state)
    result2 = await new_agent.execute_task("Second task")
    
    assert result2 is not None
    assert new_agent.task_history == magi_agent.task_history

@pytest.mark.asyncio
async def test_technique_registry_integration(magi_agent):
    """Test integration with technique registry."""
    registry = TechniqueRegistry()
    
    # Register and retrieve technique
    registry.register("test_technique", Mock())
    technique = registry.get("test_technique")
    
    assert technique is not None
    assert "test_technique" in registry.list_techniques()

@pytest.mark.asyncio
async def test_performance_monitoring(magi_agent, mock_llm):
    """Test performance monitoring across components."""
    analyzer = FeedbackAnalyzer()
    
    # Execute multiple tasks
    for i in range(3):
        result = await magi_agent.execute_task(f"Task {i}")
        analyzer.record_execution(result)
    
    # Analyze performance
    metrics = analyzer.analyze_system_performance()
    
    assert metrics.total_tasks == 3
    assert metrics.average_response_time > 0
    assert len(metrics.active_techniques) > 0

@pytest.mark.asyncio
async def test_resource_management(magi_agent):
    """Test resource management across components."""
    # Execute resource-intensive task
    result = await magi_agent.execute_task("Heavy computation")
    
    # Check resource usage
    usage = magi_agent.get_resource_usage()
    assert usage.memory > 0
    assert usage.cpu_time > 0
    
    # Verify cleanup
    await magi_agent.cleanup()
    final_usage = magi_agent.get_resource_usage()
    assert final_usage.memory < usage.memory
