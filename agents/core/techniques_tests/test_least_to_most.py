"""Unit tests for Least to Most technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.least_to_most import (
    LeastToMostTechnique,
    SubProblem,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Least to Most technique instance."""
    return LeastToMostTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_problems():
    """Create sample sub-problems."""
    return [
        SubProblem(
            number=1,
            description="Initialize data structures",
            solution="Created basic structures",
            complexity=0.3,
            dependencies=[],
            confidence=0.9
        ),
        SubProblem(
            number=2,
            description="Process input data",
            solution="Data processed",
            complexity=0.5,
            dependencies=[1],
            confidence=0.85
        ),
        SubProblem(
            number=3,
            description="Generate output",
            solution="Output generated",
            complexity=0.7,
            dependencies=[1, 2],
            confidence=0.8
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Least-to-Most"
    assert "simpler sub-problems" in technique.thought.lower()
    assert len(technique.sub_problems) == 0
    assert technique.final_answer is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_get_decomposition_prompt(technique):
    """Test problem decomposition prompt generation."""
    task = "Complex task"
    prompt = technique.get_decomposition_prompt(task)
    
    assert task in prompt
    assert "sub-problems" in prompt.lower()
    assert "dependencies" in prompt.lower()
    assert "complexity" in prompt.lower()

@pytest.mark.asyncio
async def test_problem_decomposition(technique, mock_agent):
    """Test problem decomposition functionality."""
    mock_agent.llm_response.return_value = Mock(content="""
    Sub-problem 1:
    Description: First step
    Complexity: 0.3
    Dependencies: none

    Sub-problem 2:
    Description: Second step
    Complexity: 0.5
    Dependencies: 1
    """)
    
    result = await technique.apply(mock_agent, "Test task")
    
    assert len(technique.sub_problems) >= 2
    assert technique.sub_problems[0].complexity < technique.sub_problems[1].complexity
    assert 1 in technique.sub_problems[1].dependencies

@pytest.mark.asyncio
async def test_dependency_resolution(technique, sample_problems):
    """Test dependency resolution and ordering."""
    # Shuffle problems to test sorting
    shuffled = list(reversed(sample_problems))
    sorted_problems = technique._topological_sort(shuffled)
    
    # Verify order respects dependencies
    for i, problem in enumerate(sorted_problems):
        for dep in problem.dependencies:
            # Dependencies should appear before the problem
            dep_index = next(
                j for j, p in enumerate(sorted_problems)
                if p.number == dep
            )
            assert dep_index < i

@pytest.mark.asyncio
async def test_solution_synthesis(technique, mock_agent, sample_problems):
    """Test solution synthesis from sub-problems."""
    technique.sub_problems = sample_problems
    
    mock_agent.llm_response.return_value = Mock(content="""
    Solution: Complete solution integrating all steps
    Reasoning: Combined all sub-solutions effectively
    Confidence: 0.85
    """)
    
    result = await technique.apply(mock_agent, "Test task")
    
    assert result.result is not None
    assert "solution" in result.result.lower()
    assert result.confidence > 0

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in problem solving."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_complexity_validation(technique, mock_agent):
    """Test validation of problem complexity values."""
    mock_agent.llm_response.return_value = Mock(content="""
    Sub-problem 1:
    Description: Invalid complexity
    Complexity: 1.5  # Invalid: > 1.0
    Dependencies: none
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "complexity" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_circular_dependency_detection(technique, mock_agent):
    """Test detection of circular dependencies."""
    mock_agent.llm_response.return_value = Mock(content="""
    Sub-problem 1:
    Description: Circular reference
    Complexity: 0.5
    Dependencies: 2

    Sub-problem 2:
    Description: Circular reference
    Complexity: 0.6
    Dependencies: 1
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "circular" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_confidence_aggregation(technique, sample_problems):
    """Test confidence aggregation from sub-problems."""
    technique.sub_problems = sample_problems
    technique.final_answer = "Final solution"
    
    # Calculate expected confidence (weighted by complexity)
    total_weight = sum(p.complexity for p in sample_problems)
    expected_confidence = sum(
        p.confidence * p.complexity / total_weight
        for p in sample_problems
    )
    
    result = await technique.apply(Mock(), "Test task")
    
    assert abs(result.confidence - expected_confidence) < 0.1

@pytest.mark.asyncio
async def test_incremental_solution(technique, mock_agent):
    """Test incremental solution building."""
    mock_agent.llm_response.side_effect = [
        # Problem decomposition
        Mock(content="""
        Sub-problem 1:
        Description: Basic setup
        Complexity: 0.3
        Dependencies: none

        Sub-problem 2:
        Description: Main process
        Complexity: 0.6
        Dependencies: 1
        """),
        # First solution
        Mock(content="""
        Solution: Basic setup complete
        Reasoning: Initialized system
        Confidence: 0.9
        """),
        # Second solution using first
        Mock(content="""
        Solution: Process executed on setup
        Reasoning: Used initial setup
        Confidence: 0.85
        """),
        # Final synthesis
        Mock(content="""
        Solution: Complete solution
        Reasoning: Combined all steps
        Confidence: 0.87
        """)
    ]
    
    result = await technique.apply(mock_agent, "Test task")
    
    assert len(technique.sub_problems) == 2
    assert all(p.solution is not None for p in technique.sub_problems)
    assert result.result is not None

@pytest.mark.asyncio
async def test_solution_validation(technique, mock_agent):
    """Test validation of sub-problem solutions."""
    mock_agent.llm_response.side_effect = [
        # Problem decomposition
        Mock(content="""
        Sub-problem 1:
        Description: Valid step
        Complexity: 0.5
        Dependencies: none
        """),
        # Invalid solution (empty)
        Mock(content="""
        Solution:
        Reasoning: Missing solution
        Confidence: 0.0
        """)
    ]
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "solution" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_complexity_ordering(technique, mock_agent):
    """Test ordering of sub-problems by complexity."""
    mock_agent.llm_response.side_effect = [
        # Problem decomposition with varying complexity
        Mock(content="""
        Sub-problem 1:
        Description: Complex step
        Complexity: 0.8
        Dependencies: none

        Sub-problem 2:
        Description: Simple step
        Complexity: 0.3
        Dependencies: none

        Sub-problem 3:
        Description: Medium step
        Complexity: 0.5
        Dependencies: none
        """),
        # Solutions (not relevant for this test)
        Mock(content="Solution: Test"),
        Mock(content="Solution: Test"),
        Mock(content="Solution: Test"),
        Mock(content="Final: Test")
    ]
    
    await technique.apply(mock_agent, "Test task")
    
    # Verify problems are ordered by complexity
    complexities = [p.complexity for p in technique.sub_problems]
    assert complexities == sorted(complexities)
