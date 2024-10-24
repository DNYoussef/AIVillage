"""Unit tests for Self Consistency technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.self_consistency import (
    SelfConsistencyTechnique,
    Solution,
    ConsistencyCheck,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Self Consistency technique instance."""
    return SelfConsistencyTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_solutions():
    """Create sample solutions."""
    return [
        Solution(
            id="sol1",
            content="Use hash table for O(1) lookup",
            reasoning="Constant time access needed",
            confidence=0.9,
            metadata={"approach": "hash-based"}
        ),
        Solution(
            id="sol2",
            content="Use hash table with LRU cache",
            reasoning="Constant time with memory management",
            confidence=0.85,
            metadata={"approach": "hash-based"}
        ),
        Solution(
            id="sol3",
            content="Use binary search tree",
            reasoning="Balanced tree structure",
            confidence=0.8,
            metadata={"approach": "tree-based"}
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Self-Consistency"
    assert "multiple solutions" in technique.thought.lower()
    assert len(technique.solutions) == 0
    assert technique.final_solution is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_solution_generation(technique, mock_agent):
    """Test generation of multiple solutions."""
    mock_agent.llm_response.side_effect = [
        # First solution
        Mock(content="""
        Solution:
        Use hash table
        Reasoning: O(1) lookup
        Confidence: 0.9
        """),
        # Second solution
        Mock(content="""
        Solution:
        Use binary tree
        Reasoning: Balanced structure
        Confidence: 0.8
        """),
        # Third solution
        Mock(content="""
        Solution:
        Use sorted array
        Reasoning: Memory efficient
        Confidence: 0.7
        """)
    ]
    
    result = await technique.apply(mock_agent, "Optimize data structure")
    
    assert len(technique.solutions) >= 3
    assert all(solution.content for solution in technique.solutions)
    assert all(solution.confidence > 0 for solution in technique.solutions)

@pytest.mark.asyncio
async def test_consistency_checking(technique, mock_agent, sample_solutions):
    """Test consistency checking between solutions."""
    technique.solutions = sample_solutions
    
    mock_agent.llm_response.return_value = Mock(content="""
    Consistency Analysis:
    Solutions 1 and 2 are consistent (hash-based)
    Solution 3 differs (tree-based)
    Confidence: 0.9
    """)
    
    consistency_result = await technique._check_consistency(sample_solutions[:2])
    
    assert consistency_result.score > 0
    assert consistency_result.explanation is not None
    assert "hash-based" in consistency_result.explanation.lower()

@pytest.mark.asyncio
async def test_majority_voting(technique, sample_solutions):
    """Test majority voting among solutions."""
    technique.solutions = sample_solutions
    
    # Mock consistency checks
    consistency_checks = [
        ConsistencyCheck(
            solutions=["sol1", "sol2"],
            score=0.9,
            explanation="Hash-based approaches",
            timestamp=datetime.now()
        ),
        ConsistencyCheck(
            solutions=["sol1", "sol3"],
            score=0.3,
            explanation="Different approaches",
            timestamp=datetime.now()
        ),
        ConsistencyCheck(
            solutions=["sol2", "sol3"],
            score=0.3,
            explanation="Different approaches",
            timestamp=datetime.now()
        )
    ]
    technique.consistency_checks = consistency_checks
    
    majority_solution = technique._get_majority_solution()
    
    assert majority_solution is not None
    assert "hash" in majority_solution.content.lower()
    assert majority_solution.confidence > 0.8

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in solution generation."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_confidence_calculation(technique, sample_solutions):
    """Test confidence calculation from multiple solutions."""
    technique.solutions = sample_solutions
    
    # Mock consistency checks with high agreement
    consistency_checks = [
        ConsistencyCheck(
            solutions=["sol1", "sol2"],
            score=0.9,
            explanation="Very consistent",
            timestamp=datetime.now()
        )
    ]
    technique.consistency_checks = consistency_checks
    
    confidence = technique._calculate_overall_confidence()
    
    assert confidence > 0.8  # High confidence due to consistency
    assert confidence <= 1.0

@pytest.mark.asyncio
async def test_solution_diversity(technique, mock_agent):
    """Test diversity in generated solutions."""
    mock_agent.llm_response.side_effect = [
        # Different approaches
        Mock(content="""
        Solution:
        Use hash table
        Reasoning: O(1) lookup
        Confidence: 0.9
        Approach: hash-based
        """),
        Mock(content="""
        Solution:
        Use binary tree
        Reasoning: Balanced structure
        Confidence: 0.8
        Approach: tree-based
        """),
        Mock(content="""
        Solution:
        Use array
        Reasoning: Simple implementation
        Confidence: 0.7
        Approach: array-based
        """)
    ]
    
    result = await technique.apply(mock_agent, "Implement data structure")
    
    # Check that solutions use different approaches
    approaches = set(
        solution.metadata.get('approach', '')
        for solution in technique.solutions
    )
    assert len(approaches) >= 3

@pytest.mark.asyncio
async def test_solution_refinement(technique, mock_agent):
    """Test refinement of similar solutions."""
    mock_agent.llm_response.side_effect = [
        # Initial similar solutions
        Mock(content="""
        Solution:
        Use hash table
        Reasoning: Basic implementation
        Confidence: 0.8
        """),
        Mock(content="""
        Solution:
        Use hash table with LRU
        Reasoning: Enhanced implementation
        Confidence: 0.85
        """),
        # Refinement
        Mock(content="""
        Refined Solution:
        Use hash table with LRU and size limit
        Reasoning: Optimized implementation
        Confidence: 0.9
        """)
    ]
    
    result = await technique.apply(mock_agent, "Optimize lookup")
    
    # Final solution should incorporate refinements
    assert "lru" in result.result.lower()
    assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_consistency_threshold(technique, sample_solutions):
    """Test consistency threshold enforcement."""
    technique.solutions = sample_solutions
    technique.min_consistency_score = 0.8
    
    # Mock low consistency between solutions
    consistency_checks = [
        ConsistencyCheck(
            solutions=["sol1", "sol2"],
            score=0.5,  # Below threshold
            explanation="Some differences",
            timestamp=datetime.now()
        )
    ]
    technique.consistency_checks = consistency_checks
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(Mock(), "Test task")
    assert "consistency" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_solution_validation(technique, mock_agent):
    """Test validation of generated solutions."""
    mock_agent.llm_response.return_value = Mock(content="""
    Solution:
    Invalid solution
    Reasoning:
    Confidence: 1.5  # Invalid confidence > 1.0
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "confidence" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_minimum_solutions(technique, mock_agent):
    """Test enforcement of minimum number of solutions."""
    mock_agent.llm_response.return_value = Mock(content="""
    Solution:
    Single solution
    Reasoning: Only approach
    Confidence: 0.8
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "minimum" in str(exc_info.value).lower()
