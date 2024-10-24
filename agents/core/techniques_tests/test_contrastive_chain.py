"""Unit tests for Contrastive Chain technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.contrastive_chain import (
    ContrastiveChainTechnique,
    ContrastiveStep,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Contrastive Chain technique instance."""
    return ContrastiveChainTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_steps():
    """Create sample contrastive steps."""
    return [
        ContrastiveStep(
            step_number=1,
            correct_reasoning="Use a hash table for O(1) lookup",
            incorrect_reasoning="Use linear search for lookup",
            explanation="Hash table provides constant time access vs linear time",
            confidence=0.9
        ),
        ContrastiveStep(
            step_number=2,
            correct_reasoning="Implement error handling",
            incorrect_reasoning="Assume input is always valid",
            explanation="Robust code needs error handling for edge cases",
            confidence=0.85
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Contrastive Chain"
    assert "correct and incorrect" in technique.thought.lower()
    assert len(technique.steps) == 0
    assert technique.final_answer is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_get_prompt(technique):
    """Test prompt generation."""
    task = "Optimize search algorithm"
    prompt = technique.get_prompt(task)
    
    assert task in prompt
    assert "correct approach" in prompt.lower()
    assert "incorrect approach" in prompt.lower()
    assert "explanation" in prompt.lower()

@pytest.mark.asyncio
async def test_contrastive_reasoning(technique, mock_agent):
    """Test contrastive reasoning generation."""
    mock_agent.llm_response.return_value = Mock(content="""
    Step 1:
    Correct Approach:
    Use proper error handling
    
    Incorrect Approach:
    Ignore error cases
    
    Explanation:
    Error handling ensures robustness
    
    Confidence: 0.9
    """)
    
    result = await technique.apply(mock_agent, "Implement feature")
    
    assert len(technique.steps) > 0
    assert "error handling" in technique.steps[0].correct_reasoning.lower()
    assert "ignore" in technique.steps[0].incorrect_reasoning.lower()
    assert technique.steps[0].confidence == 0.9

@pytest.mark.asyncio
async def test_explanation_quality(technique, mock_agent):
    """Test quality of contrastive explanations."""
    mock_agent.llm_response.side_effect = [
        # Step with detailed explanation
        Mock(content="""
        Step 1:
        Correct Approach:
        Use binary search
        
        Incorrect Approach:
        Use linear search
        
        Explanation:
        Binary search provides O(log n) complexity compared to O(n),
        making it more efficient for large datasets
        
        Confidence: 0.9
        """),
        # Final synthesis
        Mock(content="""
        Answer: Use binary search for efficiency
        Reasoning: Based on complexity analysis
        Confidence: 0.9
        """)
    ]
    
    result = await technique.apply(mock_agent, "Optimize search")
    
    assert "complexity" in technique.steps[0].explanation.lower()
    assert "efficient" in technique.steps[0].explanation.lower()

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in contrastive reasoning."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_confidence_calculation(technique, sample_steps):
    """Test confidence calculation from contrastive steps."""
    technique.steps = sample_steps
    technique.final_answer = "Use optimized approach"
    technique.overall_confidence = 0.9
    
    result = await technique.apply(Mock(), "Test task")
    
    # Overall confidence should consider all step confidences
    assert result.confidence > 0
    assert result.confidence <= 1.0
    # Higher confidence steps should increase overall confidence
    assert result.confidence >= min(step.confidence for step in sample_steps)

@pytest.mark.asyncio
async def test_multi_step_contrast(technique, mock_agent):
    """Test multiple contrastive steps."""
    mock_agent.llm_response.side_effect = [
        # First step
        Mock(content="""
        Step 1:
        Correct Approach:
        Initialize data structure
        
        Incorrect Approach:
        Skip initialization
        
        Explanation:
        Proper initialization prevents errors
        
        Confidence: 0.9
        """),
        # Second step
        Mock(content="""
        Step 2:
        Correct Approach:
        Validate input
        
        Incorrect Approach:
        Trust all input
        
        Explanation:
        Input validation ensures data integrity
        
        Confidence: 0.85
        """),
        # Final synthesis
        Mock(content="""
        Answer: Initialize and validate properly
        Reasoning: Based on best practices
        Confidence: 0.88
        """)
    ]
    
    result = await technique.apply(mock_agent, "Implement feature")
    
    assert len(technique.steps) == 2
    assert all(step.correct_reasoning for step in technique.steps)
    assert all(step.incorrect_reasoning for step in technique.steps)
    assert all(step.explanation for step in technique.steps)

@pytest.mark.asyncio
async def test_explanation_consistency(technique, mock_agent):
    """Test consistency between correct and incorrect explanations."""
    mock_agent.llm_response.side_effect = [
        # Step with inconsistent explanation
        Mock(content="""
        Step 1:
        Correct Approach:
        Use caching
        
        Incorrect Approach:
        Recompute every time
        
        Explanation:
        Caching improves performance by avoiding redundant computation
        
        Confidence: 0.9
        """),
        # Final synthesis
        Mock(content="""
        Answer: Implement caching
        Reasoning: Performance optimization
        Confidence: 0.9
        """)
    ]
    
    result = await technique.apply(mock_agent, "Optimize performance")
    
    step = technique.steps[0]
    # Explanation should reference both approaches
    assert "caching" in step.explanation.lower()
    assert "recompute" in step.explanation.lower() or "redundant" in step.explanation.lower()

@pytest.mark.asyncio
async def test_step_progression(technique, mock_agent):
    """Test logical progression between steps."""
    mock_agent.llm_response.side_effect = [
        # First step - design
        Mock(content="""
        Step 1:
        Correct Approach:
        Design interface first
        
        Incorrect Approach:
        Start coding immediately
        
        Explanation:
        Interface design ensures clear contract
        
        Confidence: 0.9
        """),
        # Second step - implementation
        Mock(content="""
        Step 2:
        Correct Approach:
        Implement based on interface
        
        Incorrect Approach:
        Deviate from interface
        
        Explanation:
        Following interface maintains consistency
        
        Confidence: 0.85
        """),
        # Final synthesis
        Mock(content="""
        Answer: Design then implement
        Reasoning: Structured approach
        Confidence: 0.88
        """)
    ]
    
    result = await technique.apply(mock_agent, "Create feature")
    
    # Verify steps build on each other
    assert "interface" in technique.steps[0].correct_reasoning.lower()
    assert "implement" in technique.steps[1].correct_reasoning.lower()
    assert "interface" in technique.steps[1].correct_reasoning.lower()

@pytest.mark.asyncio
async def test_invalid_contrast(technique, mock_agent):
    """Test handling of invalid contrasts."""
    mock_agent.llm_response.return_value = Mock(content="""
    Step 1:
    Correct Approach:
    Use proper approach
    
    Incorrect Approach:
    Use proper approach  # Same as correct
    
    Explanation:
    No real contrast
    
    Confidence: 0.5
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "contrast" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_confidence_thresholds(technique, mock_agent):
    """Test confidence thresholds for contrasts."""
    mock_agent.llm_response.return_value = Mock(content="""
    Step 1:
    Correct Approach:
    Some approach
    
    Incorrect Approach:
    Another approach
    
    Explanation:
    Weak explanation
    
    Confidence: 0.3  # Too low
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "confidence" in str(exc_info.value).lower()
