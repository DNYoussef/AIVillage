"""Unit tests for Chain of Thought technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.chain_of_thought import (
    ChainOfThoughtTechnique,
    ChainOfThoughtInput,
    ChainOfThoughtStep,
    ChainOfThoughtOutput
)
from ....magi.core.exceptions import ExecutionError

@pytest.fixture
def technique():
    """Create a Chain of Thought technique instance."""
    return ChainOfThoughtTechnique(
        name="test_chain_of_thought",
        description="Test Chain of Thought technique",
        confidence_threshold=0.7
    )

@pytest.fixture
def mock_llm():
    """Create a mock language model."""
    mock = AsyncMock()
    mock.complete = AsyncMock()
    return mock

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "test_chain_of_thought"
    assert technique.description == "Test Chain of Thought technique"
    assert technique.confidence_threshold == 0.7

@pytest.mark.asyncio
async def test_input_validation(technique):
    """Test input validation."""
    # Valid input
    valid_input = ChainOfThoughtInput(
        question="What is 2+2?",
        context="Basic arithmetic",
        max_steps=3,
        step_timeout=10.0,
        temperature=0.7
    )
    assert await technique.validate_input(valid_input) is True

    # Invalid inputs
    invalid_inputs = [
        ChainOfThoughtInput(
            question="",  # Empty question
            max_steps=3,
            step_timeout=10.0,
            temperature=0.7
        ),
        ChainOfThoughtInput(
            question="What is 2+2?",
            max_steps=0,  # Invalid max_steps
            step_timeout=10.0,
            temperature=0.7
        ),
        ChainOfThoughtInput(
            question="What is 2+2?",
            max_steps=3,
            step_timeout=0.0,  # Invalid timeout
            temperature=0.7
        ),
        ChainOfThoughtInput(
            question="What is 2+2?",
            max_steps=3,
            step_timeout=10.0,
            temperature=1.5  # Invalid temperature
        )
    ]
    
    for invalid_input in invalid_inputs:
        assert await technique.validate_input(invalid_input) is False

@pytest.mark.asyncio
async def test_execution_success(technique, mock_llm):
    """Test successful execution."""
    technique.llm = mock_llm
    
    # Mock responses for each step
    mock_llm.complete.side_effect = [
        Mock(text="""
        Thought: Let's break this down
        Reasoning: We need to understand the components
        Intermediate Result: Identified parts
        Confidence: 0.8
        """),
        Mock(text="""
        Thought: Now we can solve it
        Reasoning: Apply the solution
        Intermediate Result: Solution found
        Confidence: 0.9
        """),
        Mock(text="""
        Final Answer: The solution is X
        Reasoning: Based on the previous steps
        Confidence: 0.85
        """)
    ]
    
    input_data = ChainOfThoughtInput(
        question="Test question",
        max_steps=3,
        step_timeout=10.0,
        temperature=0.7
    )
    
    result = await technique.execute(input_data)
    
    assert result.output.final_answer == "The solution is X"
    assert result.output.confidence == 0.85
    assert len(result.output.steps) == 2
    assert result.metrics.success is True

@pytest.mark.asyncio
async def test_execution_timeout(technique, mock_llm):
    """Test execution timeout handling."""
    technique.llm = mock_llm
    
    # Mock timeout
    async def timeout_effect(*args, **kwargs):
        await asyncio.sleep(0.1)
        raise asyncio.TimeoutError()
    
    mock_llm.complete.side_effect = timeout_effect
    
    input_data = ChainOfThoughtInput(
        question="Test question",
        max_steps=3,
        step_timeout=0.05,  # Very short timeout
        temperature=0.7
    )
    
    with pytest.raises(ExecutionError) as exc_info:
        await technique.execute(input_data)
    assert "timed out" in str(exc_info.value)

@pytest.mark.asyncio
async def test_confidence_estimation(technique):
    """Test confidence estimation."""
    output = ChainOfThoughtOutput(
        steps=[
            ChainOfThoughtStep(
                step_number=1,
                thought="First thought",
                reasoning="First reasoning",
                confidence=0.8,
                intermediate_result="First result"
            ),
            ChainOfThoughtStep(
                step_number=2,
                thought="Second thought",
                reasoning="Second reasoning",
                confidence=0.9,
                intermediate_result="Second result"
            )
        ],
        final_answer="Final answer",
        confidence=0.85,
        execution_time=1.0
    )
    
    confidence = await technique.estimate_confidence(output)
    assert 0 <= confidence <= 1
    assert confidence == 0.85

@pytest.mark.asyncio
async def test_output_validation(technique):
    """Test output validation."""
    # Valid output
    valid_output = ChainOfThoughtOutput(
        steps=[
            ChainOfThoughtStep(
                step_number=1,
                thought="Test thought",
                reasoning="Test reasoning",
                confidence=0.8,
                intermediate_result="Test result"
            )
        ],
        final_answer="Test answer",
        confidence=0.85,
        execution_time=1.0
    )
    assert await technique.validate_output(valid_output) is True

    # Invalid outputs
    invalid_outputs = [
        ChainOfThoughtOutput(
            steps=[],  # No steps
            final_answer="Test answer",
            confidence=0.85,
            execution_time=1.0
        ),
        ChainOfThoughtOutput(
            steps=[ChainOfThoughtStep(
                step_number=1,
                thought="Test thought",
                reasoning="Test reasoning",
                confidence=0.8,
                intermediate_result="Test result"
            )],
            final_answer="",  # Empty answer
            confidence=0.85,
            execution_time=1.0
        ),
        ChainOfThoughtOutput(
            steps=[ChainOfThoughtStep(
                step_number=1,
                thought="Test thought",
                reasoning="Test reasoning",
                confidence=0.8,
                intermediate_result="Test result"
            )],
            final_answer="Test answer",
            confidence=1.5,  # Invalid confidence
            execution_time=1.0
        )
    ]
    
    for invalid_output in invalid_outputs:
        assert await technique.validate_output(invalid_output) is False

@pytest.mark.asyncio
async def test_error_handling(technique, mock_llm):
    """Test error handling."""
    technique.llm = mock_llm
    
    # Mock error in language model
    mock_llm.complete.side_effect = Exception("Test error")
    
    input_data = ChainOfThoughtInput(
        question="Test question",
        max_steps=3,
        step_timeout=10.0,
        temperature=0.7
    )
    
    with pytest.raises(ExecutionError) as exc_info:
        await technique.execute(input_data)
    assert "Test error" in str(exc_info.value)
