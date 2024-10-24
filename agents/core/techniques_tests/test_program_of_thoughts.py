"""Unit tests for Program of Thoughts technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.program_of_thoughts import (
    ProgramOfThoughtsTechnique,
    ProgramStep,
    Solution
)
from ....magi.core.exceptions import ExecutionError

@pytest.fixture
def technique():
    """Create a Program of Thoughts technique instance."""
    return ProgramOfThoughtsTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_steps():
    """Create sample program steps."""
    return [
        ProgramStep(
            step_number=1,
            description="Initialize variables",
            code="x = 5\ny = 10",
            output="Variables initialized",
            explanation="Setting up initial values",
            confidence=0.9
        ),
        ProgramStep(
            step_number=2,
            description="Perform calculation",
            code="result = x + y",
            output="result = 15",
            explanation="Adding the numbers",
            confidence=0.95
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Program-of-Thoughts"
    assert "code generation" in technique.thought.lower()
    assert len(technique.steps) == 0
    assert technique.final_solution is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_get_prompt(technique):
    """Test prompt generation."""
    task = "Calculate sum of two numbers"
    prompt = technique.get_prompt(task)
    
    # Check prompt content
    assert task in prompt
    assert "Python code" in prompt
    assert "Description:" in prompt
    assert "Code:" in prompt
    assert "Explanation:" in prompt
    assert "Confidence:" in prompt

@pytest.mark.asyncio
async def test_code_generation(technique, mock_agent):
    """Test code generation functionality."""
    mock_agent.llm_response.return_value = Mock(content="""
    Step 1:
    Description: Initialize variables
    Code:
    ```python
    x = 5
    y = 10
    ```
    Explanation: Setting up test variables
    Confidence: 0.9
    """)
    
    result = await technique.apply(mock_agent, "Add two numbers")
    
    assert len(technique.steps) > 0
    assert technique.steps[0].code.strip() == "x = 5\ny = 10"
    assert technique.steps[0].confidence == 0.9

@pytest.mark.asyncio
async def test_code_execution(technique, mock_agent):
    """Test code execution functionality."""
    # Mock successful code execution
    mock_agent.llm_response.side_effect = [
        # Code generation response
        Mock(content="""
        Step 1:
        Description: Add numbers
        Code:
        ```python
        result = 2 + 2
        print(result)
        ```
        Explanation: Basic addition
        Confidence: 0.9
        """),
        # Execution response
        Mock(content="Output: 4"),
        # Final synthesis
        Mock(content="""
        Solution: The sum is 4
        Reasoning: Direct calculation
        Confidence: 0.95
        """)
    ]
    
    result = await technique.apply(mock_agent, "Calculate 2+2")
    
    assert result.result is not None
    assert "4" in result.result
    assert result.confidence > 0

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in code execution."""
    # Mock code execution error
    mock_agent.llm_response.side_effect = [
        # Code generation with syntax error
        Mock(content="""
        Step 1:
        Description: Invalid code
        Code:
        ```python
        print(undefined_variable)
        ```
        Explanation: This will fail
        Confidence: 0.5
        """),
        # Execution error response
        Mock(content="Output: NameError: name 'undefined_variable' is not defined")
    ]
    
    with pytest.raises(ExecutionError) as exc_info:
        await technique.apply(mock_agent, "Run invalid code")
    assert "NameError" in str(exc_info.value)

@pytest.mark.asyncio
async def test_multi_step_execution(technique, mock_agent):
    """Test execution of multiple program steps."""
    mock_agent.llm_response.side_effect = [
        # First step
        Mock(content="""
        Step 1:
        Description: Initialize variable
        Code:
        ```python
        x = 5
        ```
        Explanation: Set initial value
        Confidence: 0.9
        """),
        # First execution
        Mock(content="Output: Variable initialized"),
        # Second step
        Mock(content="""
        Step 2:
        Description: Double the value
        Code:
        ```python
        result = x * 2
        ```
        Explanation: Multiply by 2
        Confidence: 0.95
        """),
        # Second execution
        Mock(content="Output: result = 10"),
        # Final synthesis
        Mock(content="""
        Solution: The result is 10
        Reasoning: Successfully doubled 5
        Confidence: 0.95
        """)
    ]
    
    result = await technique.apply(mock_agent, "Double a number")
    
    assert len(technique.steps) == 2
    assert technique.steps[0].step_number == 1
    assert technique.steps[1].step_number == 2
    assert "10" in result.result

@pytest.mark.asyncio
async def test_confidence_calculation(technique, sample_steps):
    """Test confidence calculation from multiple steps."""
    technique.steps = sample_steps
    technique.final_solution = "Final result"
    technique.overall_confidence = 0.9
    
    result = await technique.apply(Mock(), "Test task")
    
    # Overall confidence should consider all step confidences
    assert result.confidence > 0
    assert result.confidence <= 1.0
    # Higher confidence steps should increase overall confidence
    assert result.confidence >= min(step.confidence for step in sample_steps)

@pytest.mark.asyncio
async def test_result_validation(technique, mock_agent):
    """Test validation of execution results."""
    mock_agent.llm_response.side_effect = [
        # Code generation
        Mock(content="""
        Step 1:
        Description: Calculate result
        Code:
        ```python
        result = 'invalid'  # Should be a number
        ```
        Explanation: This will produce invalid output
        Confidence: 0.5
        """),
        # Execution
        Mock(content="Output: result = 'invalid'")
    ]
    
    with pytest.raises(ExecutionError) as exc_info:
        await technique.apply(mock_agent, "Calculate numeric result")
    assert "validation" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_code_safety(technique, mock_agent):
    """Test safety checks in code execution."""
    mock_agent.llm_response.return_value = Mock(content="""
    Step 1:
    Description: Unsafe operation
    Code:
    ```python
    import os
    os.system('rm -rf /')  # Dangerous command
    ```
    Explanation: This should be blocked
    Confidence: 0.5
    """)
    
    with pytest.raises(ExecutionError) as exc_info:
        await technique.apply(mock_agent, "Run system command")
    assert "security" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_intermediate_results(technique, mock_agent):
    """Test handling of intermediate results between steps."""
    mock_agent.llm_response.side_effect = [
        # First step with intermediate result
        Mock(content="""
        Step 1:
        Description: Generate intermediate value
        Code:
        ```python
        intermediate = 5
        ```
        Explanation: Will be used in next step
        Confidence: 0.9
        """),
        # First execution
        Mock(content="Output: intermediate = 5"),
        # Second step using intermediate result
        Mock(content="""
        Step 2:
        Description: Use intermediate value
        Code:
        ```python
        final = intermediate * 2
        ```
        Explanation: Using previous result
        Confidence: 0.95
        """),
        # Second execution
        Mock(content="Output: final = 10"),
        # Final synthesis
        Mock(content="""
        Solution: Successfully calculated 10
        Reasoning: Used intermediate value correctly
        Confidence: 0.95
        """)
    ]
    
    result = await technique.apply(mock_agent, "Process in steps")
    
    assert len(technique.steps) == 2
    assert technique.steps[0].output is not None
    assert technique.steps[1].output is not None
    assert "10" in result.result
