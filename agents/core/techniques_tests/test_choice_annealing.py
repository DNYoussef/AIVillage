"""Unit tests for Choice Annealing technique."""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.choice_annealing import (
    ChoiceAnnealingTechnique,
    Choice,
    AnnealingSchedule,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Choice Annealing technique instance."""
    return ChoiceAnnealingTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_choices():
    """Create sample choices."""
    return [
        Choice(
            id="choice1",
            description="Use hash table",
            score=0.8,
            temperature=1.0,
            iteration=0
        ),
        Choice(
            id="choice2",
            description="Use binary tree",
            score=0.7,
            temperature=1.0,
            iteration=0
        ),
        Choice(
            id="choice3",
            description="Use array",
            score=0.6,
            temperature=1.0,
            iteration=0
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Choice-Annealing"
    assert "temperature" in technique.thought.lower()
    assert len(technique.choices) == 0
    assert technique.current_temperature > 0
    assert technique.final_choice is None

@pytest.mark.asyncio
async def test_temperature_schedule(technique):
    """Test temperature scheduling."""
    initial_temp = technique.current_temperature
    
    # Test temperature decrease
    for _ in range(5):
        technique._update_temperature()
    
    assert technique.current_temperature < initial_temp
    assert technique.current_temperature > 0

@pytest.mark.asyncio
async def test_choice_generation(technique, mock_agent):
    """Test generation of initial choices."""
    mock_agent.llm_response.return_value = Mock(content="""
    Choice 1:
    Description: Use hash table
    Score: 0.8
    Reasoning: O(1) lookup time

    Choice 2:
    Description: Use binary tree
    Score: 0.7
    Reasoning: Balanced structure
    """)
    
    result = await technique.apply(mock_agent, "Choose data structure")
    
    assert len(technique.choices) >= 2
    assert any("hash table" in choice.description.lower() for choice in technique.choices)
    assert all(0 <= choice.score <= 1 for choice in technique.choices)

@pytest.mark.asyncio
async def test_choice_selection(technique, sample_choices):
    """Test choice selection based on temperature."""
    technique.choices = sample_choices.copy()
    
    # High temperature should allow more random selection
    technique.current_temperature = 1.0
    high_temp_selections = [
        technique._select_choice()
        for _ in range(100)
    ]
    
    # Low temperature should favor higher scores
    technique.current_temperature = 0.1
    low_temp_selections = [
        technique._select_choice()
        for _ in range(100)
    ]
    
    # High temperature should have more variety
    high_temp_unique = len(set(c.id for c in high_temp_selections))
    low_temp_unique = len(set(c.id for c in low_temp_selections))
    assert high_temp_unique >= low_temp_unique
    
    # Low temperature should favor best choice
    best_choice_id = max(sample_choices, key=lambda c: c.score).id
    low_temp_best_count = sum(1 for c in low_temp_selections if c.id == best_choice_id)
    assert low_temp_best_count > len(low_temp_selections) * 0.7  # Should be selected >70% of time

@pytest.mark.asyncio
async def test_annealing_process(technique, mock_agent):
    """Test complete annealing process."""
    mock_agent.llm_response.side_effect = [
        # Initial choices
        Mock(content="""
        Choice 1:
        Description: First approach
        Score: 0.8
        Reasoning: Initial attempt

        Choice 2:
        Description: Second approach
        Score: 0.7
        Reasoning: Alternative solution
        """),
        # Refinement
        Mock(content="""
        Choice:
        Description: Refined first approach
        Score: 0.85
        Reasoning: Improved version
        """),
        # Final selection
        Mock(content="""
        Final Choice:
        Description: Optimized approach
        Score: 0.9
        Reasoning: Best solution found
        """)
    ]
    
    result = await technique.apply(mock_agent, "Solve problem")
    
    assert result.result is not None
    assert result.confidence > 0.8
    assert technique.current_temperature < technique.initial_temperature

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in choice generation and selection."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_choice_refinement(technique, mock_agent, sample_choices):
    """Test refinement of choices during annealing."""
    technique.choices = sample_choices.copy()
    
    mock_agent.llm_response.side_effect = [
        # Refinement of best choice
        Mock(content="""
        Choice:
        Description: Enhanced hash table
        Score: 0.85
        Reasoning: Added caching
        """),
        # Final selection
        Mock(content="""
        Final Choice:
        Description: Optimized hash table
        Score: 0.9
        Reasoning: Best implementation
        """)
    ]
    
    result = await technique.apply(mock_agent, "Optimize data structure")
    
    assert "hash table" in result.result.lower()
    assert result.confidence > sample_choices[0].score

@pytest.mark.asyncio
async def test_temperature_impact(technique, sample_choices):
    """Test impact of temperature on choice probabilities."""
    technique.choices = sample_choices.copy()
    
    # Calculate selection probabilities at different temperatures
    def get_probabilities(temp: float) -> list[float]:
        technique.current_temperature = temp
        scores = np.array([c.score for c in technique.choices])
        return technique._calculate_probabilities(scores)
    
    high_temp_probs = get_probabilities(1.0)
    low_temp_probs = get_probabilities(0.1)
    
    # High temperature should have more uniform probabilities
    high_temp_std = np.std(high_temp_probs)
    low_temp_std = np.std(low_temp_probs)
    assert high_temp_std < low_temp_std

@pytest.mark.asyncio
async def test_convergence(technique, mock_agent):
    """Test convergence of choices over iterations."""
    initial_choices = [
        Choice(id=f"choice{i}", description=f"Option {i}", 
              score=0.5 + i/10, temperature=1.0, iteration=0)
        for i in range(5)
    ]
    technique.choices = initial_choices.copy()
    
    mock_agent.llm_response.side_effect = [
        # Successive refinements
        Mock(content=f"""
        Choice:
        Description: Refined option {i}
        Score: {0.6 + i/10}
        Reasoning: Improved version
        """) for i in range(3)
    ] + [
        # Final selection
        Mock(content="""
        Final Choice:
        Description: Best option
        Score: 0.9
        Reasoning: Optimal solution
        """)
    ]
    
    result = await technique.apply(mock_agent, "Find solution")
    
    # Should converge to higher scores
    assert result.confidence > max(c.score for c in initial_choices)

@pytest.mark.asyncio
async def test_annealing_schedule(technique):
    """Test different annealing schedules."""
    initial_temp = 1.0
    technique.current_temperature = initial_temp
    
    # Linear schedule
    technique.schedule = AnnealingSchedule.LINEAR
    linear_temps = [
        technique._calculate_next_temperature()
        for _ in range(5)
    ]
    assert all(t1 > t2 for t1, t2 in zip(linear_temps, linear_temps[1:]))
    
    # Reset temperature
    technique.current_temperature = initial_temp
    
    # Exponential schedule
    technique.schedule = AnnealingSchedule.EXPONENTIAL
    exp_temps = [
        technique._calculate_next_temperature()
        for _ in range(5)
    ]
    assert all(t1 > t2 for t1, t2 in zip(exp_temps, exp_temps[1:]))
    
    # Exponential should cool faster than linear
    assert exp_temps[-1] < linear_temps[-1]

@pytest.mark.asyncio
async def test_choice_validation(technique, mock_agent):
    """Test validation of generated choices."""
    mock_agent.llm_response.return_value = Mock(content="""
    Choice 1:
    Description: Valid choice
    Score: 1.5  # Invalid score > 1.0
    Reasoning: Test
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "score" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_minimum_choices(technique, mock_agent):
    """Test enforcement of minimum number of choices."""
    mock_agent.llm_response.return_value = Mock(content="""
    Choice 1:
    Description: Single choice
    Score: 0.8
    Reasoning: Only option
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "minimum" in str(exc_info.value).lower()
