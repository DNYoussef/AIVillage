"""Unit tests for Self Ask technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.self_ask import (
    SelfAskTechnique,
    Question,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Self Ask technique instance."""
    return SelfAskTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_questions():
    """Create sample questions and answers."""
    return [
        Question(
            number=1,
            question="What are the key components?",
            answer="The system has three main components",
            reasoning="Need to understand the structure",
            confidence=0.8,
            leads_to=2
        ),
        Question(
            number=2,
            question="How do they interact?",
            answer="Components communicate through APIs",
            reasoning="Understanding component relationships",
            confidence=0.85,
            leads_to=None
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Self-Ask"
    assert "self-questioning" in technique.thought.lower()
    assert len(technique.questions) == 0
    assert technique.final_answer is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_get_prompt(technique):
    """Test prompt generation."""
    task = "Analyze system architecture"
    context = "Previous analysis results"
    
    # Test initial prompt
    prompt = technique.get_prompt(task)
    assert task in prompt
    assert "Question:" in prompt
    assert "Reasoning:" in prompt
    
    # Test prompt with context
    prompt_with_context = technique.get_prompt(task, context)
    assert task in prompt_with_context
    assert context in prompt_with_context
    assert "Question:" in prompt_with_context

@pytest.mark.asyncio
async def test_question_generation(technique, mock_agent):
    """Test question generation functionality."""
    mock_agent.llm_response.return_value = Mock(content="""
    Question:
    What are the main components?
    Reasoning: Need to understand the system structure
    Confidence: 0.8
    """)
    
    result = await technique.apply(mock_agent, "Analyze system")
    
    assert len(technique.questions) > 0
    assert technique.questions[0].question == "What are the main components?"
    assert technique.questions[0].confidence == 0.8

@pytest.mark.asyncio
async def test_answer_synthesis(technique, mock_agent):
    """Test answer synthesis from questions."""
    mock_agent.llm_response.side_effect = [
        # First question
        Mock(content="""
        Question:
        What is the input?
        Reasoning: Need to understand data flow
        Confidence: 0.8
        """),
        # Answer to first question
        Mock(content="""
        Answer: System takes JSON input
        Reasoning: Based on API documentation
        Confidence: 0.9
        """),
        # Second question
        Mock(content="""
        FINAL ANSWER:
        Answer: The system processes JSON data
        Reasoning: Based on input analysis
        Confidence: 0.85
        """)
    ]
    
    result = await technique.apply(mock_agent, "Analyze data flow")
    
    assert result.result is not None
    assert "JSON" in result.result
    assert result.confidence > 0

@pytest.mark.asyncio
async def test_question_chain(technique, mock_agent):
    """Test chaining of related questions."""
    mock_agent.llm_response.side_effect = [
        # First question
        Mock(content="""
        Question:
        What is step 1?
        Reasoning: Start with basics
        Confidence: 0.8
        """),
        # Answer to first question
        Mock(content="""
        Answer: Step 1 is initialization
        Reasoning: From documentation
        Confidence: 0.9
        """),
        # Follow-up question
        Mock(content="""
        Question:
        What comes after initialization?
        Reasoning: Need next step
        Confidence: 0.85
        """),
        # Answer to follow-up
        Mock(content="""
        Answer: Data processing follows
        Reasoning: Natural sequence
        Confidence: 0.9
        """),
        # Final synthesis
        Mock(content="""
        FINAL ANSWER:
        Answer: Process starts with initialization followed by data processing
        Reasoning: Based on sequential analysis
        Confidence: 0.9
        """)
    ]
    
    result = await technique.apply(mock_agent, "Describe process")
    
    assert len(technique.questions) == 2
    assert technique.questions[0].leads_to == 2
    assert "initialization" in result.result.lower()
    assert "processing" in result.result.lower()

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in question generation."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_confidence_tracking(technique, sample_questions):
    """Test confidence tracking across questions."""
    technique.questions = sample_questions
    technique.final_answer = "Final synthesis"
    technique.overall_confidence = 0.85
    
    result = await technique.apply(Mock(), "Test task")
    
    # Overall confidence should consider all question confidences
    assert result.confidence > 0
    assert result.confidence <= 1.0
    # Higher confidence questions should increase overall confidence
    assert result.confidence >= min(q.confidence for q in sample_questions)

@pytest.mark.asyncio
async def test_context_building(technique, mock_agent):
    """Test context building from previous questions."""
    mock_agent.llm_response.side_effect = [
        # First question about context
        Mock(content="""
        Question:
        What is the context?
        Reasoning: Need background
        Confidence: 0.8
        """),
        # Answer providing context
        Mock(content="""
        Answer: Context is development environment
        Reasoning: System setting
        Confidence: 0.9
        """),
        # Question using previous context
        Mock(content="""
        Question:
        What development tools are needed?
        Reasoning: Based on dev environment
        Confidence: 0.85
        """),
        # Answer using context
        Mock(content="""
        Answer: Need IDE and compiler
        Reasoning: Standard dev tools
        Confidence: 0.9
        """),
        # Final synthesis using all context
        Mock(content="""
        FINAL ANSWER:
        Answer: Development setup requires IDE and compiler in dev environment
        Reasoning: Based on complete context
        Confidence: 0.9
        """)
    ]
    
    result = await technique.apply(mock_agent, "Setup development")
    
    assert len(technique.questions) > 1
    assert "environment" in result.result.lower()
    assert "ide" in result.result.lower()
    assert "compiler" in result.result.lower()

@pytest.mark.asyncio
async def test_max_questions_limit(technique, mock_agent):
    """Test enforcement of maximum questions limit."""
    # Set a low limit for testing
    technique.max_questions = 2
    
    mock_agent.llm_response.side_effect = [
        # First question
        Mock(content="""
        Question:
        First question?
        Reasoning: Start
        Confidence: 0.8
        """),
        # First answer
        Mock(content="""
        Answer: First answer
        Reasoning: Initial step
        Confidence: 0.9
        """),
        # Second question
        Mock(content="""
        Question:
        Second question?
        Reasoning: Continue
        Confidence: 0.85
        """),
        # Second answer
        Mock(content="""
        Answer: Second answer
        Reasoning: Follow-up
        Confidence: 0.9
        """),
        # Synthesis when limit reached
        Mock(content="""
        FINAL ANSWER:
        Answer: Combined insights from two questions
        Reasoning: Reached question limit
        Confidence: 0.85
        """)
    ]
    
    result = await technique.apply(mock_agent, "Test task")
    
    assert len(technique.questions) <= technique.max_questions
    assert result.result is not None

@pytest.mark.asyncio
async def test_question_relevance(technique, mock_agent):
    """Test question relevance to original task."""
    mock_agent.llm_response.side_effect = [
        # Question unrelated to task
        Mock(content="""
        Question:
        Unrelated question?
        Reasoning: Off-topic
        Confidence: 0.4
        """),
        # System should recognize low confidence
        Mock(content="""
        FINAL ANSWER:
        Answer: Need to focus on original task
        Reasoning: Previous question was off-topic
        Confidence: 0.3
        """)
    ]
    
    result = await technique.apply(mock_agent, "Specific task")
    
    assert result.confidence < 0.5  # Low confidence due to irrelevance
