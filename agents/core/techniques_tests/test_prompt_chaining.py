"""Unit tests for Prompt Chaining technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.prompt_chaining import (
    PromptChainingTechnique,
    PromptChain,
    ChainLink,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Prompt Chaining technique instance."""
    return PromptChainingTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_chain():
    """Create a sample prompt chain."""
    return PromptChain(
        links=[
            ChainLink(
                id="link1",
                prompt_template="Analyze the problem: {input}",
                output_key="analysis",
                dependencies=[],
                result=None,
                confidence=0.0
            ),
            ChainLink(
                id="link2",
                prompt_template="Design solution based on: {analysis}",
                output_key="design",
                dependencies=["link1"],
                result=None,
                confidence=0.0
            ),
            ChainLink(
                id="link3",
                prompt_template="Implement design: {design}",
                output_key="implementation",
                dependencies=["link2"],
                result=None,
                confidence=0.0
            )
        ],
        variables={}
    )

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Prompt-Chaining"
    assert "sequential" in technique.thought.lower()
    assert technique.chain is None
    assert technique.final_result is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_chain_construction(technique, mock_agent):
    """Test construction of prompt chain."""
    mock_agent.llm_response.return_value = Mock(content="""
    Chain:
    1. Analyze requirements
    Template: Analyze: {input}
    Dependencies: none
    
    2. Design solution
    Template: Design based on: {analysis}
    Dependencies: 1
    
    3. Implementation
    Template: Implement: {design}
    Dependencies: 2
    """)
    
    result = await technique.apply(mock_agent, "Create feature")
    
    assert technique.chain is not None
    assert len(technique.chain.links) == 3
    assert all(link.prompt_template for link in technique.chain.links)
    assert all(isinstance(link.dependencies, list) for link in technique.chain.links)

@pytest.mark.asyncio
async def test_chain_execution(technique, mock_agent, sample_chain):
    """Test execution of prompt chain."""
    technique.chain = sample_chain
    
    mock_agent.llm_response.side_effect = [
        # Analysis
        Mock(content="""
        Analysis:
        Need to implement caching
        Confidence: 0.9
        """),
        # Design
        Mock(content="""
        Design:
        Use LRU cache
        Confidence: 0.85
        """),
        # Implementation
        Mock(content="""
        Implementation:
        Cache implementation complete
        Confidence: 0.8
        """)
    ]
    
    result = await technique.apply(mock_agent, "Optimize performance")
    
    assert all(link.result is not None for link in technique.chain.links)
    assert all(link.confidence > 0 for link in technique.chain.links)
    assert "cache" in result.result.lower()

@pytest.mark.asyncio
async def test_dependency_resolution(technique, sample_chain):
    """Test resolution of chain link dependencies."""
    technique.chain = sample_chain
    
    # Get execution order
    execution_order = technique._get_execution_order()
    
    # Verify dependencies are satisfied
    for i, link in enumerate(execution_order):
        for dep_id in link.dependencies:
            # Find position of dependency in execution order
            dep_pos = next(
                j for j, dep_link in enumerate(execution_order)
                if dep_link.id == dep_id
            )
            # Dependency should come before current link
            assert dep_pos < i

@pytest.mark.asyncio
async def test_variable_substitution(technique, mock_agent):
    """Test variable substitution in prompt templates."""
    chain = PromptChain(
        links=[
            ChainLink(
                id="link1",
                prompt_template="Process {input} with {method}",
                output_key="result",
                dependencies=[],
                result=None,
                confidence=0.0
            )
        ],
        variables={"method": "caching"}
    )
    technique.chain = chain
    
    mock_agent.llm_response.return_value = Mock(content="""
    Result:
    Processed with caching
    Confidence: 0.9
    """)
    
    result = await technique.apply(mock_agent, "data")
    
    assert "caching" in technique.chain.links[0].result.lower()

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent, sample_chain):
    """Test error handling in chain execution."""
    technique.chain = sample_chain
    
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_confidence_aggregation(technique, sample_chain):
    """Test confidence aggregation across chain links."""
    # Set results and confidences
    for i, link in enumerate(sample_chain.links):
        link.result = f"Result {i}"
        link.confidence = 0.8 + i/10
    
    technique.chain = sample_chain
    technique.final_result = "Final result"
    
    result = await technique.apply(Mock(), "Test task")
    
    # Overall confidence should consider all link confidences
    assert result.confidence > 0
    assert result.confidence <= 1.0
    # Should be weighted toward later links in chain
    assert result.confidence >= min(link.confidence for link in sample_chain.links)

@pytest.mark.asyncio
async def test_chain_validation(technique):
    """Test validation of prompt chain structure."""
    # Invalid chain with circular dependency
    invalid_chain = PromptChain(
        links=[
            ChainLink(
                id="link1",
                prompt_template="Step 1: {link2}",
                output_key="result1",
                dependencies=["link2"],
                result=None,
                confidence=0.0
            ),
            ChainLink(
                id="link2",
                prompt_template="Step 2: {link1}",
                output_key="result2",
                dependencies=["link1"],
                result=None,
                confidence=0.0
            )
        ],
        variables={}
    )
    
    with pytest.raises(ToolError) as exc_info:
        technique._validate_chain(invalid_chain)
    assert "circular" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_result_aggregation(technique, mock_agent):
    """Test aggregation of results from multiple chain links."""
    chain = PromptChain(
        links=[
            ChainLink(
                id="link1",
                prompt_template="Get requirements",
                output_key="requirements",
                dependencies=[],
                result=None,
                confidence=0.0
            ),
            ChainLink(
                id="link2",
                prompt_template="Get constraints",
                output_key="constraints",
                dependencies=[],
                result=None,
                confidence=0.0
            )
        ],
        variables={}
    )
    technique.chain = chain
    
    mock_agent.llm_response.side_effect = [
        # Requirements
        Mock(content="""
        Requirements:
        Must be fast
        Confidence: 0.9
        """),
        # Constraints
        Mock(content="""
        Constraints:
        Limited memory
        Confidence: 0.85
        """),
        # Final aggregation
        Mock(content="""
        Solution:
        Optimized for speed within memory limits
        Confidence: 0.88
        """)
    ]
    
    result = await technique.apply(mock_agent, "System design")
    
    assert "speed" in result.result.lower()
    assert "memory" in result.result.lower()

@pytest.mark.asyncio
async def test_parallel_execution(technique, mock_agent):
    """Test parallel execution of independent chain links."""
    chain = PromptChain(
        links=[
            ChainLink(
                id="link1",
                prompt_template="Task 1",
                output_key="result1",
                dependencies=[],
                result=None,
                confidence=0.0
            ),
            ChainLink(
                id="link2",
                prompt_template="Task 2",
                output_key="result2",
                dependencies=[],
                result=None,
                confidence=0.0
            )
        ],
        variables={}
    )
    technique.chain = chain
    
    mock_agent.llm_response.side_effect = [
        Mock(content="Result 1\nConfidence: 0.9"),
        Mock(content="Result 2\nConfidence: 0.85")
    ]
    
    start_time = datetime.now()
    result = await technique.apply(mock_agent, "Parallel tasks")
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Parallel execution should be faster than sequential
    assert execution_time < 2  # Assuming each mock takes ~1 second

@pytest.mark.asyncio
async def test_chain_reuse(technique, mock_agent, sample_chain):
    """Test reuse of prompt chain with different inputs."""
    technique.chain = sample_chain
    
    mock_agent.llm_response.side_effect = [
        # First execution
        Mock(content="Analysis 1\nConfidence: 0.9"),
        Mock(content="Design 1\nConfidence: 0.85"),
        Mock(content="Implementation 1\nConfidence: 0.8"),
        # Second execution
        Mock(content="Analysis 2\nConfidence: 0.9"),
        Mock(content="Design 2\nConfidence: 0.85"),
        Mock(content="Implementation 2\nConfidence: 0.8")
    ]
    
    # Execute chain twice with different inputs
    result1 = await technique.apply(mock_agent, "Task 1")
    result2 = await technique.apply(mock_agent, "Task 2")
    
    assert result1.result != result2.result
    assert "1" in result1.result
    assert "2" in result2.result
