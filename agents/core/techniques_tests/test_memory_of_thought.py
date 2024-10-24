"""Unit tests for Memory of Thought technique."""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ....magi.techniques.memory_of_thought import (
    MemoryOfThoughtTechnique,
    MemoryEntry,
    ReasoningStep,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Memory of Thought technique instance."""
    return MemoryOfThoughtTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_memories():
    """Create sample memory entries."""
    return [
        MemoryEntry(
            id="mem1",
            content="Use caching to improve performance",
            embedding=np.random.randn(768),  # Common embedding size
            tags={"performance", "optimization"},
            usage_count=5,
            success_rate=0.9
        ),
        MemoryEntry(
            id="mem2",
            content="Implement error handling for edge cases",
            embedding=np.random.randn(768),
            tags={"error-handling", "robustness"},
            usage_count=3,
            success_rate=0.85
        )
    ]

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Memory-of-Thought"
    assert "memory" in technique.thought.lower()
    assert len(technique.memory_bank) == 0
    assert technique.final_output is None
    assert technique.overall_confidence == 0.0

@pytest.mark.asyncio
async def test_memory_storage(technique, mock_agent):
    """Test storing new memories."""
    mock_agent.llm_response.return_value = Mock(content="""
    Solution: Use caching
    Reasoning: Improves performance
    Confidence: 0.9
    """)
    
    # Mock embedding generation
    with patch.object(technique, '_get_embedding', return_value=np.random.randn(768)):
        result = await technique.apply(mock_agent, "Optimize performance")
        
        assert len(technique.memory_bank) > 0
        memory = next(iter(technique.memory_bank.values()))
        assert "caching" in memory.content.lower()
        assert memory.embedding is not None
        assert memory.success_rate > 0

@pytest.mark.asyncio
async def test_memory_retrieval(technique, mock_agent, sample_memories):
    """Test retrieving relevant memories."""
    technique.memory_bank = {mem.id: mem for mem in sample_memories}
    
    # Mock embedding generation
    with patch.object(technique, '_get_embedding', return_value=sample_memories[0].embedding):
        relevant_memories = technique._retrieve_relevant_memories(sample_memories[0].embedding)
        
        assert len(relevant_memories) > 0
        assert "performance" in relevant_memories[0].content.lower()

@pytest.mark.asyncio
async def test_experience_based_reasoning(technique, mock_agent, sample_memories):
    """Test reasoning based on past experiences."""
    technique.memory_bank = {mem.id: mem for mem in sample_memories}
    
    mock_agent.llm_response.side_effect = [
        # Initial reasoning with memories
        Mock(content="""
        Step 1:
        Reasoning: Apply caching based on past success
        Relevant Memories: 1
        Conclusion: Implement caching
        Confidence: 0.9
        """),
        # Final synthesis
        Mock(content="""
        Solution: Implemented caching mechanism
        Reasoning: Based on successful past experience
        Confidence: 0.9
        """)
    ]
    
    with patch.object(technique, '_get_embedding', return_value=sample_memories[0].embedding):
        result = await technique.apply(mock_agent, "Optimize system")
        
        assert "caching" in result.result.lower()
        assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_memory_update(technique, mock_agent):
    """Test updating memory with new experiences."""
    initial_memory = MemoryEntry(
        id="test",
        content="Initial approach",
        embedding=np.random.randn(768),
        tags={"test"},
        usage_count=1,
        success_rate=0.5
    )
    technique.memory_bank = {"test": initial_memory}
    
    mock_agent.llm_response.return_value = Mock(content="""
    Solution: Improved approach
    Reasoning: Better implementation
    Confidence: 0.9
    """)
    
    with patch.object(technique, '_get_embedding', return_value=initial_memory.embedding):
        await technique.apply(mock_agent, "Test task")
        
        updated_memory = technique.memory_bank["test"]
        assert updated_memory.usage_count > initial_memory.usage_count
        assert updated_memory.success_rate != initial_memory.success_rate

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in memory operations."""
    # Mock error in embedding generation
    with patch.object(technique, '_get_embedding', side_effect=Exception("Embedding error")):
        with pytest.raises(ToolError) as exc_info:
            await technique.apply(mock_agent, "Test task")
        assert "embedding" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_memory_similarity(technique):
    """Test memory similarity calculations."""
    # Create two similar and one different memory
    similar_embedding1 = np.array([1.0, 0.0, 0.0])
    similar_embedding2 = np.array([0.9, 0.1, 0.0])
    different_embedding = np.array([0.0, 0.0, 1.0])
    
    memories = [
        MemoryEntry(
            id="sim1",
            content="Similar content 1",
            embedding=similar_embedding1,
            tags={"test"},
            usage_count=1,
            success_rate=0.8
        ),
        MemoryEntry(
            id="sim2",
            content="Similar content 2",
            embedding=similar_embedding2,
            tags={"test"},
            usage_count=1,
            success_rate=0.8
        ),
        MemoryEntry(
            id="diff",
            content="Different content",
            embedding=different_embedding,
            tags={"test"},
            usage_count=1,
            success_rate=0.8
        )
    ]
    
    # Calculate similarities
    sim1_2 = technique._cosine_similarity(similar_embedding1, similar_embedding2)
    sim1_diff = technique._cosine_similarity(similar_embedding1, different_embedding)
    
    assert sim1_2 > 0.8  # Similar embeddings should have high similarity
    assert sim1_diff < 0.5  # Different embeddings should have low similarity

@pytest.mark.asyncio
async def test_memory_pruning(technique, sample_memories):
    """Test pruning of unused or low-quality memories."""
    # Add some low-quality memories
    low_quality_memory = MemoryEntry(
        id="low_quality",
        content="Poor solution",
        embedding=np.random.randn(768),
        tags={"test"},
        usage_count=1,
        success_rate=0.2
    )
    
    technique.memory_bank = {
        mem.id: mem for mem in sample_memories + [low_quality_memory]
    }
    
    # Prune memories
    pruned_memories = technique._prune_memories()
    
    assert "low_quality" not in pruned_memories
    assert all(mem.success_rate > 0.5 for mem in pruned_memories.values())

@pytest.mark.asyncio
async def test_memory_integration(technique, mock_agent, sample_memories):
    """Test integration of memories into reasoning."""
    technique.memory_bank = {mem.id: mem for mem in sample_memories}
    
    mock_agent.llm_response.side_effect = [
        # First step using memory
        Mock(content="""
        Step 1:
        Reasoning: Apply performance optimization
        Relevant Memories: mem1
        Conclusion: Use caching
        Confidence: 0.9
        """),
        # Second step using different memory
        Mock(content="""
        Step 2:
        Reasoning: Handle edge cases
        Relevant Memories: mem2
        Conclusion: Add error handling
        Confidence: 0.85
        """),
        # Final synthesis
        Mock(content="""
        Solution: Optimized and robust implementation
        Reasoning: Combined performance and reliability insights
        Confidence: 0.9
        """)
    ]
    
    with patch.object(technique, '_get_embedding', return_value=sample_memories[0].embedding):
        result = await technique.apply(mock_agent, "Implement feature")
        
        assert len(technique.steps) == 2
        assert all(step.relevant_memories for step in technique.steps)
        assert "optimized" in result.result.lower()
        assert "robust" in result.result.lower()

@pytest.mark.asyncio
async def test_memory_confidence_weighting(technique, sample_memories):
    """Test confidence weighting based on memory quality."""
    technique.memory_bank = {mem.id: mem for mem in sample_memories}
    
    # Calculate confidence for memories with different success rates
    high_success_memory = sample_memories[0]  # 0.9 success rate
    low_success_memory = MemoryEntry(
        id="low_success",
        content="Low success approach",
        embedding=np.random.randn(768),
        tags={"test"},
        usage_count=1,
        success_rate=0.3
    )
    
    # Memory with higher success rate should contribute more to confidence
    high_confidence = technique._calculate_memory_confidence([high_success_memory])
    low_confidence = technique._calculate_memory_confidence([low_success_memory])
    
    assert high_confidence > low_confidence

@pytest.mark.asyncio
async def test_memory_tags(technique, mock_agent):
    """Test memory tagging and retrieval by tags."""
    mock_agent.llm_response.return_value = Mock(content="""
    Solution: Optimize database queries
    Reasoning: Improve performance
    Tags: database, performance, optimization
    Confidence: 0.9
    """)
    
    with patch.object(technique, '_get_embedding', return_value=np.random.randn(768)):
        await technique.apply(mock_agent, "Optimize database")
        
        # Verify tags were extracted and stored
        memory = next(iter(technique.memory_bank.values()))
        assert "database" in memory.tags
        assert "performance" in memory.tags
        assert "optimization" in memory.tags
