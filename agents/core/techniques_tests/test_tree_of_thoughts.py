"""Unit tests for Tree of Thoughts technique."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.tree_of_thoughts import (
    TreeOfThoughtsTechnique,
    ThoughtNode,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create a Tree of Thoughts technique instance."""
    return TreeOfThoughtsTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_nodes():
    """Create sample thought nodes."""
    nodes = [
        ThoughtNode(
            id="root",
            content="Initial problem",
            reasoning="Starting point",
            evaluation=0.5,
            parent_id=None,
            children_ids={"child1", "child2"},
            depth=0
        ),
        ThoughtNode(
            id="child1",
            content="First approach",
            reasoning="First reasoning",
            evaluation=0.7,
            parent_id="root",
            children_ids=set(),
            depth=1
        ),
        ThoughtNode(
            id="child2",
            content="Second approach",
            reasoning="Second reasoning",
            evaluation=0.8,
            parent_id="root",
            children_ids=set(),
            depth=1
        )
    ]
    return nodes

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Tree-of-Thoughts"
    assert "tree-like search" in technique.thought.lower()
    assert technique.max_depth > 0
    assert technique.beam_width > 0

@pytest.mark.asyncio
async def test_get_prompt(technique):
    """Test prompt generation."""
    task = "Solve 2+2"
    context = "Basic arithmetic"
    
    # Test initial prompt
    prompt = technique.get_prompt(task)
    assert task in prompt
    assert str(technique.beam_width) in prompt
    
    # Test prompt with context
    prompt_with_context = technique.get_prompt(task, context)
    assert task in prompt_with_context
    assert context in prompt_with_context
    assert str(technique.beam_width) in prompt_with_context

@pytest.mark.asyncio
async def test_apply_success(technique, mock_agent):
    """Test successful application of technique."""
    # Mock responses for tree exploration
    mock_agent.llm_response.side_effect = [
        # Initial approaches
        Mock(content="""
        Approach 1:
        Reasoning: First path
        Next Step: Step 1A
        Confidence: 0.7

        Approach 2:
        Reasoning: Second path
        Next Step: Step 1B
        Confidence: 0.8
        """),
        # Evaluation of paths
        Mock(content="""
        Evaluation:
        Path: Step 1A
        Score: 0.7
        Explanation: Good progress

        Path: Step 1B
        Score: 0.8
        Explanation: Better approach
        """),
        # Final synthesis
        Mock(content="""
        Answer: The solution is X
        Reasoning: Based on explored paths
        Confidence: 0.85
        """)
    ]
    
    result = await technique.apply(mock_agent, "Test task")
    
    assert isinstance(result, TechniqueResult)
    assert result.result is not None
    assert result.confidence > 0
    assert len(technique.thought_tree) > 0

@pytest.mark.asyncio
async def test_node_expansion(technique, mock_agent, sample_nodes):
    """Test node expansion functionality."""
    technique.thought_tree = {node.id: node for node in sample_nodes}
    technique.root_id = "root"
    
    mock_agent.llm_response.return_value = Mock(content="""
    Approach 1:
    Reasoning: New path
    Next Step: New step
    Confidence: 0.75
    """)
    
    # Expand root node
    new_nodes = await technique._expand_nodes(mock_agent, "Test task", ["root"])
    
    assert len(new_nodes) > 0
    for node_id in new_nodes:
        node = technique.thought_tree[node_id]
        assert node.parent_id == "root"
        assert node.depth == 1

@pytest.mark.asyncio
async def test_beam_search(technique, sample_nodes):
    """Test beam search selection."""
    technique.beam_width = 1  # Only select best node
    nodes = [node.id for node in sample_nodes[1:]]  # Exclude root
    
    selected = await technique._select_best_nodes(Mock(), nodes)
    
    assert len(selected) == 1
    assert selected[0] == "child2"  # Should select node with highest evaluation

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_path_finding(technique, sample_nodes):
    """Test best path finding."""
    technique.thought_tree = {node.id: node for node in sample_nodes}
    technique.root_id = "root"
    
    best_path = await technique._find_best_path(Mock())
    
    assert len(best_path) > 0
    assert best_path[0] == "root"
    assert best_path[-1] in ["child1", "child2"]
    # Should select path ending in child2 as it has higher evaluation
    assert best_path[-1] == "child2"

@pytest.mark.asyncio
async def test_tree_visualization(technique, sample_nodes):
    """Test tree structure visualization."""
    technique.thought_tree = {node.id: node for node in sample_nodes}
    
    tree_structure = technique._get_tree_structure()
    
    assert 'nodes' in tree_structure
    assert 'edges' in tree_structure
    assert len(tree_structure['nodes']) == len(sample_nodes)
    # Should have edges from root to both children
    assert len(tree_structure['edges']) == 2

@pytest.mark.asyncio
async def test_depth_limit(technique, mock_agent):
    """Test maximum depth enforcement."""
    original_depth = technique.max_depth
    technique.max_depth = 1  # Set shallow depth limit
    
    mock_agent.llm_response.side_effect = [
        # Initial approaches
        Mock(content="""
        Approach 1:
        Reasoning: First path
        Next Step: Step 1A
        Confidence: 0.7
        """),
        # Evaluation
        Mock(content="""
        Evaluation:
        Path: Step 1A
        Score: 0.7
        Explanation: Good progress
        """),
        # Final synthesis
        Mock(content="""
        Answer: Quick solution
        Reasoning: Limited depth exploration
        Confidence: 0.7
        """)
    ]
    
    result = await technique.apply(mock_agent, "Test task")
    
    # Check that we didn't exceed depth limit
    max_depth = max(node.depth for node in technique.thought_tree.values())
    assert max_depth <= 1
    
    # Restore original depth
    technique.max_depth = original_depth

@pytest.mark.asyncio
async def test_beam_width_limit(technique, mock_agent):
    """Test beam width enforcement."""
    original_width = technique.beam_width
    technique.beam_width = 2  # Set narrow beam width
    
    mock_agent.llm_response.side_effect = [
        # Initial approaches (more than beam width)
        Mock(content="""
        Approach 1:
        Reasoning: First path
        Next Step: Step 1A
        Confidence: 0.7

        Approach 2:
        Reasoning: Second path
        Next Step: Step 1B
        Confidence: 0.8

        Approach 3:
        Reasoning: Third path
        Next Step: Step 1C
        Confidence: 0.6
        """),
        # Evaluation
        Mock(content="""
        Evaluation:
        Path: Step 1B
        Score: 0.8
        Explanation: Best approach

        Path: Step 1A
        Score: 0.7
        Explanation: Good approach
        """),
        # Final synthesis
        Mock(content="""
        Answer: Selected solution
        Reasoning: Based on best paths
        Confidence: 0.8
        """)
    ]
    
    result = await technique.apply(mock_agent, "Test task")
    
    # Check that we didn't exceed beam width at any level
    for node in technique.thought_tree.values():
        if node.parent_id is not None:
            parent = technique.thought_tree[node.parent_id]
            assert len(parent.children_ids) <= technique.beam_width
    
    # Restore original width
    technique.beam_width = original_width
