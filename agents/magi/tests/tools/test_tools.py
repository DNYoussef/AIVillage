"""Unit tests for MAGI tools system."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from ....magi.tools.tool_creator import ToolCreator
from ....magi.tools.tool_management import ToolManager
from ....magi.tools.tool_optimization import ToolOptimizer
from ....magi.tools.reverse_engineer import ReverseEngineer
from ....magi.core.exceptions import ToolError

@pytest.fixture
def tool_creator():
    """Create tool creator instance."""
    return ToolCreator()

@pytest.fixture
def tool_manager():
    """Create tool manager instance."""
    return ToolManager()

@pytest.fixture
def tool_optimizer():
    """Create tool optimizer instance."""
    return ToolOptimizer()

@pytest.fixture
def reverse_engineer():
    """Create reverse engineer instance."""
    return ReverseEngineer()

@pytest.fixture
def mock_llm():
    """Create mock language model."""
    mock = AsyncMock()
    mock.complete = AsyncMock()
    return mock

# Tool Creation Tests
@pytest.mark.asyncio
async def test_tool_creation(tool_creator, mock_llm):
    """Test creation of new tools."""
    mock_llm.complete.return_value = Mock(text="""
    Tool:
    def process_data(data: str) -> str:
        return data.upper()
    
    Parameters:
    - data: Input string to process
    
    Description: Converts input to uppercase
    """)
    
    tool = await tool_creator.create_tool(
        name="string_processor",
        description="Process string data",
        parameters={"data": "str"}
    )
    
    assert tool.name == "string_processor"
    assert "process_data" in tool.code
    assert tool.parameters["data"] == "str"
    assert tool.is_valid()

@pytest.mark.asyncio
async def test_tool_validation(tool_creator):
    """Test tool validation."""
    # Valid tool
    valid_tool = await tool_creator.create_tool(
        name="valid_tool",
        description="Valid tool",
        parameters={"input": "str"},
        code="def process(input: str): return input"
    )
    assert valid_tool.is_valid()
    
    # Invalid tool (missing parameters)
    with pytest.raises(ToolError):
        await tool_creator.create_tool(
            name="invalid_tool",
            description="Invalid tool",
            parameters={},
            code="def process(): pass"
        )

@pytest.mark.asyncio
async def test_tool_generation_with_examples(tool_creator, mock_llm):
    """Test tool generation with example usage."""
    mock_llm.complete.return_value = Mock(text="""
    Tool:
    def calculate_average(numbers: List[float]) -> float:
        return sum(numbers) / len(numbers)
    
    Examples:
    >>> calculate_average([1.0, 2.0, 3.0])
    2.0
    """)
    
    tool = await tool_creator.create_tool(
        name="average_calculator",
        description="Calculate average of numbers",
        parameters={"numbers": "List[float]"},
        examples=[([1.0, 2.0, 3.0], 2.0)]
    )
    
    assert tool.has_examples()
    assert tool.test_examples()

# Tool Management Tests
@pytest.mark.asyncio
async def test_tool_registration(tool_manager):
    """Test tool registration and retrieval."""
    tool = Mock(
        name="test_tool",
        description="Test tool",
        version="1.0.0"
    )
    
    tool_manager.register_tool(tool)
    retrieved_tool = tool_manager.get_tool("test_tool")
    
    assert retrieved_tool == tool
    assert "test_tool" in tool_manager.list_tools()

@pytest.mark.asyncio
async def test_tool_versioning(tool_manager):
    """Test tool version management."""
    tool_v1 = Mock(
        name="versioned_tool",
        description="V1",
        version="1.0.0"
    )
    tool_v2 = Mock(
        name="versioned_tool",
        description="V2",
        version="2.0.0"
    )
    
    tool_manager.register_tool(tool_v1)
    tool_manager.register_tool(tool_v2)
    
    assert tool_manager.get_tool_version("versioned_tool", "1.0.0") == tool_v1
    assert tool_manager.get_tool_version("versioned_tool", "2.0.0") == tool_v2
    assert tool_manager.get_latest_version("versioned_tool") == "2.0.0"

@pytest.mark.asyncio
async def test_tool_dependencies(tool_manager):
    """Test tool dependency management."""
    tool_a = Mock(
        name="tool_a",
        dependencies=[]
    )
    tool_b = Mock(
        name="tool_b",
        dependencies=["tool_a"]
    )
    
    tool_manager.register_tool(tool_a)
    tool_manager.register_tool(tool_b)
    
    assert tool_manager.check_dependencies("tool_b")
    assert tool_manager.get_dependency_graph()["tool_b"] == ["tool_a"]

# Tool Optimization Tests
@pytest.mark.asyncio
async def test_performance_optimization(tool_optimizer, mock_llm):
    """Test tool performance optimization."""
    original_code = """
    def process_list(items):
        result = []
        for item in items:
            result.append(item.upper())
        return result
    """
    
    mock_llm.complete.return_value = Mock(text="""
    def process_list(items):
        return [item.upper() for item in items]
    """)
    
    optimized_code = await tool_optimizer.optimize_performance(original_code)
    
    assert "list comprehension" in tool_optimizer.get_optimization_notes()
    assert "[item.upper() for item in items]" in optimized_code

@pytest.mark.asyncio
async def test_memory_optimization(tool_optimizer, mock_llm):
    """Test tool memory optimization."""
    original_code = """
    def process_large_data(data):
        intermediate = [x for x in data]
        processed = [x * 2 for x in intermediate]
        return processed
    """
    
    mock_llm.complete.return_value = Mock(text="""
    def process_large_data(data):
        return (x * 2 for x in data)
    """)
    
    optimized_code = await tool_optimizer.optimize_memory(original_code)
    
    assert "generator" in tool_optimizer.get_optimization_notes()
    assert "yield" in optimized_code or "generator" in optimized_code

@pytest.mark.asyncio
async def test_optimization_validation(tool_optimizer):
    """Test validation of optimized code."""
    original_code = """
    def calculate_sum(numbers):
        return sum(numbers)
    """
    
    # Test with invalid optimization
    with pytest.raises(ToolError):
        await tool_optimizer.optimize_performance(
            original_code,
            validation_data=[1, 2, 3],
            expected_output=6
        )

# Reverse Engineering Tests
@pytest.mark.asyncio
async def test_code_analysis(reverse_engineer):
    """Test code analysis capabilities."""
    code = """
    class DataProcessor:
        def __init__(self):
            self.cache = {}
        
        def process(self, data):
            if data in self.cache:
                return self.cache[data]
            result = data.upper()
            self.cache[data] = result
            return result
    """
    
    analysis = await reverse_engineer.analyze_code(code)
    
    assert "cache" in analysis.patterns
    assert "DataProcessor" in analysis.classes
    assert len(analysis.methods) > 0

@pytest.mark.asyncio
async def test_pattern_detection(reverse_engineer):
    """Test design pattern detection."""
    code = """
    class Singleton:
        _instance = None
        
        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    """
    
    patterns = await reverse_engineer.detect_patterns(code)
    
    assert "Singleton" in patterns
    assert patterns["Singleton"]["confidence"] > 0.8

@pytest.mark.asyncio
async def test_complexity_analysis(reverse_engineer):
    """Test code complexity analysis."""
    code = """
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    """
    
    complexity = await reverse_engineer.analyze_complexity(code)
    
    assert complexity.time_complexity == "O(n^2)"
    assert complexity.space_complexity == "O(1)"
    assert len(complexity.hotspots) > 0

@pytest.mark.asyncio
async def test_documentation_generation(reverse_engineer, mock_llm):
    """Test documentation generation."""
    code = """
    def process_data(data: str, options: dict = None) -> str:
        if options is None:
            options = {}
        transformed = data.upper()
        if options.get('reverse'):
            transformed = transformed[::-1]
        return transformed
    """
    
    mock_llm.complete.return_value = Mock(text="""
    Function: process_data
    
    Description:
    Processes input string data with optional transformations.
    
    Parameters:
    - data (str): Input string to process
    - options (dict, optional): Processing options
        - reverse (bool): Reverse the output if True
    
    Returns:
    str: Processed string
    """)
    
    docs = await reverse_engineer.generate_documentation(code)
    
    assert "process_data" in docs
    assert "Parameters" in docs
    assert "Returns" in docs
    assert "options" in docs

@pytest.mark.asyncio
async def test_code_reconstruction(reverse_engineer):
    """Test code reconstruction from analysis."""
    analysis = {
        "patterns": ["Singleton"],
        "methods": ["getInstance", "process"],
        "attributes": ["_instance", "data"],
        "relationships": [("getInstance", "_instance")]
    }
    
    code = await reverse_engineer.reconstruct_code(analysis)
    
    assert "class" in code
    assert "_instance" in code
    assert "getInstance" in code
    assert "process" in code

@pytest.mark.asyncio
async def test_security_analysis(reverse_engineer):
    """Test security analysis of code."""
    code = """
    def process_input(user_input):
        query = f"SELECT * FROM users WHERE id = {user_input}"
        return execute_query(query)
    """
    
    vulnerabilities = await reverse_engineer.analyze_security(code)
    
    assert "SQL Injection" in vulnerabilities
    assert vulnerabilities["SQL Injection"]["severity"] == "High"
    assert "mitigation" in vulnerabilities["SQL Injection"]
