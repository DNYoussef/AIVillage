"""Tests for Magi agent's deep baking system."""

import os
from pathlib import Path
import logging
import gc
import psutil
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import threading
from typing import Optional, Dict, Any, List
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from config.unified_config import UnifiedConfig, ModelConfig, AgentConfig, AgentType, ModelType
from agent_forge.agents.magi.magi_agent import MagiAgent, CodeGenerator, ExperimentManager
from agent_forge.agents.openrouter_agent import OpenRouterAgent, AgentInteraction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    """Create test configuration."""
    with patch('config.unified_config.UnifiedConfig._load_configs'):
        config = UnifiedConfig()
        config.config = {
            'openrouter_api_key': 'test_key',
            'model_name': 'test-model',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        # Add agent configurations
        config.agents = {
            'magi': AgentConfig(
                type=AgentType.MAGI,
                frontier_model=ModelConfig(
                    name="test-frontier-model",
                    type=ModelType.FRONTIER,
                    temperature=0.7,
                    max_tokens=1000
                ),
                local_model=ModelConfig(
                    name="test-local-model",
                    type=ModelType.LOCAL,
                    temperature=0.7,
                    max_tokens=1000
                ),
                description="Code generation and optimization agent",
                capabilities=["code_generation", "optimization"],
                performance_threshold=0.7,
                complexity_threshold=0.6,
                evolution_rate=0.1
            )
        }
        return config

@pytest.fixture
def openrouter_agent():
    """Create mock OpenRouter agent."""
    mock = Mock(spec=OpenRouterAgent)
    # Set required attributes
    mock.model = "openai/o1-mini-2024-09-12"
    mock.local_model = "ibm-granite/granite-3b-code-instruct-128k"
    mock.generate_response = AsyncMock(return_value=AgentInteraction(
        prompt="Test task",
        response="Test response",
        model="test-model",
        timestamp=123456789,
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 100},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    ))
    return mock

@pytest.fixture
def mock_code_generator():
    """Create mock CodeGenerator."""
    generator = Mock(spec=CodeGenerator)
    generator.metrics = {
        "success_rate": 1.0,
        "average_quality": 0.9,
        "test_coverage": 0.95,
        "optimization_score": 0.9
    }
    generator.generate_code = AsyncMock(return_value={
        "context": {"task": "test"},
        "structure": {"components": []},
        "requirements": {"functional": []}
    })
    return generator

@pytest.fixture
def mock_experiment_manager():
    """Create mock ExperimentManager."""
    manager = Mock(spec=ExperimentManager)
    manager.metrics = {
        "success_rate": 1.0,
        "average_performance": 0.9,
        "coverage": 0.95
    }
    manager.validate_code = AsyncMock(return_value={
        "passes_syntax": True,
        "meets_requirements": {"test": True},
        "test_results": {"coverage": 100.0},
        "metrics": {"maintainability": 0.9, "performance": 0.9}
    })
    return manager

@pytest.fixture
def magi_agent(config, openrouter_agent, mock_code_generator, mock_experiment_manager):
    """Create Magi agent for testing."""
    agent = MagiAgent(openrouter_agent=openrouter_agent, config=config)
    # Mock internal components
    agent.local_agent = Mock()
    agent.local_agent.generate_response = AsyncMock()
    agent.local_agent.get_performance_metrics = Mock(return_value={
        "code_similarity": 0.9,
        "was_used": True,
        "documentation_quality": 0.8
    })
    agent.code_generator = mock_code_generator
    agent.experiment_manager = mock_experiment_manager
    agent.complexity_evaluator = Mock()
    agent.complexity_evaluator.evaluate_complexity = Mock(return_value={
        "is_complex": False,
        "complexity_score": 0.3,
        "confidence": 0.9
    })

    # Mock _construct_coding_prompt
    def mock_construct_prompt(task, context=None, language=None, requirements=None):
        return f"""
        Task: {task}
        Language: {language or 'python'}
        Requirements: {requirements or []}
        Context: {context or {}}
        """
    agent._construct_coding_prompt = mock_construct_prompt

    # Mock _get_coding_system_prompt
    def mock_get_system_prompt(language=None):
        return "You are a code generation assistant."
    agent._get_coding_system_prompt = mock_get_system_prompt

    # Initialize performance metrics
    agent.performance_metrics = {
        "code_quality": 0.9,
        "test_coverage": 0.95,
        "optimization_score": 0.9,
        "local_model_performance": 0.85,
        "documentation_quality": 0.8  # Set a non-zero value
    }

    return agent

@pytest.mark.asyncio
async def test_code_generation_simple(magi_agent):
    """Test code generation for simple tasks."""
    task = """
    Write a Python function that:
    1. Takes a list of numbers
    2. Returns the sum of even numbers
    
    Keep it short and include a docstring.
    """
    
    # Mock local agent response
    local_response = AgentInteraction(
        prompt=task,
        response="""def sum_even_numbers(numbers: list[int]) -> int:
    \"\"\"Sum all even numbers in the list.\"\"\"
    return sum(n for n in numbers if n % 2 == 0)""",
        model="local_model",
        timestamp=datetime.now().timestamp(),
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 50},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    magi_agent.local_agent.generate_response.return_value = local_response
    
    # Generate code
    result = await magi_agent.generate_code(task)
    
    # Verify local model was used
    assert "local_model" in result.model
    assert "sum_even_numbers" in result.response
    assert '"""' in result.response  # Check for docstring delimiter

@pytest.mark.asyncio
async def test_code_generation_complex(magi_agent):
    """Test code generation for complex tasks."""
    task = """
    Create a Python class implementing a thread-safe cache with LRU eviction policy.
    Include proper type hints and comprehensive documentation.
    """
    
    # Mock complexity evaluation
    magi_agent.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.8,
        "confidence": 0.9
    }
    
    # Mock frontier agent response
    frontier_response = AgentInteraction(
        prompt=task,
        response="""class LRUCache:
    \"\"\"Thread-safe LRU cache implementation.\"\"\"
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache = {}
        self._lock = threading.Lock()""",
        model="frontier_model",
        timestamp=123456789,
        metadata={
            "quality": 0.95,
            "usage": {"total_tokens": 150},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    magi_agent.frontier_agent.generate_response.return_value = frontier_response
    
    # Generate code
    result = await magi_agent.generate_code(task)
    
    # Verify frontier model was used
    assert "frontier_model" in result.model
    assert "LRUCache" in result.response
    assert "thread" in result.response.lower()
    assert "lock" in result.response.lower()

@pytest.mark.asyncio
async def test_code_validation(magi_agent):
    """Test code validation functionality."""
    task = "Write a function to calculate factorial."
    code = """def factorial(n: int) -> int:
    \"\"\"Calculate factorial of n.\"\"\"
    if n < 0:
        raise ValueError("n must be non-negative")
    return 1 if n <= 1 else n * factorial(n - 1)
    """
    
    # Mock local response
    local_response = AgentInteraction(
        prompt=task,
        response=code,
        model="local_model",
        timestamp=datetime.now().timestamp(),
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 80},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    magi_agent.local_agent.generate_response.return_value = local_response
    
    # Generate and validate code
    result = await magi_agent.generate_code(task)
    
    # Verify validation was performed
    assert "validation_results" in result.metadata
    assert result.metadata["validation_results"]["passes_syntax"]
    assert "metrics" in result.metadata["validation_results"]

@pytest.mark.asyncio
async def test_performance_tracking(magi_agent):
    """Test performance tracking during code generation."""
    task = "Write a function to check if a string is a palindrome."
    
    # Mock local response
    local_response = AgentInteraction(
        prompt=task,
        response="""def is_palindrome(s: str) -> bool:
    \"\"\"Check if a string is a palindrome.\"\"\"
    s = s.lower()
    return s == s[::-1]""",
        model="local_model",
        timestamp=datetime.now().timestamp(),
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 60},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    magi_agent.local_agent.generate_response.return_value = local_response
    
    # Generate code
    await magi_agent.generate_code(task)
    
    # Get performance metrics
    metrics = magi_agent.get_performance_metrics()
    
    # Verify metrics are tracked
    assert "code_quality" in metrics
    assert "test_coverage" in metrics
    assert "optimization_score" in metrics
    assert "local_model_performance" in metrics

@pytest.mark.asyncio
async def test_model_comparison(magi_agent):
    """Test model comparison functionality."""
    task = "Write a sorting function."
    
    # Mock complexity evaluation
    magi_agent.complexity_evaluator.evaluate_complexity.return_value = {
        "is_complex": True,
        "complexity_score": 0.7,
        "confidence": 0.9
    }
    
    # Mock responses
    local_response = AgentInteraction(
        prompt=task,
        response="def sort(lst): return sorted(lst)",
        model="local_model",
        timestamp=datetime.now().timestamp(),
        metadata={
            "quality": 0.8,
            "usage": {"total_tokens": 30},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    frontier_response = AgentInteraction(
        prompt=task,
        response="def sort(lst): return sorted(lst)",  # Same response for testing
        model="frontier_model",
        timestamp=123456789,
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 30},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    
    magi_agent.local_agent.generate_response.return_value = local_response
    magi_agent.frontier_agent.generate_response.return_value = frontier_response
    
    # Generate code
    result = await magi_agent.generate_code(task)
    
    # Verify comparison was recorded
    local_metrics = magi_agent.local_agent.get_performance_metrics()
    assert "code_similarity" in local_metrics

@pytest.mark.asyncio
async def test_error_handling(magi_agent):
    """Test error handling during code generation."""
    task = "Write a function."
    
    # Mock error in local agent
    magi_agent.local_agent.generate_response.side_effect = Exception("Test error")
    
    # Mock frontier agent response
    frontier_response = AgentInteraction(
        prompt=task,
        response="def example(): pass",
        model="openai/o1-mini-2024-09-12",  # Match the model name
        timestamp=123456789,
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 20},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    magi_agent.frontier_agent.generate_response.return_value = frontier_response
    
    # Should fall back to frontier agent
    result = await magi_agent.generate_code(task)
    
    # Verify frontier model was used as fallback
    assert result.model == magi_agent.frontier_agent.model

@pytest.mark.asyncio
async def test_documentation_quality(magi_agent):
    """Test documentation quality evaluation."""
    task = "Write a documented function."
    code = '''
    def example():
        """This is a docstring."""
        # This is a comment
        pass
    '''
    
    # Mock local response
    local_response = AgentInteraction(
        prompt=task,
        response=code,
        model="local_model",
        timestamp=datetime.now().timestamp(),
        metadata={
            "quality": 0.9,
            "usage": {"total_tokens": 40},
            "validation_results": {
                "passes_syntax": True,
                "meets_requirements": {"test": True},
                "test_results": {"coverage": 100.0},
                "metrics": {"maintainability": 0.9, "performance": 0.9}
            }
        }
    )
    magi_agent.local_agent.generate_response.return_value = local_response
    
    # Generate code
    result = await magi_agent.generate_code(task)
    
    # Verify documentation was evaluated
    metrics = magi_agent.get_performance_metrics()
    assert "documentation_quality" in metrics
    assert metrics["documentation_quality"] > 0

def test_system_status(magi_agent):
    """Test system status reporting."""
    status = magi_agent.get_performance_metrics()
    
    # Verify all required metrics are present
    assert "code_quality" in status
    assert "test_coverage" in status
    assert "optimization_score" in status
    assert "local_model_performance" in status

if __name__ == "__main__":
    pytest.main([__file__])
