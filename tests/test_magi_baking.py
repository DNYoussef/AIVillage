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
from unittest.mock import Mock, patch, AsyncMock

from config.unified_config import UnifiedConfig, ModelConfig
from agent_forge.agents.magi.magi_agent import MagiAgent
from agent_forge.agents.openrouter_agent import OpenRouterAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

class Timer:
    """Timer class for handling timeouts."""
    def __init__(self, timeout):
        self.timeout = timeout
        self.timer = None
        self.timed_out = False
    
    def start(self):
        """Start the timer."""
        self.timer = threading.Timer(self.timeout, self._timeout)
        self.timer.start()
    
    def cancel(self):
        """Cancel the timer."""
        if self.timer:
            self.timer.cancel()
    
    def _timeout(self):
        """Called when timer expires."""
        self.timed_out = True

class ProgressTracker:
    """Track progress of long-running operations."""
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, step: int = 1):
        """Update progress."""
        self.current_step += step
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        logger.info(f"{self.description}: {self.current_step}/{self.total_steps} - "
                   f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig()

@pytest.fixture
def openrouter_agent():
    """Create mock OpenRouter agent."""
    mock = Mock(spec=OpenRouterAgent)
    # Set required attributes
    mock.model = "test-model"
    mock.generate_response = AsyncMock(return_value={
        "response": "Test response",
        "model": "test-model",
        "metadata": {"quality": 0.9}
    })
    return mock

@pytest.fixture
def magi_agent(config, openrouter_agent):
    """Create Magi agent for testing."""
    agent = MagiAgent(openrouter_agent=openrouter_agent, config=config)
    # Mock internal components
    agent.local_agent = Mock()
    agent.local_agent.generate_response = AsyncMock()
    agent.experiment_manager = Mock()
    agent.experiment_manager.validate_code = AsyncMock()
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
    local_response = {
        "response": """def sum_even_numbers(numbers: list[int]) -> int:
    \"\"\"Sum all even numbers in the list.\"\"\"
    return sum(n for n in numbers if n % 2 == 0)""",
        "model": "local_model",
        "metadata": {"quality": 0.9}
    }
    magi_agent.local_agent.generate_response.return_value = local_response
    
    # Generate code
    result = await magi_agent.generate_code(task)
    
    # Verify local model was used
    assert "local_model" in result.model
    assert "sum_even_numbers" in result.response
    assert "docstring" in result.response.lower()

@pytest.mark.asyncio
async def test_code_generation_complex(magi_agent):
    """Test code generation for complex tasks."""
    task = """
    Create a Python class implementing a thread-safe cache with LRU eviction policy.
    Include proper type hints and comprehensive documentation.
    """
    
    # Mock frontier agent response
    frontier_response = {
        "response": """class LRUCache:
    \"\"\"Thread-safe LRU cache implementation.\"\"\"
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache = {}
        self._lock = threading.Lock()""",
        "model": "frontier_model",
        "metadata": {"quality": 0.95}
    }
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
    
    # Mock validation response
    validation_results = {
        "passes_syntax": True,
        "meets_requirements": {"calculate factorial": True},
        "test_results": {"passed_tests": 1, "total_tests": 1},
        "metrics": {"complexity": 0.3, "maintainability": 0.9}
    }
    magi_agent.experiment_manager.validate_code.return_value = validation_results
    
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
    
    # Mock responses
    local_response = {
        "response": "def sort(lst): return sorted(lst)",
        "model": "local_model",
        "metadata": {}
    }
    frontier_response = {
        "response": "def sort(lst): return sorted(lst)",  # Same response for testing
        "model": "frontier_model",
        "metadata": {}
    }
    
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
    
    # Mock response
    response = {
        "response": code,
        "model": "test_model",
        "metadata": {}
    }
    magi_agent.local_agent.generate_response.return_value = response
    
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
