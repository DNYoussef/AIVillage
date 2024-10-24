"""Unit tests for MAGI utility functions."""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from ....magi.utils.helpers import (
    format_error_message,
    validate_input,
    sanitize_filename,
    merge_dicts
)
from ....magi.utils.logging import (
    setup_logger,
    LogLevel,
    format_log_message
)
from ....magi.utils.visualization import (
    generate_graph,
    plot_metrics,
    create_heatmap
)
from ....magi.utils.resource_management import (
    ResourceManager,
    ResourceUsage,
    ResourceType
)

# Helper Tests
def test_format_error_message():
    """Test error message formatting."""
    error = ValueError("Test error")
    context = {"function": "test_func", "input": "test_input"}
    
    message = format_error_message(error, context)
    
    assert "Test error" in message
    assert "test_func" in message
    assert "test_input" in message

def test_validate_input():
    """Test input validation."""
    # Valid input
    assert validate_input("test", str, min_length=1)
    assert validate_input(42, int, min_value=0, max_value=100)
    
    # Invalid input
    with pytest.raises(ValueError):
        validate_input("", str, min_length=1)
    with pytest.raises(ValueError):
        validate_input(101, int, max_value=100)
    with pytest.raises(TypeError):
        validate_input("test", int)

def test_sanitize_filename():
    """Test filename sanitization."""
    # Test invalid characters
    assert sanitize_filename("test/file.txt") == "test_file.txt"
    assert sanitize_filename("test\\file.txt") == "test_file.txt"
    assert sanitize_filename("test:file.txt") == "test_file.txt"
    
    # Test length limit
    long_name = "a" * 300
    assert len(sanitize_filename(long_name)) <= 255

def test_merge_dicts():
    """Test dictionary merging."""
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    
    merged = merge_dicts(dict1, dict2)
    
    assert merged["a"] == 1
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 3
    assert merged["e"] == 4

# Logging Tests
def test_setup_logger():
    """Test logger setup."""
    logger = setup_logger("test_logger")
    
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0

def test_log_levels():
    """Test log level enumeration."""
    assert LogLevel.DEBUG.value == logging.DEBUG
    assert LogLevel.INFO.value == logging.INFO
    assert LogLevel.WARNING.value == logging.WARNING
    assert LogLevel.ERROR.value == logging.ERROR

def test_format_log_message():
    """Test log message formatting."""
    message = format_log_message(
        "Test message",
        level=LogLevel.INFO,
        context={"function": "test_func"}
    )
    
    assert "Test message" in message
    assert "INFO" in message
    assert "test_func" in message

def test_log_file_output():
    """Test logging to file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        logger = setup_logger("test_logger", log_file=log_file)
        
        test_message = "Test log message"
        logger.info(test_message)
        
        with open(log_file) as f:
            log_content = f.read()
            assert test_message in log_content

# Visualization Tests
def test_generate_graph():
    """Test graph generation."""
    nodes = [("A", {}), ("B", {}), ("C", {})]
    edges = [("A", "B"), ("B", "C")]
    
    graph = generate_graph(nodes, edges)
    
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert ("A", "B") in graph.edges
    assert ("B", "C") in graph.edges

def test_plot_metrics():
    """Test metrics plotting."""
    metrics = {
        "accuracy": [0.8, 0.85, 0.9],
        "loss": [0.2, 0.15, 0.1]
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "metrics.png"
        plot_metrics(metrics, output_file)
        assert output_file.exists()

def test_create_heatmap():
    """Test heatmap creation."""
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    labels = ["A", "B", "C"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "heatmap.png"
        create_heatmap(data, labels, labels, output_file)
        assert output_file.exists()

# Resource Management Tests
def test_resource_manager_initialization():
    """Test resource manager initialization."""
    manager = ResourceManager()
    assert manager.resources[ResourceType.MEMORY] == 0
    assert manager.resources[ResourceType.CPU] == 0

def test_resource_allocation():
    """Test resource allocation."""
    manager = ResourceManager()
    
    # Allocate resources
    manager.allocate(ResourceType.MEMORY, 1000)
    manager.allocate(ResourceType.CPU, 50)
    
    assert manager.resources[ResourceType.MEMORY] == 1000
    assert manager.resources[ResourceType.CPU] == 50

def test_resource_limits():
    """Test resource limit enforcement."""
    manager = ResourceManager()
    
    # Set limits
    manager.set_limit(ResourceType.MEMORY, 1000)
    
    # Test within limit
    assert manager.allocate(ResourceType.MEMORY, 500)
    
    # Test exceeding limit
    assert not manager.allocate(ResourceType.MEMORY, 600)

def test_resource_cleanup():
    """Test resource cleanup."""
    manager = ResourceManager()
    
    # Allocate resources
    manager.allocate(ResourceType.MEMORY, 1000)
    manager.allocate(ResourceType.CPU, 50)
    
    # Cleanup
    manager.cleanup()
    
    assert manager.resources[ResourceType.MEMORY] == 0
    assert manager.resources[ResourceType.CPU] == 0

def test_resource_usage_tracking():
    """Test resource usage tracking."""
    manager = ResourceManager()
    
    # Track usage
    with manager.track_usage():
        # Simulate resource usage
        manager.allocate(ResourceType.MEMORY, 1000)
    
    usage = manager.get_usage_stats()
    assert usage.peak_memory >= 1000
    assert usage.current_memory == 0

def test_resource_type_validation():
    """Test resource type validation."""
    manager = ResourceManager()
    
    # Valid resource type
    assert manager.allocate(ResourceType.MEMORY, 1000)
    
    # Invalid resource type
    with pytest.raises(ValueError):
        manager.allocate("invalid_type", 1000)

def test_resource_usage_reporting():
    """Test resource usage reporting."""
    manager = ResourceManager()
    
    # Record some usage
    manager.allocate(ResourceType.MEMORY, 1000)
    manager.allocate(ResourceType.CPU, 50)
    
    # Get usage report
    report = manager.get_usage_report()
    
    assert "memory" in report.lower()
    assert "cpu" in report.lower()
    assert "1000" in report
    assert "50" in report

def test_concurrent_resource_management():
    """Test concurrent resource management."""
    manager = ResourceManager()
    
    # Simulate concurrent access
    def concurrent_allocation():
        return manager.allocate(ResourceType.MEMORY, 500)
    
    # Both allocations should succeed independently
    assert concurrent_allocation()
    assert concurrent_allocation()
    
    assert manager.resources[ResourceType.MEMORY] == 1000
