"""
Test configuration for production components.
"""

from pathlib import Path
import sys

import pytest

# Add production modules to path
production_path = Path(__file__).parent.parent
sys.path.insert(0, str(production_path))

# Test configuration


# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line("markers", "compression: mark test as compression test")
    config.addinivalue_line("markers", "evolution: mark test as evolution test")
    config.addinivalue_line("markers", "rag: mark test as RAG test")
    config.addinivalue_line("markers", "memory: mark test as memory test")
    config.addinivalue_line("markers", "benchmarking: mark test as benchmarking test")
    config.addinivalue_line("markers", "geometry: mark test as geometry test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Fixtures available to all tests
@pytest.fixture
def production_path():
    """Path to production modules."""
    return Path(__file__).parent.parent


@pytest.fixture
def mock_model():
    """Simple mock model for testing."""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
    )


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    import torch

    return {
        "input": torch.randn(32, 10),
        "target": torch.randn(32, 1),
        "documents": ["Sample document 1", "Sample document 2", "Sample document 3"],
        "queries": [
            "What is the main topic?",
            "How does this work?",
            "What are the benefits?",
        ],
    }
