"""
Unified pytest configuration and fixtures for AIVillage testing consolidation.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root, src, and packages to path for consolidated imports
project_root = Path(__file__).parent.parent
src_root = project_root / "src"
packages_root = project_root / "packages"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(packages_root))

# Environment setup for testing
os.environ.setdefault("AIVILLAGE_ENV", "test")
os.environ.setdefault("AIVILLAGE_LOG_LEVEL", "WARNING")
os.environ.setdefault("RAG_LOCAL_MODE", "1")
os.environ.setdefault("PYTHONPATH", f"{src_root}:{packages_root}:{project_root}")

# Import mocks for missing modules
try:
    from tests.mocks import install_mocks

    install_mocks()
except ImportError:
    pass


# Common fixtures
@pytest.fixture
def sample_model():
    """Provide a sample model for testing."""
    import torch

    return torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 10))


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    return {
        "model_path": "/tmp/test_model",
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 10,
    }


@pytest.fixture
def mock_dataset():
    """Provide mock dataset for tests."""
    import torch

    return torch.utils.data.TensorDataset(
        torch.randn(100, 10),  # Features
        torch.randint(0, 2, (100,)),  # Labels
    )


@pytest.fixture
def compression_test_model():
    """Provide a model specifically for compression tests."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    # Cleanup code here
    import gc

    gc.collect()

    # Clear any CUDA cache if available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except BaseException:
        pass


# Configure async tests (comment out if pytest_asyncio not installed)
# pytest_plugins = ["pytest_asyncio"]


# Additional fixtures for consolidated testing
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests"""
    import asyncio

    return asyncio.get_event_loop_policy()


@pytest.fixture
def mock_p2p_network():
    """Mock P2P network for testing"""
    return MagicMock()


@pytest.fixture
def test_config():
    """Standard test configuration"""
    return {"storage_backend": "sqlite", "db_path": ":memory:", "test_mode": True, "batch_size": 1, "max_workers": 1}


@pytest.fixture
def mock_database_path():
    """Provide mock database path for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def validation_environment():
    """Environment setup for validation tests"""
    return {"AIVILLAGE_ENV": "test", "RAG_LOCAL_MODE": "1", "PYTHONPATH": f"{src_root}:{packages_root}:{project_root}"}


# Test markers for categorization
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "validation: Validation tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
