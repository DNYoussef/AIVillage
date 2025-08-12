"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add project root and src directory to path
project_root = Path(__file__).parent.parent
src_root = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_root))

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

    return torch.nn.Sequential(
        torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 10)
    )


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


# Configure async tests
pytest_plugins = ["pytest_asyncio"]
