"""Test configuration for evomerge tests with numpy dependency guard."""
import importlib
import pytest


def pytest_configure(config):
    """Configure pytest with numpy dependency marker."""
    config.addinivalue_line(
        "markers", "requires_numpy: mark test as requiring numpy"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests requiring numpy."""
    numpy_available = importlib.util.find_spec("numpy") is not None
    skip_numpy = pytest.mark.skip(reason="numpy not available")
    
    for item in items:
        if "requires_numpy" in item.keywords and not numpy_available:
            item.add_marker(skip_numpy)


@pytest.fixture(scope="session")
def numpy_available():
    """Check if numpy is available."""
    return importlib.util.find_spec("numpy") is not None