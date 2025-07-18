"""Test configuration for king agent tests with optional dependency guards."""
import importlib
import pytest


def pytest_configure(config):
    """Configure pytest with optional dependency markers."""
    config.addinivalue_line(
        "markers", "requires_numpy: mark test as requiring numpy"
    )
    config.addinivalue_line(
        "markers", "requires_scipy: mark test as requiring scipy"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests requiring optional dependencies."""
    # Check for numpy availability
    numpy_available = importlib.util.find_spec("numpy") is not None
    
    # Check for scipy availability
    scipy_available = importlib.util.find_spec("scipy") is not None
    
    skip_numpy = pytest.mark.skip(reason="numpy not available")
    skip_scipy = pytest.mark.skip(reason="scipy not available")
    
    for item in items:
        if "requires_numpy" in item.keywords and not numpy_available:
            item.add_marker(skip_numpy)
        if "requires_scipy" in item.keywords and not scipy_available:
            item.add_marker(skip_scipy)


@pytest.fixture(scope="session")
def numpy_available():
    """Check if numpy is available."""
    return importlib.util.find_spec("numpy") is not None


@pytest.fixture(scope="session")
def scipy_available():
    """Check if scipy is available."""
    return importlib.util.find_spec("scipy") is not None