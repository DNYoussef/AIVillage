"""Sanity tests to verify basic functionality works."""
import pytest


def test_imports_work():
    """Can we even import our modules?"""
    try:
        import src.core
        import src.production
        assert True
    except ImportError as e:
        pytest.fail(f"Basic imports broken: {e}")


def test_python_environment():
    """Basic Python environment test."""
    assert True, "Python environment is working"


def test_basic_math():
    """Ensure basic operations work."""
    assert 2 + 2 == 4
