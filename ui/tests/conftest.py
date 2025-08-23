"""
Pytest configuration and fixtures for UI testing.
Provides common test fixtures and configuration for all UI components.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ui"))

# Environment setup for testing
os.environ.setdefault("AIVILLAGE_ENV", "test")
os.environ.setdefault("AIVILLAGE_LOG_LEVEL", "WARNING")


@pytest.fixture
def project_paths():
    """Provide project path information for tests."""
    return {
        "project_root": project_root,
        "ui_root": project_root / "ui",
        "web_root": project_root / "ui" / "web", 
        "mobile_root": project_root / "ui" / "mobile",
        "cli_root": project_root / "ui" / "cli"
    }


@pytest.fixture
def temp_directory():
    """Provide temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_web_api():
    """Mock web API service for testing."""
    mock_api = Mock()
    mock_api.healthCheck.return_value = True
    mock_api.setAuthToken.return_value = None
    
    # Mock API responses
    mock_api.get.return_value = {"status": "success", "data": {}}
    mock_api.post.return_value = {"status": "success", "data": {}}
    
    return mock_api


@pytest.fixture
def mock_system_metrics():
    """Mock system metrics for dashboard testing."""
    return {
        "p2pNodes": 25,
        "activeAgents": 12,
        "fogResources": 350,
        "networkHealth": 85,
        "systemUptime": "12h 34m",
        "cpuUsage": 45.2,
        "memoryUsage": 67.8,
        "diskUsage": 23.1
    }


@pytest.fixture
def mock_mobile_device():
    """Mock mobile device profile for mobile testing."""
    return {
        "device_id": "test_device_001",
        "platform": "android",
        "battery_level": 75,
        "is_charging": False,
        "network_type": "wifi",
        "available_memory": 2048,
        "cpu_usage": 30.5,
        "temperature": 35.2
    }


@pytest.fixture
def mock_cli_environment(monkeypatch):
    """Mock CLI environment for testing."""
    # Mock subprocess calls
    mock_subprocess = Mock()
    monkeypatch.setattr("subprocess.run", mock_subprocess)
    
    # Mock successful command execution
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Command executed successfully"
    mock_result.stderr = ""
    mock_subprocess.return_value = mock_result
    
    return mock_subprocess


@pytest.fixture
def sample_ui_config():
    """Sample UI configuration for testing."""
    return {
        "web": {
            "port": 3000,
            "host": "localhost",
            "api_base_url": "http://localhost:8000",
            "enable_admin": True
        },
        "mobile": {
            "enable_digital_twin": True,
            "enable_rag": True,
            "battery_optimization": True,
            "privacy_level": "high"
        },
        "cli": {
            "default_command": "dashboard",
            "log_level": "INFO",
            "auto_refresh": True
        }
    }


@pytest.fixture
def mock_react_component():
    """Mock React component for testing."""
    component_mock = MagicMock()
    component_mock.render.return_value = "<div>Mock Component</div>"
    component_mock.props = {}
    component_mock.state = {}
    
    return component_mock


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up test files after each test."""
    yield
    
    # Clean up any test files that might have been created
    test_patterns = [
        "test_*.tmp",
        "*.test.log",
        "test_output_*"
    ]
    
    for pattern in test_patterns:
        for file_path in project_root.rglob(pattern):
            try:
                file_path.unlink()
            except (OSError, PermissionError):
                pass