#!/usr/bin/env python3
"""
Integration Test Configuration
Shared fixtures and configuration for integration tests.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import pytest

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "packages"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path."""
    return project_root


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    test_env = {
        "AIVILLAGE_ENV": "testing",
        "DB_PASSWORD": "test_password",
        "REDIS_PASSWORD": "test_redis",
        "JWT_SECRET": "test_jwt_secret_key_minimum_32_characters",
        "PYTHONPATH": f"{project_root}:{project_root}/src:{project_root}/packages",
    }
    
    # Set environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield test_env
    
    # Cleanup not needed as env vars are process-scoped


@pytest.fixture
def mock_p2p_available():
    """Mock P2P availability for testing."""
    return True


@pytest.fixture
def mock_transport_manager():
    """Mock transport manager for testing."""
    class MockTransportManager:
        def __init__(self):
            self.active = True
            self.transports = ["bitchat", "bluetooth", "wifi_direct"]
        
        def get_status(self):
            return {
                "active": self.active,
                "available_transports": self.transports,
                "transport_priority": "offline_first",
            }
        
        @property
        def device_context(self):
            class MockDeviceContext:
                network_type = "wifi"
                has_internet = True
                is_metered_connection = False
                power_save_mode = False
            return MockDeviceContext()
    
    return MockTransportManager()


@pytest.fixture
async def mock_offline_coordinator():
    """Mock offline coordinator for testing."""
    class MockOfflineCoordinator:
        def __init__(self, max_storage_mb=10, daily_data_budget_usd=1.0):
            self.max_storage_mb = max_storage_mb
            self.daily_data_budget_usd = daily_data_budget_usd
            self.active = False
        
        async def start(self):
            self.active = True
            return True
        
        async def stop(self):
            self.active = False
            return True
        
        def is_active(self):
            return self.active
    
    coordinator = MockOfflineCoordinator()
    await coordinator.start()
    yield coordinator
    await coordinator.stop()


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "timeout": 30,
        "max_retries": 3,
        "enable_mocks": True,
        "enable_real_network": False,
        "log_level": "INFO",
    }


@pytest.fixture(autouse=True)
def setup_integration_test_logging():
    """Set up logging for integration tests."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary directory for test files."""
    test_dir = tmp_path / "integration_test"
    test_dir.mkdir()
    return test_dir


# Error handling fixtures
@pytest.fixture
def handle_import_errors():
    """Handle import errors gracefully in tests."""
    def _handle_import_error(module_name: str):
        try:
            return __import__(module_name)
        except ImportError as e:
            pytest.skip(f"Module {module_name} not available: {e}")
    
    return _handle_import_error


@pytest.fixture
def skip_if_no_network():
    """Skip test if network is not available."""
    import socket
    def _check_network():
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    if not _check_network():
        pytest.skip("Network not available")


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
        
        def start(self):
            self.start_time = time.time()
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
            self.start_cpu = process.cpu_percent()
        
        def stop(self):
            if self.start_time is None:
                return {}
            
            end_time = time.time()
            process = psutil.Process()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()
            
            return {
                "duration": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "cpu_usage": end_cpu,
            }
    
    return PerformanceMonitor()