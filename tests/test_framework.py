import unittest
import asyncio
import pytest
from typing import Any, Dict, List
import logging
from pathlib import Path
from unittest.mock import Mock, patch
import json
import os

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TestBase(unittest.TestCase):
    """Base class for all test cases."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_data_dir = Path('tests/test_data')
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        cls.setup_test_environment()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.cleanup_test_environment()

    @classmethod
    def setup_test_environment(cls):
        """Set up test environment with necessary configurations."""
        # Load test configuration
        config_path = cls.test_data_dir / 'test_config.json'
        if config_path.exists():
            with open(config_path) as f:
                cls.test_config = json.load(f)
        else:
            cls.test_config = {
                'test_mode': True,
                'mock_external_services': True,
                'test_database_url': 'sqlite:///:memory:',
                'test_cache_dir': str(cls.test_data_dir / 'cache')
            }
            with open(config_path, 'w') as f:
                json.dump(cls.test_config, f, indent=2)

        # Create test directories
        Path(cls.test_config['test_cache_dir']).mkdir(parents=True, exist_ok=True)

    @classmethod
    def cleanup_test_environment(cls):
        """Clean up test artifacts."""
        import shutil
        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)

    def setUp(self):
        """Set up test case."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up after test case."""
        self.loop.close()

    async def async_test(self, coroutine):
        """Helper method to run async tests."""
        return await coroutine

    def run_async(self, coroutine):
        """Run async test in the event loop."""
        return self.loop.run_until_complete(coroutine)

class MockResponse:
    """Mock response for external service calls."""
    def __init__(self, data: Any, status: int = 200):
        self.data = data
        self.status = status

    async def json(self) -> Any:
        return self.data

    async def text(self) -> str:
        return str(self.data)

class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_mock_agent():
        """Create a mock agent for testing."""
        mock_agent = Mock()
        mock_agent.process_message = Mock(return_value=None)
        mock_agent.get_state = Mock(return_value={})
        return mock_agent

    @staticmethod
    def create_mock_database():
        """Create a mock database for testing."""
        mock_db = Mock()
        mock_db.query = Mock(return_value=[])
        mock_db.insert = Mock(return_value=True)
        return mock_db

    @staticmethod
    def create_test_data(data_type: str) -> Dict[str, Any]:
        """Create test data based on type."""
        if data_type == 'task':
            return {
                'id': 'test_task_1',
                'description': 'Test task',
                'priority': 1,
                'status': 'pending'
            }
        elif data_type == 'message':
            return {
                'id': 'test_message_1',
                'content': 'Test message',
                'sender': 'test_agent',
                'receiver': 'test_agent_2'
            }
        return {}

    @staticmethod
    def compare_results(expected: Any, actual: Any) -> bool:
        """Compare test results with expected values."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            return all(
                TestUtils.compare_results(expected[k], actual[k])
                for k in expected
                if k in actual
            )
        elif isinstance(expected, list) and isinstance(actual, list):
            return len(expected) == len(actual) and all(
                TestUtils.compare_results(e, a)
                for e, a in zip(expected, actual)
            )
        return expected == actual

class TestMetrics:
    """Test metrics collection and analysis."""
    
    def __init__(self):
        self.test_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.performance_metrics: Dict[str, List[float]] = {}

    def record_test(self, component: str, success: bool):
        """Record test execution."""
        if component not in self.test_counts:
            self.test_counts[component] = 0
            self.error_counts[component] = 0
        
        self.test_counts[component] += 1
        if not success:
            self.error_counts[component] += 1

    def record_performance(self, component: str, execution_time: float):
        """Record performance metric."""
        if component not in self.performance_metrics:
            self.performance_metrics[component] = []
        self.performance_metrics[component].append(execution_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get test metrics."""
        metrics = {
            'test_coverage': {},
            'error_rates': {},
            'performance': {}
        }

        for component in self.test_counts:
            total = self.test_counts[component]
            errors = self.error_counts[component]
            metrics['test_coverage'][component] = total
            metrics['error_rates'][component] = errors / total if total > 0 else 0

            if component in self.performance_metrics:
                perf_data = self.performance_metrics[component]
                metrics['performance'][component] = {
                    'average': sum(perf_data) / len(perf_data),
                    'min': min(perf_data),
                    'max': max(perf_data)
                }

        return metrics

@pytest.fixture
def test_metrics():
    """Pytest fixture for test metrics."""
    return TestMetrics()

@pytest.fixture
def mock_agent():
    """Pytest fixture for mock agent."""
    return TestUtils.create_mock_agent()

@pytest.fixture
def mock_database():
    """Pytest fixture for mock database."""
    return TestUtils.create_mock_database()

@pytest.fixture
def test_data():
    """Pytest fixture for test data."""
    return {
        'task': TestUtils.create_test_data('task'),
        'message': TestUtils.create_test_data('message')
    }

def async_test(f):
    """Decorator for async test functions."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(f(*args, **kwargs))
    return wrapper
