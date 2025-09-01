"""
Unified pytest configuration and fixtures for AIVillage testing consolidation.
Enhanced with TDD London School patterns and MCP coordination.
"""

import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, Any, Generator
from unittest.mock import MagicMock
import asyncio

import pytest

# Import TDD London School fixtures
from .tdd_london_mocks import MockFactory, TDDLondonFixtures, ContractTestingMixin

# Add project paths for consolidated imports
project_root = Path(__file__).parent.parent.parent
src_root = project_root / "src"
packages_root = project_root / "packages"
tests_root = project_root / "tests"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_root))
sys.path.insert(0, str(packages_root))

# Environment setup for testing
os.environ.setdefault("AIVILLAGE_ENV", "test")
os.environ.setdefault("AIVILLAGE_LOG_LEVEL", "WARNING")
os.environ.setdefault("RAG_LOCAL_MODE", "1")
os.environ.setdefault("PYTHONPATH", f"{src_root}:{packages_root}:{project_root}")

# Configure asyncio for testing
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Import mocks for missing modules
try:
    from tests.mocks import install_mocks
    install_mocks()
except ImportError:
    pass


# ============================================================================
# MCP Coordination Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def mcp_coordination_config():
    """Configuration for MCP server coordination during testing."""
    return {
        'memory_mcp': {
            'enabled': True,
            'store_patterns': True,
            'category': 'testing-patterns'
        },
        'sequential_thinking_mcp': {
            'enabled': True,
            'reasoning_depth': 3
        },
        'github_mcp': {
            'enabled': True,
            'track_test_files': True
        },
        'context7_mcp': {
            'enabled': True,
            'cache_results': True,
            'ttl_seconds': 300
        }
    }


@pytest.fixture
def mcp_memory_store():
    """Mock MCP memory store for testing patterns."""
    class MockMemoryStore:
        def __init__(self):
            self.patterns = {}
            self.test_results = {}
        
        def store_pattern(self, category: str, pattern: Dict[str, Any]):
            if category not in self.patterns:
                self.patterns[category] = []
            self.patterns[category].append(pattern)
        
        def retrieve_patterns(self, category: str) -> list:
            return self.patterns.get(category, [])
        
        def store_test_result(self, test_id: str, result: Dict[str, Any]):
            self.test_results[test_id] = result
        
        def get_test_metrics(self) -> Dict[str, Any]:
            return {
                'total_patterns': sum(len(patterns) for patterns in self.patterns.values()),
                'test_results_count': len(self.test_results)
            }
    
    return MockMemoryStore()


# ============================================================================
# TDD London School Fixtures Integration
# ============================================================================

# Register all TDD London fixtures
mock_factory = TDDLondonFixtures.mock_factory
user_service_collaborators = TDDLondonFixtures.user_service_collaborators
order_processing_collaborators = TDDLondonFixtures.order_processing_collaborators
ai_agent_collaborators = TDDLondonFixtures.ai_agent_collaborators
p2p_network_collaborators = TDDLondonFixtures.p2p_network_collaborators
verify_collaborations_after_test = TDDLondonFixtures.verify_collaborations_after_test


# ============================================================================
# Enhanced Common Fixtures
# ============================================================================

@pytest.fixture
def sample_model():
    """Provide a sample model for testing."""
    try:
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )
    except ImportError:
        return MagicMock()


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def test_config():
    """Standard test configuration with MCP integration."""
    return {
        "storage_backend": "sqlite",
        "db_path": ":memory:",
        "test_mode": True,
        "batch_size": 1,
        "max_workers": 1,
        "mcp_coordination": True,
        "behavior_verification": True,
        "mock_isolation": True
    }


@pytest.fixture
def mock_database_path():
    """Provide mock database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


# ============================================================================
# P2P and Network Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_p2p_transport(mock_factory):
    """Enhanced P2P transport mock with behavior verification."""
    transport_mock = mock_factory.create_mock("p2p_transport")
    
    # Configure standard transport behaviors
    transport_mock._mock.send_message.return_value = {
        "status": "delivered",
        "message_id": "test_msg_123"
    }
    transport_mock._mock.get_status.return_value = {
        "name": "mock_transport",
        "available": True,
        "messages_sent": 0
    }
    
    return transport_mock


@pytest.fixture
def p2p_test_config():
    """Provide test configuration for P2P systems."""
    return {
        "node_id": "test_node_001",
        "device_name": "test_device",
        "max_retries": 3,
        "base_timeout_ms": 100,
        "enable_store_forward": True,
        "failure_threshold": 0.3,
        "bitchat": {
            "enable_ble": True,
            "device_name": "BitChat_Test",
            "service_uuid": "12345678-1234-5678-9012-123456789abc",
        },
        "betanet": {
            "enable_htx": True,
            "server_host": "test.betanet.ai",
            "server_port": 8443,
            "encryption_enabled": True,
        },
    }


@pytest.fixture
def mock_mesh_protocol(mock_factory):
    """Enhanced mesh protocol mock with interaction tracking."""
    mesh_mock = mock_factory.create_async_mock("mesh_protocol")
    
    # Configure standard mesh behaviors
    mesh_mock._mock.start.return_value = True
    mesh_mock._mock.stop.return_value = True
    mesh_mock._mock.send_message.return_value = "message_id_123"
    mesh_mock._mock.get_delivery_status.return_value = "ACKNOWLEDGED"
    mesh_mock._mock.get_metrics.return_value = {
        "messages_sent": 0,
        "peers_connected": 0,
        "transports_active": 0
    }
    
    return mesh_mock


@pytest.fixture
def p2p_test_messages():
    """Provide standard test messages for P2P testing."""
    messages = []
    for i in range(10):
        messages.append({
            "id": f"test_msg_{i:03d}",
            "type": "test_message",
            "payload": {"data": f"test_data_{i}", "sequence": i},
            "priority": "NORMAL" if i % 2 == 0 else "HIGH",
            "require_ack": True,
        })
    return messages


# ============================================================================
# AI/ML Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_dataset():
    """Provide mock dataset for tests."""
    try:
        import torch
        return torch.utils.data.TensorDataset(
            torch.randn(100, 10),  # Features
            torch.randint(0, 2, (100,)),  # Labels
        )
    except ImportError:
        return MagicMock()


@pytest.fixture
def compression_test_model():
    """Provide a model specifically for compression tests."""
    try:
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
    except ImportError:
        return MagicMock()


@pytest.fixture
def ai_agent_system_collaborators(mock_factory):
    """Comprehensive AI agent system collaborators."""
    return mock_factory.create_collaborator_set(
        'model_repository',
        'inference_engine',
        'memory_store',
        'communication_service',
        'metrics_collector',
        'training_coordinator',
        'hyperparameter_optimizer',
        'performance_monitor'
    )


# ============================================================================
# Security and Authentication Fixtures
# ============================================================================

@pytest.fixture
def security_test_collaborators(mock_factory):
    """Security system collaborators for testing."""
    return mock_factory.create_collaborator_set(
        'authentication_service',
        'authorization_service',
        'encryption_service',
        'audit_logger',
        'threat_detector',
        'security_policy_enforcer'
    )


@pytest.fixture
def mock_crypto_service(mock_factory):
    """Mock cryptographic service."""
    crypto_mock = mock_factory.create_mock("crypto_service")
    
    crypto_mock._mock.encrypt.return_value = b"encrypted_data"
    crypto_mock._mock.decrypt.return_value = b"decrypted_data"
    crypto_mock._mock.generate_key.return_value = b"secret_key"
    crypto_mock._mock.verify_signature.return_value = True
    
    return crypto_mock


# ============================================================================
# Performance and Monitoring Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor(mock_factory):
    """Mock performance monitoring system."""
    monitor_mock = mock_factory.create_mock("performance_monitor")
    
    monitor_mock._mock.start_timing.return_value = None
    monitor_mock._mock.end_timing.return_value = 0.123  # 123ms
    monitor_mock._mock.record_metric.return_value = None
    monitor_mock._mock.get_metrics.return_value = {
        'cpu_usage': 45.2,
        'memory_usage': 67.8,
        'network_io': 123456
    }
    
    return monitor_mock


@pytest.fixture
def resource_constraint_config():
    """Configuration for resource-constrained testing."""
    return {
        'max_memory_mb': 512,
        'max_cpu_percent': 50,
        'max_network_bandwidth_kbps': 1024,
        'timeout_seconds': 30
    }


# ============================================================================
# Validation and Environment Fixtures
# ============================================================================

@pytest.fixture
def validation_environment():
    """Environment setup for validation tests."""
    return {
        "AIVILLAGE_ENV": "test",
        "RAG_LOCAL_MODE": "1",
        "PYTHONPATH": f"{src_root}:{packages_root}:{project_root}",
        "MCP_COORDINATION_ENABLED": "1",
        "BEHAVIOR_VERIFICATION_ENABLED": "1"
    }


# ============================================================================
# Cleanup and Session Management
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Enhanced cleanup after each test."""
    yield
    
    # Standard cleanup
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    # Clear asyncio event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Cancel any pending tasks
            tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in tasks:
                task.cancel()
    except Exception:
        pass


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


# ============================================================================
# Test Markers Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers and settings."""
    
    # Core testing markers
    config.addinivalue_line("markers", "unit: Unit tests with mock isolation")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "acceptance: Acceptance tests for user scenarios")
    config.addinivalue_line("markers", "contract: Contract tests for interface verification")
    config.addinivalue_line("markers", "behavior: Behavior-driven development tests")
    
    # TDD London School markers
    config.addinivalue_line("markers", "mockist: Tests using mock objects for isolation")
    config.addinivalue_line("markers", "outside_in: Outside-in TDD implementation tests")
    config.addinivalue_line("markers", "collaboration: Tests verifying object interactions")
    config.addinivalue_line("markers", "state_verification: State-based verification tests")
    config.addinivalue_line("markers", "behavior_verification: Behavior-based verification tests")
    
    # System markers
    config.addinivalue_line("markers", "security: Security and vulnerability tests")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "e2e: End-to-end system tests")
    config.addinivalue_line("markers", "slow: Tests taking more than 5 seconds")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU resources")
    
    # MCP integration markers
    config.addinivalue_line("markers", "mcp_coordination: MCP server coordination tests")
    config.addinivalue_line("markers", "memory_mcp: Memory MCP integration tests")
    config.addinivalue_line("markers", "sequential_thinking: Sequential thinking MCP tests")


# ============================================================================
# Test Collection and Execution Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    
    for item in items:
        # Add markers based on file path
        test_file_path = str(item.fspath)
        
        if 'unit/' in test_file_path:
            item.add_marker(pytest.mark.unit)
        elif 'integration/' in test_file_path:
            item.add_marker(pytest.mark.integration)
        elif 'security/' in test_file_path:
            item.add_marker(pytest.mark.security)
        elif 'performance/' in test_file_path:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that might be slow
        if any(keyword in test_file_path for keyword in ['e2e', 'stress', 'load']):
            item.add_marker(pytest.mark.slow)
        
        # Add network marker for P2P and communication tests
        if any(keyword in test_file_path for keyword in ['p2p', 'network', 'communication']):
            item.add_marker(pytest.mark.network)


@pytest.fixture
def contract_testing_mixin():
    """Provide contract testing utilities."""
    return ContractTestingMixin()