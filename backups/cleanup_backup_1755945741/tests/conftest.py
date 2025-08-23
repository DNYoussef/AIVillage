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


# P2P-specific fixtures for consolidated testing
@pytest.fixture
def mock_p2p_transport():
    """Provide a mock P2P transport for testing."""
    
    class MockTransport:
        def __init__(self, name="mock", success_rate=0.95, latency_ms=10.0):
            self.name = name
            self.success_rate = success_rate
            self.latency_ms = latency_ms
            self.messages_sent = []
            self.is_available = True
            
        async def send_message(self, recipient_id, message_data):
            """Mock send message with configurable reliability."""
            import asyncio
            import random
            
            # Simulate network latency
            await asyncio.sleep(self.latency_ms / 1000.0)
            
            # Record message
            self.messages_sent.append({
                "recipient": recipient_id,
                "data": message_data,
                "timestamp": time.time()
            })
            
            # Simulate success/failure based on success_rate
            if random.random() < self.success_rate:
                return {"status": "delivered", "message_id": f"msg_{len(self.messages_sent)}"}
            else:
                raise Exception(f"Transport {self.name} delivery failed")
                
        def get_status(self):
            """Get transport status."""
            return {
                "name": self.name,
                "available": self.is_available,
                "messages_sent": len(self.messages_sent)
            }
    
    import time
    return MockTransport


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
            "service_uuid": "12345678-1234-5678-9012-123456789abc"
        },
        "betanet": {
            "enable_htx": True,
            "server_host": "test.betanet.ai",
            "server_port": 8443,
            "encryption_enabled": True
        }
    }


@pytest.fixture
def mock_mesh_protocol():
    """Provide a mock mesh protocol for testing."""
    
    class MockMeshProtocol:
        def __init__(self, node_id="test_node"):
            self.node_id = node_id
            self.peers = {}
            self.transports = {}
            self.messages = []
            self.is_running = False
            
        async def start(self):
            """Start the mock protocol."""
            self.is_running = True
            return True
            
        async def stop(self):
            """Stop the mock protocol."""
            self.is_running = False
            return True
            
        def add_peer(self, peer_id, transport_info):
            """Add a mock peer."""
            self.peers[peer_id] = transport_info
            
        def register_transport(self, transport_type, transport):
            """Register a mock transport."""
            self.transports[transport_type] = transport
            
        async def send_message(self, recipient_id, message_type, payload, **kwargs):
            """Mock send message."""
            import uuid
            message_id = str(uuid.uuid4())
            
            message = {
                "id": message_id,
                "recipient": recipient_id,
                "type": message_type,
                "payload": payload,
                "timestamp": time.time(),
                **kwargs
            }
            
            self.messages.append(message)
            
            # Simulate delivery through available transport
            if recipient_id in self.peers and self.transports:
                transport = list(self.transports.values())[0]
                try:
                    result = await transport.send_message(recipient_id, message)
                    return message_id
                except Exception:
                    raise Exception(f"Failed to deliver message to {recipient_id}")
            
            return message_id
            
        def get_delivery_status(self, message_id):
            """Get mock delivery status."""
            for msg in self.messages:
                if msg["id"] == message_id:
                    return "ACKNOWLEDGED"  # Mock successful delivery
            return "PENDING"
            
        def get_metrics(self):
            """Get mock metrics."""
            return {
                "messages_sent": len(self.messages),
                "peers_connected": len(self.peers),
                "transports_active": len(self.transports)
            }
    
    import time
    return MockMeshProtocol


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
            "require_ack": True
        })
    return messages


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
