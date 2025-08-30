"""
Consolidated Test Fixtures for AIVillage
========================================

Common fixtures and test utilities to reduce duplication across the codebase.
This centralizes 1,458+ mock instances into reusable, composable fixtures.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn


# ============================================================================
# Core Agent Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_config():
    """Standard agent configuration for testing."""
    return {
        'name': 'TestAgent',
        'description': 'Test agent for unit testing',
        'capabilities': ['test_capability', 'mock_capability'],
        'model': 'gpt-4',
        'instructions': 'Test instructions for agent behavior',
        'device': 'cpu',
        'batch_size': 32,
        'learning_rate': 0.001,
        'max_tokens': 1024,
    }


@pytest.fixture
def mock_base_agent():
    """Mock base agent with standard methods."""
    agent = Mock()
    agent.name = 'TestAgent'
    agent.description = 'Mock agent for testing'
    agent.capabilities = ['test_capability']
    agent.execute_task = AsyncMock(return_value={'result': 'success'})
    agent.generate = AsyncMock(return_value='Generated response')
    agent.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    agent.rerank = AsyncMock(return_value=[{'doc': 'ranked_result'}])
    agent.add_capability = Mock()
    agent.remove_capability = Mock()
    agent.add_tool = Mock()
    agent.remove_tool = Mock()
    agent.get_tool = Mock(return_value=None)
    agent.info = {
        'name': 'TestAgent',
        'description': 'Mock agent for testing',
        'capabilities': ['test_capability']
    }
    return agent


@pytest.fixture
def mock_agent_forge_config():
    """Agent Forge pipeline configuration."""
    return {
        'base_models': ['mock-model'],
        'output_dir': Path('./test_output'),
        'checkpoint_dir': Path('./test_checkpoints'),
        'device': 'cpu',
        'enable_cognate': True,
        'enable_evomerge': False,
        'enable_quietstar': False,
        'enable_initial_compression': False,
        'enable_training': False,
        'enable_tool_baking': False,
        'enable_adas': False,
        'enable_final_compression': False,
        'wandb_project': None,
    }


# ============================================================================
# Security Test Fixtures
# ============================================================================

@pytest.fixture
def mock_security_validator():
    """Mock security validator with threat detection."""
    validator = Mock()
    validator.validate_message = AsyncMock(return_value={'type': 'ping'})
    validator.security_events = []
    validator.get_security_report = Mock(return_value={
        'total_events': 0,
        'threat_type_counts': {},
        'critical_events_24h': 0
    })
    
    # Mock threat patterns
    validator.threat_patterns = {
        'CODE_INJECTION': [r'eval\s*\(', r'exec\s*\('],
        'COMMAND_INJECTION': [r'subprocess\.', r'os\.system'],
        'SCRIPT_INJECTION': [r'<script', r'javascript:'],
        'PATH_TRAVERSAL': [r'\.\./', r'\.\.\\'],
        'SQL_INJECTION': [r"'\s+OR\s+'", r'UNION\s+SELECT'],
    }
    
    return validator


@pytest.fixture
def security_payloads():
    """Common security test payloads."""
    return {
        'eval_injection': [
            '{"type": "ping", "data": {"code": "eval(\'malicious\')"}}',
            '{"eval_payload": "exec(\'danger\')"}',
            '{"injection": "__import__(\'os\').system(\'ls\')"}',
        ],
        'command_injection': [
            '{"cmd": "subprocess.call([\'rm\', \'-rf\', \'/\'])"}',
            '{"data": "os.system(\'cat /etc/passwd\')"}',
            '{"shell": "bash -c \'rm -rf /tmp/*\'"}',
        ],
        'script_injection': [
            '{"html": "<script>alert(\'XSS\')</script>"}',
            '{"js": "javascript:alert(\'Injected\')"}',
            '{"onclick": "onerror=alert(\'XSS\')"}',
        ],
        'path_traversal': [
            '{"file": "../../../etc/passwd"}',
            '{"path": "..\\\\..\\\\windows\\\\system32"}',
            '{"filename": "/etc/shadow"}',
        ],
    }


# ============================================================================
# P2P Communication Fixtures
# ============================================================================

@pytest.fixture
def mock_p2p_transport():
    """Mock P2P transport with configurable reliability."""
    class MockTransport:
        def __init__(self, name="mock", success_rate=0.95, latency_ms=10.0):
            self.name = name
            self.success_rate = success_rate
            self.latency_ms = latency_ms
            self.messages_sent = []
            self.is_available = True

        async def send_message(self, recipient_id, message_data):
            import random
            await asyncio.sleep(self.latency_ms / 1000.0)
            
            self.messages_sent.append({
                'recipient': recipient_id,
                'data': message_data,
                'timestamp': time.time()
            })
            
            if random.random() < self.success_rate:
                return {'status': 'delivered', 'message_id': f'msg_{len(self.messages_sent)}'}
            else:
                raise Exception(f"Transport {self.name} delivery failed")

        def get_status(self):
            return {
                'name': self.name,
                'available': self.is_available,
                'messages_sent': len(self.messages_sent)
            }

    return MockTransport


@pytest.fixture
def mock_mesh_protocol():
    """Mock mesh protocol for P2P testing."""
    class MockMeshProtocol:
        def __init__(self, node_id="test_node"):
            self.node_id = node_id
            self.peers = {}
            self.transports = {}
            self.messages = []
            self.is_running = False

        async def start(self):
            self.is_running = True
            return True

        async def stop(self):
            self.is_running = False
            return True

        def add_peer(self, peer_id, transport_info):
            self.peers[peer_id] = transport_info

        def register_transport(self, transport_type, transport):
            self.transports[transport_type] = transport

        async def send_message(self, recipient_id, message_type, payload, **kwargs):
            import uuid
            message_id = str(uuid.uuid4())
            
            message = {
                'id': message_id,
                'recipient': recipient_id,
                'type': message_type,
                'payload': payload,
                'timestamp': time.time(),
                **kwargs
            }
            
            self.messages.append(message)
            
            if recipient_id in self.peers and self.transports:
                transport = list(self.transports.values())[0]
                await transport.send_message(recipient_id, message)
                return message_id
            
            return message_id

        def get_metrics(self):
            return {
                'messages_sent': len(self.messages),
                'peers_connected': len(self.peers),
                'transports_active': len(self.transports)
            }

    return MockMeshProtocol


@pytest.fixture
def p2p_test_config():
    """Standard P2P test configuration."""
    return {
        'node_id': 'test_node_001',
        'device_name': 'test_device',
        'max_retries': 3,
        'base_timeout_ms': 100,
        'enable_store_forward': True,
        'failure_threshold': 0.3,
        'bitchat': {
            'enable_ble': True,
            'device_name': 'BitChat_Test',
            'service_uuid': '12345678-1234-5678-9012-123456789abc',
        },
        'betanet': {
            'enable_htx': True,
            'server_host': 'test.betanet.ai',
            'server_port': 8443,
            'encryption_enabled': True,
        },
    }


# ============================================================================
# Model and ML Fixtures
# ============================================================================

@pytest.fixture
def sample_torch_model():
    """Standard PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )


@pytest.fixture
def compression_test_model():
    """Model specifically for compression testing."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    import torch
    return torch.utils.data.TensorDataset(
        torch.randn(100, 10),  # Features
        torch.randint(0, 2, (100,)),  # Labels
    )


@pytest.fixture
def mock_embeddings():
    """Mock embedding vectors for testing."""
    return {
        'dimension': 768,
        'sample_embeddings': [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ],
        'query_embedding': [0.15] * 768,
        'similarity_scores': [0.95, 0.87, 0.73],
    }


# ============================================================================
# Database and Storage Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock vector store for RAG testing."""
    store = Mock()
    store.add = AsyncMock(return_value='doc_id_123')
    store.query = AsyncMock(return_value=[
        {'id': 'doc1', 'content': 'Test document 1', 'score': 0.95},
        {'id': 'doc2', 'content': 'Test document 2', 'score': 0.87},
    ])
    store.get = AsyncMock(return_value={'id': 'doc1', 'content': 'Test document'})
    store.delete = AsyncMock(return_value=True)
    store.update = AsyncMock(return_value=True)
    store.count = Mock(return_value=100)
    return store


@pytest.fixture
def temp_database():
    """Temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        Path(db_path).unlink()
    except FileNotFoundError:
        pass


# ============================================================================
# API and Communication Fixtures
# ============================================================================

@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    client = Mock()
    client.get = AsyncMock(return_value={'status': 'success', 'data': {}})
    client.post = AsyncMock(return_value={'status': 'success', 'data': {}})
    client.put = AsyncMock(return_value={'status': 'success', 'data': {}})
    client.delete = AsyncMock(return_value={'status': 'success'})
    client.health_check = AsyncMock(return_value=True)
    client.set_auth_token = Mock()
    return client


@pytest.fixture
def mock_communication_protocol():
    """Mock communication protocol for agent testing."""
    protocol = Mock()
    protocol.subscribe = AsyncMock()
    protocol.send_message = AsyncMock(return_value='message_sent')
    protocol.receive_message = AsyncMock(return_value={'type': 'test', 'data': 'test_data'})
    protocol.broadcast = AsyncMock()
    protocol.get_peers = Mock(return_value=['peer1', 'peer2'])
    return protocol


# ============================================================================
# Performance and Metrics Fixtures
# ============================================================================

@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics for testing."""
    return {
        'cpu_usage': 45.2,
        'memory_usage': 67.8,
        'disk_usage': 23.1,
        'network_latency': 12.5,
        'throughput': 1024.0,
        'error_rate': 0.01,
        'uptime': '12h 34m',
        'active_connections': 25,
    }


@pytest.fixture
def mock_system_metrics():
    """Mock system metrics for dashboard testing."""
    return {
        'p2p_nodes': 25,
        'active_agents': 12,
        'fog_resources': 350,
        'network_health': 85,
        'system_uptime': '12h 34m',
        'cpu_usage': 45.2,
        'memory_usage': 67.8,
        'disk_usage': 23.1,
    }


# ============================================================================
# Error Handling and Exceptions
# ============================================================================

@pytest.fixture
def mock_error_handler():
    """Mock error handler for testing exception scenarios."""
    handler = Mock()
    handler.handle_error = Mock()
    handler.log_error = Mock()
    handler.get_error_stats = Mock(return_value={
        'total_errors': 0,
        'error_types': {},
        'recent_errors': []
    })
    return handler


@pytest.fixture
def exception_scenarios():
    """Common exception scenarios for testing."""
    return {
        'network_error': ConnectionError("Network connection failed"),
        'timeout_error': TimeoutError("Operation timed out"),
        'auth_error': PermissionError("Authentication failed"),
        'data_error': ValueError("Invalid data format"),
        'resource_error': ResourceWarning("Resource limit exceeded"),
    }


# ============================================================================
# Test Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv('AIVILLAGE_ENV', 'test')
    monkeypatch.setenv('AIVILLAGE_LOG_LEVEL', 'WARNING')
    monkeypatch.setenv('RAG_LOCAL_MODE', '1')
    monkeypatch.setenv('DISABLE_WANDB', '1')
    monkeypatch.setenv('PYTORCH_DISABLE_CUDNN', '1')


@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace directory for tests."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir(exist_ok=True)
    
    # Create standard subdirectories
    (workspace / "models").mkdir()
    (workspace / "checkpoints").mkdir()
    (workspace / "logs").mkdir()
    (workspace / "data").mkdir()
    
    return workspace


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts."""
    yield
    
    # Cleanup CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    # Force garbage collection
    import gc
    gc.collect()


# ============================================================================
# Test Data Builders
# ============================================================================

class TestDataBuilder:
    """Builder for creating complex test data structures."""
    
    @staticmethod
    def create_agent_task(task_type='test', complexity='simple', **kwargs):
        """Create agent task test data."""
        base_task = {
            'id': f'task_{int(time.time())}',
            'type': task_type,
            'complexity': complexity,
            'priority': 'medium',
            'timestamp': time.time(),
            'data': {'test': True}
        }
        base_task.update(kwargs)
        return base_task
    
    @staticmethod
    def create_message_batch(count=10, message_type='test'):
        """Create batch of test messages."""
        return [
            {
                'id': f'{message_type}_msg_{i:03d}',
                'type': message_type,
                'payload': {'data': f'{message_type}_data_{i}', 'sequence': i},
                'priority': 'NORMAL' if i % 2 == 0 else 'HIGH',
                'require_ack': True,
                'timestamp': time.time() + i
            }
            for i in range(count)
        ]
    
    @staticmethod
    def create_performance_data(duration_hours=1):
        """Create performance test data."""
        import random
        
        data_points = []
        base_time = time.time()
        
        for i in range(duration_hours * 60):  # Per minute
            data_points.append({
                'timestamp': base_time + (i * 60),
                'cpu_usage': random.uniform(10, 90),
                'memory_usage': random.uniform(20, 80),
                'network_io': random.uniform(0, 1000),
                'active_connections': random.randint(1, 50),
            })
        
        return data_points


@pytest.fixture
def test_data_builder():
    """Test data builder fixture."""
    return TestDataBuilder()


# ============================================================================
# Integration Test Helpers
# ============================================================================

@pytest.fixture
def integration_test_suite():
    """Helper for running integration test suites."""
    class IntegrationTestSuite:
        def __init__(self):
            self.results = {}
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def end(self):
            self.end_time = time.time()
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
        
        def add_result(self, test_name, success, details=None):
            self.results[test_name] = {
                'success': success,
                'details': details or {},
                'timestamp': time.time()
            }
        
        def get_summary(self):
            total_tests = len(self.results)
            successful_tests = sum(1 for r in self.results.values() if r['success'])
            
            return {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'duration': self.duration,
                'results': self.results
            }
    
    return IntegrationTestSuite()


# ============================================================================
# Async Test Utilities
# ============================================================================

@pytest.fixture
def async_test_utils():
    """Utilities for async testing."""
    class AsyncTestUtils:
        @staticmethod
        async def wait_for_condition(condition_func, timeout=5.0, check_interval=0.1):
            """Wait for a condition to become true."""
            end_time = time.time() + timeout
            
            while time.time() < end_time:
                if await condition_func():
                    return True
                await asyncio.sleep(check_interval)
            
            return False
        
        @staticmethod
        async def collect_async_results(async_generators, timeout=10.0):
            """Collect results from multiple async generators."""
            results = []
            
            try:
                async with asyncio.timeout(timeout):
                    for gen in async_generators:
                        async for item in gen:
                            results.append(item)
            except asyncio.TimeoutError:
                pass
            
            return results
        
        @staticmethod
        def create_mock_async_context():
            """Create mock async context manager."""
            context = AsyncMock()
            context.__aenter__ = AsyncMock(return_value=context)
            context.__aexit__ = AsyncMock(return_value=None)
            return context
    
    return AsyncTestUtils()


# ============================================================================
# Custom Pytest Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    markers = [
        "unit: Unit tests",
        "integration: Integration tests", 
        "security: Security tests",
        "performance: Performance tests",
        "slow: Slow running tests",
        "network: Tests requiring network access",
        "gpu: Tests requiring GPU",
        "agent_forge: Agent Forge specific tests",
        "p2p: P2P communication tests",
        "rag: RAG system tests",
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


# ============================================================================
# Test Parametrization Helpers
# ============================================================================

def parametrize_agent_types():
    """Parametrize tests across different agent types."""
    return pytest.mark.parametrize("agent_type", [
        "base_agent",
        "specialized_agent", 
        "coordinator_agent",
        "security_agent",
        "performance_agent",
    ])


def parametrize_security_threats():
    """Parametrize tests across security threat types."""
    return pytest.mark.parametrize("threat_type", [
        "code_injection",
        "command_injection",
        "script_injection", 
        "path_traversal",
        "sql_injection",
    ])


def parametrize_p2p_transports():
    """Parametrize tests across P2P transport types."""
    return pytest.mark.parametrize("transport_type", [
        "bitchat_ble",
        "betanet_htx",
        "libp2p_mesh",
        "direct_tcp",
    ])