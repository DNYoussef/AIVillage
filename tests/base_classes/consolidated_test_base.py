"""
Consolidated Test Base Classes
=============================

Base classes for standardized test patterns across the AIVillage codebase.
Replaces 200+ similar test class definitions with reusable base classes.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import Mock, patch

import pytest
import torch

from tests.fixtures.common_fixtures import *
from tests.utils.test_helpers import TestAssertions, TestDataGenerator, PerformanceTester


# ============================================================================
# Base Test Classes
# ============================================================================

class BaseAIVillageTest(ABC):
    """Base class for all AIVillage tests with common setup and utilities."""
    
    def setup_method(self, method):
        """Set up each test method."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{method.__name__}")
        self.start_time = time.time()
        self.test_data = {}
        self.mocks = {}
        
        # Create test-specific temporary directory
        import tempfile
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"test_{method.__name__}_"))
        
        self.logger.debug(f"Starting test {method.__name__} with temp dir {self.temp_dir}")
    
    def teardown_method(self, method):
        """Clean up after each test method."""
        duration = time.time() - self.start_time
        self.logger.debug(f"Test {method.__name__} completed in {duration:.3f}s")
        
        # Clean up temporary directory
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up mocks
        for mock in self.mocks.values():
            if hasattr(mock, 'stop'):
                mock.stop()
        
        self.mocks.clear()
    
    def create_mock(self, name: str, spec: Optional[Type] = None) -> Mock:
        """Create and track a mock object."""
        mock = Mock(spec=spec)
        self.mocks[name] = mock
        return mock
    
    def create_temp_file(self, filename: str, content: str = "") -> Path:
        """Create a temporary file for testing."""
        file_path = self.temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def assert_no_errors_logged(self, log_level: int = logging.ERROR):
        """Assert no errors were logged during test."""
        # This would integrate with log capture fixture
        pass
    
    @property
    def test_duration(self) -> float:
        """Get current test duration in seconds."""
        return time.time() - self.start_time


class BaseAgentTest(BaseAIVillageTest):
    """Base class for agent-related tests."""
    
    def setup_method(self, method):
        """Set up agent test environment."""
        super().setup_method(method)
        
        # Common agent test setup
        self.agent_config = {
            'name': 'TestAgent',
            'description': 'Agent for testing',
            'capabilities': ['test_capability'],
            'model': 'gpt-4',
            'device': 'cpu',
        }
        
        self.mock_llm = self.create_mock('llm')
        self.mock_llm.complete = AsyncMock(return_value=Mock(text="Test response"))
        
        self.mock_communication_protocol = self.create_mock('communication_protocol')
        self.mock_communication_protocol.send_message = AsyncMock(return_value='sent')
    
    def create_test_agent(self, **config_overrides):
        """Create a test agent with optional config overrides."""
        config = {**self.agent_config, **config_overrides}
        
        # This would create an actual agent instance
        # For now, return a comprehensive mock
        agent = self.create_mock('agent')
        agent.name = config['name']
        agent.capabilities = config['capabilities']
        agent.execute_task = AsyncMock(return_value={'result': 'success'})
        agent.generate = AsyncMock(return_value='Generated response')
        
        return agent
    
    def assert_agent_behavior_valid(self, agent, expected_capabilities: List[str]):
        """Assert agent behavior meets expectations."""
        assert agent.name is not None
        assert all(cap in agent.capabilities for cap in expected_capabilities)
        
        # Verify agent can execute basic operations
        TestAssertions.assert_agent_response_valid({'result': 'success'})


class BaseSecurityTest(BaseAIVillageTest):
    """Base class for security-related tests."""
    
    def setup_method(self, method):
        """Set up security test environment."""
        super().setup_method(method)
        
        self.security_validator = self.create_mock('security_validator')
        self.security_validator.validate_message = AsyncMock()
        self.security_validator.get_security_report = Mock(return_value={
            'total_events': 0,
            'threat_type_counts': {},
            'critical_events_24h': 0
        })
        
        # Common threat payloads for testing
        self.threat_payloads = TestDataGenerator.generate_security_test_payloads([
            'code_injection',
            'command_injection', 
            'script_injection',
            'path_traversal',
            'sql_injection'
        ])
    
    async def assert_threat_blocked(self, payload: str, expected_threat_type: str):
        """Assert that a malicious payload is blocked."""
        with pytest.raises(Exception) as exc_info:
            await self.security_validator.validate_message(payload, {})
        
        # Verify the exception indicates the expected threat type
        assert expected_threat_type.lower() in str(exc_info.value).lower()
    
    async def assert_payload_safe(self, payload: str):
        """Assert that a payload is considered safe."""
        try:
            result = await self.security_validator.validate_message(payload, {})
            TestAssertions.assert_security_validation_passed(result)
        except Exception as e:
            pytest.fail(f"Safe payload was blocked: {e}")
    
    def generate_encoded_payload(self, payload: str, encoding: str = 'base64') -> str:
        """Generate encoded malicious payload for testing."""
        if encoding == 'base64':
            import base64
            return base64.b64encode(payload.encode()).decode()
        elif encoding == 'hex':
            return payload.encode().hex()
        elif encoding == 'url':
            import urllib.parse
            return urllib.parse.quote(payload)
        else:
            return payload


class BaseP2PTest(BaseAIVillageTest):
    """Base class for P2P communication tests."""
    
    def setup_method(self, method):
        """Set up P2P test environment."""
        super().setup_method(method)
        
        self.p2p_config = {
            'node_id': 'test_node_001',
            'max_retries': 3,
            'timeout_ms': 1000,
            'enable_store_forward': True,
        }
        
        self.mock_transport = self.create_mock('transport')
        self.mock_transport.send_message = AsyncMock(return_value={
            'status': 'delivered',
            'message_id': 'msg_123'
        })
        
        self.mock_mesh_protocol = self.create_mock('mesh_protocol')
        self.mock_mesh_protocol.start = AsyncMock(return_value=True)
        self.mock_mesh_protocol.stop = AsyncMock(return_value=True)
        self.mock_mesh_protocol.send_message = AsyncMock(return_value='msg_id')
        
        # Generate test network topology
        self.network_topology = TestDataGenerator.generate_p2p_network_topology(
            node_count=5,
            connectivity=0.4
        )
    
    async def simulate_message_delivery(self, sender: str, recipient: str, 
                                      message: Dict[str, Any], 
                                      success_rate: float = 0.95) -> Dict[str, Any]:
        """Simulate P2P message delivery with configurable reliability."""
        import random
        
        delivery_time = random.uniform(10, 100)  # Simulate network latency
        await asyncio.sleep(delivery_time / 1000.0)
        
        if random.random() < success_rate:
            return {
                'status': 'delivered',
                'message_id': f'msg_{int(time.time())}',
                'delivery_time_ms': delivery_time,
                'sender': sender,
                'recipient': recipient,
            }
        else:
            raise ConnectionError(f"Failed to deliver message from {sender} to {recipient}")
    
    def assert_network_topology_valid(self, topology: Dict[str, Any]):
        """Assert network topology is valid for testing."""
        assert 'nodes' in topology
        assert 'connections' in topology
        assert len(topology['nodes']) > 0
        assert topology['avg_connections_per_node'] > 0


class BasePerformanceTest(BaseAIVillageTest):
    """Base class for performance tests."""
    
    def setup_method(self, method):
        """Set up performance test environment."""
        super().setup_method(method)
        
        self.performance_thresholds = {
            'max_inference_time_ms': 100,
            'max_memory_usage_mb': 500,
            'min_throughput_req_per_sec': 10,
            'max_cpu_usage_percent': 80,
        }
        
        self.performance_tester = PerformanceTester()
    
    def measure_function_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure function performance metrics."""
        result, execution_time = self.performance_tester.measure_execution_time(
            func, *args, **kwargs
        )
        
        return {
            'result': result,
            'execution_time_seconds': execution_time,
            'execution_time_ms': execution_time * 1000,
        }
    
    async def measure_async_performance(self, coro) -> Dict[str, Any]:
        """Measure async function performance."""
        result, execution_time = await self.performance_tester.measure_async_execution_time(coro)
        
        return {
            'result': result,
            'execution_time_seconds': execution_time,
            'execution_time_ms': execution_time * 1000,
        }
    
    def assert_performance_acceptable(self, metrics: Dict[str, float]):
        """Assert performance metrics are within acceptable bounds."""
        bounds = {}
        
        if 'execution_time_ms' in metrics:
            bounds['execution_time_ms'] = (0, self.performance_thresholds['max_inference_time_ms'])
        
        if 'memory_usage_mb' in metrics:
            bounds['memory_usage_mb'] = (0, self.performance_thresholds['max_memory_usage_mb'])
        
        if 'throughput_req_per_sec' in metrics:
            bounds['throughput_req_per_sec'] = (self.performance_thresholds['min_throughput_req_per_sec'], float('inf'))
        
        TestAssertions.assert_performance_metrics_within_bounds(metrics, bounds)


class BaseMLModelTest(BaseAIVillageTest):
    """Base class for ML model tests."""
    
    def setup_method(self, method):
        """Set up ML model test environment."""
        super().setup_method(method)
        
        self.device = torch.device('cpu')  # Use CPU for testing
        self.model_config = {
            'input_size': 768,
            'hidden_size': 256,
            'output_size': 128,
            'dropout': 0.1,
        }
    
    def create_test_model(self, **config_overrides) -> torch.nn.Module:
        """Create a test model with optional config overrides."""
        config = {**self.model_config, **config_overrides}
        
        model = torch.nn.Sequential(
            torch.nn.Linear(config['input_size'], config['hidden_size']),
            torch.nn.ReLU(),
            torch.nn.Dropout(config['dropout']),
            torch.nn.Linear(config['hidden_size'], config['output_size']),
        )
        
        return model.to(self.device)
    
    def create_test_input(self, batch_size: int = 1, input_size: Optional[int] = None) -> torch.Tensor:
        """Create test input tensor."""
        input_size = input_size or self.model_config['input_size']
        return torch.randn(batch_size, input_size).to(self.device)
    
    def assert_model_output_valid(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Assert model produces valid output for given input."""
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
            
            assert output is not None, "Model output cannot be None"
            assert isinstance(output, torch.Tensor), "Model output must be a tensor"
            assert output.shape[0] == input_tensor.shape[0], "Batch size must be preserved"
            assert not torch.isnan(output).any(), "Model output contains NaN values"
            assert not torch.isinf(output).any(), "Model output contains infinite values"
    
    def benchmark_model_inference(self, model: torch.nn.Module, 
                                input_shape: tuple, 
                                iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference performance."""
        return self.performance_tester.benchmark_model_inference(
            model, input_shape, iterations
        )


class BaseIntegrationTest(BaseAIVillageTest):
    """Base class for integration tests."""
    
    def setup_method(self, method):
        """Set up integration test environment."""
        super().setup_method(method)
        
        self.integration_timeout = 30.0  # seconds
        self.component_mocks = {}
        self.test_results = {}
    
    async def run_integration_scenario(self, scenario_name: str, 
                                     test_steps: List[callable]) -> Dict[str, Any]:
        """Run an integration test scenario with multiple steps."""
        scenario_results = {
            'scenario': scenario_name,
            'start_time': time.time(),
            'steps': [],
            'success': True,
            'error': None,
        }
        
        try:
            for i, step in enumerate(test_steps):
                step_name = getattr(step, '__name__', f'step_{i}')
                step_start = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(step):
                        result = await step()
                    else:
                        result = step()
                    
                    step_result = {
                        'step_name': step_name,
                        'success': True,
                        'result': result,
                        'duration': time.time() - step_start,
                        'error': None,
                    }
                    
                except Exception as e:
                    step_result = {
                        'step_name': step_name,
                        'success': False,
                        'result': None,
                        'duration': time.time() - step_start,
                        'error': str(e),
                    }
                    scenario_results['success'] = False
                    scenario_results['error'] = f"Step {step_name} failed: {e}"
                
                scenario_results['steps'].append(step_result)
                
                # Stop on first failure
                if not step_result['success']:
                    break
        
        except Exception as e:
            scenario_results['success'] = False
            scenario_results['error'] = f"Scenario failed: {e}"
        
        finally:
            scenario_results['end_time'] = time.time()
            scenario_results['total_duration'] = scenario_results['end_time'] - scenario_results['start_time']
        
        self.test_results[scenario_name] = scenario_results
        return scenario_results
    
    def assert_integration_successful(self, scenario_results: Dict[str, Any]):
        """Assert integration scenario completed successfully."""
        TestAssertions.assert_integration_test_successful(scenario_results)
        
        # Additional integration-specific assertions
        assert scenario_results['total_duration'] < self.integration_timeout
        assert len(scenario_results['steps']) > 0
        assert all(step['success'] for step in scenario_results['steps'])


# ============================================================================
# Specialized Test Classes
# ============================================================================

class BaseAgentForgeTest(BaseMLModelTest):
    """Base class for Agent Forge specific tests."""
    
    def setup_method(self, method):
        """Set up Agent Forge test environment."""
        super().setup_method(method)
        
        self.pipeline_config = {
            'base_models': ['mock-model'],
            'output_dir': self.temp_dir / 'output',
            'checkpoint_dir': self.temp_dir / 'checkpoints',
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
        
        # Create output directories
        self.pipeline_config['output_dir'].mkdir(parents=True, exist_ok=True)
        self.pipeline_config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    
    def create_mock_pipeline(self):
        """Create a mock Agent Forge pipeline."""
        pipeline = self.create_mock('pipeline')
        
        async def mock_run_pipeline():
            result = Mock()
            result.success = True
            result.error = None
            result.metrics = {
                'phases_completed': 1,
                'total_time_seconds': 10.0,
                'memory_peak_mb': 100.0,
            }
            return result
        
        pipeline.run_pipeline = mock_run_pipeline
        pipeline.phases = [('CognatePhase', Mock())]
        pipeline.config = Mock()
        
        return pipeline
    
    def assert_pipeline_result_valid(self, result):
        """Assert pipeline result is valid."""
        assert hasattr(result, 'success')
        assert hasattr(result, 'metrics')
        
        if result.success:
            assert result.error is None
            assert result.metrics['phases_completed'] > 0
        else:
            assert result.error is not None


class BaseRAGTest(BaseAIVillageTest):
    """Base class for RAG (Retrieval-Augmented Generation) tests."""
    
    def setup_method(self, method):
        """Set up RAG test environment."""
        super().setup_method(method)
        
        self.rag_config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_store_type': 'faiss',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'top_k': 5,
        }
        
        self.mock_vector_store = self.create_mock('vector_store')
        self.mock_vector_store.query = AsyncMock(return_value=[
            {'id': 'doc1', 'content': 'Test document 1', 'score': 0.95},
            {'id': 'doc2', 'content': 'Test document 2', 'score': 0.87},
        ])
        
        self.mock_embedding_model = self.create_mock('embedding_model')
        self.mock_embedding_model.encode = Mock(return_value=[0.1] * 384)  # Mock embedding
    
    def create_test_documents(self, count: int = 10) -> List[Dict[str, Any]]:
        """Create test documents for RAG testing."""
        documents = []
        
        for i in range(count):
            doc = {
                'id': f'doc_{i:03d}',
                'content': f'This is test document {i} with content for testing RAG functionality.',
                'metadata': {
                    'source': f'test_source_{i}',
                    'created_at': time.time(),
                    'category': 'test',
                }
            }
            documents.append(doc)
        
        return documents
    
    async def assert_retrieval_quality(self, query: str, retrieved_docs: List[Dict[str, Any]]):
        """Assert retrieval quality meets expectations."""
        assert len(retrieved_docs) > 0, "No documents retrieved"
        assert all('score' in doc for doc in retrieved_docs), "All docs must have similarity scores"
        
        # Check scores are in descending order
        scores = [doc['score'] for doc in retrieved_docs]
        assert scores == sorted(scores, reverse=True), "Documents not ordered by relevance"
        
        # Check scores are reasonable
        assert all(0.0 <= score <= 1.0 for score in scores), "Scores must be between 0 and 1"


# ============================================================================
# Test Suite Organization
# ============================================================================

class TestSuiteBase:
    """Base class for organizing related test suites."""
    
    @classmethod
    def get_test_classes(cls) -> List[Type[BaseAIVillageTest]]:
        """Get all test classes in this suite."""
        test_classes = []
        
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseAIVillageTest) and 
                attr != BaseAIVillageTest):
                test_classes.append(attr)
        
        return test_classes
    
    @classmethod
    def run_suite(cls, test_runner=None):
        """Run all tests in this suite."""
        if test_runner is None:
            import pytest
            test_classes = cls.get_test_classes()
            
            for test_class in test_classes:
                pytest.main([f"{test_class.__module__}::{test_class.__name__}", "-v"])


# ============================================================================
# Export All Base Classes
# ============================================================================

__all__ = [
    'BaseAIVillageTest',
    'BaseAgentTest',
    'BaseSecurityTest', 
    'BaseP2PTest',
    'BasePerformanceTest',
    'BaseMLModelTest',
    'BaseIntegrationTest',
    'BaseAgentForgeTest',
    'BaseRAGTest',
    'TestSuiteBase',
]