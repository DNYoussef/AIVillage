"""
TDD London School Mock Factory and Behavior Verification System
==============================================================

Centralized mock creation and interaction verification following London School TDD principles.
Focuses on behavior verification and object collaboration testing.
"""

import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Type
from unittest.mock import MagicMock, Mock, AsyncMock, patch, call
from collections import defaultdict
import asyncio
import pytest


class BehaviorVerificationMock:
    """
    Enhanced mock that tracks interactions and provides behavior verification
    following London School TDD principles.
    """
    
    def __init__(self, name: str, spec: Optional[Type] = None, **kwargs):
        self.name = name
        self.spec = spec
        self._interactions: List[Dict] = []
        self._expectations: List[Dict] = []
        self._mock = MagicMock(spec=spec, **kwargs)
        self._setup_interaction_tracking()
    
    def _setup_interaction_tracking(self):
        """Setup automatic interaction tracking on all mock methods."""
        original_getattr = self._mock.__getattribute__
        
        def tracking_getattr(attr_name):
            attr = original_getattr(attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                return self._create_tracking_wrapper(attr_name, attr)
            return attr
        
        self._mock.__getattribute__ = tracking_getattr
    
    def _create_tracking_wrapper(self, method_name: str, method: Callable):
        """Create a wrapper that tracks method calls."""
        def wrapper(*args, **kwargs):
            interaction = {
                'method': method_name,
                'args': args,
                'kwargs': kwargs,
                'timestamp': asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
            }
            self._interactions.append(interaction)
            return method(*args, **kwargs)
        
        return wrapper
    
    def expect_interaction(self, method_name: str, *args, **kwargs) -> 'BehaviorVerificationMock':
        """Set expectation for a specific interaction."""
        self._expectations.append({
            'method': method_name,
            'args': args,
            'kwargs': kwargs
        })
        return self
    
    def verify_interactions(self) -> bool:
        """Verify that all expected interactions occurred."""
        for expectation in self._expectations:
            if not self._find_matching_interaction(expectation):
                raise AssertionError(f"Expected interaction not found: {expectation}")
        return True
    
    def _find_matching_interaction(self, expectation: Dict) -> bool:
        """Find if an interaction matches the expectation."""
        for interaction in self._interactions:
            if (interaction['method'] == expectation['method'] and
                interaction['args'] == expectation['args'] and
                interaction['kwargs'] == expectation['kwargs']):
                return True
        return False
    
    def get_interaction_sequence(self) -> List[str]:
        """Get the sequence of method calls."""
        return [f"{i['method']}({i['args']}, {i['kwargs']})" for i in self._interactions]
    
    def reset_interactions(self):
        """Reset interaction history."""
        self._interactions.clear()
        self._expectations.clear()
    
    def __getattr__(self, name):
        """Delegate to underlying mock."""
        return getattr(self._mock, name)


class MockFactory:
    """
    Centralized factory for creating standardized mocks following London School patterns.
    """
    
    def __init__(self):
        self._mock_registry: Dict[str, BehaviorVerificationMock] = {}
        self._collaboration_graph: Dict[str, List[str]] = defaultdict(list)
    
    def create_mock(self, name: str, spec: Optional[Type] = None, **kwargs) -> BehaviorVerificationMock:
        """Create a behavior verification mock."""
        mock = BehaviorVerificationMock(name, spec, **kwargs)
        self._mock_registry[name] = mock
        return mock
    
    def create_async_mock(self, name: str, spec: Optional[Type] = None, **kwargs) -> BehaviorVerificationMock:
        """Create an async behavior verification mock."""
        mock = BehaviorVerificationMock(name, spec, **kwargs)
        mock._mock = AsyncMock(spec=spec, **kwargs)
        self._mock_registry[name] = mock
        return mock
    
    def create_service_mock(self, service_name: str, methods: List[str]) -> BehaviorVerificationMock:
        """Create a service mock with predefined methods."""
        mock = self.create_mock(service_name)
        
        for method_name in methods:
            setattr(mock._mock, method_name, MagicMock())
        
        return mock
    
    def create_repository_mock(self, entity_name: str) -> BehaviorVerificationMock:
        """Create a repository mock with standard CRUD operations."""
        methods = ['save', 'find_by_id', 'find_all', 'update', 'delete', 'exists']
        return self.create_service_mock(f"{entity_name}Repository", methods)
    
    def create_gateway_mock(self, gateway_name: str, endpoints: List[str]) -> BehaviorVerificationMock:
        """Create a gateway mock with specified endpoints."""
        return self.create_service_mock(f"{gateway_name}Gateway", endpoints)
    
    def create_collaborator_set(self, *collaborator_specs) -> Dict[str, BehaviorVerificationMock]:
        """Create a set of collaborating mocks."""
        collaborators = {}
        
        for spec in collaborator_specs:
            if isinstance(spec, tuple):
                name, mock_type = spec
                collaborators[name] = self.create_mock(name, mock_type)
            else:
                collaborators[spec] = self.create_mock(spec)
        
        # Register collaboration relationships
        for name in collaborators.keys():
            self._collaboration_graph[name].extend([n for n in collaborators.keys() if n != name])
        
        return collaborators
    
    def verify_all_collaborations(self):
        """Verify interactions across all registered collaborators."""
        for name, mock in self._mock_registry.items():
            try:
                mock.verify_interactions()
            except AssertionError as e:
                raise AssertionError(f"Collaboration verification failed for {name}: {e}")
    
    def get_interaction_diagram(self) -> Dict[str, List[str]]:
        """Get a diagram of interactions between collaborators."""
        diagram = {}
        for name, mock in self._mock_registry.items():
            diagram[name] = mock.get_interaction_sequence()
        return diagram
    
    def reset_all_mocks(self):
        """Reset all mocks and their interaction histories."""
        for mock in self._mock_registry.values():
            mock.reset_interactions()
            mock._mock.reset_mock()
        
        self._collaboration_graph.clear()


class TDDLondonFixtures:
    """
    Pytest fixtures for TDD London School testing patterns.
    """
    
    @staticmethod
    @pytest.fixture
    def mock_factory():
        """Provide a mock factory for behavior verification."""
        return MockFactory()
    
    @staticmethod
    @pytest.fixture
    def user_service_collaborators(mock_factory):
        """Standard collaborators for UserService testing."""
        return mock_factory.create_collaborator_set(
            'user_repository',
            'email_service',
            'notification_service',
            'audit_logger'
        )
    
    @staticmethod
    @pytest.fixture
    def order_processing_collaborators(mock_factory):
        """Standard collaborators for order processing."""
        return mock_factory.create_collaborator_set(
            'order_repository',
            'payment_gateway',
            'inventory_service',
            'shipping_service',
            'notification_service'
        )
    
    @staticmethod
    @pytest.fixture
    def ai_agent_collaborators(mock_factory):
        """Standard collaborators for AI agent testing."""
        return mock_factory.create_collaborator_set(
            'model_repository',
            'inference_engine',
            'memory_store',
            'communication_service',
            'metrics_collector'
        )
    
    @staticmethod
    @pytest.fixture
    def p2p_network_collaborators(mock_factory):
        """Standard collaborators for P2P network testing."""
        return mock_factory.create_collaborator_set(
            'transport_manager',
            'peer_discovery',
            'message_router',
            'encryption_service',
            'reputation_tracker'
        )
    
    @staticmethod
    @pytest.fixture(autouse=True)
    def verify_collaborations_after_test(mock_factory):
        """Automatically verify all collaborations after each test."""
        yield
        try:
            mock_factory.verify_all_collaborations()
        except AssertionError as e:
            pytest.fail(f"Collaboration verification failed: {e}")


class ContractTestingMixin:
    """
    Mixin for contract testing following London School principles.
    """
    
    def assert_collaboration_sequence(self, mock: BehaviorVerificationMock, expected_sequence: List[str]):
        """Assert that interactions occurred in expected sequence."""
        actual_sequence = mock.get_interaction_sequence()
        assert actual_sequence == expected_sequence, (
            f"Expected sequence: {expected_sequence}\n"
            f"Actual sequence: {actual_sequence}"
        )
    
    def assert_interaction_count(self, mock: BehaviorVerificationMock, method_name: str, expected_count: int):
        """Assert number of calls to a specific method."""
        actual_count = sum(1 for i in mock._interactions if i['method'] == method_name)
        assert actual_count == expected_count, (
            f"Expected {method_name} to be called {expected_count} times, "
            f"but was called {actual_count} times"
        )
    
    def assert_never_called(self, mock: BehaviorVerificationMock, method_name: str):
        """Assert that a method was never called."""
        self.assert_interaction_count(mock, method_name, 0)
    
    def assert_called_with_contract(self, mock: BehaviorVerificationMock, method_name: str, **contract_kwargs):
        """Assert method was called with parameters matching a contract."""
        for interaction in mock._interactions:
            if interaction['method'] == method_name:
                for key, expected_value in contract_kwargs.items():
                    if key in interaction['kwargs']:
                        actual_value = interaction['kwargs'][key]
                        assert actual_value == expected_value, (
                            f"Contract violation: {key} expected {expected_value}, got {actual_value}"
                        )
                    elif interaction['args']:
                        # Try to match positional args by parameter name
                        method_sig = inspect.signature(getattr(mock._mock, method_name))
                        param_names = list(method_sig.parameters.keys())
                        if key in param_names:
                            param_index = param_names.index(key)
                            if param_index < len(interaction['args']):
                                actual_value = interaction['args'][param_index]
                                assert actual_value == expected_value, (
                                    f"Contract violation: {key} expected {expected_value}, got {actual_value}"
                                )


# Global factory instance for use in fixtures
global_mock_factory = MockFactory()


def create_behavior_mock(name: str, spec: Optional[Type] = None, **kwargs) -> BehaviorVerificationMock:
    """Convenience function to create behavior verification mocks."""
    return global_mock_factory.create_mock(name, spec, **kwargs)


def create_service_collaborators(*service_names) -> Dict[str, BehaviorVerificationMock]:
    """Convenience function to create service collaborators."""
    return global_mock_factory.create_collaborator_set(*service_names)