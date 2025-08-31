"""
Constitutional Testing Configuration and Fixtures

Shared pytest configuration, fixtures, and utilities for constitutional
fog compute system testing.
"""

import pytest
import asyncio
import os
import json
import tempfile
from typing import Dict, List, Any, Optional, Generator
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from pathlib import Path

# Import test data fixtures
from tests.constitutional.fixtures.test_data import (
    ConstitutionalTestDataFixtures,
    ALL_TEST_SAMPLES,
    HARMLESS_SAMPLES,
    MODERATE_HARM_SAMPLES,
    SEVERE_HARM_SAMPLES,
    BIAS_TESTING_DATASETS
)

# Mock enums for testing
from enum import Enum

class HarmLevel(Enum):
    H0 = "harmless"
    H1 = "minor_harm"
    H2 = "moderate_harm"
    H3 = "severe_harm"

class UserTier(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"

class ConstitutionalPrinciple(Enum):
    FREE_SPEECH = "free_speech"
    DUE_PROCESS = "due_process"
    EQUAL_PROTECTION = "equal_protection"
    PRIVACY_RIGHTS = "privacy_rights"
    DEMOCRATIC_PARTICIPATION = "democratic_participation"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    HUMAN_DIGNITY = "human_dignity"
    VIEWPOINT_NEUTRALITY = "viewpoint_neutrality"
    PROCEDURAL_FAIRNESS = "procedural_fairness"


@dataclass
class MockConstitutionalSystem:
    """Mock constitutional system for testing"""
    harm_classifier: Mock
    constitutional_enforcer: Mock
    governance: Mock
    tee_manager: Mock
    moderation_pipeline: Mock
    pricing_manager: Mock
    mixnode_client: Mock
    auction_engine: Mock
    edge_bridge: Mock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def constitutional_test_config():
    """Configuration for constitutional testing"""
    return {
        'testing_mode': True,
        'mock_external_services': True,
        'enable_performance_tracking': True,
        'strict_constitutional_compliance': True,
        'log_level': 'DEBUG',
        'test_data_validation': True,
        'parallel_test_execution': True,
        'timeout_seconds': 30,
        'max_retries': 3,
        'enable_bias_testing': True,
        'enable_edge_case_testing': True,
        'enable_adversarial_testing': True,
        'performance_benchmarks': {
            'max_latency_ms': 200,
            'min_throughput_rps': 100,
            'max_memory_mb': 1024,
            'max_cpu_percent': 80
        }
    }


@pytest.fixture
def mock_constitutional_system():
    """Mock constitutional system components for testing"""
    return MockConstitutionalSystem(
        harm_classifier=Mock(),
        constitutional_enforcer=Mock(),
        governance=Mock(),
        tee_manager=Mock(),
        moderation_pipeline=Mock(),
        pricing_manager=Mock(),
        mixnode_client=Mock(),
        auction_engine=Mock(),
        edge_bridge=Mock()
    )


@pytest.fixture
def constitutional_test_data():
    """Comprehensive constitutional test data"""
    return {
        'all_samples': ALL_TEST_SAMPLES,
        'harmless_samples': HARMLESS_SAMPLES,
        'moderate_harm_samples': MODERATE_HARM_SAMPLES,
        'severe_harm_samples': SEVERE_HARM_SAMPLES,
        'bias_testing_datasets': BIAS_TESTING_DATASETS,
        'edge_cases': ConstitutionalTestDataFixtures.get_constitutional_edge_cases(),
        'protected_speech': ConstitutionalTestDataFixtures.get_protected_speech_samples(),
        'performance_samples': ConstitutionalTestDataFixtures.get_performance_testing_samples()
    }


@pytest.fixture
def mock_harm_classifier():
    """Mock harm classifier with realistic responses"""
    classifier = Mock()
    
    async def mock_classify_harm(content: str, context: Dict[str, Any] = None):
        """Mock harm classification with realistic logic"""
        context = context or {}
        
        # Simple heuristic-based classification for testing
        content_lower = content.lower()
        
        # Severe harm indicators
        if any(word in content_lower for word in ['kill', 'bomb', 'explode', 'murder', 'terrorist']):
            return {
                'harm_level': HarmLevel.H3,
                'confidence': 0.95,
                'categories': ['violence', 'dangerous_activities'],
                'constitutional_concerns': ['public_safety', 'clear_and_present_danger']
            }
        
        # Moderate harm indicators
        elif any(word in content_lower for word in ['hate', 'discriminate', 'stereotype', 'harmful']):
            return {
                'harm_level': HarmLevel.H2,
                'confidence': 0.85,
                'categories': ['hate_speech', 'discrimination'],
                'constitutional_concerns': ['equal_protection', 'human_dignity']
            }
        
        # Minor harm indicators  
        elif any(word in content_lower for word in ['disagree', 'criticize', 'opinion', 'politics']):
            return {
                'harm_level': HarmLevel.H1,
                'confidence': 0.75,
                'categories': [],
                'constitutional_concerns': ['political_expression', 'free_speech']
            }
        
        # Default to harmless
        else:
            return {
                'harm_level': HarmLevel.H0,
                'confidence': 0.90,
                'categories': [],
                'constitutional_concerns': ['information_access']
            }
    
    classifier.classify_harm = AsyncMock(side_effect=mock_classify_harm)
    return classifier


@pytest.fixture
def mock_constitutional_enforcer():
    """Mock constitutional enforcer with realistic responses"""
    enforcer = Mock()
    
    async def mock_enforce_standards(harm_result: Dict, user_tier: UserTier, context: Dict):
        """Mock constitutional enforcement with tier-based logic"""
        harm_level = harm_result.get('harm_level', HarmLevel.H0)
        
        # Determine action based on harm level and user tier
        if harm_level == HarmLevel.H3:
            if user_tier == UserTier.PLATINUM:
                action = 'restrict_with_constitutional_review'
            else:
                action = 'block_with_explanation'
        elif harm_level == HarmLevel.H2:
            if user_tier in [UserTier.GOLD, UserTier.PLATINUM]:
                action = 'allow_with_context'
            else:
                action = 'warn_and_educate'
        elif harm_level == HarmLevel.H1:
            action = 'allow'
        else:
            action = 'allow'
        
        return {
            'enforcement_action': action,
            'constitutional_compliance': True,
            'principles_applied': harm_result.get('constitutional_concerns', []),
            'escalation_required': harm_level in [HarmLevel.H2, HarmLevel.H3],
            'appeal_rights_granted': user_tier in [UserTier.GOLD, UserTier.PLATINUM],
            'tier_protections_applied': True
        }
    
    enforcer.enforce_standards = AsyncMock(side_effect=mock_enforce_standards)
    return enforcer


@pytest.fixture
def mock_governance_system():
    """Mock democratic governance system"""
    governance = Mock()
    
    async def mock_validate_democratic_process(action: str, stakeholders: List[str], context: Dict):
        """Mock democratic governance validation"""
        return {
            'process_valid': True,
            'voting_conducted': context.get('voting_required', False),
            'public_input_gathered': context.get('public_input_required', False),
            'transparency_maintained': True,
            'stakeholder_participation': len(stakeholders),
            'democratic_legitimacy_score': 0.90,
            'appeal_mechanism_available': True
        }
    
    governance.validate_democratic_process = AsyncMock(side_effect=mock_validate_democratic_process)
    return governance


@pytest.fixture
def mock_tee_manager():
    """Mock TEE security manager"""
    tee_manager = Mock()
    
    async def mock_process_in_enclave(content: str):
        """Mock TEE secure processing"""
        return {
            'processed_securely': True,
            'attestation_verified': True,
            'tamper_detection': 'none',
            'security_level': 'maximum',
            'enclave_id': 'enc_test_123',
            'processing_integrity': True
        }
    
    tee_manager.process_in_enclave = AsyncMock(side_effect=mock_process_in_enclave)
    return tee_manager


@pytest.fixture
def mock_pricing_manager():
    """Mock constitutional pricing manager"""
    pricing_manager = Mock()
    
    async def mock_calculate_pricing(user_tier: UserTier, processing_data: Dict, context: Dict):
        """Mock pricing calculation with tier-based rates"""
        base_rates = {
            UserTier.BRONZE: 1.00,
            UserTier.SILVER: 1.25,
            UserTier.GOLD: 1.75,
            UserTier.PLATINUM: 2.50
        }
        
        base_cost = base_rates.get(user_tier, 1.00)
        constitutional_premium = 0.15 if context.get('constitutional_analysis_required') else 0.0
        
        return {
            'base_cost': base_cost,
            'constitutional_premium': constitutional_premium,
            'total_cost': base_cost + constitutional_premium,
            'billing_breakdown': f"Tier: {user_tier.value}, Base: ${base_cost:.2f}",
            'billing_compliant': True
        }
    
    pricing_manager.calculate_pricing = AsyncMock(side_effect=mock_calculate_pricing)
    return pricing_manager


@pytest.fixture
def test_database():
    """Mock database for testing"""
    db = {}
    
    def get(key: str, default=None):
        return db.get(key, default)
    
    def set(key: str, value: Any):
        db[key] = value
    
    def delete(key: str):
        db.pop(key, None)
    
    def clear():
        db.clear()
    
    mock_db = Mock()
    mock_db.get = Mock(side_effect=get)
    mock_db.set = Mock(side_effect=set)
    mock_db.delete = Mock(side_effect=delete)
    mock_db.clear = Mock(side_effect=clear)
    
    return mock_db


@pytest.fixture
def temp_test_directory():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for constitutional compliance"""
    logger = Mock()
    logged_events = []
    
    def log_event(event_type: str, details: Dict[str, Any]):
        logged_events.append({
            'event_type': event_type,
            'details': details,
            'timestamp': '2023-12-01T00:00:00Z'
        })
    
    logger.log_event = Mock(side_effect=log_event)
    logger.logged_events = logged_events
    
    return logger


@pytest.fixture
def performance_tracker():
    """Performance tracking fixture for benchmarking"""
    import time
    
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_timer(self, operation: str):
            self.start_times[operation] = time.perf_counter()
        
        def end_timer(self, operation: str):
            if operation in self.start_times:
                duration = time.perf_counter() - self.start_times[operation]
                self.metrics[operation] = duration * 1000  # Convert to milliseconds
                del self.start_times[operation]
                return self.metrics[operation]
            return None
        
        def get_metric(self, operation: str):
            return self.metrics.get(operation)
        
        def get_all_metrics(self):
            return self.metrics.copy()
    
    return PerformanceTracker()


@pytest.fixture
def constitutional_compliance_validator():
    """Constitutional compliance validation utility"""
    
    class ComplianceValidator:
        def __init__(self):
            self.required_principles = [
                ConstitutionalPrinciple.FREE_SPEECH,
                ConstitutionalPrinciple.DUE_PROCESS,
                ConstitutionalPrinciple.EQUAL_PROTECTION
            ]
        
        def validate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
            """Validate constitutional compliance of a decision"""
            validation_result = {
                'compliant': True,
                'violations': [],
                'recommendations': []
            }
            
            # Check for required constitutional principles
            applied_principles = decision.get('principles_applied', [])
            for principle in self.required_principles:
                if principle.value not in applied_principles and self._principle_relevant(decision, principle):
                    validation_result['violations'].append(f"Missing principle: {principle.value}")
                    validation_result['compliant'] = False
            
            # Check for due process requirements
            if decision.get('harm_level') in ['H2', 'H3']:
                if not decision.get('human_review_available'):
                    validation_result['violations'].append("Human review not available for significant harm")
                    validation_result['compliant'] = False
                
                if not decision.get('appeal_rights_granted'):
                    validation_result['violations'].append("Appeal rights not granted for significant decisions")
                    validation_result['compliant'] = False
            
            # Generate recommendations
            if not validation_result['compliant']:
                validation_result['recommendations'].append("Review constitutional principle application")
                validation_result['recommendations'].append("Ensure due process requirements are met")
            
            return validation_result
        
        def _principle_relevant(self, decision: Dict[str, Any], principle: ConstitutionalPrinciple) -> bool:
            """Check if a constitutional principle is relevant to the decision"""
            content_type = decision.get('content_type', '')
            
            if principle == ConstitutionalPrinciple.FREE_SPEECH:
                return 'political' in content_type or 'opinion' in content_type
            elif principle == ConstitutionalPrinciple.DUE_PROCESS:
                return decision.get('enforcement_action') in ['restrict', 'block', 'suspend']
            elif principle == ConstitutionalPrinciple.EQUAL_PROTECTION:
                return 'bias_test' in decision.get('context', {})
            
            return False
    
    return ComplianceValidator()


@pytest.fixture(autouse=True)
def setup_test_environment(constitutional_test_config):
    """Automatic test environment setup"""
    # Set test mode environment variables
    os.environ['CONSTITUTIONAL_TESTING_MODE'] = 'true'
    os.environ['MOCK_EXTERNAL_SERVICES'] = 'true'
    os.environ['TEST_LOG_LEVEL'] = constitutional_test_config['log_level']
    
    yield
    
    # Cleanup environment
    test_env_vars = [
        'CONSTITUTIONAL_TESTING_MODE',
        'MOCK_EXTERNAL_SERVICES', 
        'TEST_LOG_LEVEL'
    ]
    
    for var in test_env_vars:
        os.environ.pop(var, None)


# Pytest marks for test categorization
def pytest_configure(config):
    """Configure custom pytest marks"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end system tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "adversarial: Adversarial and security tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: Edge case and boundary condition tests"
    )
    config.addinivalue_line(
        "markers", "constitutional: Constitutional compliance tests"
    )
    config.addinivalue_line(
        "markers", "bias: Bias detection and fairness tests"
    )
    config.addinivalue_line(
        "markers", "robustness: System robustness and stress tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (disabled by default)"
    )


# Test collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and configuration"""
    # Skip slow tests by default
    skip_slow = pytest.mark.skip(reason="Slow test skipped by default (use --run-slow to enable)")
    
    for item in items:
        if "slow" in item.keywords:
            if not config.getoption("--run-slow"):
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-adversarial",
        action="store_true", 
        default=False,
        help="Run adversarial security tests"
    )
    parser.addoption(
        "--performance-only",
        action="store_true",
        default=False,
        help="Run only performance tests"
    )
    parser.addoption(
        "--constitutional-only",
        action="store_true",
        default=False,
        help="Run only constitutional compliance tests"
    )


# Async test utilities
@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests"""
    return 30.0  # 30 seconds


@pytest.fixture
def mock_async_context_manager():
    """Mock async context manager for testing"""
    class MockAsyncContextManager:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockAsyncContextManager


# Test data validation utilities
@pytest.fixture
def test_data_validator():
    """Validate test data integrity"""
    
    class TestDataValidator:
        def validate_harm_level_distribution(self, samples: List) -> bool:
            """Validate harm level distribution in test samples"""
            harm_levels = [sample.expected_harm_level for sample in samples]
            unique_levels = set(harm_levels)
            
            # Ensure all harm levels are represented
            return len(unique_levels) == len(HarmLevel)
        
        def validate_constitutional_coverage(self, samples: List) -> bool:
            """Validate constitutional principle coverage"""
            all_principles = set()
            for sample in samples:
                all_principles.update(sample.constitutional_considerations)
            
            # Ensure core principles are covered
            core_principles = {
                'free_speech', 'due_process', 'equal_protection'
            }
            
            return core_principles.issubset(all_principles)
        
        def validate_bias_test_symmetry(self, bias_datasets: Dict) -> bool:
            """Validate bias test dataset symmetry"""
            for bias_type, samples in bias_datasets.items():
                if len(samples) < 2:
                    return False
                
                # Check that samples have same expected classification
                expected_levels = {sample.expected_harm_level for sample in samples}
                if len(expected_levels) > 1:
                    return False
            
            return True
    
    return TestDataValidator()


if __name__ == "__main__":
    # Run configuration validation
    print("Constitutional testing configuration loaded successfully")
    print(f"Total test samples available: {len(ALL_TEST_SAMPLES)}")
    print(f"Bias testing datasets: {len(BIAS_TESTING_DATASETS)}")
    print("Mock system components initialized")
    print("Custom pytest marks configured")
    print("Test environment ready")