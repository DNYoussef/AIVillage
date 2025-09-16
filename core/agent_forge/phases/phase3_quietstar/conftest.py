"""
Pytest configuration and fixtures for Quiet-STaR test suite.

Provides shared fixtures, test configuration, and utilities for comprehensive
testing of Quiet-STaR reasoning enhancement algorithms.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings
import os
import sys
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock

# Add source path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

# Import Quiet-STaR components
from quiet_star.algorithms import (
    QuietSTaRConfig,
    ThoughtGenerator,
    CoherenceScorer,
    MixingHead,
    ThoughtInjector,
    OptimizationStrategies
)


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "property: mark test as property-based test"
    )
    config.addinivalue_line(
        "markers", "contract: mark test as contract test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (>10s)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_sessionstart(session):
    """Session startup configuration."""
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)

    # Configure PyTorch for testing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    print("\n" + "="*80)
    print("QUIET-STAR TEST SUITE INITIALIZATION")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Random seed: 42")
    print("="*80)


def pytest_sessionfinish(session, exitstatus):
    """Session cleanup and summary."""
    print("\n" + "="*80)
    print("QUIET-STAR TEST SUITE COMPLETED")
    print("="*80)

    # Print session statistics
    if hasattr(session, 'testscollected'):
        print(f"Tests collected: {session.testscollected}")

    if exitstatus == 0:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ Tests failed with exit status: {exitstatus}")

    print("="*80)


@pytest.fixture(scope="session")
def device():
    """Provide computing device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_config():
    """Standard test configuration for Quiet-STaR."""
    return QuietSTaRConfig(
        thought_length=4,
        num_thoughts=8,
        coherence_threshold=0.6,
        mixing_head_hidden_dim=128,
        temperature=1.0,
        top_p=0.9
    )


@pytest.fixture(scope="session")
def small_config():
    """Smaller configuration for fast tests."""
    return QuietSTaRConfig(
        thought_length=2,
        num_thoughts=4,
        coherence_threshold=0.5,
        mixing_head_hidden_dim=64,
        temperature=0.8,
        top_p=0.8
    )


@pytest.fixture(scope="session")
def large_config():
    """Larger configuration for stress tests."""
    return QuietSTaRConfig(
        thought_length=12,
        num_thoughts=32,
        coherence_threshold=0.8,
        mixing_head_hidden_dim=512,
        temperature=1.2,
        top_p=0.95
    )


@pytest.fixture
def mock_language_model():
    """Mock language model for testing."""
    class MockLanguageModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=512):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size

            # Simple linear transformation for testing
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.output_projection = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask=None):
            # Simple forward pass for testing
            embedded = self.embedding(input_ids)
            logits = self.output_projection(embedded)

            # Return object with logits attribute (like HuggingFace models)
            return type('ModelOutput', (), {'logits': logits})()

    return MockLanguageModel()


@pytest.fixture
def fast_mock_model():
    """Fast mock model for performance tests."""
    class FastMockModel(nn.Module):
        def __init__(self, vocab_size=500):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            # Return random logits quickly
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            return type('ModelOutput', (), {'logits': logits})()

    return FastMockModel()


@pytest.fixture
def sample_input_data():
    """Sample input data for testing."""
    return {
        'small': {
            'input_ids': torch.randint(0, 100, (1, 8)),
            'attention_mask': torch.ones(1, 8),
            'position': 4
        },
        'medium': {
            'input_ids': torch.randint(0, 500, (2, 16)),
            'attention_mask': torch.ones(2, 16),
            'position': 8
        },
        'large': {
            'input_ids': torch.randint(0, 1000, (4, 32)),
            'attention_mask': torch.ones(4, 32),
            'position': 16
        }
    }


@pytest.fixture
def thought_generator_components(test_config):
    """Complete set of ThoughtGenerator components."""
    return {
        'config': test_config,
        'thought_generator': ThoughtGenerator(test_config),
        'coherence_scorer': CoherenceScorer(test_config),
        'mixing_head': MixingHead(test_config),
        'thought_injector': ThoughtInjector(test_config),
        'optimizer': OptimizationStrategies(test_config)
    }


@pytest.fixture
def sample_thoughts(test_config):
    """Sample thought tensors for testing."""
    batch_size, num_thoughts, thought_length = 2, test_config.num_thoughts, test_config.thought_length
    vocab_size = 1000

    return {
        'thoughts': torch.randint(0, vocab_size, (batch_size, num_thoughts, thought_length)),
        'logits': torch.randn(batch_size, num_thoughts, thought_length, vocab_size),
        'coherence_scores': torch.rand(batch_size, num_thoughts),
        'context': torch.randint(0, vocab_size, (batch_size, 10))
    }


@pytest.fixture
def performance_monitor():
    """Performance monitoring utilities."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None

        def start(self):
            import time
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.start_memory = torch.cuda.memory_allocated()
            else:
                import psutil
                process = psutil.Process()
                self.start_memory = process.memory_info().rss

        def stop(self):
            import time
            end_time = time.time()
            duration = end_time - self.start_time if self.start_time else 0

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - (self.start_memory or 0)
            else:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss
                memory_used = end_memory - (self.start_memory or 0)

            return {
                'duration': duration,
                'memory_used': memory_used,
                'peak_memory': peak_memory if torch.cuda.is_available() else end_memory
            }

    return PerformanceMonitor()


@pytest.fixture
def test_data_generator():
    """Generate test data with various characteristics."""
    class TestDataGenerator:
        @staticmethod
        def create_input_ids(batch_size, seq_len, vocab_size=1000, pattern=None):
            """Create input IDs with optional patterns."""
            if pattern == 'repetitive':
                # Create repetitive patterns
                base_seq = torch.randint(0, vocab_size // 10, (1, seq_len))
                return base_seq.repeat(batch_size, 1)
            elif pattern == 'diverse':
                # Create diverse sequences
                return torch.randint(0, vocab_size, (batch_size, seq_len))
            else:
                return torch.randint(0, vocab_size, (batch_size, seq_len))

        @staticmethod
        def create_logits(batch_size, seq_len, vocab_size=1000, distribution='normal'):
            """Create logits with different distributions."""
            if distribution == 'uniform':
                return torch.zeros(batch_size, seq_len, vocab_size)
            elif distribution == 'peaked':
                logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
                # Make first token highly probable
                logits[:, :, 0] = 10.0
                return logits
            else:  # normal
                return torch.randn(batch_size, seq_len, vocab_size)

        @staticmethod
        def create_attention_weights(batch_size, num_heads, seq_len, pattern='random'):
            """Create attention weights with different patterns."""
            if pattern == 'focused':
                # Focused attention on first token
                weights = torch.zeros(batch_size, num_heads, seq_len, seq_len)
                weights[:, :, :, 0] = 1.0
                return weights
            elif pattern == 'dispersed':
                # Uniform attention
                weights = torch.ones(batch_size, num_heads, seq_len, seq_len)
                return weights / seq_len
            else:  # random
                weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
                return weights / weights.sum(dim=-1, keepdim=True)

    return TestDataGenerator()


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def temporary_files(tmp_path):
    """Provide temporary file paths for testing."""
    return {
        'model_checkpoint': tmp_path / "model.pt",
        'config_file': tmp_path / "config.json",
        'results_file': tmp_path / "results.json",
        'log_file': tmp_path / "test.log"
    }


class MockTrainingLoop:
    """Mock training loop for testing optimization strategies."""

    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.performance_history = []

    def simulate_epoch(self, optimizer):
        """Simulate one training epoch."""
        # Get curriculum parameters
        curriculum_params = optimizer.curriculum_scheduler(self.epoch, 0.7)

        # Simulate training performance
        base_performance = 0.6 + (self.epoch * 0.002)  # Gradual improvement
        noise = np.random.normal(0, 0.05)  # Add some noise
        performance = max(0.0, min(1.0, base_performance + noise))

        self.performance_history.append(performance)

        # Get adaptive parameters
        adaptive_params = optimizer.adaptive_sampling_schedule({
            'coherence_rate': performance
        })

        self.epoch += 1

        return {
            'curriculum': curriculum_params,
            'adaptive': adaptive_params,
            'performance': performance
        }


@pytest.fixture
def mock_training_loop(test_config):
    """Provide mock training loop for testing."""
    return MockTrainingLoop(test_config)


def pytest_runtest_setup(item):
    """Setup for individual tests."""
    # Skip GPU tests if CUDA not available
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # Skip slow tests if running in fast mode
    if item.get_closest_marker("slow") and item.config.getoption("--fast", default=False):
        pytest.skip("Skipping slow test in fast mode")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Run only GPU tests"
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run performance benchmarks"
    )


@pytest.fixture
def test_environment_info():
    """Provide test environment information."""
    return {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        'python_version': sys.version,
        'platform': sys.platform
    }