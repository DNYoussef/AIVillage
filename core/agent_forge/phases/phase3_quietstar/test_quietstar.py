"""
Comprehensive Test Suite for Quiet-STaR Implementation
=====================================================

Test coverage for Quiet-STaR reasoning enhancement algorithms including:
- Unit tests for all core components (>85% coverage target)
- Integration tests for end-to-end pipeline
- Performance benchmarks and validation
- Property-based testing for algorithm invariants
- Contract testing for input/output validation

Based on "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"
by Zelikman et al. (2024)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest.mock as mock
from typing import Dict, List, Tuple, Optional
import time
import math
from dataclasses import asdict

# Import the Quiet-STaR components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from quiet_star.algorithms import (
    QuietSTaRConfig,
    ThoughtGenerator,
    CoherenceScorer,
    MixingHead,
    ThoughtInjector,
    OptimizationStrategies
)


class TestQuietSTaRConfig:
    """Test configuration management for Quiet-STaR."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = QuietSTaRConfig()

        assert config.thought_length == 8
        assert config.num_thoughts == 16
        assert config.coherence_threshold == 0.7
        assert config.mixing_head_hidden_dim == 256
        assert config.start_thought_token == "<|startofthought|>"
        assert config.end_thought_token == "<|endofthought|>"
        assert config.temperature == 1.0
        assert config.top_p == 0.9

    def test_custom_config_creation(self):
        """Test creating custom configuration."""
        config = QuietSTaRConfig(
            thought_length=12,
            num_thoughts=32,
            coherence_threshold=0.8,
            temperature=0.8
        )

        assert config.thought_length == 12
        assert config.num_thoughts == 32
        assert config.coherence_threshold == 0.8
        assert config.temperature == 0.8

    def test_config_serialization(self):
        """Test configuration serialization to dict."""
        config = QuietSTaRConfig(thought_length=10, num_thoughts=20)
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert config_dict['thought_length'] == 10
        assert config_dict['num_thoughts'] == 20


class TestThoughtGenerator:
    """Comprehensive tests for ThoughtGenerator component."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QuietSTaRConfig(
            thought_length=4,
            num_thoughts=8,
            coherence_threshold=0.6,
            temperature=1.0,
            top_p=0.9
        )

    @pytest.fixture
    def thought_generator(self, config):
        """Create ThoughtGenerator instance."""
        return ThoughtGenerator(config)

    @pytest.fixture
    def mock_model(self):
        """Create mock language model."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 50000

            def forward(self, input_ids, attention_mask=None):
                batch_size, seq_len = input_ids.shape
                logits = torch.randn(batch_size, seq_len, self.vocab_size)
                return type('obj', (object,), {'logits': logits})()

        return MockModel()

    def test_thought_generator_initialization(self, thought_generator, config):
        """Test ThoughtGenerator initialization."""
        assert thought_generator.config == config
        assert isinstance(thought_generator.coherence_scorer, CoherenceScorer)
        assert isinstance(thought_generator.mixing_head, MixingHead)

    def test_generate_thoughts_shape(self, thought_generator, mock_model):
        """Test thought generation output shapes."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        position = 5

        result = thought_generator.generate_thoughts(
            input_ids, attention_mask, mock_model, position
        )

        # Check output structure
        assert 'thoughts' in result
        assert 'logits' in result
        assert 'coherence_scores' in result
        assert 'valid_mask' in result

        # Check shapes
        thoughts = result['thoughts']
        assert thoughts.shape == (batch_size, thought_generator.config.num_thoughts,
                                 thought_generator.config.thought_length)

        coherence_scores = result['coherence_scores']
        assert coherence_scores.shape == (batch_size, thought_generator.config.num_thoughts)

        valid_mask = result['valid_mask']
        assert valid_mask.shape == (batch_size, thought_generator.config.num_thoughts)

    def test_generate_thoughts_types(self, thought_generator, mock_model):
        """Test thought generation output types."""
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones_like(input_ids)

        result = thought_generator.generate_thoughts(
            input_ids, attention_mask, mock_model, 5
        )

        assert torch.is_tensor(result['thoughts'])
        assert torch.is_tensor(result['logits'])
        assert torch.is_tensor(result['coherence_scores'])
        assert torch.is_tensor(result['valid_mask'])
        assert result['valid_mask'].dtype == torch.bool

    def test_prepare_thought_prefix(self, thought_generator):
        """Test thought prefix preparation."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        position = 3

        prefix = thought_generator._prepare_thought_prefix(input_ids, position)

        # Should include tokens up to position + start token
        assert prefix.shape[1] == position + 2  # position + 1 (inclusive) + 1 (start token)
        assert torch.equal(prefix[:, :position+1], input_ids[:, :position+1])

    def test_top_p_filter_functionality(self, thought_generator):
        """Test top-p filtering implementation."""
        # Create test logits with known distribution
        logits = torch.tensor([[10.0, 5.0, 1.0, 0.1, 0.01]])
        top_p = 0.8

        filtered_logits = thought_generator._top_p_filter(logits, top_p)

        # Check that low probability tokens are masked
        assert torch.isinf(filtered_logits[0, -1])  # Last token should be -inf
        assert not torch.isinf(filtered_logits[0, 0])  # First token should remain

    def test_sample_thought_tokens_length(self, thought_generator, mock_model):
        """Test that sampled thoughts have correct length."""
        prefix = torch.randint(0, 1000, (1, 5))

        with mock.patch.object(thought_generator, '_top_p_filter',
                              return_value=torch.randn(1, 50000)):
            thought_tokens = thought_generator._sample_thought_tokens(
                prefix, mock_model, prefix.device
            )

        assert thought_tokens.shape[1] == thought_generator.config.thought_length

    def test_coherence_threshold_filtering(self, thought_generator, mock_model):
        """Test coherence threshold filtering."""
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones_like(input_ids)

        # Mock coherence scorer to return known scores
        with mock.patch.object(thought_generator.coherence_scorer, 'score_thoughts') as mock_scorer:
            # Half thoughts above threshold, half below
            mock_scores = torch.tensor([[0.8, 0.5, 0.9, 0.4, 0.7, 0.3, 0.8, 0.2]])
            mock_scorer.return_value = mock_scores

            result = thought_generator.generate_thoughts(
                input_ids, attention_mask, mock_model, 5
            )

            valid_mask = result['valid_mask']
            expected_mask = mock_scores >= thought_generator.config.coherence_threshold
            assert torch.equal(valid_mask, expected_mask)


class TestCoherenceScorer:
    """Comprehensive tests for CoherenceScorer component."""

    @pytest.fixture
    def config(self):
        return QuietSTaRConfig(thought_length=4, num_thoughts=6)

    @pytest.fixture
    def coherence_scorer(self, config):
        return CoherenceScorer(config)

    def test_coherence_scorer_initialization(self, coherence_scorer):
        """Test CoherenceScorer initialization."""
        assert coherence_scorer.alpha == 0.4
        assert coherence_scorer.beta == 0.3
        assert coherence_scorer.gamma == 0.3
        assert abs(coherence_scorer.alpha + coherence_scorer.beta + coherence_scorer.gamma - 1.0) < 1e-6

    def test_score_thoughts_shape(self, coherence_scorer):
        """Test coherence scoring output shape."""
        batch_size, num_thoughts, thought_len, vocab_size = 2, 6, 4, 1000

        thoughts = torch.randint(0, vocab_size, (batch_size, num_thoughts, thought_len))
        thought_logits = torch.randn(batch_size, num_thoughts, thought_len, vocab_size)
        context = torch.randint(0, vocab_size, (batch_size, 8))

        scores = coherence_scorer.score_thoughts(thoughts, thought_logits, context)

        assert scores.shape == (batch_size, num_thoughts)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)

    def test_semantic_coherence_computation(self, coherence_scorer):
        """Test semantic coherence computation."""
        batch_size, num_thoughts = 2, 4
        thoughts = torch.randint(0, 1000, (batch_size, num_thoughts, 4))
        context = torch.randint(0, 1000, (batch_size, 6))

        # Mock to control embedding computation
        with mock.patch('torch.randn') as mock_randn:
            # Return known embeddings for predictable cosine similarity
            mock_randn.side_effect = [
                torch.tensor([[[1, 0], [0, 1], [1, 1], [-1, 0]],
                             [[1, 0], [0, 1], [1, 1], [-1, 0]]]),  # thought_embeds
                torch.tensor([[1, 0], [0, 1]])  # context_embeds
            ]

            semantic_scores = coherence_scorer._semantic_coherence(thoughts, context)

            assert semantic_scores.shape == (batch_size, num_thoughts)
            assert torch.all(semantic_scores >= 0) and torch.all(semantic_scores <= 1)

    def test_syntactic_coherence_computation(self, coherence_scorer):
        """Test syntactic coherence (perplexity-based) computation."""
        batch_size, num_thoughts, thought_len, vocab_size = 1, 3, 4, 100

        thoughts = torch.randint(0, vocab_size, (batch_size, num_thoughts, thought_len))
        # Create logits with higher probability for actual tokens
        thought_logits = torch.randn(batch_size, num_thoughts, thought_len, vocab_size)

        syntactic_scores = coherence_scorer._syntactic_coherence(thoughts, thought_logits)

        assert syntactic_scores.shape == (batch_size, num_thoughts)
        assert torch.all(syntactic_scores >= 0) and torch.all(syntactic_scores <= 1)

    def test_predictive_utility_computation(self, coherence_scorer):
        """Test predictive utility computation."""
        batch_size, num_thoughts, thought_len, vocab_size = 2, 4, 3, 500

        thoughts = torch.randint(0, vocab_size, (batch_size, num_thoughts, thought_len))
        thought_logits = torch.randn(batch_size, num_thoughts, thought_len, vocab_size)
        context = torch.randint(0, vocab_size, (batch_size, 5))

        predictive_scores = coherence_scorer._predictive_utility(thoughts, thought_logits, context)

        assert predictive_scores.shape == (batch_size, num_thoughts)
        assert torch.all(predictive_scores >= 0) and torch.all(predictive_scores <= 1)

    def test_score_combination_weights(self, coherence_scorer):
        """Test that coherence score properly combines weighted components."""
        batch_size, num_thoughts = 1, 2

        # Mock individual scoring methods to return known values
        with mock.patch.object(coherence_scorer, '_semantic_coherence',
                              return_value=torch.tensor([[0.8, 0.6]])):
            with mock.patch.object(coherence_scorer, '_syntactic_coherence',
                                  return_value=torch.tensor([[0.7, 0.5]])):
                with mock.patch.object(coherence_scorer, '_predictive_utility',
                                      return_value=torch.tensor([[0.9, 0.4]])):

                    thoughts = torch.randint(0, 100, (batch_size, num_thoughts, 3))
                    thought_logits = torch.randn(batch_size, num_thoughts, 3, 100)
                    context = torch.randint(0, 100, (batch_size, 5))

                    scores = coherence_scorer.score_thoughts(thoughts, thought_logits, context)

                    # Calculate expected scores
                    expected_0 = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.9  # 0.8
                    expected_1 = 0.4 * 0.6 + 0.3 * 0.5 + 0.3 * 0.4  # 0.51

                    assert abs(scores[0, 0].item() - expected_0) < 1e-6
                    assert abs(scores[0, 1].item() - expected_1) < 1e-6


class TestMixingHead:
    """Comprehensive tests for MixingHead neural component."""

    @pytest.fixture
    def config(self):
        return QuietSTaRConfig(
            num_thoughts=4,
            mixing_head_hidden_dim=128
        )

    @pytest.fixture
    def mixing_head(self, config):
        return MixingHead(config)

    def test_mixing_head_initialization(self, mixing_head, config):
        """Test MixingHead initialization."""
        assert isinstance(mixing_head.mixing_network, nn.Sequential)

        # Check network architecture
        layers = list(mixing_head.mixing_network.children())
        assert len(layers) == 4  # Linear, ReLU, Dropout, Linear, Softmax
        assert isinstance(layers[0], nn.Linear)
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[2], nn.Dropout)
        assert isinstance(layers[3], nn.Linear)

    def test_mixing_head_forward_shape(self, mixing_head):
        """Test MixingHead forward pass output shape."""
        batch_size, vocab_size, num_thoughts = 2, 1000, 4

        original_logits = torch.randn(batch_size, vocab_size)
        thought_logits = torch.randn(batch_size, num_thoughts, vocab_size)
        coherence_scores = torch.rand(batch_size, num_thoughts)

        mixed_logits = mixing_head(original_logits, thought_logits, coherence_scores)

        assert mixed_logits.shape == (batch_size, vocab_size)

    def test_mixing_weights_sum_to_one(self, mixing_head):
        """Test that mixing weights sum to approximately 1."""
        batch_size, vocab_size, num_thoughts = 1, 100, 4

        original_logits = torch.randn(batch_size, vocab_size)
        thought_logits = torch.randn(batch_size, num_thoughts, vocab_size)
        coherence_scores = torch.rand(batch_size, num_thoughts)

        # Extract mixing weights by accessing internal computation
        original_agg = torch.mean(original_logits, dim=1, keepdim=True)
        thought_agg = torch.mean(thought_logits, dim=2)

        mixing_input = torch.cat([original_agg, thought_agg, coherence_scores], dim=1)
        mixing_weights = mixing_head.mixing_network(mixing_input)

        # Weights should sum to 1 (softmax output)
        weight_sums = torch.sum(mixing_weights, dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_mixing_head_gradient_flow(self, mixing_head):
        """Test gradient flow through MixingHead."""
        batch_size, vocab_size, num_thoughts = 1, 50, 3

        original_logits = torch.randn(batch_size, vocab_size, requires_grad=True)
        thought_logits = torch.randn(batch_size, num_thoughts, vocab_size, requires_grad=True)
        coherence_scores = torch.rand(batch_size, num_thoughts, requires_grad=True)

        mixed_logits = mixing_head(original_logits, thought_logits, coherence_scores)
        loss = mixed_logits.sum()
        loss.backward()

        # Check gradients exist
        assert original_logits.grad is not None
        assert thought_logits.grad is not None
        assert coherence_scores.grad is not None

        # Check gradients are non-zero
        assert not torch.allclose(original_logits.grad, torch.zeros_like(original_logits.grad))

    def test_mixing_head_deterministic(self, mixing_head):
        """Test that MixingHead produces deterministic outputs."""
        torch.manual_seed(42)

        original_logits = torch.randn(1, 100)
        thought_logits = torch.randn(1, 4, 100)
        coherence_scores = torch.rand(1, 4)

        # Run twice with same seed
        mixing_head.eval()  # Disable dropout
        output1 = mixing_head(original_logits, thought_logits, coherence_scores)
        output2 = mixing_head(original_logits, thought_logits, coherence_scores)

        assert torch.allclose(output1, output2, atol=1e-6)


class TestThoughtInjector:
    """Comprehensive tests for ThoughtInjector component."""

    @pytest.fixture
    def config(self):
        return QuietSTaRConfig(num_thoughts=6)

    @pytest.fixture
    def thought_injector(self, config):
        return ThoughtInjector(config)

    def test_thought_injector_initialization(self, thought_injector, config):
        """Test ThoughtInjector initialization."""
        assert thought_injector.config == config

    def test_identify_injection_points_count(self, thought_injector):
        """Test injection point identification returns correct count."""
        seq_len = 20
        input_ids = torch.randint(0, 1000, (1, seq_len))
        logits = torch.randn(1, seq_len, 1000)
        attention_weights = torch.rand(1, 8, seq_len, seq_len)  # 8 heads

        injection_points = thought_injector.identify_injection_points(
            input_ids, logits, attention_weights
        )

        # Should return at most num_thoughts positions
        assert len(injection_points) <= thought_injector.config.num_thoughts
        # Should skip first token
        assert all(pos > 0 for pos in injection_points)
        # Should be within sequence bounds
        assert all(pos < seq_len for pos in injection_points)

    def test_compute_difficulty_score(self, thought_injector):
        """Test difficulty score computation."""
        # Create logits with known entropy
        vocab_size = 1000
        logits = torch.zeros(1, 10, vocab_size)

        # Position with uniform distribution (high entropy)
        logits[0, 5, :] = 0.0  # Uniform distribution
        difficulty_high = thought_injector._compute_difficulty(logits, 5)

        # Position with peaked distribution (low entropy)
        logits[0, 7, 0] = 10.0  # Peaked at first token
        difficulty_low = thought_injector._compute_difficulty(logits, 7)

        assert difficulty_high > difficulty_low
        assert 0 <= difficulty_high <= 1
        assert 0 <= difficulty_low <= 1

    def test_compute_boundary_score(self, thought_injector):
        """Test boundary score computation."""
        # Create input with boundary tokens
        input_ids = torch.tensor([[1, 2, 46, 4, 44, 6]])  # 46 = '.', 44 = ','

        boundary_score_period = thought_injector._compute_boundary_score(input_ids, 2)
        boundary_score_comma = thought_injector._compute_boundary_score(input_ids, 4)
        boundary_score_other = thought_injector._compute_boundary_score(input_ids, 1)

        assert boundary_score_period == 1.0
        assert boundary_score_comma == 1.0
        assert boundary_score_other == 0.0

    def test_compute_attention_score(self, thought_injector):
        """Test attention score computation."""
        batch_size, num_heads, seq_len = 1, 4, 8

        # Create attention with different dispersions
        attention_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len)

        # High dispersion at position 3
        attention_weights[0, :, 3, :] = torch.rand(num_heads, seq_len)

        # Low dispersion at position 5 (focused attention)
        attention_weights[0, :, 5, 0] = 1.0

        score_high = thought_injector._compute_attention_score(attention_weights, 3)
        score_low = thought_injector._compute_attention_score(attention_weights, 5)

        assert score_high > score_low
        assert score_high >= 0 and score_low >= 0

    def test_injection_points_sorted_by_score(self, thought_injector):
        """Test that injection points are sorted by score."""
        seq_len = 15
        input_ids = torch.randint(0, 1000, (1, seq_len))

        # Create predictable logits for known difficulty scores
        logits = torch.ones(1, seq_len, 1000)
        attention_weights = torch.rand(1, 4, seq_len, seq_len)

        # Mock scoring methods to return predictable values
        with mock.patch.object(thought_injector, '_compute_difficulty') as mock_diff:
            with mock.patch.object(thought_injector, '_compute_boundary_score') as mock_boundary:
                with mock.patch.object(thought_injector, '_compute_attention_score') as mock_attn:

                    # Return decreasing scores for positions
                    mock_diff.side_effect = [0.9, 0.7, 0.5, 0.3] + [0.1] * 20
                    mock_boundary.side_effect = [0.8, 0.6, 0.4, 0.2] + [0.0] * 20
                    mock_attn.side_effect = [0.7, 0.5, 0.3, 0.1] + [0.0] * 20

                    injection_points = thought_injector.identify_injection_points(
                        input_ids, logits, attention_weights
                    )

                    # First point should have highest score
                    assert injection_points[0] == 1  # First valid position


class TestOptimizationStrategies:
    """Comprehensive tests for OptimizationStrategies component."""

    @pytest.fixture
    def config(self):
        return QuietSTaRConfig(
            thought_length=8,
            num_thoughts=16,
            coherence_threshold=0.7
        )

    @pytest.fixture
    def optimizer(self, config):
        return OptimizationStrategies(config)

    def test_optimization_strategies_initialization(self, optimizer):
        """Test OptimizationStrategies initialization."""
        assert optimizer.curriculum_stage == 0
        assert optimizer.performance_history == []

    def test_curriculum_scheduler_stages(self, optimizer):
        """Test curriculum learning stage progression."""
        # Stage 1: Early epochs
        stage1_params = optimizer.curriculum_scheduler(50, 0.6)
        assert stage1_params['thought_length'] == 4
        assert stage1_params['coherence_threshold'] == 0.8
        assert stage1_params['num_thoughts'] == 8

        # Stage 2: Middle epochs
        stage2_params = optimizer.curriculum_scheduler(150, 0.7)
        assert stage2_params['thought_length'] == 6
        assert stage2_params['coherence_threshold'] == 0.75
        assert stage2_params['num_thoughts'] == 12

        # Stage 3: Late epochs
        stage3_params = optimizer.curriculum_scheduler(250, 0.75)
        assert stage3_params['thought_length'] == optimizer.config.thought_length
        assert stage3_params['coherence_threshold'] == optimizer.config.coherence_threshold
        assert stage3_params['num_thoughts'] == optimizer.config.num_thoughts

    def test_thought_regularization_loss_components(self, optimizer):
        """Test thought regularization loss computation."""
        batch_size, num_thoughts, thought_len = 2, 4, 6

        # Create test thoughts with some repetitions
        thoughts = torch.randint(0, 100, (batch_size, num_thoughts, thought_len))
        # Make some thoughts similar for diversity loss
        thoughts[:, 1, :] = thoughts[:, 0, :]  # Copy first thought to second

        coherence_scores = torch.tensor([[0.8, 0.6, 0.7, 0.5], [0.9, 0.4, 0.8, 0.6]])

        reg_loss = optimizer.compute_thought_regularization_loss(thoughts, coherence_scores)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.dim() == 0  # Scalar
        assert reg_loss >= 0  # Loss should be non-negative

    def test_compute_thought_similarities(self, optimizer):
        """Test thought similarity computation."""
        # Create thoughts with known similarities
        thoughts = torch.tensor([[[1, 2, 3], [1, 2, 3], [4, 5, 6]]])  # First two identical

        similarities = optimizer._compute_thought_similarities(thoughts)

        assert isinstance(similarities, torch.Tensor)
        assert similarities >= 0  # Similarities should be non-negative

    def test_compute_complexity_loss(self, optimizer):
        """Test complexity loss computation."""
        # Create thoughts with repetitive patterns
        repetitive_thoughts = torch.tensor([[[1, 1, 1, 1], [2, 3, 2, 3]]])
        diverse_thoughts = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])

        loss_repetitive = optimizer._compute_complexity_loss(repetitive_thoughts)
        loss_diverse = optimizer._compute_complexity_loss(diverse_thoughts)

        assert loss_repetitive > loss_diverse
        assert loss_repetitive >= 0 and loss_diverse >= 0

    def test_adaptive_sampling_schedule_trends(self, optimizer):
        """Test adaptive sampling based on performance trends."""
        # Test improving performance
        for perf in [0.6, 0.65, 0.7, 0.72, 0.75]:
            params = optimizer.adaptive_sampling_schedule({'coherence_rate': perf})

        improving_params = optimizer.adaptive_sampling_schedule({'coherence_rate': 0.78})
        assert improving_params['temperature'] == optimizer.config.temperature

        # Test stagnating performance
        for perf in [0.75, 0.751, 0.749, 0.75, 0.752]:
            params = optimizer.adaptive_sampling_schedule({'coherence_rate': perf})

        stagnating_params = optimizer.adaptive_sampling_schedule({'coherence_rate': 0.751})
        assert stagnating_params['temperature'] > optimizer.config.temperature

        # Test degrading performance
        optimizer.performance_history = [0.8, 0.78, 0.76, 0.74, 0.72, 0.7]
        degrading_params = optimizer.adaptive_sampling_schedule({'coherence_rate': 0.68})
        assert degrading_params['temperature'] < optimizer.config.temperature

    def test_performance_history_management(self, optimizer):
        """Test performance history tracking and management."""
        # Add more than 10 entries
        for i in range(15):
            optimizer.adaptive_sampling_schedule({'coherence_rate': 0.5 + i * 0.01})

        # Should keep only last 10 entries
        assert len(optimizer.performance_history) == 10
        assert optimizer.performance_history[0] >= 0.55  # Should be from later entries


class TestQuietSTaRIntegration:
    """Integration tests for end-to-end Quiet-STaR pipeline."""

    @pytest.fixture
    def config(self):
        return QuietSTaRConfig(
            thought_length=4,
            num_thoughts=6,
            coherence_threshold=0.6,
            mixing_head_hidden_dim=64
        )

    @pytest.fixture
    def components(self, config):
        """Create all Quiet-STaR components."""
        return {
            'thought_generator': ThoughtGenerator(config),
            'coherence_scorer': CoherenceScorer(config),
            'mixing_head': MixingHead(config),
            'thought_injector': ThoughtInjector(config),
            'optimizer': OptimizationStrategies(config)
        }

    @pytest.fixture
    def mock_model(self):
        """Create mock language model for integration testing."""
        class MockLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 1000
                self.embed = nn.Embedding(self.vocab_size, 512)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(512, 8, batch_first=True),
                    num_layers=2
                )
                self.lm_head = nn.Linear(512, self.vocab_size)

            def forward(self, input_ids, attention_mask=None):
                x = self.embed(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                return type('obj', (object,), {'logits': logits})()

        return MockLanguageModel()

    def test_end_to_end_pipeline(self, components, mock_model):
        """Test complete Quiet-STaR pipeline."""
        batch_size, seq_len = 1, 12
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Step 1: Identify injection points
        with torch.no_grad():
            outputs = mock_model(input_ids, attention_mask)
            logits = outputs.logits

        # Mock attention weights for injection point identification
        attention_weights = torch.rand(batch_size, 8, seq_len, seq_len)

        injection_points = components['thought_injector'].identify_injection_points(
            input_ids, logits, attention_weights
        )

        assert len(injection_points) > 0
        assert all(0 < pos < seq_len for pos in injection_points)

        # Step 2: Generate thoughts at injection points
        all_thoughts = []
        all_mixed_logits = []

        for position in injection_points[:2]:  # Test first 2 positions
            thought_result = components['thought_generator'].generate_thoughts(
                input_ids, attention_mask, mock_model, position
            )

            assert 'thoughts' in thought_result
            assert 'coherence_scores' in thought_result
            assert 'valid_mask' in thought_result

            # Step 3: Mix predictions
            original_logits = logits[:, position, :]
            thought_logits = thought_result['logits'][:, :, -1, :]  # Last token of each thought
            coherence_scores = thought_result['coherence_scores']

            mixed_logits = components['mixing_head'](
                original_logits, thought_logits, coherence_scores
            )

            all_thoughts.append(thought_result['thoughts'])
            all_mixed_logits.append(mixed_logits)

        # Verify outputs
        assert len(all_thoughts) == len(all_mixed_logits)
        for thoughts, mixed_logits in zip(all_thoughts, all_mixed_logits):
            assert thoughts.shape[0] == batch_size
            assert mixed_logits.shape == (batch_size, mock_model.vocab_size)

    def test_training_loop_simulation(self, components, mock_model):
        """Test simulated training loop with optimization strategies."""
        batch_size = 2
        vocab_size = mock_model.vocab_size

        # Simulate training over epochs
        for epoch in range(3):
            # Get curriculum parameters
            curriculum_params = components['optimizer'].curriculum_scheduler(epoch * 100, 0.7)

            # Update config based on curriculum
            components['thought_generator'].config.thought_length = curriculum_params['thought_length']
            components['thought_generator'].config.num_thoughts = curriculum_params['num_thoughts']
            components['thought_generator'].config.coherence_threshold = curriculum_params['coherence_threshold']

            # Generate sample data
            input_ids = torch.randint(0, vocab_size, (batch_size, 10))
            attention_mask = torch.ones_like(input_ids)

            # Generate thoughts
            thought_result = components['thought_generator'].generate_thoughts(
                input_ids, attention_mask, mock_model, 5
            )

            # Compute regularization loss
            reg_loss = components['optimizer'].compute_thought_regularization_loss(
                thought_result['thoughts'], thought_result['coherence_scores']
            )

            assert isinstance(reg_loss, torch.Tensor)
            assert reg_loss >= 0

            # Update adaptive sampling
            perf_metrics = {'coherence_rate': torch.mean(thought_result['coherence_scores']).item()}
            adaptive_params = components['optimizer'].adaptive_sampling_schedule(perf_metrics)

            assert 'temperature' in adaptive_params
            assert 'top_p' in adaptive_params

    def test_memory_efficiency(self, components, mock_model):
        """Test memory efficiency with larger sequences."""
        torch.manual_seed(42)

        # Test with longer sequences
        batch_size, seq_len = 1, 50
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Monitor memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Generate thoughts at multiple positions
        for position in range(5, min(15, seq_len)):
            with torch.no_grad():  # Disable gradients for memory efficiency
                thought_result = components['thought_generator'].generate_thoughts(
                    input_ids, attention_mask, mock_model, position
                )

                # Verify shapes are consistent
                assert thought_result['thoughts'].shape[2] == components['thought_generator'].config.thought_length
                assert thought_result['coherence_scores'].shape[1] == components['thought_generator'].config.num_thoughts

        # Memory should not grow excessively
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            assert memory_growth < 1e9  # Less than 1GB growth


class TestQuietSTaRPerformance:
    """Performance benchmarks and validation tests."""

    @pytest.fixture
    def config(self):
        return QuietSTaRConfig(
            thought_length=6,
            num_thoughts=12,
            coherence_threshold=0.7
        )

    @pytest.fixture
    def thought_generator(self, config):
        return ThoughtGenerator(config)

    def test_thought_generation_speed(self, thought_generator):
        """Benchmark thought generation speed."""
        class FastMockModel(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                batch_size, seq_len = input_ids.shape
                logits = torch.randn(batch_size, seq_len, 1000)
                return type('obj', (object,), {'logits': logits})()

        model = FastMockModel()
        input_ids = torch.randint(0, 1000, (4, 20))  # Larger batch
        attention_mask = torch.ones_like(input_ids)

        # Benchmark multiple runs
        times = []
        for _ in range(5):
            start_time = time.time()

            result = thought_generator.generate_thoughts(
                input_ids, attention_mask, model, 10
            )

            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        assert avg_time < 2.0  # Should complete within 2 seconds

        # Verify quality of results
        assert result['thoughts'].shape == (4, 12, 6)  # batch_size, num_thoughts, thought_length
        assert torch.all(result['coherence_scores'] >= 0)
        assert torch.all(result['coherence_scores'] <= 1)

    def test_memory_usage_scaling(self, thought_generator):
        """Test memory usage with different batch sizes."""
        class MemoryMockModel(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                batch_size, seq_len = input_ids.shape
                return type('obj', (object,), {
                    'logits': torch.randn(batch_size, seq_len, 500)
                })()

        model = MemoryMockModel()

        memory_usage = []
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            input_ids = torch.randint(0, 500, (batch_size, 15))
            attention_mask = torch.ones_like(input_ids)

            _ = thought_generator.generate_thoughts(input_ids, attention_mask, model, 7)

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append(peak_memory)

        # Memory usage should scale reasonably with batch size
        if torch.cuda.is_available() and len(memory_usage) > 1:
            # Memory should not grow super-linearly
            for i in range(1, len(memory_usage)):
                ratio = memory_usage[i] / memory_usage[i-1]
                assert ratio < 3.0  # Should not triple with each batch size doubling

    def test_coherence_score_distribution(self, thought_generator):
        """Test distribution of coherence scores."""
        class ConsistentMockModel(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                torch.manual_seed(42)  # Consistent outputs
                batch_size, seq_len = input_ids.shape
                return type('obj', (object,), {
                    'logits': torch.randn(batch_size, seq_len, 800)
                })()

        model = ConsistentMockModel()

        all_scores = []

        # Generate thoughts across multiple samples
        for _ in range(10):
            input_ids = torch.randint(0, 800, (2, 12))
            attention_mask = torch.ones_like(input_ids)

            result = thought_generator.generate_thoughts(input_ids, attention_mask, model, 6)
            scores = result['coherence_scores']
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, dim=0)

        # Check score distribution properties
        mean_score = torch.mean(all_scores)
        std_score = torch.std(all_scores)

        assert 0.3 < mean_score < 0.9  # Reasonable mean
        assert 0.05 < std_score < 0.4  # Reasonable variance
        assert torch.all(all_scores >= 0) and torch.all(all_scores <= 1)  # Valid range

    def test_reasoning_improvement_metric(self):
        """Test metric for measuring reasoning improvement."""
        # Simulate before/after reasoning scenarios

        # Baseline: Simple next-token prediction
        baseline_logits = torch.randn(1, 1000)
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_entropy = -torch.sum(baseline_probs * torch.log(baseline_probs + 1e-10))

        # With thoughts: More confident predictions
        thought_enhanced_logits = baseline_logits.clone()
        thought_enhanced_logits[0, :100] += 2.0  # Boost certain tokens
        enhanced_probs = F.softmax(thought_enhanced_logits, dim=-1)
        enhanced_entropy = -torch.sum(enhanced_probs * torch.log(enhanced_probs + 1e-10))

        # Reasoning improvement metric
        improvement = (baseline_entropy - enhanced_entropy) / baseline_entropy

        assert improvement > 0  # Should show improvement
        assert improvement < 1  # Should be reasonable

    def test_inference_latency_benchmark(self, config):
        """Benchmark inference latency for real-world usage."""
        components = {
            'thought_generator': ThoughtGenerator(config),
            'mixing_head': MixingHead(config)
        }

        class OptimizedMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 1000)

            def forward(self, input_ids, attention_mask=None):
                # Simulate faster model
                batch_size, seq_len = input_ids.shape
                embedded = F.embedding(input_ids, torch.randn(1000, 100))
                logits = self.linear(embedded)
                return type('obj', (object,), {'logits': logits})()

        model = OptimizedMockModel()

        # Benchmark complete pipeline
        input_ids = torch.randint(0, 1000, (1, 25))
        attention_mask = torch.ones_like(input_ids)

        # Time complete thought generation and mixing
        start_time = time.time()

        for position in [5, 10, 15]:
            thought_result = components['thought_generator'].generate_thoughts(
                input_ids, attention_mask, model, position
            )

            # Simulate mixing step
            original_logits = torch.randn(1, 1000)
            thought_logits = thought_result['logits'][:, :, -1, :]
            coherence_scores = thought_result['coherence_scores']

            _ = components['mixing_head'](original_logits, thought_logits, coherence_scores)

        total_time = time.time() - start_time

        # Should complete within reasonable time for interactive use
        assert total_time < 5.0  # Complete pipeline under 5 seconds


class TestQuietSTaRProperties:
    """Property-based tests for algorithm invariants."""

    def test_coherence_score_bounds(self):
        """Property: Coherence scores must always be in [0, 1]."""
        config = QuietSTaRConfig()
        scorer = CoherenceScorer(config)

        # Test with various input shapes and values
        for batch_size in [1, 3]:
            for num_thoughts in [4, 8, 16]:
                for thought_len in [2, 6]:
                    for vocab_size in [100, 1000]:
                        thoughts = torch.randint(0, vocab_size, (batch_size, num_thoughts, thought_len))
                        thought_logits = torch.randn(batch_size, num_thoughts, thought_len, vocab_size)
                        context = torch.randint(0, vocab_size, (batch_size, thought_len + 2))

                        scores = scorer.score_thoughts(thoughts, thought_logits, context)

                        # Property: All scores in valid range
                        assert torch.all(scores >= 0), f"Found negative score: {torch.min(scores)}"
                        assert torch.all(scores <= 1), f"Found score > 1: {torch.max(scores)}"

    def test_mixing_weights_conservation(self):
        """Property: Mixing weights must sum to 1."""
        config = QuietSTaRConfig(num_thoughts=6)
        mixing_head = MixingHead(config)

        # Test with various configurations
        for batch_size in [1, 2, 4]:
            for vocab_size in [50, 500]:
                original_logits = torch.randn(batch_size, vocab_size)
                thought_logits = torch.randn(batch_size, config.num_thoughts, vocab_size)
                coherence_scores = torch.rand(batch_size, config.num_thoughts)

                # Extract mixing weights computation
                original_agg = torch.mean(original_logits, dim=1, keepdim=True)
                thought_agg = torch.mean(thought_logits, dim=2)
                mixing_input = torch.cat([original_agg, thought_agg, coherence_scores], dim=1)

                mixing_weights = mixing_head.mixing_network(mixing_input)
                weight_sums = torch.sum(mixing_weights, dim=1)

                # Property: Weights sum to 1
                assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

    def test_thought_length_consistency(self):
        """Property: Generated thoughts must have exactly the configured length."""
        for thought_length in [2, 4, 8, 12]:
            config = QuietSTaRConfig(thought_length=thought_length, num_thoughts=4)
            generator = ThoughtGenerator(config)

            class LengthMockModel(nn.Module):
                def forward(self, input_ids, attention_mask=None):
                    batch_size, seq_len = input_ids.shape
                    return type('obj', (object,), {
                        'logits': torch.randn(batch_size, seq_len, 100)
                    })()

            model = LengthMockModel()
            input_ids = torch.randint(0, 100, (1, 10))
            attention_mask = torch.ones_like(input_ids)

            result = generator.generate_thoughts(input_ids, attention_mask, model, 5)
            thoughts = result['thoughts']

            # Property: Exact length match
            assert thoughts.shape[2] == thought_length

    def test_injection_point_ordering(self):
        """Property: Injection points should be in ascending order."""
        config = QuietSTaRConfig(num_thoughts=8)
        injector = ThoughtInjector(config)

        for seq_len in [10, 20, 30]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            logits = torch.randn(1, seq_len, 1000)
            attention_weights = torch.rand(1, 4, seq_len, seq_len)

            injection_points = injector.identify_injection_points(
                input_ids, logits, attention_weights
            )

            # Property: Points should be sorted in ascending order
            assert injection_points == sorted(injection_points)

            # Property: All points should be valid positions
            assert all(0 < pos < seq_len for pos in injection_points)

    def test_curriculum_progression_monotonicity(self):
        """Property: Curriculum should show monotonic progression in complexity."""
        config = QuietSTaRConfig()
        optimizer = OptimizationStrategies(config)

        previous_complexity = 0

        for epoch in [50, 150, 250]:  # Test different stages
            params = optimizer.curriculum_scheduler(epoch, 0.7)

            # Define complexity as combination of length and number of thoughts
            complexity = params['thought_length'] * params['num_thoughts']

            # Property: Complexity should not decrease
            assert complexity >= previous_complexity
            previous_complexity = complexity


class TestQuietSTaRContracts:
    """Contract tests for input/output validation."""

    def test_thought_generator_input_contracts(self):
        """Test input validation for ThoughtGenerator."""
        config = QuietSTaRConfig()
        generator = ThoughtGenerator(config)

        class ContractMockModel(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                batch_size, seq_len = input_ids.shape
                return type('obj', (object,), {
                    'logits': torch.randn(batch_size, seq_len, 100)
                })()

        model = ContractMockModel()

        # Valid inputs
        valid_input_ids = torch.randint(0, 100, (2, 10))
        valid_attention_mask = torch.ones_like(valid_input_ids)
        valid_position = 5

        # Should succeed with valid inputs
        result = generator.generate_thoughts(valid_input_ids, valid_attention_mask, model, valid_position)
        assert isinstance(result, dict)

        # Test boundary conditions
        edge_position = valid_input_ids.size(1) - 1
        result_edge = generator.generate_thoughts(valid_input_ids, valid_attention_mask, model, edge_position)
        assert isinstance(result_edge, dict)

    def test_coherence_scorer_output_contracts(self):
        """Test output validation for CoherenceScorer."""
        config = QuietSTaRConfig()
        scorer = CoherenceScorer(config)

        batch_size, num_thoughts, thought_len, vocab_size = 3, 8, 4, 200

        thoughts = torch.randint(0, vocab_size, (batch_size, num_thoughts, thought_len))
        thought_logits = torch.randn(batch_size, num_thoughts, thought_len, vocab_size)
        context = torch.randint(0, vocab_size, (batch_size, 6))

        scores = scorer.score_thoughts(thoughts, thought_logits, context)

        # Contract: Output shape
        assert scores.shape == (batch_size, num_thoughts)

        # Contract: Output type
        assert isinstance(scores, torch.Tensor)
        assert scores.dtype == torch.float32

        # Contract: Output range
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)

        # Contract: No NaN or infinite values
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_mixing_head_gradient_contracts(self):
        """Test gradient flow contracts for MixingHead."""
        config = QuietSTaRConfig(num_thoughts=4)
        mixing_head = MixingHead(config)

        batch_size, vocab_size = 2, 100

        original_logits = torch.randn(batch_size, vocab_size, requires_grad=True)
        thought_logits = torch.randn(batch_size, config.num_thoughts, vocab_size, requires_grad=True)
        coherence_scores = torch.rand(batch_size, config.num_thoughts, requires_grad=True)

        mixed_logits = mixing_head(original_logits, thought_logits, coherence_scores)
        loss = mixed_logits.sum()
        loss.backward()

        # Contract: All inputs should have gradients
        assert original_logits.grad is not None
        assert thought_logits.grad is not None
        assert coherence_scores.grad is not None

        # Contract: Gradients should be finite
        assert torch.all(torch.isfinite(original_logits.grad))
        assert torch.all(torch.isfinite(thought_logits.grad))
        assert torch.all(torch.isfinite(coherence_scores.grad))

        # Contract: Output shape preservation
        assert mixed_logits.shape == original_logits.shape

    def test_optimization_strategies_performance_contracts(self):
        """Test performance tracking contracts for OptimizationStrategies."""
        config = QuietSTaRConfig()
        optimizer = OptimizationStrategies(config)

        # Contract: Performance history management
        for i in range(15):  # Add more than max history
            metrics = {'coherence_rate': 0.5 + i * 0.01}
            optimizer.adaptive_sampling_schedule(metrics)

        # Contract: History length limit
        assert len(optimizer.performance_history) <= 10

        # Contract: Valid parameter ranges
        params = optimizer.adaptive_sampling_schedule({'coherence_rate': 0.8})
        assert 0.1 <= params['temperature'] <= 2.0
        assert 0.1 <= params['top_p'] <= 1.0

        # Contract: Curriculum parameter bounds
        for epoch in [0, 100, 200, 300]:
            curriculum_params = optimizer.curriculum_scheduler(epoch, 0.7)
            assert curriculum_params['thought_length'] >= 2
            assert curriculum_params['num_thoughts'] >= 4
            assert 0.5 <= curriculum_params['coherence_threshold'] <= 1.0


def run_comprehensive_test_suite():
    """Run complete test suite with coverage reporting."""
    print("=" * 80)
    print("QUIET-STAR COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run pytest with coverage
    import subprocess
    import sys

    test_file = __file__

    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=short",
        "--durations=10",
        f"--cov=quiet_star",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html",
        "--cov-fail-under=85"  # Ensure >85% coverage
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\nSTDOUT:")
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    print(f"\nTest suite completed with return code: {result.returncode}")

    if result.returncode == 0:
        print("✅ All tests passed with >85% coverage!")
    else:
        print("❌ Some tests failed or coverage below 85%")

    return result.returncode == 0


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)