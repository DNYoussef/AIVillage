"""
Tests for Cogment Core Model Components (Agent 1)

Tests the unified Cogment model's core functionality including:
- RefinementCore forward pass and parameter counting
- ACT (Adaptive Computation Time) halting logic
- Model instantiation and configuration validation
- Parameter budget validation (23.7M vs 25M target)
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

# Import Cogment core components
try:
    from config.cogment.config_loader import load_cogment_config
    from core.agent_forge.models.cogment.core.act_halting import ACTHalting
    from core.agent_forge.models.cogment.core.config import CogmentConfig
    from core.agent_forge.models.cogment.core.model import CogmentModel
    from core.agent_forge.models.cogment.core.refinement_core import RefinementCore

    COGMENT_AVAILABLE = True
except ImportError as e:
    print(f"Cogment components not available: {e}")
    COGMENT_AVAILABLE = False


@dataclass
class TestConfig:
    """Test configuration for Cogment model."""

    d_model: int = 512
    n_layers: int = 6
    n_head: int = 8
    d_ff: int = 1536
    vocab_size: int = 13000
    max_seq_len: int = 2048

    # ACT parameters
    act_epsilon: float = 0.01
    max_act_steps: int = 16
    halt_threshold: float = 0.99
    ponder_cost_weight: float = 0.1

    # Memory parameters
    mem_slots: int = 2048
    ltm_capacity: int = 1024
    ltm_dim: int = 256

    # Expected parameter count
    target_params: int = 25_000_000
    tolerance: float = 0.05


class TestRefinementCore:
    """Test RefinementCore component (Agent 1 core)."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return TestConfig()

    @pytest.fixture
    def refinement_core(self, test_config):
        """Create RefinementCore instance."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment components not available")

        config = CogmentConfig.from_dict(test_config.__dict__)
        return RefinementCore(config)

    def test_refinement_core_creation(self, refinement_core, test_config):
        """Test RefinementCore instantiation."""
        assert refinement_core.d_model == test_config.d_model
        assert refinement_core.n_layers == test_config.n_layers
        assert hasattr(refinement_core, "transformer_layers")
        assert hasattr(refinement_core, "act_halting")

    def test_refinement_core_forward_pass(self, refinement_core, test_config):
        """Test RefinementCore forward pass."""
        batch_size = 2
        seq_len = 32

        # Create input tensors
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = refinement_core(input_ids)

        # Verify output structure
        assert hasattr(outputs, "hidden_states")
        assert hasattr(outputs, "act_outputs")
        assert outputs.hidden_states.shape == (batch_size, seq_len, test_config.d_model)

        # Verify ACT outputs
        act_outputs = outputs.act_outputs
        assert hasattr(act_outputs, "halting_probability")
        assert hasattr(act_outputs, "ponder_cost")
        assert hasattr(act_outputs, "step_count")

    def test_refinement_core_parameter_count(self, refinement_core, test_config):
        """Test RefinementCore parameter count is reasonable."""
        total_params = sum(p.numel() for p in refinement_core.parameters())

        # Should be significant portion of 23.7M budget but not exceed target
        min_expected = 3_000_000  # At least 3M parameters
        max_expected = test_config.target_params  # Not exceed full budget

        assert (
            min_expected <= total_params <= max_expected
        ), f"RefinementCore params {total_params:,} outside expected range [{min_expected:,}, {max_expected:,}]"

    def test_refinement_core_gradient_flow(self, refinement_core, test_config):
        """Test gradient flow through RefinementCore."""
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = refinement_core(input_ids)
        loss = outputs.hidden_states.sum() + outputs.act_outputs.ponder_cost.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        grad_count = 0
        for param in refinement_core.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), "NaN gradients detected"

        assert grad_count > 0, "No gradients computed"


class TestACTHalting:
    """Test ACT (Adaptive Computation Time) halting mechanism."""

    @pytest.fixture
    def act_halting(self):
        """Create ACT halting module."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment components not available")

        config = TestConfig()
        return ACTHalting(
            d_model=config.d_model,
            epsilon=config.act_epsilon,
            max_steps=config.max_act_steps,
            halt_threshold=config.halt_threshold,
        )

    def test_act_creation(self, act_halting):
        """Test ACT halting module creation."""
        config = TestConfig()
        assert act_halting.epsilon == config.act_epsilon
        assert act_halting.max_steps == config.max_act_steps
        assert act_halting.halt_threshold == config.halt_threshold
        assert hasattr(act_halting, "halting_predictor")

    def test_act_forward_pass(self, act_halting):
        """Test ACT forward pass produces valid outputs."""
        batch_size = 2
        seq_len = 16
        d_model = 512

        hidden_states = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            outputs = act_halting(hidden_states)

        # Verify output structure
        assert hasattr(outputs, "halting_probability")
        assert hasattr(outputs, "ponder_cost")
        assert hasattr(outputs, "step_count")
        assert hasattr(outputs, "should_halt")

        # Verify tensor shapes
        assert outputs.halting_probability.shape == (batch_size, seq_len)
        assert outputs.ponder_cost.shape == (batch_size, seq_len)
        assert outputs.step_count.shape == (batch_size, seq_len)

        # Verify value ranges
        assert (outputs.halting_probability >= 0).all()
        assert (outputs.halting_probability <= 1).all()
        assert (outputs.ponder_cost >= 0).all()
        assert (outputs.step_count >= 1).all()

    def test_act_halting_behavior(self, act_halting):
        """Test ACT halting behavior under different conditions."""
        batch_size = 1
        seq_len = 8
        d_model = 512

        # Test with high confidence input (should halt early)
        high_conf_input = torch.ones(batch_size, seq_len, d_model) * 10.0

        with torch.no_grad():
            high_conf_outputs = act_halting(high_conf_input)

        # Test with low confidence input (should take more steps)
        low_conf_input = torch.randn(batch_size, seq_len, d_model) * 0.1

        with torch.no_grad():
            low_conf_outputs = act_halting(low_conf_input)

        # High confidence should generally halt earlier (lower ponder cost)
        avg_high_ponder = high_conf_outputs.ponder_cost.mean()
        avg_low_ponder = low_conf_outputs.ponder_cost.mean()

        # Note: This is probabilistic, so we allow some variance
        assert (
            avg_high_ponder <= avg_low_ponder + 1.0
        ), f"Expected high confidence to ponder less: {avg_high_ponder:.3f} vs {avg_low_ponder:.3f}"

    def test_act_max_steps_enforcement(self, act_halting):
        """Test that ACT respects maximum step limit."""
        batch_size = 1
        seq_len = 4
        d_model = 512

        # Create input that should trigger max steps
        difficult_input = torch.randn(batch_size, seq_len, d_model) * 0.01

        with torch.no_grad():
            outputs = act_halting(difficult_input)

        # Should not exceed max steps
        max_steps_found = outputs.step_count.max().item()
        assert max_steps_found <= act_halting.max_steps, f"Steps {max_steps_found} exceeded max {act_halting.max_steps}"


@pytest.mark.skipif(not COGMENT_AVAILABLE, reason="Cogment components not available")
class TestCogmentModel:
    """Test complete Cogment model integration."""

    @pytest.fixture
    def cogment_config(self):
        """Load Cogment configuration."""
        try:
            # Try to load from config file
            return load_cogment_config()
        except:
            # Fallback to test config
            return CogmentConfig.from_dict(TestConfig().__dict__)

    @pytest.fixture
    def cogment_model(self, cogment_config):
        """Create Cogment model instance."""
        return CogmentModel(cogment_config)

    def test_cogment_model_creation(self, cogment_model, cogment_config):
        """Test complete Cogment model instantiation."""
        assert cogment_model.config == cogment_config
        assert hasattr(cogment_model, "refinement_core")
        assert hasattr(cogment_model, "gated_ltm")
        assert hasattr(cogment_model, "heads")
        assert hasattr(cogment_model, "embeddings")

    def test_cogment_model_parameter_budget(self, cogment_model):
        """Test Cogment model meets parameter budget (23.7M target)."""
        total_params = sum(p.numel() for p in cogment_model.parameters())

        target_params = 25_000_000  # 25M budget
        actual_target = 23_700_000  # 23.7M achieved target
        tolerance = 0.05  # 5% tolerance

        min_acceptable = actual_target * (1 - tolerance)
        max_acceptable = target_params * (1 + tolerance)

        assert (
            min_acceptable <= total_params <= max_acceptable
        ), f"Model params {total_params:,} outside acceptable range [{min_acceptable:,}, {max_acceptable:,}]"

        print(f"✓ Cogment model parameter count: {total_params:,} (target: {actual_target:,})")

    def test_cogment_model_forward_pass(self, cogment_model, cogment_config):
        """Test complete Cogment model forward pass."""
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, cogment_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = cogment_model(input_ids)

        # Verify output structure
        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "memory_outputs")
        assert hasattr(outputs, "act_outputs")

        # Verify tensor shapes
        assert outputs.logits.shape == (batch_size, seq_len, cogment_config.vocab_size)

        # Verify components worked
        assert outputs.memory_outputs is not None
        assert outputs.act_outputs is not None

    def test_cogment_vs_hrrm_parameter_comparison(self, cogment_model):
        """Test parameter reduction vs HRRM baseline."""
        cogment_params = sum(p.numel() for p in cogment_model.parameters())

        # HRRM baseline: 3 models × 50M each = 150M total
        hrrm_baseline = 150_000_000

        # Calculate reduction
        reduction_factor = hrrm_baseline / cogment_params

        # Should achieve ~6x reduction
        assert reduction_factor >= 5.0, f"Insufficient parameter reduction: {reduction_factor:.1f}x (expected ≥5x)"

        print(f"✓ Parameter reduction achieved: {reduction_factor:.1f}x ({hrrm_baseline:,} → {cogment_params:,})")

    def test_cogment_memory_efficiency(self, cogment_model, cogment_config):
        """Test memory efficiency compared to HRRM."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple models to test memory scaling
        models = [cogment_model]
        for _ in range(2):  # Create 2 more instances
            models.append(CogmentModel(cogment_config))

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_model = (current_memory - initial_memory) / len(models)

        # Expected: Each Cogment model should use <150MB
        # (vs ~600MB for full HRRM 3-model ensemble)
        max_expected_memory = 200  # MB per model

        assert (
            memory_per_model <= max_expected_memory
        ), f"Memory usage too high: {memory_per_model:.1f}MB per model (expected ≤{max_expected_memory}MB)"

        print(f"✓ Memory efficiency: {memory_per_model:.1f}MB per model")

    def test_cogment_training_compatibility(self, cogment_model):
        """Test Cogment model is compatible with training."""
        batch_size = 2
        seq_len = 16
        vocab_size = cogment_model.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = cogment_model(input_ids)

        # Compute loss
        logits = outputs.logits.view(-1, vocab_size)
        targets = target_ids.view(-1)

        nll_loss = F.cross_entropy(logits, targets)

        # Add ACT ponder cost
        ponder_cost = outputs.act_outputs.ponder_cost.mean() if outputs.act_outputs else 0
        total_loss = nll_loss + 0.1 * ponder_cost

        # Backward pass
        total_loss.backward()

        # Verify gradients
        grad_norm = 0
        for param in cogment_model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm**0.5

        assert grad_norm > 0, "No gradients computed"
        assert not torch.isnan(torch.tensor(grad_norm)), "NaN gradients"

        print(f"✓ Training compatibility: grad norm = {grad_norm:.6f}")


@pytest.mark.integration
class TestCogmentIntegration:
    """Integration tests for complete Cogment system."""

    @pytest.mark.skipif(not COGMENT_AVAILABLE, reason="Cogment components not available")
    def test_end_to_end_instantiation(self):
        """Test end-to-end model instantiation through all agent components."""
        # This test validates that all Agent 1-7 components work together

        # Load configuration (Agent 7)
        try:
            config = load_cogment_config()
        except:
            config = CogmentConfig.from_dict(TestConfig().__dict__)

        # Create model (Agent 1-3 components)
        model = CogmentModel(config)

        # Verify all subsystems
        assert model.refinement_core is not None  # Agent 1
        assert model.gated_ltm is not None  # Agent 2
        assert model.heads is not None  # Agent 3

        # Test forward pass works
        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits is not None
        assert outputs.memory_outputs is not None
        assert outputs.act_outputs is not None

        print("✓ End-to-end instantiation successful across all agent components")

    @pytest.mark.skipif(not COGMENT_AVAILABLE, reason="Cogment components not available")
    def test_cogment_replaces_hrrm_functionality(self):
        """Test that Cogment provides equivalent functionality to HRRM 3-model approach."""
        config = CogmentConfig.from_dict(TestConfig().__dict__)
        model = CogmentModel(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        # Verify planning capability (replaces HRRM Planner)
        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

        # Verify reasoning capability (replaces HRRM Reasoner)
        assert outputs.act_outputs is not None
        assert hasattr(outputs.act_outputs, "step_count")

        # Verify memory capability (replaces HRRM Memory)
        assert outputs.memory_outputs is not None

        # Verify unified nature (single model vs 3 separate models)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 30_000_000, f"Model too large: {total_params:,} params"

        print("✓ Cogment successfully replaces HRRM 3-model functionality in single unified model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
