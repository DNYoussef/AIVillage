#!/usr/bin/env python3
"""
Test Suite for Cognate Model Core

This module tests the core Cognate model implementation including:
- Model architecture and parameter counting
- Forward pass functionality
- ACT halting behavior
- Memory system integration
- Save/load functionality
"""

import logging
import os
from pathlib import Path

# Import the canonical Cognate implementation
import sys
import tempfile
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cognate_model import CognateConfig, CognateModel, create_cognate_model

from config.cognate_config import CognateModelConfig

logger = logging.getLogger(__name__)


class TestCognateConfig(unittest.TestCase):
    """Test Cognate configuration system."""

    def setUp(self):
        self.config = CognateConfig()

    def test_default_config(self):
        """Test default configuration values."""
        self.assertEqual(self.config.vocab_size, 32000)
        self.assertEqual(self.config.d_model, 216)
        self.assertEqual(self.config.n_layers, 11)
        self.assertEqual(self.config.n_heads, 4)
        self.assertEqual(self.config.ffn_mult, 4)

        # Check derived values
        self.assertEqual(self.config.d_model // self.config.n_heads, 54)  # head_dim

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        valid_config = CognateConfig(d_model=256, n_heads=8)
        self.assertEqual(valid_config.d_model // valid_config.n_heads, 32)

        # Invalid config should raise
        with self.assertRaises(ValueError):
            # d_model not divisible by n_heads
            CognateConfig(d_model=256, n_heads=7)

    def test_parameter_estimation(self):
        """Test parameter count estimation."""
        config = CognateModelConfig()
        estimates = config.estimate_parameters()

        # Should be close to target 25M
        target = 25_069_534
        estimated = estimates["total"]
        error_pct = abs(estimated - target) / target * 100

        self.assertLess(error_pct, 5.0, f"Parameter estimate {estimated} is {error_pct:.1f}% off target {target}")


class TestCognateModel(unittest.TestCase):
    """Test Cognate model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CognateConfig(
            # Smaller model for faster testing
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            n_heads=4,
            max_seq_len=256,
            mem_capacity=64,
        )
        self.model = CognateModel(self.config)
        self.batch_size = 2
        self.seq_len = 32

        # Test inputs
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
        self.labels = torch.randint(0, 1000, (self.batch_size, self.seq_len))

    def test_model_creation(self):
        """Test model creation and parameter counting."""
        param_count = self.model.count_parameters()
        self.assertGreater(param_count, 0)

        breakdown = self.model.get_parameter_breakdown()
        self.assertIn("embed_tokens", breakdown)
        self.assertIn("layers", breakdown)
        self.assertIn("lm_head", breakdown)
        self.assertIn("act_head", breakdown)

        # Total should match sum of components
        component_sum = sum(v for k, v in breakdown.items() if k != "total")
        self.assertEqual(breakdown["total"], component_sum)

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        outputs = self.model(self.input_ids, return_dict=True)

        # Check output structure
        self.assertIn("logits", outputs)
        self.assertIn("act_steps", outputs)
        self.assertIn("halt_probs", outputs)
        self.assertIn("memory_stats", outputs)

        # Check output shapes
        logits = outputs["logits"]
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))

        halt_probs = outputs["halt_probs"]
        self.assertEqual(halt_probs.shape, (self.batch_size,))

        # ACT steps should be reasonable
        act_steps = outputs["act_steps"]
        self.assertGreater(act_steps, 0)
        self.assertLessEqual(act_steps, self.config.train_max_steps)

    def test_forward_pass_with_labels(self):
        """Test forward pass with loss computation."""
        outputs = self.model(self.input_ids, attention_mask=self.attention_mask, labels=self.labels, return_dict=True)

        # Should include loss
        self.assertIn("loss", outputs)
        loss = outputs["loss"]
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Positive loss

    def test_act_paradigm_switching(self):
        """Test train-many/infer-few paradigm switching."""
        # Training mode (train-many)
        self.model.set_inference_mode(False)
        train_outputs = self.model(self.input_ids, return_dict=True)
        train_steps = train_outputs["act_steps"]

        # Inference mode (infer-few)
        self.model.set_inference_mode(True)
        infer_outputs = self.model(self.input_ids, return_dict=True)
        infer_steps = infer_outputs["act_steps"]

        # Inference should use fewer steps (generally)
        # Note: This might not always be true due to randomness, but on average it should be
        self.assertLessEqual(infer_steps, self.config.infer_max_steps)
        self.assertLessEqual(train_steps, self.config.train_max_steps)

    def test_memory_integration(self):
        """Test memory system integration."""
        outputs = self.model(
            self.input_ids, labels=self.labels, return_dict=True  # Provides loss signal for memory writing
        )

        memory_stats = outputs["memory_stats"]
        self.assertIn("memory_size", memory_stats)
        self.assertIn("memory_utilization", memory_stats)

        # Memory stats should be reasonable
        memory_size = memory_stats["memory_size"]
        self.assertGreaterEqual(memory_size, 0)
        self.assertLessEqual(memory_size, self.config.mem_capacity)

    def test_generation(self):
        """Test text generation functionality."""
        # Simple generation test
        input_ids = torch.randint(0, 1000, (1, 10))  # Single sequence

        generated = self.model.generate(
            input_ids, max_new_tokens=5, do_sample=False, temperature=1.0  # Greedy for determinism
        )

        # Should generate additional tokens
        self.assertEqual(generated.shape[0], 1)  # Batch size preserved
        self.assertEqual(generated.shape[1], 15)  # Original + new tokens

        # First 10 tokens should match input
        self.assertTrue(torch.equal(generated[:, :10], input_ids))

    def test_save_load_functionality(self):
        """Test model save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"

            # Save model
            self.model.save_pretrained(str(save_path))

            # Check files were created
            self.assertTrue((save_path / "pytorch_model.bin").exists())
            self.assertTrue((save_path / "config.json").exists())
            self.assertTrue((save_path / "parameter_breakdown.json").exists())

            # Load model
            loaded_model = CognateModel.from_pretrained(str(save_path))

            # Models should have same parameter count
            orig_params = self.model.count_parameters()
            loaded_params = loaded_model.count_parameters()
            self.assertEqual(orig_params, loaded_params)

            # Should produce same outputs (in eval mode)
            self.model.eval()
            loaded_model.eval()

            with torch.no_grad():
                orig_out = self.model(self.input_ids, return_dict=True)
                loaded_out = loaded_model(self.input_ids, return_dict=True)

                # Logits should be close (within numerical precision)
                torch.testing.assert_close(orig_out["logits"], loaded_out["logits"], rtol=1e-4, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Enable gradient computation
        self.model.train()

        # Forward pass with loss
        outputs = self.model(self.input_ids, labels=self.labels, return_dict=True)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Check that gradients exist and are non-zero
        has_grad = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients found in model parameters")

        # Check for reasonable gradient magnitudes
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2

        total_grad_norm = total_grad_norm**0.5
        self.assertGreater(total_grad_norm, 0.0)
        self.assertLess(total_grad_norm, 100.0, "Gradient norm seems too large")

    def test_attention_mask_handling(self):
        """Test attention mask handling."""
        # Create mask with some positions masked out
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        attention_mask[0, -5:] = 0  # Mask last 5 positions of first sequence

        outputs = self.model(self.input_ids, attention_mask=attention_mask, return_dict=True)

        # Should complete without errors
        self.assertIn("logits", outputs)
        logits = outputs["logits"]
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.config.vocab_size))

    def test_model_modes(self):
        """Test training/evaluation mode switching."""
        # Training mode
        self.model.train()
        self.assertTrue(self.model.training)

        train_outputs = self.model(self.input_ids, return_dict=True)

        # Evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)

        eval_outputs = self.model(self.input_ids, return_dict=True)

        # Both should work
        self.assertIn("logits", train_outputs)
        self.assertIn("logits", eval_outputs)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for model creation."""

    def test_create_cognate_model(self):
        """Test the create_cognate_model factory function."""
        model = create_cognate_model(variant_name="test-model", seed=42, d_model=128, n_layers=4)  # Override config

        self.assertIsInstance(model, CognateModel)
        self.assertEqual(model.variant_name, "test-model")
        self.assertEqual(model.config.d_model, 128)
        self.assertEqual(model.config.n_layers, 4)

    def test_reproducible_initialization(self):
        """Test that models are initialized reproducibly with same seed."""
        model1 = create_cognate_model(seed=42, d_model=128, n_layers=4)
        model2 = create_cognate_model(seed=42, d_model=128, n_layers=4)

        # Parameters should be identical
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(param1, param2)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)
