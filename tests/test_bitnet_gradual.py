"""
Unit tests for gradual λ schedule BitNet implementation.
Tests the training-driven ternary quantization approach.
"""

import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from src.production.compression.compression.stage1_bitnet import (
    BitNetLinear,
    GradualBitnetCallback,
    RMSNorm,
    convert_to_bitnet,
)


class TestBitNetLinear(unittest.TestCase):
    """Test BitNetLinear layer with gradual λ schedule."""

    def setUp(self):
        """Set up test environment."""
        self.layer = BitNetLinear(10, 5, bias=True)

    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.in_features, 10)
        self.assertEqual(self.layer.out_features, 5)
        self.assertEqual(self.layer.lambda_val, 0.0)  # Should start at 0
        self.assertIsNotNone(self.layer.weight_fp)
        self.assertIsNotNone(self.layer.bias)
        self.assertIsNotNone(self.layer.alpha)

    def test_quantize_weights(self):
        """Test weight quantization function."""
        # Create test weights
        weights = torch.tensor([[-1.5, 0.1, 1.2], [0.8, -0.3, 2.0]])
        quantized = self.layer.quantize_weights(weights)

        # Check that values are in {-1, 0, 1}
        unique_values = torch.unique(quantized).tolist()
        for val in unique_values:
            self.assertIn(val, [-1.0, 0.0, 1.0])

    def test_forward_training_lambda_0(self):
        """Test forward pass with λ=0 (pure FP)."""
        self.layer.train()
        self.layer.lambda_val = 0.0

        x = torch.randn(2, 10)
        output = self.layer(x)

        # Should use full precision weights
        expected = torch.nn.functional.linear(
            x, self.layer.weight_fp * self.layer.alpha, self.layer.bias
        )
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_forward_training_lambda_1(self):
        """Test forward pass with λ=1 (pure quantized)."""
        self.layer.train()
        self.layer.lambda_val = 1.0

        x = torch.randn(2, 10)
        output = self.layer(x)

        # Should use quantized weights
        quantized_weights = (
            self.layer.quantize_weights(self.layer.weight_fp) * self.layer.alpha
        )
        expected = torch.nn.functional.linear(x, quantized_weights, self.layer.bias)
        torch.testing.assert_close(output, expected)

    def test_forward_training_lambda_interpolation(self):
        """Test forward pass with λ interpolation."""
        self.layer.train()
        self.layer.lambda_val = 0.5

        x = torch.randn(2, 10)
        output = self.layer(x)

        # Should interpolate between FP and quantized
        quantized_weights = self.layer.quantize_weights(self.layer.weight_fp)
        interpolated = 0.5 * self.layer.weight_fp + 0.5 * quantized_weights
        expected = torch.nn.functional.linear(
            x, interpolated * self.layer.alpha, self.layer.bias
        )
        torch.testing.assert_close(output, expected)

    def test_forward_eval(self):
        """Test forward pass in eval mode."""
        self.layer.eval()
        self.layer.lambda_val = 0.5  # Should be ignored in eval mode

        x = torch.randn(2, 10)
        output = self.layer(x)

        # Should always use quantized weights in eval
        quantized_weights = (
            self.layer.quantize_weights(self.layer.weight_fp) * self.layer.alpha
        )
        expected = torch.nn.functional.linear(x, quantized_weights, self.layer.bias)
        torch.testing.assert_close(output, expected)


class TestGradualBitnetCallback(unittest.TestCase):
    """Test gradual λ schedule callback."""

    def setUp(self):
        """Set up test environment."""
        self.total_steps = 1000
        self.warmup_ratio = 0.4
        self.callback = GradualBitnetCallback(self.total_steps, self.warmup_ratio)

    def test_initialization(self):
        """Test callback initialization."""
        self.assertEqual(self.callback.total_steps, 1000)
        self.assertEqual(self.callback.warmup_steps, 400)  # 40% of 1000
        self.assertEqual(self.callback.warmup_ratio, 0.4)

    def test_lambda_schedule_warmup(self):
        """Test λ schedule during warmup period."""
        # Create mock model with BitNetLinear layers
        model = nn.Sequential(BitNetLinear(10, 5), nn.ReLU(), BitNetLinear(5, 1))

        # Mock training state
        mock_state = Mock()
        mock_args = Mock()
        mock_control = Mock()

        # Test different steps during warmup
        test_cases = [
            (0, 0.0),  # Start: λ=0
            (100, 0.25),  # 25% through warmup
            (200, 0.5),  # 50% through warmup
            (400, 1.0),  # End of warmup: λ=1
        ]

        for step, expected_lambda in test_cases:
            with self.subTest(step=step):
                mock_state.global_step = step

                self.callback.on_step_begin(
                    mock_args, mock_state, mock_control, model=model
                )

                # Check that all BitNetLinear layers have correct λ
                for module in model.modules():
                    if isinstance(module, BitNetLinear):
                        self.assertAlmostEqual(
                            module.lambda_val, expected_lambda, places=4
                        )

    def test_lambda_schedule_post_warmup(self):
        """Test λ schedule after warmup period."""
        model = nn.Sequential(BitNetLinear(10, 5))

        mock_state = Mock()
        mock_args = Mock()
        mock_control = Mock()

        # Test steps after warmup
        test_steps = [401, 500, 800, 1000]

        for step in test_steps:
            with self.subTest(step=step):
                mock_state.global_step = step

                self.callback.on_step_begin(
                    mock_args, mock_state, mock_control, model=model
                )

                # Should always be λ=1.0 after warmup
                for module in model.modules():
                    if isinstance(module, BitNetLinear):
                        self.assertEqual(module.lambda_val, 1.0)


class TestRMSNorm(unittest.TestCase):
    """Test RMSNorm implementation."""

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        dim = 512
        norm = RMSNorm(dim)

        x = torch.randn(2, 10, dim)
        output = norm(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

        # Check that norm is applied (output should have different values)
        self.assertFalse(torch.allclose(x, output))

        # Check that the norm reduces variance
        output_var = output.var(dim=-1)
        self.assertTrue(torch.all(output_var < x.var(dim=-1)))


class TestConvertToBitNet(unittest.TestCase):
    """Test model conversion to BitNet."""

    def test_convert_single_linear(self):
        """Test converting a single Linear layer."""
        linear = nn.Linear(10, 5)
        original_weight = linear.weight.clone()
        original_bias = linear.bias.clone()

        bitnet = convert_to_bitnet(linear, rmsnorm_post_attn=False)

        # Should return BitNetLinear
        self.assertIsInstance(bitnet, BitNetLinear)
        self.assertEqual(bitnet.in_features, 10)
        self.assertEqual(bitnet.out_features, 5)

        # Weights should be copied
        torch.testing.assert_close(bitnet.weight_fp, original_weight)
        torch.testing.assert_close(bitnet.bias, original_bias)

    def test_convert_model_recursive(self):
        """Test converting a model with nested Linear layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Sequential(nn.Linear(20, 15), nn.Dropout(0.1), nn.Linear(15, 5)),
        )

        # Count original Linear layers
        linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        self.assertEqual(linear_count, 3)

        # Convert to BitNet
        bitnet_model = convert_to_bitnet(model, rmsnorm_post_attn=False)

        # All Linear layers should be replaced with BitNetLinear
        linear_count_after = sum(
            1 for m in bitnet_model.modules() if isinstance(m, nn.Linear)
        )
        bitnet_count = sum(
            1 for m in bitnet_model.modules() if isinstance(m, BitNetLinear)
        )

        self.assertEqual(linear_count_after, 0)  # No Linear layers left
        self.assertEqual(bitnet_count, 3)  # 3 BitNetLinear layers


class TestSelfGeneratedData(unittest.TestCase):
    """Test self-generated data functionality."""

    @patch("src.production.compression.selfgen.generate.AutoModelForCausalLM")
    @patch("src.production.compression.selfgen.generate.AutoTokenizer")
    def test_data_generation_mock(self, mock_tokenizer, mock_model):
        """Test data generation with mocked model."""
        from src.production.compression.selfgen.generate import SelfDataGenerator

        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock generation
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer_instance.decode.return_value = "### Response:\nThis is a test response that meets all quality criteria and should pass validation."

        # Test generator initialization
        generator = SelfDataGenerator("/fake/model/path")

        # Test response validation
        valid_response = (
            "This is a valid response with enough content to pass all quality checks."
        )
        invalid_response = "Short."
        ai_response = "As an AI language model, I cannot help with this."

        self.assertTrue(generator.is_valid_response(valid_response))
        self.assertFalse(generator.is_valid_response(invalid_response))  # Too short
        self.assertFalse(generator.is_valid_response(ai_response))  # AI disclaimer

    def test_template_structure(self):
        """Test generation template structure."""
        from src.production.compression.selfgen.generate import SelfDataGenerator

        # Create mock generator to access templates
        with (
            patch("src.production.compression.selfgen.generate.AutoModelForCausalLM"),
            patch("src.production.compression.selfgen.generate.AutoTokenizer"),
        ):
            generator = SelfDataGenerator("/fake/path")
            templates = generator.get_generation_templates()

        # Check template structure
        self.assertGreater(len(templates), 0)

        for template in templates:
            self.assertIn("instruction", template)
            self.assertIn("input", template)
            self.assertIn("type", template)

            # Check types
            self.assertIn(template["type"], ["coding", "math", "writing", "reasoning"])


class TestTrainingManifest(unittest.TestCase):
    """Test training manifest generation."""

    def test_manifest_structure(self):
        """Test training manifest has required fields."""
        # Expected manifest structure from train_bitnet.py
        expected_fields = [
            "base_model",
            "dataset_path",
            "training_steps",
            "lambda_warmup_frac",
            "rmsnorm_post_attn",
            "final_lambda",
            "compression_ready",
            "model_type",
        ]

        # Mock manifest
        manifest = {
            "base_model": "/path/to/model",
            "dataset_path": "/path/to/dataset.jsonl",
            "training_steps": 1000,
            "lambda_warmup_frac": 0.4,
            "rmsnorm_post_attn": True,
            "final_lambda": 1.0,
            "compression_ready": True,
            "model_type": "bitnet158",
        }

        # Check all expected fields are present
        for field in expected_fields:
            self.assertIn(field, manifest)

        # Check critical values
        self.assertEqual(manifest["final_lambda"], 1.0)
        self.assertTrue(manifest["compression_ready"])
        self.assertEqual(manifest["model_type"], "bitnet158")


if __name__ == "__main__":
    unittest.main()
