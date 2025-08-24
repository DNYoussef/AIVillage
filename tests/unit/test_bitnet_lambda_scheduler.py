#!/usr/bin/env python3
"""
Unit tests for BitNet λ scheduler and quantizer functionality.
Tests the gradual λ schedule and quantization behavior.
"""

# Import the classes we're testing
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent / "src" / "production" / "compression"))

from compression.stage1_bitnet import BitNetLinear, GradualBitnetCallback


class TestBitNetLinear:
    """Test BitNetLinear layer functionality."""

    def test_initialization(self):
        """Test BitNetLinear initialization."""
        layer = BitNetLinear(128, 256, bias=True)

        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.bias is not None
        assert layer.lambda_val == 0.0
        assert layer.weight_fp.shape == (256, 128)
        assert layer.alpha.shape == (1,)

    def test_initialization_no_bias(self):
        """Test BitNetLinear initialization without bias."""
        layer = BitNetLinear(128, 256, bias=False)

        assert layer.bias is None

    def test_quantize_weights_threshold(self):
        """Test quantize_weights uses proper threshold."""
        layer = BitNetLinear(4, 4)

        # Create test weights with known values
        test_weights = torch.tensor(
            [
                [1.0, 0.1, -0.1, -1.0],
                [0.5, -0.5, 0.8, -0.8],
                [0.0, 0.2, -0.2, 0.0],
                [2.0, -2.0, 0.0, 1.5],
            ]
        )

        quantized = layer.quantize_weights(test_weights)

        # Check that values are ternary {-1, 0, 1}
        unique_values = torch.unique(quantized)
        assert all(val in [-1.0, 0.0, 1.0] for val in unique_values)

        # Check that large values become ±1, small values become 0
        threshold = test_weights.abs().mean()
        expected = torch.zeros_like(test_weights)
        mask = test_weights.abs() > threshold
        expected = torch.sign(test_weights) * mask.float()

        torch.testing.assert_close(quantized, expected)

    def test_quantize_weights_monotonicity(self):
        """Test that quantizer respects magnitude ordering."""
        layer = BitNetLinear(3, 3)

        # Weights with clear magnitude ordering
        weights = torch.tensor(
            [
                [2.0, 1.0, 0.1],  # large, medium, small
                [-2.0, -1.0, -0.1],  # large neg, medium neg, small neg
                [0.0, 0.5, -0.5],  # zero, medium pos, medium neg
            ]
        )

        quantized = layer.quantize_weights(weights)

        # Large magnitude values should not be zero
        assert quantized[0, 0] != 0  # 2.0
        assert quantized[1, 0] != 0  # -2.0

        # Check signs are preserved for non-zero quantized values
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if quantized[i, j] != 0:
                    assert torch.sign(weights[i, j]) == torch.sign(quantized[i, j])

    def test_forward_training_interpolation(self):
        """Test forward pass interpolation during training."""
        layer = BitNetLinear(2, 2)
        layer.train()  # Set to training mode

        # Set specific weights for predictable behavior
        with torch.no_grad():
            layer.weight_fp.copy_(torch.tensor([[1.0, -1.0], [0.5, -0.5]]))
            layer.alpha.copy_(torch.tensor([1.0]))

        # Test with λ = 0 (pure floating point)
        layer.lambda_val = 0.0
        x = torch.tensor([[1.0, 1.0]])
        output_fp = layer(x)

        # Test with λ = 1 (pure quantized)
        layer.lambda_val = 1.0
        output_q = layer(x)

        # Test with λ = 0.5 (interpolation)
        layer.lambda_val = 0.5
        output_interp = layer(x)

        # Interpolated result should be between fp and quantized
        # This is a basic sanity check - exact values depend on quantization
        assert output_interp.shape == output_fp.shape == output_q.shape

        # λ = 0 should give different result than λ = 1
        assert not torch.allclose(output_fp, output_q, atol=1e-6)

    def test_forward_inference_mode(self):
        """Test forward pass uses pure quantized weights in inference."""
        layer = BitNetLinear(2, 2)
        layer.eval()  # Set to inference mode

        with torch.no_grad():
            layer.weight_fp.copy_(torch.tensor([[1.0, -1.0], [0.5, -0.5]]))
            layer.alpha.copy_(torch.tensor([1.0]))

        # Lambda value should be ignored in inference
        layer.lambda_val = 0.5
        x = torch.tensor([[1.0, 1.0]])
        output_eval = layer(x)

        # Should be same as λ = 1.0 in training mode
        layer.train()
        layer.lambda_val = 1.0
        output_train_q = layer(x)

        torch.testing.assert_close(output_eval, output_train_q, atol=1e-6)

    def test_zeroing_behavior(self):
        """Test that small weights are properly zeroed."""
        layer = BitNetLinear(3, 1)

        # Weights where some should be zeroed
        small_weights = torch.tensor([[0.01], [0.02], [0.001]])
        large_weights = torch.tensor([[1.0], [2.0], [0.5]])

        quantized_small = layer.quantize_weights(small_weights)
        quantized_large = layer.quantize_weights(large_weights)

        # Small weights should have more zeros
        zeros_small = (quantized_small == 0).sum()
        zeros_large = (quantized_large == 0).sum()

        # This is a probabilistic test, but should generally hold
        assert zeros_small >= zeros_large


class TestGradualBitnetCallback:
    """Test GradualBitnetCallback functionality."""

    def test_initialization(self):
        """Test callback initialization."""
        total_steps = 1000
        warmup_ratio = 0.4

        callback = GradualBitnetCallback(total_steps, warmup_ratio)

        assert callback.total_steps == total_steps
        assert callback.warmup_steps == int(total_steps * warmup_ratio)
        assert callback.warmup_ratio == warmup_ratio
        assert callback.current_lambda == 0.0

    def test_lambda_schedule_boundaries(self):
        """Test λ schedule at key boundaries."""
        total_steps = 1000
        warmup_ratio = 0.4
        callback = GradualBitnetCallback(total_steps, warmup_ratio)

        # Create mock training state and model
        mock_state = Mock()
        mock_model = Mock()
        mock_bitnet_layer = Mock(spec=BitNetLinear)
        mock_model.modules.return_value = [mock_bitnet_layer]

        # Test step 0 (start)
        mock_state.global_step = 0
        callback.on_step_begin(None, mock_state, None, model=mock_model)
        assert callback.current_lambda == 0.0
        assert mock_bitnet_layer.lambda_val == 0.0

        # Test warmup end (step 400)
        mock_state.global_step = callback.warmup_steps
        callback.on_step_begin(None, mock_state, None, model=mock_model)
        assert callback.current_lambda == 1.0
        assert mock_bitnet_layer.lambda_val == 1.0

        # Test after warmup (step 500)
        mock_state.global_step = 500
        callback.on_step_begin(None, mock_state, None, model=mock_model)
        assert callback.current_lambda == 1.0
        assert mock_bitnet_layer.lambda_val == 1.0

        # Test final step
        mock_state.global_step = total_steps
        callback.on_step_begin(None, mock_state, None, model=mock_model)
        assert callback.current_lambda == 1.0
        assert mock_bitnet_layer.lambda_val == 1.0

    def test_lambda_schedule_progression(self):
        """Test λ schedule progresses smoothly."""
        total_steps = 100
        warmup_ratio = 0.5
        callback = GradualBitnetCallback(total_steps, warmup_ratio)

        mock_state = Mock()
        mock_model = Mock()
        mock_bitnet_layer = Mock(spec=BitNetLinear)
        mock_model.modules.return_value = [mock_bitnet_layer]

        lambda_values = []

        # Test progression through warmup
        for step in range(0, callback.warmup_steps + 1, 10):
            mock_state.global_step = step
            callback.on_step_begin(None, mock_state, None, model=mock_model)
            lambda_values.append(callback.current_lambda)

        # Should be monotonically increasing
        for i in range(1, len(lambda_values)):
            assert lambda_values[i] >= lambda_values[i - 1]

        # Should start at 0 and end at 1
        assert lambda_values[0] == 0.0
        assert lambda_values[-1] == 1.0

    def test_multiple_bitnet_layers(self):
        """Test callback updates multiple BitNet layers."""
        callback = GradualBitnetCallback(100, 0.5)

        mock_state = Mock()
        mock_state.global_step = 25  # Halfway through warmup

        # Create multiple mock BitNet layers
        mock_layer1 = Mock(spec=BitNetLinear)
        mock_layer2 = Mock(spec=BitNetLinear)
        mock_other_layer = Mock(spec=nn.Linear)  # Non-BitNet layer

        mock_model = Mock()
        mock_model.modules.return_value = [mock_layer1, mock_other_layer, mock_layer2]

        callback.on_step_begin(None, mock_state, None, model=mock_model)

        # Both BitNet layers should be updated
        expected_lambda = 0.5  # 25/50
        assert mock_layer1.lambda_val == expected_lambda
        assert mock_layer2.lambda_val == expected_lambda

        # Non-BitNet layer should not have lambda_val set
        assert not hasattr(mock_other_layer, "lambda_val") or not callable(mock_other_layer.lambda_val)

    def test_log_callback(self):
        """Test logging callback adds λ values."""
        callback = GradualBitnetCallback(100, 0.4)
        callback.current_lambda = 0.75

        logs = {}
        callback.on_log(None, None, None, logs=logs)

        assert "lambda_val" in logs
        assert logs["lambda_val"] == 0.75
        assert "bitnet_phase" in logs
        assert logs["bitnet_phase"] == "warmup"  # < 1.0

        # Test when λ = 1.0
        callback.current_lambda = 1.0
        logs = {}
        callback.on_log(None, None, None, logs=logs)
        assert logs["bitnet_phase"] == "ternary"

    def test_edge_case_zero_warmup_steps(self):
        """Test behavior when warmup_steps = 0."""
        callback = GradualBitnetCallback(100, 0.0)  # No warmup
        assert callback.warmup_steps == 0

        mock_state = Mock()
        mock_model = Mock()
        mock_bitnet_layer = Mock(spec=BitNetLinear)
        mock_model.modules.return_value = [mock_bitnet_layer]

        # Even at step 0, should be λ = 1.0
        mock_state.global_step = 0
        callback.on_step_begin(None, mock_state, None, model=mock_model)
        assert callback.current_lambda == 1.0
        assert mock_bitnet_layer.lambda_val == 1.0


class TestQuantizerThresholdBehavior:
    """Test specific quantizer threshold and zeroing behaviors."""

    def test_threshold_calculation_consistency(self):
        """Test that threshold calculation is consistent."""
        layer = BitNetLinear(1, 1)

        # Test with different weight distributions
        test_cases = [
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[-1.0, -2.0, 0.5, 0.1]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[0.1, -0.1, 0.2, -0.2]]),
        ]

        for weights in test_cases:
            quantized1 = layer.quantize_weights(weights)
            quantized2 = layer.quantize_weights(weights)

            # Should be deterministic
            torch.testing.assert_close(quantized1, quantized2)

            # Threshold should be mean absolute value
            expected_threshold = weights.abs().mean()

            # Verify sparsity pattern matches threshold
            mask = weights.abs() > expected_threshold
            expected_nonzero = mask.sum()
            actual_nonzero = (quantized1 != 0).sum()

            assert actual_nonzero == expected_nonzero

    def test_sparsity_increases_with_smaller_weights(self):
        """Test that smaller weights lead to more sparsity."""
        layer = BitNetLinear(1, 1)

        # Create weights with decreasing magnitude
        large_weights = torch.randn(10, 10) * 2.0
        small_weights = torch.randn(10, 10) * 0.1

        quantized_large = layer.quantize_weights(large_weights)
        quantized_small = layer.quantize_weights(small_weights)

        sparsity_large = (quantized_large == 0).float().mean()
        sparsity_small = (quantized_small == 0).float().mean()

        # Smaller weights should generally lead to higher sparsity
        assert sparsity_small >= sparsity_large


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
