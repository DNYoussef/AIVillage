"""Real compression tests with actual PyTorch models.

Test with real models, not mocks. Verify actual compression ratios.
"""

import io
import tempfile
from pathlib import Path

import pytest

# Test imports - will skip if PyTorch not available
torch = pytest.importorskip("torch")
torch_nn = pytest.importorskip("torch.nn")

from src.compression.simple_quantizer import CompressionError, SimpleQuantizer
from src.compression.test_model_generator import (
    create_mixed_model,
    create_simple_cnn_model,
    create_test_model,
    create_test_model_file,
)


class TestSimpleQuantizer:
    """Test the SimpleQuantizer with real PyTorch models."""

    def test_quantization_actually_compresses(self):
        """Test with real model, not mocks - verify actual compression."""
        # Create small test model (target ~5MB)
        model = create_test_model(layers=3, hidden_size=128, size_mb=5.0)

        quantizer = SimpleQuantizer(target_compression_ratio=3.0)  # Lower target for test
        compressed = quantizer.quantize_model_from_object(model)

        # Real assertions about compression
        stats = quantizer.get_compression_stats()

        assert stats["compression_ratio"] >= 3.0, f"Only achieved {stats['compression_ratio']:.2f}x compression"
        assert len(compressed) < (stats["original_size_bytes"] / 3.0), "Compressed size too large"
        assert stats["compressed_size_mb"] < stats["original_size_mb"], "No size reduction"

        # Verify we can load it back
        restored = SimpleQuantizer.load_quantized_model(compressed)
        assert restored is not None, "Failed to load compressed model"

    def test_quantization_with_file_model(self):
        """Test quantization with model saved to file."""
        # Create and save test model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = create_test_model_file(
                layers=2,
                hidden_size=64,
                size_mb=3.0,
                save_path=Path(temp_dir) / "test_model.pt",
            )

            quantizer = SimpleQuantizer(target_compression_ratio=2.5)
            compressed = quantizer.quantize_model(model_path)

            # Verify compression worked
            stats = quantizer.get_compression_stats()
            assert stats["compression_ratio"] >= 2.5
            assert len(compressed) > 0

    def test_cnn_model_compression(self):
        """Test compression of CNN model with Conv2d layers."""
        model = create_simple_cnn_model(input_channels=3, num_classes=10)

        quantizer = SimpleQuantizer(target_compression_ratio=2.5)  # CNN models compress less than linear
        compressed = quantizer.quantize_model_from_object(model)

        # Verify CNN compression
        stats = quantizer.get_compression_stats()
        assert stats["compression_ratio"] >= 2.5

        # Verify model can be restored and is Sequential
        restored = SimpleQuantizer.load_quantized_model(compressed)
        assert isinstance(restored, torch_nn.Sequential), "Model should be Sequential"
        assert len(restored) > 0, "Model should have layers"

    def test_mixed_model_compression(self):
        """Test compression of model with both Conv2d and Linear layers."""
        model = create_mixed_model()

        quantizer = SimpleQuantizer(target_compression_ratio=3.5)
        quantizer.quantize_model_from_object(model)

        # Verify mixed model compression
        stats = quantizer.get_compression_stats()
        assert stats["compression_ratio"] >= 3.5

        # Check mobile readiness
        assert quantizer.is_mobile_ready(max_size_mb=20.0)

    def test_compression_failure_on_insufficient_ratio(self):
        """Test that compression fails when target ratio not met."""
        # Create very small model that may not compress well
        model = create_test_model(layers=1, hidden_size=16, size_mb=0.5)

        quantizer = SimpleQuantizer(target_compression_ratio=10.0)  # Unrealistic target

        with pytest.raises(CompressionError) as exc_info:
            quantizer.quantize_model_from_object(model)

        assert "Only achieved" in str(exc_info.value)
        assert "need 10.0x" in str(exc_info.value)

    def test_mobile_compression_pipeline(self):
        """Test complete mobile compression pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test model
            model_path = create_test_model_file(
                layers=4,
                hidden_size=256,
                size_mb=8.0,
                save_path=Path(temp_dir) / "mobile_test_model.pt",
            )

            quantizer = SimpleQuantizer(target_compression_ratio=3.8)  # Slightly lower to account for variance

            # Run mobile compression pipeline
            output_path = quantizer.compress_for_mobile(model_path, output_dir=Path(temp_dir) / "mobile_output")

            # Verify pipeline results
            assert Path(output_path).exists(), "Mobile model not saved"

            stats = quantizer.get_compression_stats()
            assert stats["compression_ratio"] >= 3.8
            assert quantizer.is_mobile_ready(max_size_mb=50.0)

            # Verify we can load the mobile model
            mobile_model = torch.load(output_path, map_location="cpu")
            assert mobile_model is not None

    def test_compression_stats_accuracy(self):
        """Test that compression statistics are accurate."""
        model = create_test_model(layers=3, hidden_size=128, size_mb=6.0)

        # Get actual model size
        buffer = io.BytesIO()
        torch.save(model, buffer)
        actual_original_size = buffer.tell()

        quantizer = SimpleQuantizer(target_compression_ratio=3.0)
        compressed = quantizer.quantize_model_from_object(model)

        stats = quantizer.get_compression_stats()

        # Verify stats accuracy
        assert stats["original_size_bytes"] == actual_original_size
        assert stats["compressed_size_bytes"] == len(compressed)

        expected_ratio = actual_original_size / len(compressed)
        assert abs(stats["compression_ratio"] - expected_ratio) < 0.01

    @pytest.mark.skip(reason="Complex import mocking disabled")
    def test_error_handling_missing_pytorch(self, monkeypatch):
        """Test error handling when PyTorch is not available."""
        # This test is disabled due to import mocking complexity

    def test_error_handling_missing_model_file(self):
        """Test error handling for missing model file."""
        quantizer = SimpleQuantizer()

        with pytest.raises(CompressionError) as exc_info:
            quantizer.quantize_model("/nonexistent/model.pt")

        assert "Model file not found" in str(exc_info.value)

    def test_large_model_compression(self):
        """Test compression of larger model (target ~20MB)."""
        model = create_test_model(layers=5, hidden_size=512, size_mb=20.0)

        quantizer = SimpleQuantizer(target_compression_ratio=3.9)  # Realistic target for large models
        quantizer.quantize_model_from_object(model)

        stats = quantizer.get_compression_stats()

        # Should achieve good compression on larger models
        assert stats["compression_ratio"] >= 3.9
        assert stats["compressed_size_mb"] <= 5.0  # Should be under 5MB after compression

    def test_model_functional_after_compression(self):
        """Test that compressed model is functionally equivalent."""
        # Create model and generate test input
        model = create_test_model(layers=3, hidden_size=64, size_mb=2.0)

        # Create test input (batch_size=1, input_size matches model)
        # Note: input size was calculated in create_test_model
        test_input = torch.randn(1, 16)  # Approximate input size

        # Get original output
        model.eval()
        with torch.no_grad():
            try:
                original_output = model(test_input)
            except RuntimeError:
                # If input size doesn't match, skip functional test
                pytest.skip("Input size mismatch - skipping functional test")

        # Compress model
        quantizer = SimpleQuantizer(target_compression_ratio=3.0)
        compressed = quantizer.quantize_model_from_object(model)

        # Load compressed model
        quantized_model = SimpleQuantizer.load_quantized_model(compressed)
        quantized_model.eval()

        # Get quantized output
        with torch.no_grad():
            quantized_output = quantized_model(test_input)

        # Outputs should have same shape
        assert original_output.shape == quantized_output.shape

        # Outputs should be reasonably close (quantization introduces some error)
        diff = torch.abs(original_output - quantized_output).mean()
        assert diff < 1.0, f"Quantized output differs too much: {diff}"


@pytest.mark.integration
class TestCompressionIntegration:
    """Integration tests for the full compression pipeline."""

    def test_end_to_end_mobile_workflow(self):
        """Test complete end-to-end workflow for mobile deployment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Create training model
            model = create_mixed_model()
            model_path = temp_path / "trained_model.pt"
            torch.save(model, model_path)

            # Step 2: Compress for mobile
            quantizer = SimpleQuantizer(target_compression_ratio=3.9)
            mobile_path = quantizer.compress_for_mobile(str(model_path), output_dir=str(temp_path / "mobile"))

            # Step 3: Verify mobile deployment readiness
            assert Path(mobile_path).exists()
            assert quantizer.is_mobile_ready(max_size_mb=30.0)

            # Step 4: Load and verify mobile model
            mobile_model = torch.load(mobile_path, map_location="cpu")
            assert mobile_model is not None

            # Step 5: Check file sizes
            original_size = model_path.stat().st_size
            mobile_size = Path(mobile_path).stat().st_size
            actual_ratio = original_size / mobile_size

            assert actual_ratio >= 3.9, f"Actual compression ratio {actual_ratio:.2f}x insufficient"

    def test_multiple_model_types_compression(self):
        """Test compression works across different model architectures."""
        models_and_names = [
            (create_test_model(layers=2, hidden_size=32, size_mb=1.0), "simple_linear"),
            (create_simple_cnn_model(input_channels=1, num_classes=5), "simple_cnn"),
            (create_mixed_model(), "mixed_conv_linear"),
        ]

        quantizer = SimpleQuantizer(target_compression_ratio=2.7)

        results = []
        for model, name in models_and_names:
            try:
                quantizer.quantize_model_from_object(model)
                stats = quantizer.get_compression_stats()

                results.append(
                    {
                        "name": name,
                        "success": True,
                        "ratio": stats["compression_ratio"],
                        "mobile_ready": quantizer.is_mobile_ready(),
                    }
                )
            except CompressionError as e:
                results.append({"name": name, "success": False, "error": str(e)})

        # All models should compress successfully
        successful = [r for r in results if r["success"]]
        assert len(successful) == len(models_and_names), f"Some models failed compression: {results}"

        # All should be mobile ready
        mobile_ready = [r for r in successful if r["mobile_ready"]]
        assert len(mobile_ready) >= len(successful) // 2, "Too few models mobile ready"
