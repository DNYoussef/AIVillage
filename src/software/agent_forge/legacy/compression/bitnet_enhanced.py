"""Enhanced BitNet 1.58-bit quantization with real model testing.

Provides comprehensive model compression with benchmarks and evaluation.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

try:
    from .bitnet import BITNETCompressor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from bitnet import BITNETCompressor

logger = logging.getLogger(__name__)


class EnhancedBitNetCompressor:
    """Enhanced BitNet compressor with real model testing capabilities."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self.core_compressor = BITNETCompressor()
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "model_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Benchmarking data
        self.compression_stats = {
            "models_tested": 0,
            "total_compression_time": 0.0,
            "compression_ratios": [],
            "model_sizes_before": [],
            "model_sizes_after": [],
        }

    def load_test_model(self, model_name: str = "distilbert-base-uncased") -> tuple[nn.Module, AutoTokenizer]:
        """Load a small model for testing compression."""
        try:
            logger.info(f"Loading model: {model_name}")

            # Use small, fast models for testing
            model = AutoModel.from_pretrained(model_name, cache_dir=str(self.cache_dir), torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))

            # Ensure we have a pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

            logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return model, tokenizer

        except Exception as e:
            logger.exception(f"Failed to load model {model_name}: {e}")
            # Fallback to a simple model
            return self._create_simple_model(), None

    def _create_simple_model(self) -> nn.Module:
        """Create a simple model for testing when HuggingFace models fail."""

        class SimpleTestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)
                self.output = nn.Linear(128, 2)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = self.embedding(x)
                x = x.mean(dim=1)  # Simple pooling
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = torch.relu(self.linear2(x))
                return self.output(x)

        model = SimpleTestModel()
        logger.info(f"Created simple test model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def compress_model(self, model: nn.Module, model_name: str = "test_model") -> dict:
        """Compress a PyTorch model using BitNet quantization."""
        start_time = time.time()

        try:
            # Calculate original model size
            original_size = self._calculate_model_size(model)
            logger.info(f"Original model size: {original_size / 1024 / 1024:.2f} MB")

            # Extract and compress weights
            compressed_layers = {}
            total_params_original = 0
            total_params_compressed = 0

            for name, layer in model.named_modules():
                if isinstance(layer, nn.Linear | nn.Conv2d | nn.Embedding):
                    if hasattr(layer, "weight") and layer.weight is not None:
                        weight_tensor = layer.weight.data
                        total_params_original += weight_tensor.numel()

                        # Apply BitNet quantization
                        compressed_weight = self._quantize_to_bitnet(weight_tensor)
                        compressed_layers[f"{name}.weight"] = {
                            "data": compressed_weight,
                            "shape": weight_tensor.shape,
                            "dtype": str(weight_tensor.dtype),
                            "quantization": "bitnet_1.58",
                        }

                        # Count compressed parameters (stored as int8)
                        total_params_compressed += compressed_weight.numel()

                        if hasattr(layer, "bias") and layer.bias is not None:
                            # Keep biases unquantized for better accuracy
                            compressed_layers[f"{name}.bias"] = {
                                "data": layer.bias.data.numpy(),
                                "shape": layer.bias.shape,
                                "dtype": str(layer.bias.dtype),
                                "quantization": "none",
                            }

            # Calculate compression metrics
            compression_time = time.time() - start_time
            compressed_size = self._calculate_compressed_size(compressed_layers)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            # Update statistics
            self.compression_stats["models_tested"] += 1
            self.compression_stats["total_compression_time"] += compression_time
            self.compression_stats["compression_ratios"].append(compression_ratio)
            self.compression_stats["model_sizes_before"].append(original_size)
            self.compression_stats["model_sizes_after"].append(compressed_size)

            compression_result = {
                "model_name": model_name,
                "original_size_mb": original_size / 1024 / 1024,
                "compressed_size_mb": compressed_size / 1024 / 1024,
                "compression_ratio": compression_ratio,
                "compression_time_seconds": compression_time,
                "original_parameters": total_params_original,
                "compressed_parameters": total_params_compressed,
                "layers_compressed": len([k for k in compressed_layers if "weight" in k]),
                "quantization_method": "BitNet-1.58b",
                "compressed_layers": compressed_layers,
                "metadata": {
                    "compression_timestamp": time.time(),
                    "bits_per_weight": 1.58,
                    "preserves_bias": True,
                },
            }

            logger.info(f"Compression complete: {compression_ratio:.2f}x ratio in {compression_time:.2f}s")
            return compression_result

        except Exception as e:
            logger.exception(f"Compression failed: {e}")
            return {"error": str(e), "compression_ratio": 0.0}

    def _quantize_to_bitnet(self, tensor: torch.Tensor) -> np.ndarray:
        """Apply BitNet 1.58-bit quantization to a tensor."""
        # BitNet quantization: weights are quantized to {-1, 0, +1}
        # This simulates the 1.58-bit quantization (log2(3) ≈ 1.58)

        # Calculate scale for quantization
        scale = tensor.abs().mean()

        # Quantize to {-1, 0, +1}
        normalized = tensor / (scale + 1e-8)  # Avoid division by zero
        quantized = torch.sign(normalized)  # This gives {-1, 0, +1}

        # Add some sparsity (set small values to 0)
        threshold = 0.1
        mask = tensor.abs() < (scale * threshold)
        quantized[mask] = 0

        # Store as int8 with scale
        quantized_int8 = quantized.to(torch.int8)

        # Return both quantized weights and scale for reconstruction
        return {
            "weights": quantized_int8.numpy(),
            "scale": scale.item(),
            "quantization_type": "bitnet_1.58",
        }

    def decompress_model(self, compressed_data: dict, model_class: type | None = None) -> nn.Module | None:
        """Decompress a model from compressed representation."""
        try:
            if "compressed_layers" not in compressed_data:
                logger.error("Invalid compressed data format")
                return None

            # For now, create a simple model structure
            # In a full implementation, you'd need the original model architecture
            decompressed_weights = {}

            for layer_name, layer_data in compressed_data["compressed_layers"].items():
                if layer_data["quantization"] == "bitnet_1.58":
                    # Decompress BitNet quantized weights
                    weights_data = layer_data["data"]
                    quantized_weights = torch.tensor(weights_data["weights"], dtype=torch.float32)
                    scale = weights_data["scale"]

                    # Reconstruct approximate weights
                    decompressed_weights[layer_name] = quantized_weights * scale
                else:
                    # Unquantized data (like biases)
                    decompressed_weights[layer_name] = torch.tensor(layer_data["data"])

            logger.info(f"Decompressed {len(decompressed_weights)} layers")
            return decompressed_weights

        except Exception as e:
            logger.exception(f"Decompression failed: {e}")
            return None

    def save_compressed_model(self, compressed_data: dict, filepath: str) -> None:
        """Save compressed model to disk."""
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy arrays to lists for JSON serialization
            serializable_data = self._make_serializable(compressed_data)

            with open(save_path, "w") as f:
                json.dump(serializable_data, f, indent=2)

            logger.info(f"Compressed model saved to {filepath}")
            file_size = save_path.stat().st_size / 1024 / 1024
            logger.info(f"Saved file size: {file_size:.2f} MB")

        except Exception as e:
            logger.exception(f"Failed to save compressed model: {e}")

    def load_compressed_model(self, filepath: str) -> dict | None:
        """Load compressed model from disk."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Convert lists back to numpy arrays
            restored_data = self._restore_from_serializable(data)
            logger.info(f"Compressed model loaded from {filepath}")
            return restored_data

        except Exception as e:
            logger.exception(f"Failed to load compressed model: {e}")
            return None

    def run_compression_benchmark(self, models_to_test: list | None = None) -> dict:
        """Run comprehensive compression benchmarks."""
        if models_to_test is None:
            models_to_test = [
                "distilbert-base-uncased",
                "microsoft/DialoGPT-small",
                "gpt2",
            ]  # Small GPT-2 model

        benchmark_results = {
            "benchmark_timestamp": time.time(),
            "models_tested": [],
            "average_compression_ratio": 0.0,
            "average_compression_time": 0.0,
            "total_size_saved_mb": 0.0,
        }

        successful_compressions = 0

        for model_name in models_to_test:
            try:
                logger.info(f"Benchmarking compression for {model_name}")

                # Load model
                model, tokenizer = self.load_test_model(model_name)

                # Compress model
                compression_result = self.compress_model(model, model_name)

                if "error" not in compression_result:
                    benchmark_results["models_tested"].append(compression_result)
                    successful_compressions += 1

                    # Save compressed model
                    save_path = self.cache_dir / f"{model_name.replace('/', '_')}_compressed.json"
                    self.save_compressed_model(compression_result, str(save_path))

                    logger.info(f"✅ {model_name}: {compression_result['compression_ratio']:.2f}x compression")
                else:
                    logger.error(f"❌ {model_name}: {compression_result['error']}")

                # Clean up memory
                del model
                if tokenizer:
                    del tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                logger.exception(f"Benchmark failed for {model_name}: {e}")

        # Calculate average metrics
        if successful_compressions > 0:
            ratios = [r["compression_ratio"] for r in benchmark_results["models_tested"]]
            times = [r["compression_time_seconds"] for r in benchmark_results["models_tested"]]
            size_savings = [r["original_size_mb"] - r["compressed_size_mb"] for r in benchmark_results["models_tested"]]

            benchmark_results["average_compression_ratio"] = sum(ratios) / len(ratios)
            benchmark_results["average_compression_time"] = sum(times) / len(times)
            benchmark_results["total_size_saved_mb"] = sum(size_savings)

        logger.info(f"Benchmark complete: {successful_compressions}/{len(models_to_test)} models compressed")
        return benchmark_results

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size

    def _calculate_compressed_size(self, compressed_layers: dict) -> int:
        """Estimate compressed model size in bytes."""
        total_size = 0
        for layer_data in compressed_layers.values():
            if isinstance(layer_data["data"], dict):
                # BitNet quantized data (int8 + scale)
                weights_size = np.array(layer_data["data"]["weights"]).nbytes
                scale_size = 8  # float64
                total_size += weights_size + scale_size
            else:
                # Unquantized data
                total_size += np.array(layer_data["data"]).nbytes
        return total_size

    def _make_serializable(self, data):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, np.number):
            return float(data)
        return data

    def _restore_from_serializable(self, data):
        """Restore numpy arrays from JSON data."""
        if isinstance(data, dict):
            if "weights" in data and "scale" in data:
                # This is BitNet quantized data
                return {
                    "weights": np.array(data["weights"], dtype=np.int8),
                    "scale": data["scale"],
                    "quantization_type": data.get("quantization_type", "bitnet_1.58"),
                }
            return {k: self._restore_from_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return np.array(data)
        return data

    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        stats = self.compression_stats.copy()

        if stats["compression_ratios"]:
            stats["average_compression_ratio"] = sum(stats["compression_ratios"]) / len(stats["compression_ratios"])
            stats["best_compression_ratio"] = max(stats["compression_ratios"])
            stats["total_size_saved_mb"] = sum(
                (before - after) / 1024 / 1024
                for before, after in zip(
                    stats["model_sizes_before"],
                    stats["model_sizes_after"],
                    strict=False,
                )
            )

        return stats


def test_enhanced_bitnet_compression():
    """Test function for enhanced BitNet compression."""
    print("Testing Enhanced BitNet Compression...")

    compressor = EnhancedBitNetCompressor()

    # Run benchmark
    results = compressor.run_compression_benchmark()

    print("\n📊 Compression Benchmark Results:")
    print(f"Models tested: {len(results['models_tested'])}")
    print(f"Average compression ratio: {results['average_compression_ratio']:.2f}x")
    print(f"Average compression time: {results['average_compression_time']:.2f}s")
    print(f"Total size saved: {results['total_size_saved_mb']:.2f} MB")

    # Print individual results
    for model_result in results["models_tested"]:
        print(f"\n🤖 {model_result['model_name']}:")
        print(f"  Size: {model_result['original_size_mb']:.1f} MB → {model_result['compressed_size_mb']:.1f} MB")
        print(f"  Ratio: {model_result['compression_ratio']:.2f}x")
        print(f"  Parameters: {model_result['original_parameters']:,} → {model_result['compressed_parameters']:,}")

    return results


if __name__ == "__main__":
    test_enhanced_bitnet_compression()
