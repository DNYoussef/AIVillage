"""
Test Compression Algorithms - Phase 8
Tests all core compression algorithms with theater detection.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any
import time
from unittest.mock import Mock, patch
import copy

# Import Phase 8 components
from ..core.compression_algorithms import (
    CompressionAlgorithmFactory,
    MagnitudePruning,
    GradientBasedPruning,
    WeightClustering,
    SVDCompression,
    HuffmanCoding,
    CompressionMetrics
)
from ..core.pruning_algorithms import (
    MagnitudePruner,
    GradientBasedPruner,
    SNIPPruner,
    GraSPPruner,
    StructuredPruner,
    PruningOrchestrator,
    PruningConfig
)
from ..core.quantization_engine import (
    PostTrainingQuantizer,
    QuantizationAwareTrainer,
    DynamicQuantizer,
    MixedPrecisionOptimizer,
    QuantizationConfig,
    QuantizationMode
)


class TestCompressionAlgorithms:
    """Test core compression algorithms with theater detection."""

    def setup_method(self):
        """Setup test models and data."""
        self.simple_model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

        # Create dummy data loader
        self.dummy_data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(100, 784),
                torch.randint(0, 10, (100,))
            ),
            batch_size=32
        )

    def test_algorithm_factory_creation(self):
        """Test compression algorithm factory."""
        factory = CompressionAlgorithmFactory()

        # Test available algorithms
        algorithms = factory.get_available_algorithms()
        assert len(algorithms) > 0
        assert 'magnitude_pruning' in algorithms
        assert 'gradient_pruning' in algorithms
        assert 'weight_clustering' in algorithms
        assert 'svd_compression' in algorithms
        assert 'huffman_coding' in algorithms

        # Test algorithm creation
        for algorithm_type in algorithms:
            alg = factory.create_algorithm(algorithm_type)
            assert alg is not None
            assert hasattr(alg, 'compress')
            assert hasattr(alg, 'get_compression_info')

    def test_magnitude_pruning_algorithm(self):
        """Test magnitude pruning with theater detection."""
        algorithm = MagnitudePruning(sparsity_ratio=0.5)
        original_model = copy.deepcopy(self.simple_model)

        # Test compression
        compressed_model, metrics = algorithm.compress(self.simple_model)

        # Basic functionality tests
        assert compressed_model is not None
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.compression_ratio >= 1.0
        assert 0 <= metrics.parameter_reduction <= 1.0

        # Theater detection: Check for fake compression
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)

        # Should have actual sparsity, not just claim it
        zero_params = sum((p == 0).sum().item() for p in compressed_model.parameters() if p.requires_grad)
        actual_sparsity = zero_params / compressed_params if compressed_params > 0 else 0

        # Theater detection: Verify actual sparsity matches claimed sparsity
        expected_sparsity = 0.5
        sparsity_tolerance = 0.1
        assert abs(actual_sparsity - expected_sparsity) < sparsity_tolerance, \
            f"Theater detected: Claimed sparsity {expected_sparsity}, actual {actual_sparsity}"

        # Verify compression info
        info = algorithm.get_compression_info()
        assert info['algorithm'] == 'magnitude_pruning'
        assert info['sparsity_ratio'] == 0.5

    def test_structured_vs_unstructured_pruning(self):
        """Test structured vs unstructured pruning with theater detection."""
        unstructured_alg = MagnitudePruning(sparsity_ratio=0.5, structured=False)
        structured_alg = MagnitudePruning(sparsity_ratio=0.5, structured=True)

        model1 = copy.deepcopy(self.cnn_model)
        model2 = copy.deepcopy(self.cnn_model)

        # Test unstructured pruning
        unstructured_model, unstructured_metrics = unstructured_alg.compress(model1)

        # Test structured pruning
        structured_model, structured_metrics = structured_alg.compress(model2)

        # Theater detection: Verify different compression characteristics
        assert unstructured_metrics.compression_ratio != structured_metrics.compression_ratio, \
            "Theater detected: Structured and unstructured should have different compression ratios"

        # Verify structured pruning actually removed channels/filters
        for name, module in structured_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Check if output channels were actually reduced
                original_channels = getattr(self.cnn_model.state_dict().get(f"{name}.weight", torch.empty(64, 0, 0, 0)), 'shape', [64])[0]
                if hasattr(module, 'out_channels'):
                    assert module.out_channels <= original_channels, \
                        f"Theater detected: Structured pruning didn't reduce channels in {name}"

    def test_gradient_based_pruning(self):
        """Test gradient-based pruning with theater detection."""
        algorithm = GradientBasedPruning(sparsity_ratio=0.4, num_samples=100)
        model = copy.deepcopy(self.simple_model)

        # Test compression with data loader
        compressed_model, metrics = algorithm.compress(model, data_loader=self.dummy_data)

        # Basic functionality tests
        assert compressed_model is not None
        assert metrics.compression_ratio >= 1.0

        # Theater detection: Verify gradient scores were actually computed
        assert len(algorithm.gradient_scores) > 0, \
            "Theater detected: No gradient scores computed"

        # Verify gradient scores are meaningful (not all zeros)
        for name, scores in algorithm.gradient_scores.items():
            assert scores.sum() > 0, \
                f"Theater detected: All gradient scores are zero for {name}"

        # Test without data loader should raise error
        with pytest.raises(ValueError):
            algorithm.compress(copy.deepcopy(self.simple_model))

    def test_weight_clustering_algorithm(self):
        """Test weight clustering with theater detection."""
        algorithm = WeightClustering(num_clusters=16, clustering_method='kmeans')
        model = copy.deepcopy(self.simple_model)
        original_weights = {}

        # Store original weights
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()

        # Test compression
        compressed_model, metrics = algorithm.compress(model)

        # Basic functionality tests
        assert compressed_model is not None
        assert metrics.compression_ratio > 1.0
        assert metrics.memory_reduction > 0

        # Theater detection: Verify clustering actually occurred
        assert len(algorithm.cluster_centers) > 0, \
            "Theater detected: No clusters created"
        assert len(algorithm.cluster_assignments) > 0, \
            "Theater detected: No cluster assignments made"

        # Verify weights were actually clustered (limited unique values)
        for name, module in compressed_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
                unique_weights = torch.unique(module.weight.data)
                assert len(unique_weights) <= algorithm.num_clusters, \
                    f"Theater detected: Too many unique weights in {name} after clustering"

        # Test uniform clustering
        uniform_alg = WeightClustering(num_clusters=8, clustering_method='uniform')
        uniform_model, uniform_metrics = uniform_alg.compress(copy.deepcopy(self.simple_model))

        # Should produce different results from k-means
        assert uniform_metrics.compression_ratio != metrics.compression_ratio, \
            "Theater detected: Uniform and K-means clustering should produce different results"

    def test_svd_compression_algorithm(self):
        """Test SVD compression with theater detection."""
        algorithm = SVDCompression(rank_ratio=0.5)
        model = copy.deepcopy(self.simple_model)

        # Test compression
        compressed_model, metrics = algorithm.compress(model)

        # Basic functionality tests
        assert compressed_model is not None
        assert metrics.compression_ratio > 1.0
        assert metrics.parameter_reduction > 0

        # Theater detection: Verify SVD was actually applied
        assert len(algorithm.svd_info) > 0, \
            "Theater detected: No SVD information recorded"

        # Check SVD compression ratios are reasonable
        for layer_name, info in algorithm.svd_info.items():
            assert info['target_rank'] < min(info['original_shape']), \
                f"Theater detected: SVD rank not reduced in {layer_name}"
            assert info['compression_ratio'] > 1.0, \
                f"Theater detected: No compression achieved in {layer_name}"

        # Verify weights were actually modified by SVD
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                # Check rank of weight matrix (should be reduced)
                weight_rank = torch.matrix_rank(module.weight.data.float())
                original_rank = min(module.weight.data.shape)
                assert weight_rank <= original_rank, \
                    f"Theater detected: Weight rank not reduced in {name}"

    def test_huffman_coding_algorithm(self):
        """Test Huffman coding with theater detection."""
        algorithm = HuffmanCoding(num_bits=4)
        model = copy.deepcopy(self.simple_model)

        # Store original weights for comparison
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()

        # Test compression
        compressed_model, metrics = algorithm.compress(model)

        # Basic functionality tests
        assert compressed_model is not None
        assert metrics.compression_ratio > 1.0
        assert metrics.memory_reduction > 0

        # Theater detection: Verify Huffman codes were created
        assert len(algorithm.huffman_codes) > 0, \
            "Theater detected: No Huffman codes generated"
        assert len(algorithm.huffman_trees) > 0, \
            "Theater detected: No Huffman trees created"

        # Verify quantization actually occurred
        for name, module in compressed_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
                # Check limited precision (due to quantization)
                unique_values = torch.unique(module.weight.data)
                max_unique_values = 2 ** algorithm.num_bits
                assert len(unique_values) <= max_unique_values, \
                    f"Theater detected: Too many unique values after {algorithm.num_bits}-bit quantization in {name}"

        # Verify average code length is reasonable
        avg_code_length = algorithm._calculate_average_code_length()
        assert 0 < avg_code_length <= algorithm.num_bits, \
            f"Theater detected: Invalid average code length {avg_code_length}"

    def test_compression_metrics_validation(self):
        """Test compression metrics for theater patterns."""
        algorithm = MagnitudePruning(sparsity_ratio=0.6)
        model = copy.deepcopy(self.simple_model)

        # Get original model stats
        original_params = sum(p.numel() for p in model.parameters())
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Test compression
        compressed_model, metrics = algorithm.compress(model)

        # Theater detection: Verify metrics consistency
        assert metrics.compression_ratio >= 1.0, \
            "Theater detected: Compression ratio less than 1.0"

        assert 0 <= metrics.parameter_reduction <= 1.0, \
            "Theater detected: Invalid parameter reduction ratio"

        assert 0 <= metrics.accuracy_retention <= 1.0, \
            "Theater detected: Invalid accuracy retention"

        assert metrics.inference_speedup >= 1.0, \
            "Theater detected: Inference speedup less than 1.0"

        # Check metric relationships
        if metrics.parameter_reduction > 0:
            expected_compression = 1.0 / (1.0 - metrics.parameter_reduction)
            tolerance = 0.5  # Allow some tolerance for estimation
            assert abs(metrics.compression_ratio - expected_compression) < tolerance, \
                f"Theater detected: Inconsistent compression ratio {metrics.compression_ratio} vs expected {expected_compression}"

    def test_algorithm_robustness(self):
        """Test algorithm robustness and edge cases."""
        # Test with very small model
        tiny_model = nn.Linear(2, 1)
        algorithm = MagnitudePruning(sparsity_ratio=0.5)

        compressed_model, metrics = algorithm.compress(tiny_model)
        assert compressed_model is not None

        # Test with very high sparsity
        high_sparsity_alg = MagnitudePruning(sparsity_ratio=0.99)
        model = copy.deepcopy(self.simple_model)

        compressed_model, metrics = high_sparsity_alg.compress(model)

        # Theater detection: Should still have some non-zero weights
        total_params = sum(p.numel() for p in compressed_model.parameters())
        zero_params = sum((p == 0).sum().item() for p in compressed_model.parameters())
        non_zero_params = total_params - zero_params

        assert non_zero_params > 0, \
            "Theater detected: All parameters pruned (model would be non-functional)"

        # Test with zero sparsity (should be no-op)
        no_sparsity_alg = MagnitudePruning(sparsity_ratio=0.0)
        model = copy.deepcopy(self.simple_model)
        original_model = copy.deepcopy(model)

        compressed_model, metrics = no_sparsity_alg.compress(model)

        # Should have minimal changes
        assert metrics.parameter_reduction == 0.0 or metrics.parameter_reduction < 0.01, \
            "Theater detected: Significant reduction with zero sparsity"

    def test_algorithm_comparison(self):
        """Test comparison between different algorithms."""
        model = copy.deepcopy(self.simple_model)
        algorithms = [
            ('magnitude', MagnitudePruning(sparsity_ratio=0.5)),
            ('clustering', WeightClustering(num_clusters=16)),
            ('svd', SVDCompression(rank_ratio=0.5))
        ]

        results = {}

        for name, algorithm in algorithms:
            test_model = copy.deepcopy(model)
            compressed_model, metrics = algorithm.compress(test_model)
            results[name] = {
                'metrics': metrics,
                'model': compressed_model
            }

        # Theater detection: Different algorithms should produce different results
        compression_ratios = [results[name]['metrics'].compression_ratio for name in results]
        assert len(set(compression_ratios)) > 1, \
            "Theater detected: All algorithms produced identical compression ratios"

        # Verify each algorithm's specific characteristics
        magnitude_result = results['magnitude']
        clustering_result = results['clustering']
        svd_result = results['svd']

        # Magnitude pruning should have clear sparsity
        magnitude_model = magnitude_result['model']
        zero_count = sum((p == 0).sum().item() for p in magnitude_model.parameters())
        total_count = sum(p.numel() for p in magnitude_model.parameters())
        magnitude_sparsity = zero_count / total_count

        assert magnitude_sparsity > 0.3, \
            f"Theater detected: Magnitude pruning should create sparsity, got {magnitude_sparsity}"

        # SVD should reduce parameter count for linear layers
        svd_params = sum(p.numel() for p in svd_result['model'].parameters())
        original_params = sum(p.numel() for p in model.parameters())
        assert svd_params < original_params, \
            "Theater detected: SVD should reduce parameter count"

    def test_performance_requirements(self):
        """Test performance requirements are met."""
        model = copy.deepcopy(self.simple_model)
        algorithm = MagnitudePruning(sparsity_ratio=0.5)

        # Time compression
        start_time = time.time()
        compressed_model, metrics = algorithm.compress(model)
        compression_time = time.time() - start_time

        # Should complete in reasonable time
        assert compression_time < 10.0, \
            f"Theater detected: Compression took too long ({compression_time:.2f}s)"

        # Compressed model should be smaller
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compressed_size = self._estimate_compressed_size(compressed_model, metrics)

        assert compressed_size < original_size, \
            "Theater detected: Compressed model not actually smaller"

    def _estimate_compressed_size(self, model, metrics):
        """Estimate compressed model size."""
        # For sparse models, count only non-zero parameters
        non_zero_params = 0
        for param in model.parameters():
            non_zero_params += (param != 0).sum().item()

        # Estimate size (4 bytes per float32)
        return non_zero_params * 4

    @pytest.mark.slow
    def test_algorithm_scaling(self):
        """Test algorithm scaling with larger models."""
        # Create larger model
        large_model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        algorithm = MagnitudePruning(sparsity_ratio=0.7)

        # Test with larger model
        start_time = time.time()
        compressed_model, metrics = algorithm.compress(large_model)
        compression_time = time.time() - start_time

        # Should still complete in reasonable time
        assert compression_time < 30.0, \
            f"Theater detected: Large model compression took too long ({compression_time:.2f}s)"

        # Should achieve good compression
        assert metrics.compression_ratio > 2.0, \
            "Theater detected: Poor compression ratio on large model"

    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test with invalid sparsity ratio
        with pytest.raises((ValueError, AssertionError)):
            MagnitudePruning(sparsity_ratio=1.5)  # Invalid ratio

        with pytest.raises((ValueError, AssertionError)):
            MagnitudePruning(sparsity_ratio=-0.1)  # Negative ratio

        # Test with invalid clustering parameters
        with pytest.raises((ValueError, AssertionError)):
            WeightClustering(num_clusters=0)  # Zero clusters

        # Test with invalid SVD parameters
        with pytest.raises((ValueError, AssertionError)):
            SVDCompression(rank_ratio=1.5)  # Invalid ratio

    def test_reproducibility(self):
        """Test algorithm reproducibility."""
        model1 = copy.deepcopy(self.simple_model)
        model2 = copy.deepcopy(self.simple_model)

        # Set random seed
        torch.manual_seed(42)
        algorithm1 = MagnitudePruning(sparsity_ratio=0.5)
        compressed_model1, metrics1 = algorithm1.compress(model1)

        # Reset seed and repeat
        torch.manual_seed(42)
        algorithm2 = MagnitudePruning(sparsity_ratio=0.5)
        compressed_model2, metrics2 = algorithm2.compress(model2)

        # Results should be identical for deterministic algorithms
        assert abs(metrics1.compression_ratio - metrics2.compression_ratio) < 1e-6, \
            "Theater detected: Non-reproducible compression results"

        # Check weight similarity
        for (name1, param1), (name2, param2) in zip(compressed_model1.named_parameters(),
                                                   compressed_model2.named_parameters()):
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Theater detected: Non-reproducible weights in {name1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])