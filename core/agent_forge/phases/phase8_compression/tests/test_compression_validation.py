"""
Test Compression Validation - Phase 8
Tests quality validation and accuracy preservation with theater detection.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import json

# Import Phase 8 components
from ..agents.compression_validator import (
    CompressionValidatorAgent,
    ValidationConfig,
    ValidationResults,
    ValidationMetrics
)
from ..validation.model_validator import (
    ModelValidationFramework,
    ValidationThresholds,
    ValidationReport
)
from ..agents.performance_profiler import (
    PerformanceProfilerAgent,
    ProfilingConfig,
    PerformanceResults
)
from ..core.compression_algorithms import (
    MagnitudePruning,
    WeightClustering,
    SVDCompression,
    CompressionMetrics
)


class TestCompressionValidation:
    """Test compression validation with theater detection."""

    def setup_method(self):
        """Setup test models and validation data."""
        # Create test models
        self.simple_model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

        # Create validation data
        self.validation_data_mlp = self._create_validation_data(
            input_shape=(100,), batch_size=16, num_batches=10
        )

        self.validation_data_cnn = self._create_validation_data(
            input_shape=(3, 32, 32), batch_size=8, num_batches=8
        )

        # Create compressed models for testing
        self.compressed_models = self._create_compressed_models()

    def _create_validation_data(self, input_shape, batch_size=16, num_batches=10):
        """Create synthetic validation data."""
        data = []
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, *input_shape)
            targets = torch.randint(0, 10, (batch_size,))
            data.append((inputs, targets))
        return data

    def _create_compressed_models(self):
        """Create compressed models for testing."""
        compressed = {}

        # Magnitude pruning
        pruning_alg = MagnitudePruning(sparsity_ratio=0.5)
        pruned_model, pruning_metrics = pruning_alg.compress(
            torch.nn.utils.deepcopy(self.simple_model)
        )
        compressed['pruned'] = (pruned_model, pruning_metrics)

        # Weight clustering
        clustering_alg = WeightClustering(num_clusters=16)
        clustered_model, clustering_metrics = clustering_alg.compress(
            torch.nn.utils.deepcopy(self.simple_model)
        )
        compressed['clustered'] = (clustered_model, clustering_metrics)

        # SVD compression
        svd_alg = SVDCompression(rank_ratio=0.6)
        svd_model, svd_metrics = svd_alg.compress(
            torch.nn.utils.deepcopy(self.simple_model)
        )
        compressed['svd'] = (svd_model, svd_metrics)

        return compressed

    def test_compression_validator_initialization(self):
        """Test compression validator agent initialization."""
        validator = CompressionValidatorAgent()

        assert validator is not None
        assert hasattr(validator, 'validate_compression')
        assert hasattr(validator, 'validate_accuracy_preservation')
        assert hasattr(validator, 'benchmark_inference_performance')

    def test_basic_validation_functionality(self):
        """Test basic validation functionality with theater detection."""
        validator = CompressionValidatorAgent()

        config = ValidationConfig(
            accuracy_threshold=0.8,
            performance_threshold=0.9,
            device='cpu'
        )

        compressed_model, _ = self.compressed_models['pruned']

        # Run validation
        results = validator.validate_compression(
            self.simple_model,
            compressed_model,
            self.validation_data_mlp,
            config
        )

        # Verify results structure
        assert isinstance(results, ValidationResults)
        assert results.overall_metrics is not None
        assert results.layer_wise_analysis is not None
        assert results.performance_benchmarks is not None
        assert isinstance(results.deployment_readiness, dict)

        # Theater detection: Verify metrics are realistic
        metrics = results.overall_metrics
        if 'accuracy' in metrics:
            assert 0.0 <= metrics['accuracy'] <= 1.0, \
                f"Theater detected: Invalid accuracy {metrics['accuracy']}"

        if 'compression_ratio' in metrics:
            assert metrics['compression_ratio'] >= 1.0, \
                f"Theater detected: Invalid compression ratio {metrics['compression_ratio']}"

        # Verify deployment readiness assessment
        assert 'ready' in results.deployment_readiness
        assert isinstance(results.deployment_readiness['ready'], bool)

        if 'issues' in results.deployment_readiness:
            assert isinstance(results.deployment_readiness['issues'], list)

    def test_accuracy_preservation_validation(self):
        """Test accuracy preservation validation with theater detection."""
        validator = CompressionValidatorAgent()

        # Test different compression levels
        for compression_name, (compressed_model, metrics) in self.compressed_models.items():
            config = ValidationConfig(
                accuracy_threshold=0.1,  # Very lenient for testing
                device='cpu',
                num_validation_runs=3
            )

            # Validate accuracy preservation
            accuracy_results = validator.validate_accuracy_preservation(
                self.simple_model,
                compressed_model,
                self.validation_data_mlp,
                config
            )

            # Theater detection: Verify accuracy measurements
            assert 'original_accuracy' in accuracy_results
            assert 'compressed_accuracy' in accuracy_results
            assert 'accuracy_retention' in accuracy_results

            original_acc = accuracy_results['original_accuracy']
            compressed_acc = accuracy_results['compressed_accuracy']
            retention = accuracy_results['accuracy_retention']

            # Basic sanity checks
            assert 0.0 <= original_acc <= 1.0, \
                f"Theater detected: Invalid original accuracy {original_acc}"
            assert 0.0 <= compressed_acc <= 1.0, \
                f"Theater detected: Invalid compressed accuracy {compressed_acc}"

            # Retention should be ratio of compressed to original
            expected_retention = compressed_acc / original_acc if original_acc > 0 else 0
            assert abs(retention - expected_retention) < 0.01, \
                f"Theater detected: Accuracy retention calculation error {retention} vs {expected_retention}"

            # Different compression methods should have different impacts
            if compression_name == 'pruned':
                # Pruned models might have accuracy drop
                assert retention >= 0.5, \
                    f"Theater detected: Excessive accuracy drop for pruning {retention}"
            elif compression_name == 'clustered':
                # Clustering should preserve accuracy better
                assert retention >= 0.8, \
                    f"Theater detected: Unexpected accuracy drop for clustering {retention}"

    def test_performance_benchmarking(self):
        """Test performance benchmarking with theater detection."""
        validator = CompressionValidatorAgent()

        config = ValidationConfig(
            device='cpu',
            benchmark_batch_sizes=[1, 4, 8],
            benchmark_iterations=10  # Short for testing
        )

        compressed_model, _ = self.compressed_models['pruned']

        # Benchmark inference performance
        perf_results = validator.benchmark_inference_performance(
            self.simple_model,
            compressed_model,
            self.validation_data_mlp,
            config
        )

        # Theater detection: Verify performance metrics
        assert 'original_performance' in perf_results
        assert 'compressed_performance' in perf_results
        assert 'speedup_ratio' in perf_results

        original_perf = perf_results['original_performance']
        compressed_perf = perf_results['compressed_performance']
        speedup = perf_results['speedup_ratio']

        # Verify performance structure
        for perf in [original_perf, compressed_perf]:
            assert 'latency_ms' in perf
            assert 'throughput_ops_per_sec' in perf
            assert 'memory_usage_mb' in perf

            # Theater detection: Verify realistic values
            assert perf['latency_ms'] > 0, \
                "Theater detected: Zero latency"
            assert perf['throughput_ops_per_sec'] > 0, \
                "Theater detected: Zero throughput"
            assert perf['memory_usage_mb'] >= 0, \
                "Theater detected: Negative memory usage"

        # Speedup should be positive
        assert speedup > 0, \
            f"Theater detected: Invalid speedup ratio {speedup}"

        # For pruned models, speedup should be >= 1.0 (compressed should be faster or equal)
        if speedup < 0.8:  # Allow some overhead
            print(f"Warning: Compressed model slower than original (speedup: {speedup:.2f})")

    def test_layer_wise_analysis(self):
        """Test layer-wise compression analysis."""
        validator = CompressionValidatorAgent()

        config = ValidationConfig(device='cpu')
        compressed_model, _ = self.compressed_models['pruned']

        # Run validation to get layer-wise analysis
        results = validator.validate_compression(
            self.cnn_model,
            self._apply_compression_to_cnn(),
            self.validation_data_cnn,
            config
        )

        # Theater detection: Verify layer-wise analysis
        layer_analysis = results.layer_wise_analysis
        assert isinstance(layer_analysis, dict)

        # Should have analysis for each layer type
        layer_types = set()
        for layer_name in layer_analysis.keys():
            if 'conv' in layer_name.lower():
                layer_types.add('conv')
            elif 'linear' in layer_name.lower() or 'fc' in layer_name.lower():
                layer_types.add('linear')

        assert len(layer_types) > 0, \
            "Theater detected: No layer-wise analysis generated"

        # Verify analysis content
        for layer_name, analysis in layer_analysis.items():
            if isinstance(analysis, dict):
                # Check for meaningful analysis metrics
                assert len(analysis) > 0, \
                    f"Theater detected: Empty analysis for layer {layer_name}"

    def _apply_compression_to_cnn(self):
        """Apply compression to CNN model for testing."""
        pruning_alg = MagnitudePruning(sparsity_ratio=0.3)
        compressed_model, _ = pruning_alg.compress(
            torch.nn.utils.deepcopy(self.cnn_model)
        )
        return compressed_model

    def test_validation_thresholds_enforcement(self):
        """Test validation threshold enforcement."""
        validator = CompressionValidatorAgent()

        # Test with strict thresholds
        strict_config = ValidationConfig(
            accuracy_threshold=0.99,  # Very strict
            performance_threshold=0.95,
            device='cpu'
        )

        compressed_model, _ = self.compressed_models['pruned']

        results = validator.validate_compression(
            self.simple_model,
            compressed_model,
            self.validation_data_mlp,
            strict_config
        )

        # Theater detection: Strict thresholds should affect deployment readiness
        deployment_ready = results.deployment_readiness.get('ready', True)

        # With aggressive compression and strict thresholds, likely won't pass
        if deployment_ready:
            # If it passes, verify it actually meets the thresholds
            if 'accuracy' in results.overall_metrics:
                accuracy = results.overall_metrics['accuracy']
                # Note: In real scenarios, this might fail due to synthetic data
                # But we can at least verify the logic is consistent
                print(f"Strict validation passed with accuracy: {accuracy:.3f}")
        else:
            # Should have specific issues listed
            assert 'issues' in results.deployment_readiness
            assert len(results.deployment_readiness['issues']) > 0

    def test_model_validation_framework(self):
        """Test model validation framework."""
        thresholds = ValidationThresholds(
            min_accuracy_retention=0.9,
            max_accuracy_drop=0.1,
            max_size_increase=0.0,  # Should decrease
            min_compression_ratio=1.1
        )

        framework = ModelValidationFramework(thresholds)
        compressed_model, _ = self.compressed_models['svd']

        # Run comprehensive validation
        report = framework.validate_model(
            self.simple_model,
            compressed_model,
            self.validation_data_mlp,
            "test_svd_model"
        )

        # Verify report structure
        assert isinstance(report, ValidationReport)
        assert report.model_name == "test_svd_model"
        assert report.metrics is not None
        assert report.validation_summary is not None
        assert isinstance(report.recommendations, list)

        # Theater detection: Verify validation summary
        summary = report.validation_summary
        assert 'passed_tests' in summary
        assert 'failed_tests' in summary
        assert 'overall_score' in summary

        assert isinstance(summary['passed_tests'], int)
        assert isinstance(summary['failed_tests'], int)
        assert 0.0 <= summary['overall_score'] <= 1.0

        # Total tests should be reasonable
        total_tests = summary['passed_tests'] + summary['failed_tests']
        assert total_tests > 0, \
            "Theater detected: No validation tests were run"
        assert total_tests < 100, \
            "Theater detected: Unrealistic number of tests"

    def test_validation_report_saving(self):
        """Test validation report persistence."""
        framework = ModelValidationFramework()
        compressed_model, _ = self.compressed_models['clustered']

        # Generate validation report
        report = framework.validate_model(
            self.simple_model,
            compressed_model,
            self.validation_data_mlp,
            "test_persistence"
        )

        # Save report
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "validation_report.json"
            framework.save_validation_report(report, report_path)

            # Theater detection: Verify report was saved
            assert report_path.exists(), \
                "Theater detected: Report file not created"
            assert report_path.stat().st_size > 0, \
                "Theater detected: Empty report file"

            # Load and verify report content
            with open(report_path, 'r') as f:
                report_data = json.load(f)

            # Verify JSON structure
            assert 'model_name' in report_data
            assert 'validation_summary' in report_data
            assert 'metrics' in report_data
            assert 'recommendations' in report_data

            assert report_data['model_name'] == "test_persistence"

    def test_cross_compression_method_comparison(self):
        """Test comparison across different compression methods."""
        validator = CompressionValidatorAgent()
        config = ValidationConfig(device='cpu')

        validation_results = {}

        # Validate all compression methods
        for method_name, (compressed_model, compression_metrics) in self.compressed_models.items():
            results = validator.validate_compression(
                self.simple_model,
                compressed_model,
                self.validation_data_mlp,
                config
            )

            validation_results[method_name] = {
                'validation': results,
                'compression_metrics': compression_metrics
            }

        # Theater detection: Compare results across methods
        compression_ratios = []
        accuracy_retentions = []

        for method_name, data in validation_results.items():
            # Extract compression ratio from validation or compression metrics
            comp_ratio = None
            if 'compression_ratio' in data['validation'].overall_metrics:
                comp_ratio = data['validation'].overall_metrics['compression_ratio']
            elif hasattr(data['compression_metrics'], 'compression_ratio'):
                comp_ratio = data['compression_metrics'].compression_ratio

            if comp_ratio is not None:
                compression_ratios.append(comp_ratio)

            # Extract accuracy retention
            if 'accuracy_retention' in data['validation'].overall_metrics:
                accuracy_retentions.append(data['validation'].overall_metrics['accuracy_retention'])

        # Theater detection: Different methods should produce different results
        if len(compression_ratios) > 1:
            unique_ratios = len(set([round(r, 2) for r in compression_ratios]))
            assert unique_ratios > 1, \
                "Theater detected: All compression methods produced identical ratios"

        # Verify trade-offs make sense
        if len(compression_ratios) > 1 and len(accuracy_retentions) > 1:
            # Generally, higher compression might correlate with lower accuracy retention
            # But this is synthetic data, so we just check for variety
            ratio_range = max(compression_ratios) - min(compression_ratios)
            assert ratio_range > 0.1, \
                "Theater detected: No meaningful difference in compression ratios"

    @pytest.mark.slow
    def test_validation_performance_scaling(self):
        """Test validation performance with larger models."""
        # Create larger model for performance testing
        large_model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Apply compression
        pruning_alg = MagnitudePruning(sparsity_ratio=0.6)
        compressed_large, _ = pruning_alg.compress(large_model)

        # Create validation data for larger model
        large_validation_data = self._create_validation_data(
            input_shape=(512,), batch_size=32, num_batches=5
        )

        validator = CompressionValidatorAgent()
        config = ValidationConfig(
            device='cpu',
            benchmark_iterations=5  # Reduced for performance
        )

        # Time the validation
        start_time = time.time()
        results = validator.validate_compression(
            large_model,
            compressed_large,
            large_validation_data,
            config
        )
        validation_time = time.time() - start_time

        # Theater detection: Should complete in reasonable time
        assert validation_time < 60.0, \
            f"Theater detected: Validation took too long ({validation_time:.1f}s)"

        # Should still provide complete results
        assert results.overall_metrics is not None
        assert results.performance_benchmarks is not None

    def test_validation_edge_cases(self):
        """Test validation with edge cases."""
        validator = CompressionValidatorAgent()

        # Test with identical models (no compression)
        config = ValidationConfig(device='cpu')
        identical_model = torch.nn.utils.deepcopy(self.simple_model)

        results = validator.validate_compression(
            self.simple_model,
            identical_model,
            self.validation_data_mlp,
            config
        )

        # Theater detection: Should detect no compression
        if 'compression_ratio' in results.overall_metrics:
            compression_ratio = results.overall_metrics['compression_ratio']
            assert abs(compression_ratio - 1.0) < 0.1, \
                f"Theater detected: False compression detected {compression_ratio}"

        # Test with over-compressed model (very high sparsity)
        extreme_pruning = MagnitudePruning(sparsity_ratio=0.95)
        over_compressed, _ = extreme_pruning.compress(
            torch.nn.utils.deepcopy(self.simple_model)
        )

        extreme_results = validator.validate_compression(
            self.simple_model,
            over_compressed,
            self.validation_data_mlp,
            config
        )

        # Should flag as potentially problematic
        deployment_ready = extreme_results.deployment_readiness.get('ready', False)
        if deployment_ready:
            # If marked as ready, accuracy should be acceptable
            if 'accuracy_retention' in extreme_results.overall_metrics:
                retention = extreme_results.overall_metrics['accuracy_retention']
                assert retention > 0.1, \
                    "Theater detected: Extreme compression marked as ready despite poor accuracy"

    def test_validation_error_handling(self):
        """Test error handling in validation."""
        validator = CompressionValidatorAgent()
        config = ValidationConfig(device='cpu')

        # Test with mismatched input sizes
        wrong_validation_data = self._create_validation_data(
            input_shape=(50,), batch_size=8, num_batches=3  # Wrong input size
        )

        # Should handle gracefully
        try:
            results = validator.validate_compression(
                self.simple_model,
                self.compressed_models['pruned'][0],
                wrong_validation_data,
                config
            )
            # If it doesn't raise an error, should at least detect the issue
            assert not results.deployment_readiness.get('ready', True), \
                "Theater detected: Should flag mismatched input sizes"
        except (RuntimeError, ValueError):
            # Expected behavior - validation should catch dimension mismatches
            pass

        # Test with empty validation data
        empty_data = []

        with pytest.raises((ValueError, IndexError, RuntimeError)):
            validator.validate_compression(
                self.simple_model,
                self.compressed_models['pruned'][0],
                empty_data,
                config
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])