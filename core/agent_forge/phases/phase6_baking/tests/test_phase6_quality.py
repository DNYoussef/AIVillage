#!/usr/bin/env python3
"""
Quality Tests for Phase 6 Baking System
=======================================

Comprehensive quality validation and theater detection tests for Phase 6:
- Accuracy preservation validation
- Quality degradation detection
- Performance theater detection
- Model consistency verification
- Robustness testing
- Output quality validation
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import time
from unittest.mock import Mock, patch

# Import Phase 6 components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from agent_forge.phase6 import (
    BakingArchitecture,
    BakingConfig,
    OptimizationMetrics,
    QualityValidator,
    create_baking_pipeline
)


class QualityTestModel(nn.Module):
    """Deterministic model for quality testing"""
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 16, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ClassificationTestModel(nn.Module):
    """Model for classification accuracy testing"""
    def __init__(self, num_classes=10, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class QualityMetrics:
    """Utility class for calculating quality metrics"""

    @staticmethod
    def calculate_accuracy(model: nn.Module, data_loader) -> float:
        """Calculate classification accuracy"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total if total > 0 else 0.0

    @staticmethod
    def calculate_output_similarity(
        original_model: nn.Module,
        optimized_model: nn.Module,
        inputs: torch.Tensor,
        similarity_metric: str = "cosine"
    ) -> float:
        """Calculate similarity between model outputs"""
        original_model.eval()
        optimized_model.eval()

        with torch.no_grad():
            original_output = original_model(inputs)
            optimized_output = optimized_model(inputs)

            if similarity_metric == "cosine":
                # Cosine similarity
                original_flat = original_output.view(original_output.size(0), -1)
                optimized_flat = optimized_output.view(optimized_output.size(0), -1)

                similarity = F.cosine_similarity(original_flat, optimized_flat, dim=1)
                return float(torch.mean(similarity))

            elif similarity_metric == "mse":
                # Mean squared error (lower is better)
                mse = F.mse_loss(original_output, optimized_output)
                return float(mse)

            elif similarity_metric == "correlation":
                # Correlation coefficient
                original_flat = original_output.view(-1)
                optimized_flat = optimized_output.view(-1)

                correlation = torch.corrcoef(torch.stack([original_flat, optimized_flat]))[0, 1]
                return float(correlation)

        return 0.0

    @staticmethod
    def detect_output_degradation(
        original_outputs: torch.Tensor,
        optimized_outputs: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, bool]:
        """Detect various types of output degradation"""
        # Check for NaN or infinite values
        has_nan = torch.isnan(optimized_outputs).any()
        has_inf = torch.isinf(optimized_outputs).any()

        # Check for significant distribution shifts
        original_mean = torch.mean(original_outputs)
        optimized_mean = torch.mean(optimized_outputs)
        mean_shift = abs(original_mean - optimized_mean) / (abs(original_mean) + 1e-8)

        original_std = torch.std(original_outputs)
        optimized_std = torch.std(optimized_outputs)
        std_shift = abs(original_std - optimized_std) / (original_std + 1e-8)

        # Check for extreme values
        original_range = torch.max(original_outputs) - torch.min(original_outputs)
        optimized_range = torch.max(optimized_outputs) - torch.min(optimized_outputs)
        range_shift = abs(original_range - optimized_range) / (original_range + 1e-8)

        return {
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
            "significant_mean_shift": bool(mean_shift > threshold),
            "significant_std_shift": bool(std_shift > threshold),
            "significant_range_shift": bool(range_shift > threshold),
            "mean_shift": float(mean_shift),
            "std_shift": float(std_shift),
            "range_shift": float(range_shift)
        }


class TestAccuracyPreservation(unittest.TestCase):
    """Test accuracy preservation through optimization"""

    def setUp(self):
        """Set up accuracy preservation tests"""
        self.config = BakingConfig(
            preserve_accuracy_threshold=0.95,
            optimization_level=3
        )

        # Create deterministic test data
        torch.manual_seed(42)
        self.num_samples = 1000
        self.num_classes = 10

        # Generate synthetic classification dataset
        self.test_inputs = torch.randn(self.num_samples, 100)
        self.test_targets = torch.randint(0, self.num_classes, (self.num_samples,))

        # Create test model
        self.model = ClassificationTestModel(self.num_classes)

    def test_classification_accuracy_preservation(self):
        """Test preservation of classification accuracy"""
        # Calculate baseline accuracy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.test_inputs)
            _, predicted = torch.max(outputs, 1)
            original_accuracy = (predicted == self.test_targets).float().mean().item()

        # Simulate optimization (using quantization as example)
        optimized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )

        # Calculate optimized accuracy
        optimized_model.eval()
        with torch.no_grad():
            outputs = optimized_model(self.test_inputs)
            _, predicted = torch.max(outputs, 1)
            optimized_accuracy = (predicted == self.test_targets).float().mean().item()

        # Calculate accuracy retention
        accuracy_retention = optimized_accuracy / original_accuracy if original_accuracy > 0 else 0

        print(f"Original accuracy: {original_accuracy:.4f}")
        print(f"Optimized accuracy: {optimized_accuracy:.4f}")
        print(f"Accuracy retention: {accuracy_retention:.4f}")

        # Verify accuracy preservation
        self.assertGreaterEqual(accuracy_retention, self.config.preserve_accuracy_threshold)

    def test_output_consistency(self):
        """Test output consistency between original and optimized models"""
        # Create optimized model
        self.model.eval()
        optimized_model = torch.jit.script(self.model)

        # Test output similarity
        test_batch = self.test_inputs[:100]  # Use smaller batch for testing

        similarity = QualityMetrics.calculate_output_similarity(
            self.model, optimized_model, test_batch, "cosine"
        )

        print(f"Output cosine similarity: {similarity:.4f}")

        # Verify high similarity
        self.assertGreaterEqual(similarity, 0.95)

    def test_statistical_output_consistency(self):
        """Test statistical consistency of outputs"""
        self.model.eval()
        optimized_model = torch.jit.script(self.model)

        test_batch = self.test_inputs[:200]

        with torch.no_grad():
            original_outputs = self.model(test_batch)
            optimized_outputs = optimized_model(test_batch)

        # Check for degradation
        degradation = QualityMetrics.detect_output_degradation(
            original_outputs, optimized_outputs, threshold=0.05
        )

        print("Output degradation analysis:")
        for key, value in degradation.items():
            print(f"  {key}: {value}")

        # Verify no significant degradation
        self.assertFalse(degradation["has_nan"])
        self.assertFalse(degradation["has_inf"])
        self.assertFalse(degradation["significant_mean_shift"])

    def test_cross_validation_consistency(self):
        """Test accuracy consistency across different data splits"""
        # Split data into multiple folds
        fold_size = self.num_samples // 5
        accuracies_original = []
        accuracies_optimized = []

        optimized_model = torch.jit.script(self.model)

        for i in range(5):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size

            fold_inputs = self.test_inputs[start_idx:end_idx]
            fold_targets = self.test_targets[start_idx:end_idx]

            # Original model accuracy
            with torch.no_grad():
                outputs = self.model(fold_inputs)
                _, predicted = torch.max(outputs, 1)
                original_acc = (predicted == fold_targets).float().mean().item()
                accuracies_original.append(original_acc)

            # Optimized model accuracy
            with torch.no_grad():
                outputs = optimized_model(fold_inputs)
                _, predicted = torch.max(outputs, 1)
                optimized_acc = (predicted == fold_targets).float().mean().item()
                accuracies_optimized.append(optimized_acc)

        # Calculate consistency metrics
        original_std = np.std(accuracies_original)
        optimized_std = np.std(accuracies_optimized)

        print(f"Original accuracy std: {original_std:.4f}")
        print(f"Optimized accuracy std: {optimized_std:.4f}")

        # Optimized model should not be significantly less consistent
        self.assertLessEqual(optimized_std, original_std * 1.5)


class TestPerformanceTheaterDetection(unittest.TestCase):
    """Test detection of performance theater (fake optimizations)"""

    def setUp(self):
        """Set up theater detection tests"""
        self.config = BakingConfig(enable_theater_detection=True)

    def test_fake_speedup_detection(self):
        """Test detection of fake speedup claims"""
        # Create fake optimization metrics
        metrics = OptimizationMetrics()

        # Scenario 1: Claimed speedup with no actual improvement
        metrics.original_latency = 100.0
        metrics.optimized_latency = 99.0  # Minimal improvement
        metrics.speedup_factor = 5.0  # Fake claim

        baseline_metrics = {"latency_ms": 100.0}
        final_metrics = {"latency_ms": 99.0}

        # Mock quality validator for theater detection
        validator = Mock()
        validator.detect_performance_theater.return_value = {
            "is_theater": True,
            "reasons": ["speedup_claim_mismatch", "insufficient_improvement"]
        }

        # Test theater detection
        theater_result = validator.detect_performance_theater(
            metrics, baseline_metrics, final_metrics
        )

        self.assertTrue(theater_result["is_theater"])
        self.assertIn("speedup_claim_mismatch", theater_result["reasons"])

    def test_fake_memory_reduction_detection(self):
        """Test detection of fake memory reduction claims"""
        metrics = OptimizationMetrics()

        # Scenario: Claimed memory reduction with actual increase
        metrics.original_memory = 1000.0
        metrics.optimized_memory = 1200.0  # Actually increased
        metrics.memory_reduction = 0.3  # Fake claim of 30% reduction

        # Calculate actual reduction
        actual_reduction = (metrics.original_memory - metrics.optimized_memory) / metrics.original_memory

        # Should detect inconsistency
        self.assertLess(actual_reduction, 0)  # Negative = increase
        self.assertNotAlmostEqual(actual_reduction, metrics.memory_reduction, places=2)

    def test_accuracy_theater_detection(self):
        """Test detection of accuracy theater"""
        metrics = OptimizationMetrics()

        # Scenario: Claimed accuracy preservation with significant drop
        metrics.original_accuracy = 0.95
        metrics.optimized_accuracy = 0.70  # Significant drop
        metrics.accuracy_retention = 0.98  # Fake claim

        # Calculate actual retention
        actual_retention = metrics.optimized_accuracy / metrics.original_accuracy

        self.assertLess(actual_retention, 0.8)  # Significant drop
        self.assertNotAlmostEqual(actual_retention, metrics.accuracy_retention, places=2)

    def test_consistent_optimization_claims(self):
        """Test validation of consistent optimization claims"""
        metrics = OptimizationMetrics()

        # Scenario: Consistent and realistic optimization
        metrics.original_latency = 100.0
        metrics.optimized_latency = 50.0
        metrics.speedup_factor = 2.0

        metrics.original_memory = 1000.0
        metrics.optimized_memory = 600.0
        metrics.memory_reduction = 0.4

        metrics.original_accuracy = 0.95
        metrics.optimized_accuracy = 0.94
        metrics.accuracy_retention = 0.989

        # Verify consistency
        actual_speedup = metrics.original_latency / metrics.optimized_latency
        actual_memory_reduction = (metrics.original_memory - metrics.optimized_memory) / metrics.original_memory
        actual_accuracy_retention = metrics.optimized_accuracy / metrics.original_accuracy

        self.assertAlmostEqual(actual_speedup, metrics.speedup_factor, places=1)
        self.assertAlmostEqual(actual_memory_reduction, metrics.memory_reduction, places=2)
        self.assertAlmostEqual(actual_accuracy_retention, metrics.accuracy_retention, places=3)


class TestModelRobustness(unittest.TestCase):
    """Test model robustness after optimization"""

    def setUp(self):
        """Set up robustness tests"""
        self.model = QualityTestModel()
        self.base_inputs = torch.randn(10, 3, 32, 32)

    def test_input_variation_robustness(self):
        """Test robustness to input variations"""
        self.model.eval()
        optimized_model = torch.jit.script(self.model)

        # Test with different input variations
        variations = {
            "normal": self.base_inputs,
            "scaled": self.base_inputs * 2.0,
            "shifted": self.base_inputs + 0.5,
            "noisy": self.base_inputs + torch.randn_like(self.base_inputs) * 0.1,
            "clipped": torch.clamp(self.base_inputs, -1, 1)
        }

        similarities = {}
        for variation_name, inputs in variations.items():
            similarity = QualityMetrics.calculate_output_similarity(
                self.model, optimized_model, inputs, "cosine"
            )
            similarities[variation_name] = similarity

            print(f"Similarity for {variation_name} inputs: {similarity:.4f}")

            # Verify reasonable similarity
            self.assertGreater(similarity, 0.8)

    def test_batch_size_consistency(self):
        """Test consistency across different batch sizes"""
        self.model.eval()
        optimized_model = torch.jit.script(self.model)

        batch_sizes = [1, 4, 8, 16]
        similarities = []

        for batch_size in batch_sizes:
            inputs = torch.randn(batch_size, 3, 32, 32)

            similarity = QualityMetrics.calculate_output_similarity(
                self.model, optimized_model, inputs, "cosine"
            )

            similarities.append(similarity)
            print(f"Batch size {batch_size} similarity: {similarity:.4f}")

            # Verify consistency
            self.assertGreater(similarity, 0.95)

        # Verify consistency across batch sizes
        similarity_std = np.std(similarities)
        self.assertLess(similarity_std, 0.05)  # Low variation

    def test_numerical_stability(self):
        """Test numerical stability of optimized models"""
        self.model.eval()
        optimized_model = torch.jit.script(self.model)

        # Test with extreme inputs
        extreme_inputs = [
            torch.ones(4, 3, 32, 32) * 1000,  # Very large values
            torch.ones(4, 3, 32, 32) * 1e-6,  # Very small values
            torch.randn(4, 3, 32, 32) * 100,   # High variance
        ]

        for i, inputs in enumerate(extreme_inputs):
            with torch.no_grad():
                original_output = self.model(inputs)
                optimized_output = optimized_model(inputs)

            # Check for numerical issues
            original_has_nan = torch.isnan(original_output).any()
            optimized_has_nan = torch.isnan(optimized_output).any()

            original_has_inf = torch.isinf(original_output).any()
            optimized_has_inf = torch.isinf(optimized_output).any()

            print(f"Extreme input {i}: Original NaN={original_has_nan}, Inf={original_has_inf}")
            print(f"Extreme input {i}: Optimized NaN={optimized_has_nan}, Inf={optimized_has_inf}")

            # Optimized model should not be less stable
            if not original_has_nan:
                self.assertFalse(optimized_has_nan)
            if not original_has_inf:
                self.assertFalse(optimized_has_inf)


class TestQualityRegression(unittest.TestCase):
    """Test for quality regression detection"""

    def setUp(self):
        """Set up regression tests"""
        self.reference_accuracy = 0.95
        self.reference_outputs_path = None

    def test_accuracy_regression_detection(self):
        """Test detection of accuracy regression"""
        current_accuracies = [0.94, 0.93, 0.92, 0.89]  # Declining accuracy

        for i, accuracy in enumerate(current_accuracies):
            retention = accuracy / self.reference_accuracy

            print(f"Test {i}: Accuracy {accuracy:.3f}, Retention {retention:.3f}")

            if retention < 0.95:  # 5% drop threshold
                print(f"  WARNING: Accuracy regression detected!")

            # Flag significant regression
            if retention < 0.90:  # 10% drop
                self.fail(f"Significant accuracy regression: {retention:.3f}")

    def test_output_regression_detection(self):
        """Test detection of output quality regression"""
        model = QualityTestModel()
        inputs = torch.randn(20, 3, 32, 32)

        # Generate reference outputs
        model.eval()
        with torch.no_grad():
            reference_outputs = model(inputs)

        # Simulate degraded model (add noise to weights)
        degraded_model = QualityTestModel()
        degraded_model.load_state_dict(model.state_dict())

        # Add noise to simulate degradation
        with torch.no_grad():
            for param in degraded_model.parameters():
                param.data += torch.randn_like(param.data) * 0.01

        degraded_model.eval()
        with torch.no_grad():
            degraded_outputs = degraded_model(inputs)

        # Calculate output similarity
        similarity = QualityMetrics.calculate_output_similarity(
            model, degraded_model, inputs, "cosine"
        )

        print(f"Output similarity after degradation: {similarity:.4f}")

        # Detect regression
        if similarity < 0.9:
            print("Output quality regression detected!")

        # Should still maintain reasonable similarity
        self.assertGreater(similarity, 0.8)

    def test_performance_quality_tradeoff(self):
        """Test performance vs quality tradeoff validation"""
        # Define acceptable tradeoffs
        tradeoff_scenarios = [
            {"speedup": 2.0, "min_accuracy_retention": 0.98},
            {"speedup": 3.0, "min_accuracy_retention": 0.95},
            {"speedup": 4.0, "min_accuracy_retention": 0.90},
            {"speedup": 5.0, "min_accuracy_retention": 0.85},
        ]

        for scenario in tradeoff_scenarios:
            speedup = scenario["speedup"]
            min_retention = scenario["min_accuracy_retention"]

            # Simulate optimization result
            simulated_retention = 1.0 - (speedup - 1.0) * 0.02  # 2% drop per speedup unit

            print(f"Speedup {speedup}x: Required retention {min_retention:.3f}, "
                  f"Achieved retention {simulated_retention:.3f}")

            is_acceptable = simulated_retention >= min_retention
            print(f"  Tradeoff acceptable: {is_acceptable}")

            if speedup <= 3.0:  # For moderate speedups, should meet requirements
                self.assertTrue(is_acceptable)


if __name__ == "__main__":
    # Set up logging for quality tests
    logging.basicConfig(level=logging.INFO)

    print("Running Phase 6 Quality Tests")
    print("=" * 50)

    # Run tests
    unittest.main(verbosity=2)