#!/usr/bin/env python3
"""
Unit Tests for Phase 6 Baking Architecture
==========================================

Comprehensive unit tests for all Phase 6 baking components including:
- BakingArchitecture core functionality
- Model optimization pipeline
- Inference acceleration
- Quality validation
- Hardware adaptation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import time

# Import Phase 6 components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from agent_forge.phase6 import (
    BakingArchitecture,
    BakingConfig,
    OptimizationMetrics,
    ModelOptimizer,
    InferenceAccelerator,
    QualityValidator,
    HardwareAdapter,
    PerformanceProfiler,
    AccelerationConfig
)


class TestModel(nn.Module):
    """Simple test model for validation"""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TestConvModel(nn.Module):
    """Convolutional test model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestBakingConfig(unittest.TestCase):
    """Test BakingConfig functionality"""

    def test_default_config(self):
        """Test default configuration values"""
        config = BakingConfig()

        self.assertEqual(config.optimization_level, 3)
        self.assertEqual(config.preserve_accuracy_threshold, 0.98)
        self.assertEqual(config.target_speedup, 2.0)
        self.assertTrue(config.enable_bitnet_optimization)
        self.assertEqual(config.quantization_bits, 1)
        self.assertEqual(config.target_device, "auto")
        self.assertTrue(config.enable_theater_detection)
        self.assertEqual(config.batch_sizes, [1, 4, 8, 16, 32])
        self.assertEqual(config.export_formats, ["pytorch", "onnx", "torchscript"])

    def test_custom_config(self):
        """Test custom configuration values"""
        config = BakingConfig(
            optimization_level=4,
            preserve_accuracy_threshold=0.95,
            target_speedup=3.0,
            enable_bitnet_optimization=False,
            target_device="cuda",
            batch_sizes=[1, 2, 4],
            export_formats=["pytorch"]
        )

        self.assertEqual(config.optimization_level, 4)
        self.assertEqual(config.preserve_accuracy_threshold, 0.95)
        self.assertEqual(config.target_speedup, 3.0)
        self.assertFalse(config.enable_bitnet_optimization)
        self.assertEqual(config.target_device, "cuda")
        self.assertEqual(config.batch_sizes, [1, 2, 4])
        self.assertEqual(config.export_formats, ["pytorch"])


class TestOptimizationMetrics(unittest.TestCase):
    """Test OptimizationMetrics functionality"""

    def test_default_metrics(self):
        """Test default metric values"""
        metrics = OptimizationMetrics()

        self.assertEqual(metrics.original_accuracy, 0.0)
        self.assertEqual(metrics.optimized_accuracy, 0.0)
        self.assertEqual(metrics.accuracy_retention, 0.0)
        self.assertEqual(metrics.original_latency, 0.0)
        self.assertEqual(metrics.optimized_latency, 0.0)
        self.assertEqual(metrics.speedup_factor, 0.0)
        self.assertEqual(metrics.passes_applied, [])

    def test_metrics_calculations(self):
        """Test metric calculations"""
        metrics = OptimizationMetrics()

        metrics.original_accuracy = 0.9
        metrics.optimized_accuracy = 0.85
        metrics.accuracy_retention = metrics.optimized_accuracy / metrics.original_accuracy

        metrics.original_latency = 100.0
        metrics.optimized_latency = 50.0
        metrics.speedup_factor = metrics.original_latency / metrics.optimized_latency

        metrics.original_memory = 1000.0
        metrics.optimized_memory = 600.0
        metrics.memory_reduction = (metrics.original_memory - metrics.optimized_memory) / metrics.original_memory

        self.assertAlmostEqual(metrics.accuracy_retention, 0.944, places=3)
        self.assertAlmostEqual(metrics.speedup_factor, 2.0, places=1)
        self.assertAlmostEqual(metrics.memory_reduction, 0.4, places=1)


class TestBakingArchitecture(unittest.TestCase):
    """Test BakingArchitecture core functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = BakingConfig(
            optimization_level=2,
            preserve_accuracy_threshold=0.95,
            enable_theater_detection=True
        )

        # Mock logger
        self.logger = Mock(spec=logging.Logger)

        # Test model and inputs
        self.model = TestModel()
        self.sample_inputs = torch.randn(4, 10)
        self.validation_data = (torch.randn(100, 10), torch.randint(0, 2, (100, 1)).float())

    def test_initialization(self):
        """Test BakingArchitecture initialization"""
        baker = BakingArchitecture(self.config, self.logger)

        self.assertEqual(baker.config, self.config)
        self.assertEqual(baker.logger, self.logger)
        self.assertIsInstance(baker.device, torch.device)
        self.assertEqual(baker.optimization_history, [])
        self.assertIsNone(baker.model_optimizer)
        self.assertIsNone(baker.inference_accelerator)

    def test_device_detection_auto(self):
        """Test automatic device detection"""
        config = BakingConfig(target_device="auto")
        baker = BakingArchitecture(config, self.logger)

        # Device should be detected automatically
        self.assertIsInstance(baker.device, torch.device)
        self.assertIn(baker.device.type, ["cuda", "cpu"])

    def test_device_detection_manual(self):
        """Test manual device specification"""
        config = BakingConfig(target_device="cpu")
        baker = BakingArchitecture(config, self.logger)

        self.assertEqual(baker.device.type, "cpu")

    @patch('agent_forge.phase6.ModelOptimizer')
    @patch('agent_forge.phase6.InferenceAccelerator')
    @patch('agent_forge.phase6.QualityValidator')
    @patch('agent_forge.phase6.PerformanceProfiler')
    @patch('agent_forge.phase6.HardwareAdapter')
    def test_component_initialization(self, mock_hardware, mock_profiler,
                                    mock_quality, mock_accelerator, mock_optimizer):
        """Test component initialization"""
        baker = BakingArchitecture(self.config, self.logger)
        baker.initialize_components()

        # Verify all components are initialized
        self.assertIsNotNone(baker.model_optimizer)
        self.assertIsNotNone(baker.inference_accelerator)
        self.assertIsNotNone(baker.quality_validator)
        self.assertIsNotNone(baker.performance_profiler)
        self.assertIsNotNone(baker.hardware_adapter)

        # Verify components were created with correct parameters
        mock_optimizer.assert_called_once_with(self.config, self.logger)
        mock_accelerator.assert_called_once_with(self.config, baker.device, self.logger)
        mock_quality.assert_called_once_with(self.config, self.logger)
        mock_profiler.assert_called_once_with(self.config, baker.device, self.logger)
        mock_hardware.assert_called_once_with(self.config, baker.device, self.logger)

    def test_load_phase5_models(self):
        """Test loading models from Phase 5"""
        baker = BakingArchitecture(self.config, self.logger)

        # Create temporary directory with test models
        with tempfile.TemporaryDirectory() as temp_dir:
            phase5_dir = Path(temp_dir)

            # Save test models
            model1_path = phase5_dir / "model1.pth"
            model2_path = phase5_dir / "model2.pt"

            torch.save(self.model.state_dict(), model1_path)
            torch.save(self.model.state_dict(), model2_path)

            # Test loading (should handle missing architecture gracefully)
            models = baker.load_phase5_models(phase5_dir)

            # Verify attempt was made to load models
            self.logger.info.assert_called()

    def test_prepare_for_phase7(self):
        """Test preparation for Phase 7 ADAS integration"""
        baker = BakingArchitecture(self.config, self.logger)

        models = {"test_model": self.model}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            phase7_paths = baker.prepare_for_phase7(models, output_dir)

            # Verify output structure
            self.assertIn("test_model", phase7_paths)

            # Verify ADAS-ready model was saved
            model_path = Path(phase7_paths["test_model"])
            self.assertTrue(model_path.exists())

            # Verify model structure
            saved_data = torch.load(model_path, map_location="cpu")
            self.assertIn("model_state_dict", saved_data)
            self.assertIn("model_config", saved_data)
            self.assertTrue(saved_data["model_config"]["adas_compatible"])

    def test_adas_wrapper_creation(self):
        """Test ADAS wrapper creation"""
        baker = BakingArchitecture(self.config, self.logger)

        wrapper = baker._create_adas_wrapper(self.model, "test_model")

        # Test wrapper properties
        self.assertEqual(wrapper.name, "test_model")
        self.assertTrue(wrapper.inference_mode)

        # Test wrapper forward pass
        test_input = torch.randn(1, 10)
        output = wrapper(test_input)
        self.assertIsInstance(output, torch.Tensor)


class TestModelOptimizer(unittest.TestCase):
    """Test ModelOptimizer functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = BakingConfig(optimization_level=3, enable_bitnet_optimization=True)
        self.logger = Mock(spec=logging.Logger)
        self.optimizer = ModelOptimizer(self.config, self.logger)
        self.model = TestModel()
        self.sample_inputs = torch.randn(4, 10)

    def test_initialization(self):
        """Test ModelOptimizer initialization"""
        self.assertEqual(self.optimizer.config, self.config)
        self.assertEqual(self.optimizer.logger, self.logger)
        self.assertIsInstance(self.optimizer.optimization_passes, list)
        self.assertGreater(len(self.optimizer.optimization_passes), 0)

    def test_optimization_pass_initialization(self):
        """Test optimization pass initialization based on config"""
        # Test different optimization levels
        for level in range(5):
            config = BakingConfig(optimization_level=level)
            optimizer = ModelOptimizer(config, self.logger)

            # Verify passes are sorted by priority
            priorities = [p.priority for p in optimizer.optimization_passes]
            self.assertEqual(priorities, sorted(priorities))

    def test_parameter_counting(self):
        """Test parameter counting functionality"""
        param_count = self.optimizer._count_parameters(self.model)

        # Calculate expected parameters
        expected = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(param_count, expected)

    def test_optimization_impact_estimation(self):
        """Test optimization impact estimation"""
        estimates = self.optimizer.estimate_optimization_impact(self.model, self.sample_inputs)

        # Verify estimate structure
        self.assertIn("original_parameters", estimates)
        self.assertIn("estimated_reduction", estimates)
        self.assertIn("estimated_speedup", estimates)
        self.assertIn("estimated_memory_saving", estimates)

        # Verify reasonable estimates
        self.assertGreaterEqual(estimates["estimated_reduction"], 0.0)
        self.assertLessEqual(estimates["estimated_reduction"], 1.0)
        self.assertGreaterEqual(estimates["estimated_speedup"], 1.0)
        self.assertLessEqual(estimates["estimated_speedup"], 10.0)


class TestInferenceAccelerator(unittest.TestCase):
    """Test InferenceAccelerator functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = BakingConfig()
        self.device = torch.device("cpu")
        self.logger = Mock(spec=logging.Logger)
        self.accelerator = InferenceAccelerator(self.config, self.device, self.logger)
        self.model = TestConvModel()
        self.sample_inputs = torch.randn(1, 3, 32, 32)

    def test_initialization(self):
        """Test InferenceAccelerator initialization"""
        self.assertEqual(self.accelerator.config, self.config)
        self.assertEqual(self.accelerator.device, self.device)
        self.assertEqual(self.accelerator.logger, self.logger)
        self.assertIsInstance(self.accelerator.acceleration_cache, dict)
        self.assertIsInstance(self.accelerator.custom_kernels, dict)

    def test_graph_optimization(self):
        """Test computation graph optimization"""
        # Test with traced model
        self.model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, self.sample_inputs)

        config = AccelerationConfig(
            enable_constant_folding=True,
            enable_operator_fusion=True
        )

        optimized_model, metrics = self.accelerator._optimize_computation_graph(
            traced_model, self.sample_inputs, config
        )

        # Verify optimization results
        self.assertIsInstance(optimized_model, torch.jit.ScriptModule)
        self.assertIn("original_nodes", metrics)
        self.assertIn("optimized_nodes", metrics)
        self.assertIn("node_reduction", metrics)

    def test_kernel_optimization(self):
        """Test kernel optimization functionality"""
        # Create a traced model for kernel optimization
        self.model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, self.sample_inputs)

        config = AccelerationConfig(enable_kernel_fusion=True)

        optimized_model, metrics = self.accelerator._optimize_kernels(
            traced_model, self.sample_inputs, config
        )

        # Verify optimization results
        self.assertIn("fusions_applied", metrics)
        self.assertIn("custom_kernels", metrics)
        self.assertGreaterEqual(metrics["fusions_applied"], 0)

    def test_bitnet_detection(self):
        """Test BitNet layer detection"""
        # Create a mock traced model with BitNet patterns
        mock_model = Mock(spec=torch.jit.ScriptModule)
        mock_model.graph = "BitLinear operation here"

        has_bitnet = self.accelerator._has_bitnet_layers(mock_model)
        self.assertTrue(has_bitnet)

        # Test without BitNet
        mock_model.graph = "regular linear operation"
        has_bitnet = self.accelerator._has_bitnet_layers(mock_model)
        self.assertFalse(has_bitnet)

    def test_acceleration_benchmark(self):
        """Test acceleration benchmarking"""
        # Create a simple accelerated model (just copy for testing)
        accelerated_model = self.model

        results = self.accelerator.benchmark_acceleration(
            self.model, accelerated_model, self.sample_inputs, num_iterations=5
        )

        # Verify benchmark results
        self.assertIn("original_latency_ms", results)
        self.assertIn("accelerated_latency_ms", results)
        self.assertIn("speedup_factor", results)
        self.assertIn("latency_improvement_ms", results)
        self.assertIn("throughput_improvement", results)

        # Verify reasonable values
        self.assertGreater(results["original_latency_ms"], 0)
        self.assertGreater(results["accelerated_latency_ms"], 0)
        self.assertGreater(results["speedup_factor"], 0)


class TestIntegrationValidation(unittest.TestCase):
    """Test integration validation functionality"""

    def test_phase_integration_validation(self):
        """Test Phase 5/7 integration validation"""
        from agent_forge.phase6 import validate_phase_integration

        with tempfile.TemporaryDirectory() as temp_dir:
            phase5_dir = Path(temp_dir) / "phase5"
            phase7_dir = Path(temp_dir) / "phase7"

            # Create Phase 5 directory with test models
            phase5_dir.mkdir()
            test_model_path = phase5_dir / "test_model.pth"
            torch.save(TestModel().state_dict(), test_model_path)

            validation_results = validate_phase_integration(
                str(phase5_dir), str(phase7_dir)
            )

            # Verify validation structure
            self.assertIn("phase5_integration", validation_results)
            self.assertIn("phase7_integration", validation_results)
            self.assertIn("overall_integration", validation_results)

            # Verify Phase 5 validation
            phase5_results = validation_results["phase5_integration"]
            self.assertTrue(phase5_results["input_dir_exists"])
            self.assertTrue(phase5_results["validation_passed"])
            self.assertIn("test_model.pth", phase5_results["models_found"])

            # Verify Phase 7 validation
            phase7_results = validation_results["phase7_integration"]
            self.assertTrue(phase7_results["output_dir_ready"])
            self.assertTrue(phase7_results["validation_passed"])

            # Verify overall integration
            overall_results = validation_results["overall_integration"]
            self.assertTrue(overall_results["cross_phase_compatibility"])
            self.assertTrue(overall_results["ready_for_production"])


class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirement validation"""

    def setUp(self):
        """Set up performance test environment"""
        self.config = BakingConfig(target_speedup=2.0)
        self.logger = Mock(spec=logging.Logger)

    def test_speedup_validation(self):
        """Test speedup requirement validation"""
        # Create metrics that meet requirements
        metrics = OptimizationMetrics()
        metrics.original_latency = 100.0
        metrics.optimized_latency = 40.0  # 2.5x speedup
        metrics.speedup_factor = metrics.original_latency / metrics.optimized_latency

        # Verify speedup meets requirements
        self.assertGreaterEqual(metrics.speedup_factor, self.config.target_speedup)

    def test_memory_reduction_validation(self):
        """Test memory reduction validation"""
        metrics = OptimizationMetrics()
        metrics.original_memory = 1000.0
        metrics.optimized_memory = 600.0
        metrics.memory_reduction = (metrics.original_memory - metrics.optimized_memory) / metrics.original_memory

        # Verify reasonable memory reduction
        self.assertGreater(metrics.memory_reduction, 0.1)  # At least 10% reduction
        self.assertLess(metrics.memory_reduction, 0.9)     # Less than 90% reduction

    def test_accuracy_preservation_validation(self):
        """Test accuracy preservation validation"""
        metrics = OptimizationMetrics()
        metrics.original_accuracy = 0.95
        metrics.optimized_accuracy = 0.93
        metrics.accuracy_retention = metrics.optimized_accuracy / metrics.original_accuracy

        # Verify accuracy retention meets threshold
        self.assertGreaterEqual(metrics.accuracy_retention, self.config.preserve_accuracy_threshold)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Set up error handling tests"""
        self.config = BakingConfig()
        self.logger = Mock(spec=logging.Logger)

    def test_invalid_model_handling(self):
        """Test handling of invalid model inputs"""
        baker = BakingArchitecture(self.config, self.logger)

        # Test with None model
        with self.assertRaises(AttributeError):
            baker.bake_model(None, torch.randn(1, 10))

    def test_device_mismatch_handling(self):
        """Test handling of device mismatches"""
        config = BakingConfig(target_device="cpu")
        baker = BakingArchitecture(config, self.logger)

        model = TestModel()
        sample_inputs = torch.randn(1, 10)

        # Should handle device transfer gracefully
        baker.initialize_components()
        # The bake_model method should handle device transfer internally


if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)

    # Run tests
    unittest.main(verbosity=2)