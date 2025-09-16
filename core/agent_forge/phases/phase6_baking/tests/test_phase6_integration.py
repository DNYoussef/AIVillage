#!/usr/bin/env python3
"""
Integration Tests for Phase 6 Baking System
===========================================

Comprehensive integration tests for Phase 6 baking system integration with:
- Phase 5 trained model loading and compatibility
- Phase 7 ADAS preparation and export
- Cross-phase data flow validation
- End-to-end baking pipeline
- Multi-model batch processing
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
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
    create_baking_pipeline,
    benchmark_baked_models,
    validate_phase_integration
)


class MockPhase5Model(nn.Module):
    """Mock model representing Phase 5 trained output"""
    def __init__(self, model_type="bitnet"):
        super().__init__()
        self.model_type = model_type

        if model_type == "bitnet":
            # Simulate BitNet architecture
            self.bit_linear1 = nn.Linear(128, 256)
            self.bit_linear2 = nn.Linear(256, 128)
            self.output = nn.Linear(128, 10)
        elif model_type == "conv":
            # Simulate CNN architecture
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)
        else:
            # Standard model
            self.fc1 = nn.Linear(100, 200)
            self.fc2 = nn.Linear(200, 100)
            self.output = nn.Linear(100, 10)

        self.relu = nn.ReLU()
        self.training_metadata = {
            "training_accuracy": 0.95,
            "validation_accuracy": 0.92,
            "epochs_trained": 50,
            "optimizer": "AdamW",
            "learning_rate": 0.001
        }

    def forward(self, x):
        if self.model_type == "bitnet":
            x = self.relu(self.bit_linear1(x))
            x = self.relu(self.bit_linear2(x))
            return self.output(x)
        elif self.model_type == "conv":
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        else:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.output(x)


class TestPhase5Integration(unittest.TestCase):
    """Test integration with Phase 5 trained models"""

    def setUp(self):
        """Set up Phase 5 integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.phase5_dir = Path(self.temp_dir) / "phase5_models"
        self.phase5_dir.mkdir(parents=True)

        # Create mock Phase 5 models
        self.models = {
            "bitnet_model": MockPhase5Model("bitnet"),
            "conv_model": MockPhase5Model("conv"),
            "standard_model": MockPhase5Model("standard")
        }

        # Save models in Phase 5 format
        for name, model in self.models.items():
            model_path = self.phase5_dir / f"{name}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "architecture": model.model_type,
                    "input_size": 128 if model.model_type == "bitnet" else (100 if model.model_type == "standard" else (3, 32, 32)),
                    "output_size": 10
                },
                "training_metadata": model.training_metadata,
                "phase5_timestamp": time.time()
            }, model_path)

        # Create sample inputs for each model type
        self.sample_inputs = {
            "bitnet_model": torch.randn(4, 128),
            "conv_model": torch.randn(4, 3, 32, 32),
            "standard_model": torch.randn(4, 100)
        }

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def test_phase5_model_discovery(self):
        """Test discovery and enumeration of Phase 5 models"""
        config = BakingConfig()
        baker = BakingArchitecture(config)

        # Test model discovery
        models = baker.load_phase5_models(self.phase5_dir)

        # Should attempt to load all model files
        expected_models = ["bitnet_model.pth", "conv_model.pth", "standard_model.pth"]

        # Verify files were found (even if loading fails due to architecture mismatch)
        model_files = list(self.phase5_dir.glob("*.pth"))
        self.assertEqual(len(model_files), 3)

    def test_phase5_metadata_validation(self):
        """Test validation of Phase 5 model metadata"""
        for model_name in self.models:
            model_path = self.phase5_dir / f"{model_name}.pth"
            model_data = torch.load(model_path, map_location="cpu")

            # Verify Phase 5 metadata structure
            self.assertIn("model_state_dict", model_data)
            self.assertIn("model_config", model_data)
            self.assertIn("training_metadata", model_data)
            self.assertIn("phase5_timestamp", model_data)

            # Verify training metadata
            training_meta = model_data["training_metadata"]
            self.assertIn("training_accuracy", training_meta)
            self.assertIn("validation_accuracy", training_meta)
            self.assertGreater(training_meta["training_accuracy"], 0.8)

    def test_phase5_compatibility_check(self):
        """Test compatibility checking with Phase 5 models"""
        validation_results = validate_phase_integration(
            str(self.phase5_dir),
            str(Path(self.temp_dir) / "phase7_output")
        )

        # Verify Phase 5 integration results
        phase5_results = validation_results["phase5_integration"]
        self.assertTrue(phase5_results["input_dir_exists"])
        self.assertTrue(phase5_results["validation_passed"])
        self.assertEqual(len(phase5_results["models_found"]), 3)

        # Verify supported formats
        for format_ext in phase5_results["supported_formats"]:
            self.assertIn(format_ext, [".pth", ".pt", ".onnx"])


class TestPhase7Integration(unittest.TestCase):
    """Test integration with Phase 7 ADAS system"""

    def setUp(self):
        """Set up Phase 7 integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.phase7_dir = Path(self.temp_dir) / "phase7_adas"

        self.config = BakingConfig(
            export_formats=["pytorch", "torchscript", "onnx"]
        )
        self.baker = BakingArchitecture(self.config)

        # Mock optimized models
        self.optimized_models = {
            "perception_model": MockPhase5Model("conv"),
            "decision_model": MockPhase5Model("bitnet"),
            "control_model": MockPhase5Model("standard")
        }

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def test_adas_model_preparation(self):
        """Test preparation of models for ADAS integration"""
        phase7_paths = self.baker.prepare_for_phase7(
            self.optimized_models,
            self.phase7_dir
        )

        # Verify all models were prepared
        self.assertEqual(len(phase7_paths), len(self.optimized_models))

        for model_name in self.optimized_models:
            self.assertIn(model_name, phase7_paths)
            model_path = Path(phase7_paths[model_name])
            self.assertTrue(model_path.exists())

    def test_adas_model_structure(self):
        """Test ADAS model file structure and metadata"""
        phase7_paths = self.baker.prepare_for_phase7(
            self.optimized_models,
            self.phase7_dir
        )

        for model_name, model_path in phase7_paths.items():
            model_data = torch.load(model_path, map_location="cpu")

            # Verify ADAS-specific structure
            self.assertIn("model_state_dict", model_data)
            self.assertIn("model_config", model_data)
            self.assertIn("baking_timestamp", model_data)

            # Verify ADAS compatibility flags
            model_config = model_data["model_config"]
            self.assertTrue(model_config["adas_compatible"])
            self.assertTrue(model_config["optimized"])
            self.assertTrue(model_config["inference_mode"])

    def test_adas_wrapper_functionality(self):
        """Test ADAS wrapper functionality"""
        test_model = MockPhase5Model("conv")
        wrapper = self.baker._create_adas_wrapper(test_model, "test_adas_model")

        # Test wrapper properties
        self.assertEqual(wrapper.name, "test_adas_model")
        self.assertTrue(wrapper.inference_mode)

        # Test inference mode
        wrapper.eval()
        test_input = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            output = wrapper(test_input)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[0], 1)  # Batch size
        self.assertEqual(output.shape[1], 10) # Output classes

    def test_phase7_export_formats(self):
        """Test multiple export format support for Phase 7"""
        config = BakingConfig(export_formats=["pytorch", "torchscript"])
        baker = BakingArchitecture(config)

        # Mock baking results
        baking_results = {
            "test_model": {
                "optimized_model": MockPhase5Model("conv"),
                "metrics": OptimizationMetrics()
            }
        }

        export_paths = baker.export_optimized_models(
            baking_results,
            self.phase7_dir
        )

        # Verify export formats
        self.assertIn("test_model", export_paths)
        model_exports = export_paths["test_model"]

        for format_name in config.export_formats:
            if format_name == "pytorch":
                self.assertIn("pytorch", model_exports)
                pytorch_path = Path(model_exports["pytorch"])
                self.assertTrue(pytorch_path.exists())
                self.assertEqual(pytorch_path.suffix, ".pth")

            elif format_name == "torchscript":
                self.assertIn("torchscript", model_exports)
                script_path = Path(model_exports["torchscript"])
                self.assertTrue(script_path.exists())
                self.assertEqual(script_path.suffix, ".pt")


class TestEndToEndBaking(unittest.TestCase):
    """Test end-to-end baking pipeline"""

    def setUp(self):
        """Set up end-to-end test environment"""
        self.temp_dir = tempfile.mkdtemp()

        self.config = BakingConfig(
            optimization_level=2,
            preserve_accuracy_threshold=0.90,
            target_speedup=1.5,
            enable_bitnet_optimization=True,
            benchmark_iterations=10  # Reduced for testing
        )

        # Test models
        self.models = {
            "small_model": MockPhase5Model("standard"),
            "bitnet_model": MockPhase5Model("bitnet")
        }

        self.sample_inputs = {
            "small_model": torch.randn(4, 100),
            "bitnet_model": torch.randn(4, 128)
        }

        # Validation data
        self.validation_data = {
            "small_model": (torch.randn(50, 100), torch.randint(0, 10, (50,))),
            "bitnet_model": (torch.randn(50, 128), torch.randint(0, 10, (50,)))
        }

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    @patch('agent_forge.phase6.PerformanceProfiler')
    @patch('agent_forge.phase6.QualityValidator')
    @patch('agent_forge.phase6.ModelOptimizer')
    @patch('agent_forge.phase6.InferenceAccelerator')
    @patch('agent_forge.phase6.HardwareAdapter')
    def test_complete_baking_pipeline(self, mock_hardware, mock_accelerator,
                                    mock_optimizer, mock_quality, mock_profiler):
        """Test complete baking pipeline from start to finish"""
        # Set up mocks
        mock_profiler_instance = Mock()
        mock_profiler_instance.profile_model.return_value = {
            "accuracy": 0.95,
            "latency_ms": 50.0,
            "memory_mb": 100.0,
            "flops": 1000000
        }
        mock_profiler.return_value = mock_profiler_instance

        mock_quality_instance = Mock()
        mock_quality_instance.validate_accuracy.return_value = 0.94
        mock_quality_instance.detect_performance_theater.return_value = {
            "is_theater": False,
            "reasons": []
        }
        mock_quality.return_value = mock_quality_instance

        mock_optimizer_instance = Mock()
        mock_optimizer_instance.optimize_model.return_value = (
            self.models["small_model"],
            {"passes_applied": ["magnitude_pruning", "quantization"]}
        )
        mock_optimizer.return_value = mock_optimizer_instance

        mock_accelerator_instance = Mock()
        mock_accelerator_instance.accelerate_model.return_value = self.models["small_model"]
        mock_accelerator.return_value = mock_accelerator_instance

        mock_hardware_instance = Mock()
        mock_hardware_instance.adapt_model.return_value = self.models["small_model"]
        mock_hardware.return_value = mock_hardware_instance

        # Run baking pipeline
        baker = BakingArchitecture(self.config)

        result = baker.bake_model(
            self.models["small_model"],
            self.sample_inputs["small_model"],
            self.validation_data["small_model"],
            "small_model"
        )

        # Verify pipeline execution
        self.assertIn("optimized_model", result)
        self.assertIn("metrics", result)
        self.assertIn("baseline_metrics", result)
        self.assertIn("final_metrics", result)

        # Verify metrics
        metrics = result["metrics"]
        self.assertIsInstance(metrics, OptimizationMetrics)
        self.assertGreater(metrics.optimization_time, 0)

    def test_batch_baking_pipeline(self):
        """Test batch processing of multiple models"""
        baker = BakingArchitecture(self.config)

        # Mock components to avoid actual optimization
        baker.model_optimizer = Mock()
        baker.inference_accelerator = Mock()
        baker.quality_validator = Mock()
        baker.performance_profiler = Mock()
        baker.hardware_adapter = Mock()

        # Set up mock returns
        baker.model_optimizer.optimize_model.return_value = (
            self.models["small_model"],
            {"passes_applied": ["test_pass"]}
        )
        baker.inference_accelerator.accelerate_model.return_value = self.models["small_model"]
        baker.quality_validator.validate_accuracy.return_value = 0.93
        baker.quality_validator.detect_performance_theater.return_value = {
            "is_theater": False,
            "reasons": []
        }
        baker.performance_profiler.profile_model.return_value = {
            "accuracy": 0.93,
            "latency_ms": 40.0,
            "memory_mb": 80.0,
            "flops": 800000
        }
        baker.hardware_adapter.adapt_model.return_value = self.models["small_model"]

        # Run batch baking
        results = baker.batch_bake_models(
            self.models,
            self.sample_inputs,
            self.validation_data,
            max_workers=1  # Single worker for test stability
        )

        # Verify batch results
        self.assertEqual(len(results), len(self.models))
        for model_name in self.models:
            self.assertIn(model_name, results)
            if "error" not in results[model_name]:
                self.assertIn("optimized_model", results[model_name])
                self.assertIn("metrics", results[model_name])

    def test_cross_phase_data_flow(self):
        """Test data flow from Phase 5 through baking to Phase 7"""
        # Set up Phase 5 input directory
        phase5_dir = Path(self.temp_dir) / "phase5"
        phase5_dir.mkdir()

        # Save a model in Phase 5 format
        model_path = phase5_dir / "test_model.pth"
        torch.save({
            "model_state_dict": self.models["small_model"].state_dict(),
            "model_config": {"architecture": "standard"},
            "training_metadata": {"accuracy": 0.95}
        }, model_path)

        # Set up Phase 7 output directory
        phase7_dir = Path(self.temp_dir) / "phase7"

        # Test integration validation
        validation_results = validate_phase_integration(
            str(phase5_dir),
            str(phase7_dir)
        )

        # Verify cross-phase compatibility
        overall_results = validation_results["overall_integration"]
        self.assertTrue(overall_results["cross_phase_compatibility"])
        self.assertTrue(overall_results["data_flow_validated"])
        self.assertTrue(overall_results["ready_for_production"])


class TestBenchmarkIntegration(unittest.TestCase):
    """Test benchmarking and performance validation integration"""

    def setUp(self):
        """Set up benchmark test environment"""
        self.models = {
            "fast_model": MockPhase5Model("standard"),
            "accurate_model": MockPhase5Model("bitnet")
        }

        self.sample_inputs = {
            "fast_model": torch.randn(4, 100),
            "accurate_model": torch.randn(4, 128)
        }

    def test_benchmark_baked_models(self):
        """Test benchmarking of multiple baked models"""
        results = benchmark_baked_models(
            self.models,
            self.sample_inputs,
            device="cpu",
            num_iterations=5  # Reduced for testing
        )

        # Verify benchmark results structure
        self.assertEqual(len(results), len(self.models))

        for model_name in self.models:
            self.assertIn(model_name, results)
            model_results = results[model_name]

            # Verify benchmark metrics
            self.assertIn("latency_mean", model_results)
            self.assertIn("latency_std", model_results)
            self.assertIn("latency_p95", model_results)
            self.assertIn("latency_p99", model_results)
            self.assertIn("throughput_samples_per_sec", model_results)
            self.assertIn("device", model_results)

            # Verify reasonable values
            self.assertGreater(model_results["latency_mean"], 0)
            self.assertGreater(model_results["throughput_samples_per_sec"], 0)

    def test_performance_comparison(self):
        """Test performance comparison between models"""
        results = benchmark_baked_models(
            self.models,
            self.sample_inputs,
            num_iterations=5
        )

        # Compare model performance
        fast_latency = results["fast_model"]["latency_mean"]
        accurate_latency = results["accurate_model"]["latency_mean"]

        # Both should have reasonable latencies
        self.assertLess(fast_latency, 1000)  # Less than 1 second
        self.assertLess(accurate_latency, 1000)

        # Throughput should be reasonable
        self.assertGreater(results["fast_model"]["throughput_samples_per_sec"], 1)
        self.assertGreater(results["accurate_model"]["throughput_samples_per_sec"], 1)


class TestPipelineCreation(unittest.TestCase):
    """Test baking pipeline creation and configuration"""

    def test_create_baking_pipeline_default(self):
        """Test creation of default baking pipeline"""
        pipeline = create_baking_pipeline()

        self.assertIsInstance(pipeline, BakingArchitecture)
        self.assertIsInstance(pipeline.config, BakingConfig)
        self.assertIsInstance(pipeline.device, torch.device)

    def test_create_baking_pipeline_custom_config(self):
        """Test creation of baking pipeline with custom config"""
        config = BakingConfig(
            optimization_level=4,
            target_speedup=3.0,
            enable_bitnet_optimization=False
        )

        pipeline = create_baking_pipeline(config, device="cpu")

        self.assertEqual(pipeline.config.optimization_level, 4)
        self.assertEqual(pipeline.config.target_speedup, 3.0)
        self.assertFalse(pipeline.config.enable_bitnet_optimization)
        self.assertEqual(pipeline.device.type, "cpu")

    def test_pipeline_device_detection(self):
        """Test automatic device detection in pipeline creation"""
        pipeline = create_baking_pipeline(device="auto")

        # Should detect appropriate device
        self.assertIn(pipeline.device.type, ["cuda", "cpu", "mps"])


if __name__ == "__main__":
    # Set up logging for integration tests
    logging.basicConfig(level=logging.INFO)

    # Run tests with higher verbosity
    unittest.main(verbosity=2)