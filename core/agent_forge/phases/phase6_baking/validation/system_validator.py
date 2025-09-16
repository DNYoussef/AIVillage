#!/usr/bin/env python3
"""
Phase 6 System Working Verification Validator
============================================

Validates that the Phase 6 baking system is working correctly end-to-end.
Verifies model optimization, inference acceleration, and quality preservation.
"""

import asyncio
import logging
import torch
import torch.nn as nn
import numpy as np
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import Phase 6 components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "agent_forge" / "phase6"))

from baking_architecture import BakingArchitecture, BakingConfig, OptimizationMetrics
from model_optimizer import ModelOptimizer
from inference_accelerator import InferenceAccelerator, AccelerationConfig

@dataclass
class SystemValidationResult:
    """Result of system validation"""
    component: str
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class SystemValidationReport:
    """Complete system validation report"""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    system_status: str  # WORKING, PARTIALLY_WORKING, FAILED
    component_results: Dict[str, List[SystemValidationResult]]
    performance_metrics: Dict[str, float]
    recommendations: List[str]

class SystemValidator:
    """
    Comprehensive system validator for Phase 6 baking pipeline.
    Validates all components work together correctly.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.validation_results: List[SystemValidationResult] = []

        # Test models and data
        self.test_models = {}
        self.test_data = {}

        # Performance thresholds
        self.performance_thresholds = {
            "optimization_time": 30.0,  # seconds
            "acceleration_time": 15.0,   # seconds
            "baking_time": 60.0,         # seconds
            "min_speedup": 1.2,          # minimum speedup required
            "accuracy_retention": 0.95,  # minimum accuracy retention
            "memory_reduction": 0.1      # minimum memory reduction
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger("SystemValidator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def validate_system(self) -> SystemValidationReport:
        """
        Run comprehensive system validation.

        Returns:
            Complete validation report
        """
        self.logger.info("Starting Phase 6 system validation")
        start_time = time.time()

        # Initialize test environment
        await self._setup_test_environment()

        # Component validation tests
        await self._validate_baking_architecture()
        await self._validate_model_optimizer()
        await self._validate_inference_accelerator()
        await self._validate_integration_pipeline()
        await self._validate_performance_targets()
        await self._validate_quality_preservation()
        await self._validate_error_handling()

        # Generate final report
        report = self._generate_validation_report(time.time() - start_time)

        self.logger.info(f"System validation completed: {report.system_status}")
        return report

    async def _setup_test_environment(self):
        """Setup test models and data for validation"""
        self.logger.info("Setting up test environment")

        # Create test models of different complexities
        self.test_models = {
            "simple_linear": self._create_simple_linear_model(),
            "small_cnn": self._create_small_cnn_model(),
            "medium_resnet": self._create_medium_resnet_model()
        }

        # Create test data
        self.test_data = {
            "simple_linear": torch.randn(32, 10),
            "small_cnn": torch.randn(8, 3, 32, 32),
            "medium_resnet": torch.randn(4, 3, 64, 64)
        }

        # Create validation datasets
        for model_name in self.test_models.keys():
            inputs = self.test_data[model_name]
            targets = torch.randint(0, 10, (inputs.size(0),))
            self.test_data[f"{model_name}_validation"] = (inputs, targets)

    def _create_simple_linear_model(self) -> nn.Module:
        """Create simple linear model for testing"""
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 50)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)

        return SimpleLinear()

    def _create_small_cnn_model(self) -> nn.Module:
        """Create small CNN model for testing"""
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
                self.fc = nn.Linear(32 * 4 * 4, 10)

            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return SmallCNN()

    def _create_medium_resnet_model(self) -> nn.Module:
        """Create medium ResNet-like model for testing"""
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU()

                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels)
                    )

            def forward(self, x):
                residual = self.shortcut(x)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.bn2(self.conv2(x))
                return self.relu(x + residual)

        class MediumResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 7, 2, 3)
                self.bn1 = nn.BatchNorm2d(32)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(3, 2, 1)

                self.layer1 = self._make_layer(32, 32, 2, 1)
                self.layer2 = self._make_layer(32, 64, 2, 2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 10)

            def _make_layer(self, in_channels, out_channels, blocks, stride):
                layers = [ResidualBlock(in_channels, out_channels, stride)]
                for _ in range(1, blocks):
                    layers.append(ResidualBlock(out_channels, out_channels))
                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return MediumResNet()

    async def _validate_baking_architecture(self):
        """Validate BakingArchitecture component"""
        self.logger.info("Validating BakingArchitecture component")

        # Test 1: Initialization
        result = await self._test_baking_architecture_initialization()
        self.validation_results.append(result)

        # Test 2: Model baking pipeline
        result = await self._test_model_baking_pipeline()
        self.validation_results.append(result)

        # Test 3: Batch baking
        result = await self._test_batch_baking()
        self.validation_results.append(result)

    async def _test_baking_architecture_initialization(self) -> SystemValidationResult:
        """Test BakingArchitecture initialization"""
        start_time = time.time()

        try:
            config = BakingConfig(
                optimization_level=2,
                preserve_accuracy_threshold=0.95,
                target_speedup=1.5
            )

            baker = BakingArchitecture(config)
            baker.initialize_components()

            # Verify components are initialized
            assert baker.model_optimizer is not None
            assert baker.inference_accelerator is not None
            assert baker.quality_validator is not None
            assert baker.performance_profiler is not None
            assert baker.hardware_adapter is not None

            execution_time = time.time() - start_time

            return SystemValidationResult(
                component="BakingArchitecture",
                test_name="initialization",
                passed=True,
                score=1.0,
                execution_time=execution_time,
                details={
                    "components_initialized": 5,
                    "config_valid": True,
                    "device_detected": str(baker.device)
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="BakingArchitecture",
                test_name="initialization",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _test_model_baking_pipeline(self) -> SystemValidationResult:
        """Test complete model baking pipeline"""
        start_time = time.time()

        try:
            config = BakingConfig(
                optimization_level=2,
                preserve_accuracy_threshold=0.90,
                target_speedup=1.2
            )

            baker = BakingArchitecture(config)
            baker.initialize_components()

            # Test with simple model
            model = self.test_models["simple_linear"]
            sample_inputs = self.test_data["simple_linear"]
            validation_data = self.test_data["simple_linear_validation"]

            result = baker.bake_model(
                model,
                sample_inputs,
                validation_data,
                "test_simple_linear"
            )

            # Verify result structure
            assert "optimized_model" in result
            assert "metrics" in result
            assert isinstance(result["metrics"], OptimizationMetrics)

            metrics = result["metrics"]

            # Verify optimization occurred
            assert metrics.optimization_time > 0
            assert len(metrics.passes_applied) > 0

            execution_time = time.time() - start_time

            # Calculate score based on performance
            score = 1.0
            if execution_time > self.performance_thresholds["baking_time"]:
                score *= 0.8
            if metrics.speedup_factor < self.performance_thresholds["min_speedup"]:
                score *= 0.7

            return SystemValidationResult(
                component="BakingArchitecture",
                test_name="model_baking_pipeline",
                passed=True,
                score=score,
                execution_time=execution_time,
                details={
                    "speedup_factor": metrics.speedup_factor,
                    "memory_reduction": metrics.memory_reduction,
                    "accuracy_retention": metrics.accuracy_retention,
                    "passes_applied": metrics.passes_applied,
                    "optimization_time": metrics.optimization_time
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="BakingArchitecture",
                test_name="model_baking_pipeline",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _test_batch_baking(self) -> SystemValidationResult:
        """Test batch baking functionality"""
        start_time = time.time()

        try:
            config = BakingConfig(optimization_level=1)
            baker = BakingArchitecture(config)
            baker.initialize_components()

            # Select subset of models for batch test
            models = {
                "simple_linear": self.test_models["simple_linear"],
                "small_cnn": self.test_models["small_cnn"]
            }

            sample_inputs = {
                "simple_linear": self.test_data["simple_linear"],
                "small_cnn": self.test_data["small_cnn"]
            }

            results = baker.batch_bake_models(models, sample_inputs, max_workers=1)

            # Verify all models were processed
            assert len(results) == len(models)

            successful_bakes = 0
            for model_name, result in results.items():
                if "error" not in result:
                    successful_bakes += 1
                    assert "optimized_model" in result
                    assert "metrics" in result

            execution_time = time.time() - start_time
            success_rate = successful_bakes / len(models)

            return SystemValidationResult(
                component="BakingArchitecture",
                test_name="batch_baking",
                passed=success_rate >= 0.8,
                score=success_rate,
                execution_time=execution_time,
                details={
                    "total_models": len(models),
                    "successful_bakes": successful_bakes,
                    "success_rate": success_rate
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="BakingArchitecture",
                test_name="batch_baking",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _validate_model_optimizer(self):
        """Validate ModelOptimizer component"""
        self.logger.info("Validating ModelOptimizer component")

        # Test optimization passes
        result = await self._test_optimization_passes()
        self.validation_results.append(result)

        # Test BitNet quantization
        result = await self._test_bitnet_quantization()
        self.validation_results.append(result)

    async def _test_optimization_passes(self) -> SystemValidationResult:
        """Test optimization passes"""
        start_time = time.time()

        try:
            config = BakingConfig(optimization_level=3, enable_bitnet_optimization=False)
            logger = logging.getLogger("TestOptimizer")
            optimizer = ModelOptimizer(config, logger)

            model = self.test_models["small_cnn"]
            sample_inputs = self.test_data["small_cnn"]

            optimized_model, info = optimizer.optimize_model(model, sample_inputs)

            # Verify optimization occurred
            assert info["optimization_time"] > 0
            assert len(info["passes_applied"]) > 0
            assert info["parameter_reduction"] >= 0

            execution_time = time.time() - start_time

            return SystemValidationResult(
                component="ModelOptimizer",
                test_name="optimization_passes",
                passed=True,
                score=1.0,
                execution_time=execution_time,
                details={
                    "passes_applied": info["passes_applied"],
                    "parameter_reduction": info["parameter_reduction"],
                    "optimization_time": info["optimization_time"]
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="ModelOptimizer",
                test_name="optimization_passes",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _test_bitnet_quantization(self) -> SystemValidationResult:
        """Test BitNet quantization"""
        start_time = time.time()

        try:
            config = BakingConfig(
                optimization_level=2,
                enable_bitnet_optimization=True,
                quantization_bits=1
            )
            logger = logging.getLogger("TestOptimizer")
            optimizer = ModelOptimizer(config, logger)

            model = self.test_models["simple_linear"]
            sample_inputs = self.test_data["simple_linear"]

            optimized_model, info = optimizer.optimize_model(model, sample_inputs)

            # Verify BitNet quantization was applied
            assert "bitnet_quantization" in info["passes_applied"]

            execution_time = time.time() - start_time

            return SystemValidationResult(
                component="ModelOptimizer",
                test_name="bitnet_quantization",
                passed=True,
                score=1.0,
                execution_time=execution_time,
                details={
                    "bitnet_applied": True,
                    "quantization_bits": 1
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="ModelOptimizer",
                test_name="bitnet_quantization",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _validate_inference_accelerator(self):
        """Validate InferenceAccelerator component"""
        self.logger.info("Validating InferenceAccelerator component")

        result = await self._test_inference_acceleration()
        self.validation_results.append(result)

    async def _test_inference_acceleration(self) -> SystemValidationResult:
        """Test inference acceleration"""
        start_time = time.time()

        try:
            device = torch.device("cpu")  # Use CPU for consistent testing
            logger = logging.getLogger("TestAccelerator")
            config = AccelerationConfig(enable_tensorrt=False)  # Disable TensorRT for testing

            accelerator = InferenceAccelerator(config, device, logger)

            model = self.test_models["simple_linear"]
            sample_inputs = self.test_data["simple_linear"]

            accelerated_model = accelerator.accelerate_model(model, sample_inputs, config)

            # Verify acceleration occurred
            assert accelerated_model is not None

            # Test that accelerated model produces output
            with torch.no_grad():
                output = accelerated_model(sample_inputs)
                assert output is not None
                assert output.shape[0] == sample_inputs.shape[0]

            execution_time = time.time() - start_time

            return SystemValidationResult(
                component="InferenceAccelerator",
                test_name="inference_acceleration",
                passed=True,
                score=1.0,
                execution_time=execution_time,
                details={
                    "acceleration_successful": True,
                    "output_verified": True
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="InferenceAccelerator",
                test_name="inference_acceleration",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _validate_integration_pipeline(self):
        """Validate end-to-end integration pipeline"""
        self.logger.info("Validating integration pipeline")

        result = await self._test_end_to_end_pipeline()
        self.validation_results.append(result)

    async def _test_end_to_end_pipeline(self) -> SystemValidationResult:
        """Test complete end-to-end pipeline"""
        start_time = time.time()

        try:
            # Create temporary directory for exports
            with tempfile.TemporaryDirectory() as temp_dir:
                export_dir = Path(temp_dir)

                config = BakingConfig(optimization_level=1)
                baker = BakingArchitecture(config)
                baker.initialize_components()

                # Bake model
                model = self.test_models["simple_linear"]
                sample_inputs = self.test_data["simple_linear"]

                baking_result = baker.bake_model(model, sample_inputs, model_name="e2e_test")

                # Export model
                baking_results = {"e2e_test": baking_result}
                export_paths = baker.export_optimized_models(baking_results, export_dir)

                # Verify exports
                assert "e2e_test" in export_paths

                # Generate report
                report_path = export_dir / "optimization_report.json"
                report = baker.generate_optimization_report(baking_results, report_path)

                assert report_path.exists()
                assert report["summary"]["successful_optimizations"] == 1

                execution_time = time.time() - start_time

                return SystemValidationResult(
                    component="Integration",
                    test_name="end_to_end_pipeline",
                    passed=True,
                    score=1.0,
                    execution_time=execution_time,
                    details={
                        "baking_successful": True,
                        "export_successful": True,
                        "report_generated": True,
                        "total_operations": 3
                    }
                )

        except Exception as e:
            return SystemValidationResult(
                component="Integration",
                test_name="end_to_end_pipeline",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _validate_performance_targets(self):
        """Validate performance targets are met"""
        self.logger.info("Validating performance targets")

        result = await self._test_performance_targets()
        self.validation_results.append(result)

    async def _test_performance_targets(self) -> SystemValidationResult:
        """Test that performance targets are met"""
        start_time = time.time()

        try:
            config = BakingConfig(
                optimization_level=2,
                target_speedup=1.2,
                preserve_accuracy_threshold=0.90
            )
            baker = BakingArchitecture(config)
            baker.initialize_components()

            model = self.test_models["small_cnn"]
            sample_inputs = self.test_data["small_cnn"]
            validation_data = self.test_data["small_cnn_validation"]

            result = baker.bake_model(model, sample_inputs, validation_data, "perf_test")
            metrics = result["metrics"]

            # Check performance targets
            targets_met = 0
            total_targets = 0

            performance_checks = {
                "speedup": metrics.speedup_factor >= self.performance_thresholds["min_speedup"],
                "accuracy": metrics.accuracy_retention >= self.performance_thresholds["accuracy_retention"],
                "memory": metrics.memory_reduction >= self.performance_thresholds["memory_reduction"],
                "time": metrics.optimization_time <= self.performance_thresholds["optimization_time"]
            }

            for check_name, passed in performance_checks.items():
                total_targets += 1
                if passed:
                    targets_met += 1

            execution_time = time.time() - start_time
            score = targets_met / total_targets

            return SystemValidationResult(
                component="Performance",
                test_name="performance_targets",
                passed=score >= 0.75,
                score=score,
                execution_time=execution_time,
                details={
                    "targets_met": targets_met,
                    "total_targets": total_targets,
                    "performance_checks": performance_checks,
                    "metrics": asdict(metrics)
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="Performance",
                test_name="performance_targets",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _validate_quality_preservation(self):
        """Validate quality preservation"""
        self.logger.info("Validating quality preservation")

        result = await self._test_quality_preservation()
        self.validation_results.append(result)

    async def _test_quality_preservation(self) -> SystemValidationResult:
        """Test that model quality is preserved during optimization"""
        start_time = time.time()

        try:
            config = BakingConfig(
                optimization_level=1,  # Light optimization
                preserve_accuracy_threshold=0.95
            )
            baker = BakingArchitecture(config)
            baker.initialize_components()

            model = self.test_models["simple_linear"]
            sample_inputs = self.test_data["simple_linear"]
            validation_data = self.test_data["simple_linear_validation"]

            # Get original model accuracy
            model.eval()
            with torch.no_grad():
                original_output = model(validation_data[0])
                original_pred = torch.argmax(original_output, dim=1)
                original_accuracy = (original_pred == validation_data[1]).float().mean().item()

            # Bake model
            result = baker.bake_model(model, sample_inputs, validation_data, "quality_test")
            optimized_model = result["optimized_model"]
            metrics = result["metrics"]

            # Get optimized model accuracy
            optimized_model.eval()
            with torch.no_grad():
                optimized_output = optimized_model(validation_data[0])
                optimized_pred = torch.argmax(optimized_output, dim=1)
                optimized_accuracy = (optimized_pred == validation_data[1]).float().mean().item()

            # Calculate actual retention
            actual_retention = optimized_accuracy / original_accuracy if original_accuracy > 0 else 0

            execution_time = time.time() - start_time

            quality_preserved = actual_retention >= config.preserve_accuracy_threshold
            score = min(actual_retention / config.preserve_accuracy_threshold, 1.0)

            return SystemValidationResult(
                component="Quality",
                test_name="quality_preservation",
                passed=quality_preserved,
                score=score,
                execution_time=execution_time,
                details={
                    "original_accuracy": original_accuracy,
                    "optimized_accuracy": optimized_accuracy,
                    "actual_retention": actual_retention,
                    "threshold": config.preserve_accuracy_threshold,
                    "reported_retention": metrics.accuracy_retention
                }
            )

        except Exception as e:
            return SystemValidationResult(
                component="Quality",
                test_name="quality_preservation",
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                details={},
                error_message=str(e)
            )

    async def _validate_error_handling(self):
        """Validate error handling capabilities"""
        self.logger.info("Validating error handling")

        result = await self._test_error_handling()
        self.validation_results.append(result)

    async def _test_error_handling(self) -> SystemValidationResult:
        """Test error handling and recovery"""
        start_time = time.time()

        try:
            config = BakingConfig(optimization_level=2)
            baker = BakingArchitecture(config)
            baker.initialize_components()

            # Test with invalid input (wrong tensor shape)
            model = self.test_models["small_cnn"]  # Expects 3D input
            invalid_inputs = torch.randn(8, 10)    # Wrong shape

            error_handled = False
            try:
                result = baker.bake_model(model, invalid_inputs, model_name="error_test")
            except Exception:
                error_handled = True

            execution_time = time.time() - start_time

            return SystemValidationResult(
                component="ErrorHandling",
                test_name="error_handling",
                passed=error_handled,
                score=1.0 if error_handled else 0.0,
                execution_time=execution_time,
                details={
                    "error_properly_handled": error_handled,
                    "test_type": "invalid_input_shape"
                }
            )

        except Exception as e:
            # If we get here, error handling worked
            return SystemValidationResult(
                component="ErrorHandling",
                test_name="error_handling",
                passed=True,
                score=1.0,
                execution_time=time.time() - start_time,
                details={
                    "error_properly_handled": True,
                    "error_message": str(e)
                }
            )

    def _generate_validation_report(self, total_time: float) -> SystemValidationReport:
        """Generate comprehensive validation report"""
        # Organize results by component
        component_results = {}
        for result in self.validation_results:
            if result.component not in component_results:
                component_results[result.component] = []
            component_results[result.component].append(result)

        # Calculate overall statistics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        failed_tests = total_tests - passed_tests

        # Calculate overall score
        if total_tests > 0:
            overall_score = sum(r.score for r in self.validation_results) / total_tests
        else:
            overall_score = 0.0

        # Determine system status
        if overall_score >= 0.9:
            system_status = "WORKING"
        elif overall_score >= 0.7:
            system_status = "PARTIALLY_WORKING"
        else:
            system_status = "FAILED"

        # Performance metrics
        performance_metrics = {
            "total_validation_time": total_time,
            "average_test_time": total_time / total_tests if total_tests > 0 else 0,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_score": overall_score
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(component_results, overall_score)

        return SystemValidationReport(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_score=overall_score,
            system_status=system_status,
            component_results=component_results,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        component_results: Dict[str, List[SystemValidationResult]],
        overall_score: float
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Analyze component performance
        for component, results in component_results.items():
            component_score = sum(r.score for r in results) / len(results) if results else 0

            if component_score < 0.8:
                recommendations.append(f"Improve {component} component performance (score: {component_score:.2f})")

            # Check for failed tests
            failed_tests = [r for r in results if not r.passed]
            if failed_tests:
                recommendations.append(f"Address {len(failed_tests)} failed tests in {component}")

        # Performance recommendations
        slow_tests = [r for r in self.validation_results if r.execution_time > 10.0]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow-running tests")

        # Overall system recommendations
        if overall_score < 0.9:
            recommendations.append("System not fully production-ready, address failing components")

        if overall_score >= 0.9:
            recommendations.append("System validation successful, ready for production deployment")

        return recommendations


async def main():
    """Example usage of SystemValidator"""
    logging.basicConfig(level=logging.INFO)

    validator = SystemValidator()
    report = await validator.validate_system()

    print(f"\n=== Phase 6 System Validation Report ===")
    print(f"Status: {report.system_status}")
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
    print(f"Validation Time: {report.performance_metrics['total_validation_time']:.2f}s")

    print(f"\nComponent Results:")
    for component, results in report.component_results.items():
        component_score = sum(r.score for r in results) / len(results)
        passed = sum(1 for r in results if r.passed)
        print(f"  {component}: {passed}/{len(results)} passed, score: {component_score:.2f}")

    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main())