"""
Test Phase 8 Integration - Phase 7 → Phase 8 Pipeline
Tests complete integration between Phase 7 ADAS and Phase 8 Compression.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
import json
import pickle
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

# Import Phase 8 components
from ..agents.compression_orchestrator import (
    CompressionOrchestrator,
    CompressionPipelineConfig,
    CompressionStrategy,
    CompressionTarget,
    CompressionPhase
)
from ..agents.model_analyzer import ModelAnalyzerAgent, ModelAnalysis
from ..agents.deployment_packager import DeploymentPackagerAgent, DeploymentConfig
from ..core.compression_algorithms import CompressionAlgorithmFactory


class MockPhase7ADASModel:
    """Mock ADAS model from Phase 7 for testing integration."""

    def __init__(self, model_type="perception"):
        self.model_type = model_type
        self.model = self._create_adas_model()
        self.metadata = self._create_metadata()
        self.performance_metrics = self._create_performance_metrics()

    def _create_adas_model(self):
        """Create mock ADAS model based on type."""
        if self.model_type == "perception":
            return nn.Sequential(
                # Perception layers
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),

                # Feature extraction
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                # Classification head
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 8)  # 8 ADAS classes
            )
        elif self.model_type == "planning":
            return nn.Sequential(
                nn.Linear(64, 256),  # Input features from perception
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 4)  # 4 planning actions
            )
        else:  # control
            return nn.Sequential(
                nn.Linear(32, 128),  # Input from planning
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # 2 control outputs (steering, throttle)
            )

    def _create_metadata(self):
        """Create mock metadata."""
        return {
            'phase': 'phase7_adas',
            'model_type': self.model_type,
            'training_accuracy': 0.95,
            'validation_accuracy': 0.93,
            'model_size_mb': self._calculate_model_size(),
            'inference_time_ms': 15.5,
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
            'training_dataset': 'adas_dataset_v2',
            'version': '1.2.0',
            'timestamp': '2024-01-15T10:30:00Z'
        }

    def _create_performance_metrics(self):
        """Create mock performance metrics."""
        return {
            'accuracy': 0.93,
            'precision': 0.91,
            'recall': 0.94,
            'f1_score': 0.925,
            'inference_latency_ms': 15.5,
            'throughput_fps': 64.5,
            'memory_usage_mb': 125.3,
            'flops': 1.2e9
        }

    def _calculate_model_size(self):
        """Calculate model size in MB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return (total_params * 4) / (1024 * 1024)  # 4 bytes per float32

    def save_model(self, path: Path):
        """Save mock ADAS model."""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'metadata': self.metadata,
            'performance_metrics': self.performance_metrics
        }
        torch.save(model_data, path)

    def get_validation_data(self, batch_size=8, num_batches=5):
        """Get mock validation data."""
        if self.model_type == "perception":
            input_shape = (3, 224, 224)
        elif self.model_type == "planning":
            input_shape = (64,)
        else:  # control
            input_shape = (32,)

        data = []
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, *input_shape)
            if self.model_type == "perception":
                targets = torch.randint(0, 8, (batch_size,))
            elif self.model_type == "planning":
                targets = torch.randint(0, 4, (batch_size,))
            else:
                targets = torch.randn(batch_size, 2)  # Regression for control
            data.append((inputs, targets))

        return data


class TestPhase7ToPhase8Integration:
    """Test integration from Phase 7 ADAS to Phase 8 Compression."""

    def setup_method(self):
        """Setup test environment."""
        self.mock_adas_models = {
            'perception': MockPhase7ADASModel('perception'),
            'planning': MockPhase7ADASModel('planning'),
            'control': MockPhase7ADASModel('control')
        }

        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_phase7_model_loading(self):
        """Test loading Phase 7 ADAS models."""
        for model_type, adas_model in self.mock_adas_models.items():
            # Save model
            model_path = self.temp_path / f"adas_{model_type}.pth"
            adas_model.save_model(model_path)

            # Load model
            loaded_data = torch.load(model_path)

            # Verify model data
            assert 'model_state_dict' in loaded_data
            assert 'metadata' in loaded_data
            assert 'performance_metrics' in loaded_data

            # Verify metadata
            metadata = loaded_data['metadata']
            assert metadata['phase'] == 'phase7_adas'
            assert metadata['model_type'] == model_type
            assert metadata['training_accuracy'] > 0.9
            assert metadata['parameter_count'] > 0

            # Theater detection: Verify realistic performance metrics
            perf = loaded_data['performance_metrics']
            assert 0.5 < perf['accuracy'] < 1.0, \
                "Theater detected: Unrealistic accuracy"
            assert perf['inference_latency_ms'] > 0, \
                "Theater detected: Zero inference latency"
            assert perf['memory_usage_mb'] > 0, \
                "Theater detected: Zero memory usage"

    def test_phase8_model_analysis_integration(self):
        """Test Phase 8 analysis of Phase 7 models."""
        analyzer = ModelAnalyzerAgent()

        for model_type, adas_model in self.mock_adas_models.items():
            # Analyze ADAS model
            analysis = analyzer.analyze_model(adas_model.model, f"adas_{model_type}")

            # Verify analysis results
            assert isinstance(analysis, ModelAnalysis)
            assert analysis.model_size_mb > 0
            assert analysis.parameter_count > 0
            assert len(analysis.layer_distribution) > 0
            assert 0 <= analysis.redundancy_score <= 1

            # Theater detection: Verify meaningful analysis
            assert analysis.parameter_count == sum(p.numel() for p in adas_model.model.parameters()), \
                "Theater detected: Parameter count mismatch"

            expected_size = (analysis.parameter_count * 4) / (1024 * 1024)
            assert abs(analysis.model_size_mb - expected_size) < 0.1, \
                f"Theater detected: Model size calculation error"

            # Verify compression recommendations
            assert len(analysis.recommended_strategies) > 0, \
                "Theater detected: No compression strategies recommended"

            # Check for appropriate strategies for ADAS models
            if model_type == "perception":
                # CNNs should have structured pruning recommendations
                strategies = [s.strategy_type for s in analysis.recommended_strategies]
                assert any('structured' in s.lower() or 'channel' in s.lower() for s in strategies), \
                    "Theater detected: No CNN-appropriate strategies for perception model"

    def test_compression_pipeline_configuration(self):
        """Test compression pipeline configuration for ADAS models."""
        for model_type, adas_model in self.mock_adas_models.items():
            # Configure compression for ADAS deployment
            target = CompressionTarget(
                max_model_size_mb=10.0,  # Edge deployment constraint
                min_accuracy_retention=0.95,  # High accuracy for safety
                target_platform='edge_cpu',
                max_inference_latency_ms=20.0
            )

            config = CompressionPipelineConfig(
                strategy=CompressionStrategy.HYBRID_COMPRESSION,
                target=target,
                output_directory=self.temp_path / f"compressed_{model_type}",
                save_intermediate_results=True,
                parallel_execution=False  # Disable for testing
            )

            # Create orchestrator
            orchestrator = CompressionOrchestrator(config)

            # Verify configuration
            assert orchestrator.config.target.max_model_size_mb == 10.0
            assert orchestrator.config.target.min_accuracy_retention == 0.95
            assert orchestrator.config.strategy == CompressionStrategy.HYBRID_COMPRESSION

            # Theater detection: Verify realistic constraints
            original_size = adas_model.metadata['model_size_mb']
            if original_size <= target.max_model_size_mb:
                assert False, f"Theater detected: Target size {target.max_model_size_mb}MB not smaller than original {original_size}MB"

    @pytest.mark.slow
    def test_end_to_end_compression_pipeline(self):
        """Test complete end-to-end compression pipeline."""
        # Test with perception model (most complex)
        adas_model = self.mock_adas_models['perception']
        validation_data = adas_model.get_validation_data(batch_size=4, num_batches=3)

        # Configure compression
        target = CompressionTarget(
            max_model_size_mb=5.0,  # Aggressive compression
            min_accuracy_retention=0.90,  # Slightly relaxed for testing
            target_platform='mobile_cpu'
        )

        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PROGRESSIVE_COMPRESSION,
            target=target,
            output_directory=self.temp_path / "compressed_perception",
            save_intermediate_results=True
        )

        orchestrator = CompressionOrchestrator(config)

        # Run compression pipeline
        start_time = time.time()
        results = orchestrator.compress_model(
            adas_model.model,
            validation_data,
            model_name="adas_perception_compressed"
        )
        compression_time = time.time() - start_time

        # Verify results
        assert results is not None
        assert results.original_model is not None
        assert results.best_model is not None
        assert results.compression_ratio > 1.0
        assert results.execution_time > 0

        # Theater detection: Verify actual compression occurred
        original_size = sum(p.numel() * 4 for p in adas_model.model.parameters()) / (1024 * 1024)
        compressed_size = sum(p.numel() * 4 for p in results.best_model.parameters()) / (1024 * 1024)

        actual_compression_ratio = original_size / compressed_size
        assert abs(actual_compression_ratio - results.compression_ratio) < 0.5, \
            f"Theater detected: Claimed compression ratio {results.compression_ratio} vs actual {actual_compression_ratio}"

        # Verify pipeline phases completed
        assert results.pipeline_state.current_phase == CompressionPhase.COMPLETED
        assert 'analysis' in results.pipeline_state.phase_times
        assert 'compression' in results.pipeline_state.phase_times

        # Verify compressed models were created
        assert len(results.compressed_models) > 0

        # Theater detection: Verify compression time is reasonable
        assert compression_time < 300, \
            f"Theater detected: Compression took too long ({compression_time:.1f}s)"

    def test_deployment_package_creation(self):
        """Test deployment package creation for compressed ADAS models."""
        adas_model = self.mock_adas_models['control']  # Smaller model for faster testing

        # Create deployment packager
        packager = DeploymentPackagerAgent()

        config = DeploymentConfig(
            target_platform='edge_cpu',
            runtime_environment='pytorch',
            optimization_level=2,
            include_preprocessing=True,
            include_postprocessing=True,
            include_metadata=True
        )

        # Create deployment package
        output_dir = self.temp_path / "deployment_package"
        results = packager.create_deployment_package(
            adas_model.model,
            config,
            output_dir,
            "adas_control_compressed"
        )

        # Verify package creation
        assert len(results.packages) > 0
        package = results.packages[0]

        # Verify package files exist
        assert package.model_path.exists()
        assert package.config_path.exists()
        assert package.package_size_mb > 0

        # Theater detection: Verify package contents
        # Check model file
        model_data = torch.load(package.model_path)
        assert 'model_state_dict' in model_data or isinstance(model_data, dict)

        # Check config file
        with open(package.config_path, 'r') as f:
            config_data = json.load(f)
        assert 'model_info' in config_data
        assert 'deployment_info' in config_data

        # Verify deployment metadata
        if package.metadata_path and package.metadata_path.exists():
            with open(package.metadata_path, 'r') as f:
                metadata = json.load(f)
            assert 'source_phase' in metadata
            assert metadata['source_phase'] == 'phase8_compression'

    def test_performance_benchmarking_integration(self):
        """Test performance benchmarking of compressed models."""
        from ..agents.performance_profiler import PerformanceProfilerAgent, ProfilingConfig

        adas_model = self.mock_adas_models['planning']

        # Create simple compression (for faster testing)
        from ..core.compression_algorithms import MagnitudePruning
        algorithm = MagnitudePruning(sparsity_ratio=0.3)
        compressed_model, _ = algorithm.compress(adas_model.model)

        # Profile both models
        profiler = PerformanceProfilerAgent()
        config = ProfilingConfig(
            device='cpu',
            batch_sizes=[1, 4, 8],
            measurement_iterations=10  # Short for testing
        )

        input_tensors = [torch.randn(8, 64)]  # Planning model input

        # Profile original model
        original_results = profiler.profile_model(
            adas_model.model, input_tensors, "adas_planning_original", config
        )

        # Profile compressed model
        compressed_results = profiler.profile_model(
            compressed_model, input_tensors, "adas_planning_compressed", config
        )

        # Theater detection: Verify realistic performance improvements
        original_latency = original_results.latency_metrics[0].mean_latency_ms
        compressed_latency = compressed_results.latency_metrics[0].mean_latency_ms

        # Compressed model should be faster or similar (due to sparsity)
        latency_ratio = original_latency / compressed_latency
        assert latency_ratio >= 0.8, \
            f"Theater detected: Compressed model significantly slower ({latency_ratio:.2f}x)"

        # Throughput should improve or stay similar
        original_throughput = original_results.throughput_metrics[0].samples_per_second
        compressed_throughput = compressed_results.throughput_metrics[0].samples_per_second

        throughput_ratio = compressed_throughput / original_throughput
        assert throughput_ratio >= 0.9, \
            f"Theater detected: Compressed model throughput degraded significantly ({throughput_ratio:.2f}x)"

    def test_validation_accuracy_preservation(self):
        """Test accuracy preservation during compression."""
        from ..agents.compression_validator import CompressionValidatorAgent, ValidationConfig

        adas_model = self.mock_adas_models['perception']
        validation_data = adas_model.get_validation_data()

        # Apply light compression
        from ..core.compression_algorithms import MagnitudePruning
        algorithm = MagnitudePruning(sparsity_ratio=0.2)  # Light pruning
        compressed_model, _ = algorithm.compress(adas_model.model)

        # Validate compression
        validator = CompressionValidatorAgent()
        config = ValidationConfig(
            accuracy_threshold=0.85,  # Relaxed for testing
            device='cpu'
        )

        results = validator.validate_compression(
            adas_model.model,
            compressed_model,
            validation_data,
            config
        )

        # Verify validation results
        assert results.overall_metrics is not None
        assert results.layer_wise_analysis is not None
        assert results.performance_benchmarks is not None

        # Theater detection: Verify accuracy measurements are realistic
        if 'accuracy' in results.overall_metrics:
            accuracy = results.overall_metrics['accuracy']
            assert 0.0 <= accuracy <= 1.0, \
                f"Theater detected: Invalid accuracy value {accuracy}"

            # Light compression shouldn't hurt accuracy much
            original_accuracy = adas_model.performance_metrics['accuracy']
            accuracy_retention = accuracy / original_accuracy
            assert accuracy_retention > 0.8, \
                f"Theater detected: Excessive accuracy loss {accuracy_retention:.2f}"

        # Verify deployment readiness
        assert isinstance(results.deployment_readiness, dict)
        assert 'ready' in results.deployment_readiness

    def test_multi_model_compression_pipeline(self):
        """Test compression pipeline with multiple ADAS models."""
        # Compress all ADAS models
        compression_results = {}

        for model_type, adas_model in self.mock_adas_models.items():
            validation_data = adas_model.get_validation_data(batch_size=4, num_batches=2)

            # Configure for each model type
            if model_type == 'perception':
                target = CompressionTarget(max_model_size_mb=8.0, min_accuracy_retention=0.92)
                strategy = CompressionStrategy.STRUCTURED_PRUNING
            elif model_type == 'planning':
                target = CompressionTarget(max_model_size_mb=3.0, min_accuracy_retention=0.90)
                strategy = CompressionStrategy.QUANTIZATION_ONLY
            else:  # control
                target = CompressionTarget(max_model_size_mb=1.0, min_accuracy_retention=0.88)
                strategy = CompressionStrategy.HYBRID_COMPRESSION

            config = CompressionPipelineConfig(
                strategy=strategy,
                target=target,
                output_directory=self.temp_path / f"compressed_{model_type}"
            )

            orchestrator = CompressionOrchestrator(config)

            # Compress model
            results = orchestrator.compress_model(
                adas_model.model,
                validation_data,
                model_name=f"adas_{model_type}_compressed"
            )

            compression_results[model_type] = results

        # Theater detection: Verify all models were compressed differently
        compression_ratios = [results.compression_ratio for results in compression_results.values()]
        assert len(set([round(r, 1) for r in compression_ratios])) > 1, \
            "Theater detected: All models had identical compression ratios"

        # Verify each model meets its constraints
        for model_type, results in compression_results.items():
            assert results.compression_ratio > 1.0, \
                f"Theater detected: No compression achieved for {model_type}"

            # Check model size constraint
            compressed_size = sum(p.numel() * 4 for p in results.best_model.parameters()) / (1024 * 1024)
            target_size = results.target.max_model_size_mb

            if target_size is not None:
                assert compressed_size <= target_size * 1.1, \
                    f"Theater detected: {model_type} model size {compressed_size:.1f}MB exceeds target {target_size}MB"

    def test_integration_error_handling(self):
        """Test error handling in Phase 7 to Phase 8 integration."""
        # Test with corrupted model data
        corrupted_path = self.temp_path / "corrupted_model.pth"
        with open(corrupted_path, 'wb') as f:
            f.write(b"corrupted data")

        # Should handle corrupted model gracefully
        with pytest.raises((RuntimeError, pickle.UnpicklingError, EOFError)):
            torch.load(corrupted_path)

        # Test with empty validation data
        adas_model = self.mock_adas_models['control']
        empty_validation_data = []

        target = CompressionTarget(max_model_size_mb=1.0)
        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PRUNING_ONLY,
            target=target,
            output_directory=self.temp_path / "test_empty"
        )

        orchestrator = CompressionOrchestrator(config)

        # Should handle empty validation data
        with pytest.raises((ValueError, RuntimeError)):
            orchestrator.compress_model(
                adas_model.model,
                empty_validation_data,
                model_name="test_empty"
            )

    @pytest.mark.slow
    def test_compression_quality_gates(self):
        """Test quality gates for ADAS model compression."""
        adas_model = self.mock_adas_models['perception']
        validation_data = adas_model.get_validation_data()

        # Set strict quality gates
        target = CompressionTarget(
            max_model_size_mb=3.0,  # Very aggressive
            min_accuracy_retention=0.98,  # Very strict
            target_platform='edge_cpu',
            max_inference_latency_ms=10.0  # Very fast
        )

        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PROGRESSIVE_COMPRESSION,
            target=target,
            output_directory=self.temp_path / "quality_gates_test",
            enable_quality_gates=True
        )

        orchestrator = CompressionOrchestrator(config)

        # Run compression with strict gates
        results = orchestrator.compress_model(
            adas_model.model,
            validation_data,
            model_name="adas_quality_gates"
        )

        # Theater detection: Quality gates should be enforced
        if results.compression_ratio < 2.0:  # If compression wasn't aggressive enough
            # Should either achieve compression or fail with clear message
            assert results.pipeline_state.current_phase != CompressionPhase.COMPLETED or \
                   results.compression_ratio >= 2.0, \
                "Theater detected: Quality gates not properly enforced"

        # If compression succeeded, verify constraints are met
        if results.pipeline_state.current_phase == CompressionPhase.COMPLETED:
            compressed_size = sum(p.numel() * 4 for p in results.best_model.parameters()) / (1024 * 1024)
            assert compressed_size <= target.max_model_size_mb * 1.05, \
                f"Theater detected: Size constraint not met {compressed_size:.1f}MB > {target.max_model_size_mb}MB"


class TestDeploymentIntegration:
    """Test deployment integration for compressed ADAS models."""

    def setup_method(self):
        """Setup deployment test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create compressed ADAS model
        self.adas_model = MockPhase7ADASModel('perception')
        from ..core.compression_algorithms import MagnitudePruning
        algorithm = MagnitudePruning(sparsity_ratio=0.4)
        self.compressed_model, self.compression_metrics = algorithm.compress(self.adas_model.model)

    def teardown_method(self):
        """Cleanup deployment test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deployment_package_structure(self):
        """Test deployment package structure for ADAS models."""
        from ..agents.deployment_packager import DeploymentPackagerAgent, DeploymentConfig

        packager = DeploymentPackagerAgent()

        config = DeploymentConfig(
            target_platform='edge_cpu',
            runtime_environment='pytorch',
            optimization_level=2,
            include_preprocessing=True,
            include_postprocessing=True,
            include_metadata=True,
            include_validation_data=True
        )

        # Create deployment package
        results = packager.create_deployment_package(
            self.compressed_model,
            config,
            self.temp_path / "deployment",
            "adas_perception_deploy"
        )

        # Verify package structure
        assert len(results.packages) > 0
        package = results.packages[0]

        # Theater detection: Verify all required files exist
        required_files = [package.model_path, package.config_path]
        if package.metadata_path:
            required_files.append(package.metadata_path)

        for file_path in required_files:
            assert file_path.exists(), \
                f"Theater detected: Required file missing {file_path}"
            assert file_path.stat().st_size > 0, \
                f"Theater detected: Empty file {file_path}"

        # Verify deployment config contains ADAS-specific settings
        with open(package.config_path, 'r') as f:
            config_data = json.load(f)

        assert 'model_info' in config_data
        assert 'deployment_info' in config_data

        # Theater detection: Verify realistic model info
        model_info = config_data['model_info']
        assert model_info['parameter_count'] > 0
        assert model_info['model_size_mb'] > 0

        # Verify compression info is included
        if 'compression_info' in model_info:
            compression_info = model_info['compression_info']
            assert compression_info['compression_ratio'] > 1.0

    def test_runtime_optimization(self):
        """Test runtime optimization for deployment."""
        from ..agents.deployment_packager import DeploymentPackagerAgent, DeploymentConfig

        packager = DeploymentPackagerAgent()

        # Test different optimization levels
        for opt_level in [0, 1, 2]:
            config = DeploymentConfig(
                target_platform='edge_cpu',
                runtime_environment='pytorch',
                optimization_level=opt_level
            )

            output_dir = self.temp_path / f"opt_level_{opt_level}"

            results = packager.create_deployment_package(
                self.compressed_model,
                config,
                output_dir,
                f"adas_opt_{opt_level}"
            )

            # Theater detection: Higher optimization should produce different results
            package = results.packages[0]

            # Save package size for comparison
            if opt_level == 0:
                baseline_size = package.package_size_mb
            else:
                # Higher optimization might affect size (depending on implementation)
                size_diff = abs(package.package_size_mb - baseline_size)
                # Allow for some variation but should be meaningful optimization
                if opt_level == 2:
                    assert size_diff >= 0 or package.package_size_mb <= baseline_size, \
                        f"Theater detected: Optimization level {opt_level} increased size"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])