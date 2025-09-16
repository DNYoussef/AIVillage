"""
Phase 8 Compression - Comprehensive Test Suite

Tests for the complete compression pipeline including all agents,
algorithms, optimization, and validation components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
import json
import copy

# Import Phase 8 components
from ..agents.model_analyzer import ModelAnalyzerAgent
from ..agents.pruning_agent import PruningAgent, PruningConfig
from ..agents.quantization_agent import QuantizationAgent, QuantizationConfig
from ..agents.knowledge_distiller import KnowledgeDistillationAgent, DistillationConfig
from ..agents.architecture_optimizer import ArchitectureOptimizerAgent, ArchitectureConfig
from ..agents.compression_validator import CompressionValidatorAgent, ValidationConfig
from ..agents.deployment_packager import DeploymentPackagerAgent, DeploymentConfig
from ..agents.performance_profiler import PerformanceProfilerAgent, ProfilingConfig
from ..agents.compression_orchestrator import (
    CompressionOrchestrator,
    CompressionPipelineConfig,
    CompressionStrategy,
    CompressionTarget
)

from ..core.compression_algorithms import (
    CompressionAlgorithmFactory,
    MagnitudePruning,
    WeightClustering,
    SVDCompression
)

from ..optimization.compression_optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    OptimizationObjective
)

from ..validation.model_validator import (
    ModelValidationFramework,
    ValidationThresholds
)


class TestModelFixtures:
    """Test model fixtures for compression testing."""

    @staticmethod
    def create_simple_cnn():
        """Create simple CNN for testing."""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )

    @staticmethod
    def create_simple_mlp():
        """Create simple MLP for testing."""
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    @staticmethod
    def create_validation_data(batch_size=8, num_batches=5, input_size=(3, 32, 32)):
        """Create dummy validation data."""
        data = []
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, *input_size)
            targets = torch.randint(0, 10, (batch_size,))
            data.append((inputs, targets))
        return data

    @staticmethod
    def create_training_data(batch_size=16, num_batches=10, input_size=(3, 32, 32)):
        """Create dummy training data."""
        return TestModelFixtures.create_validation_data(batch_size, num_batches, input_size)


class TestModelAnalyzer:
    """Test ModelAnalyzerAgent."""

    def test_model_analyzer_initialization(self):
        """Test model analyzer initialization."""
        analyzer = ModelAnalyzerAgent()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_model')

    def test_cnn_analysis(self):
        """Test CNN model analysis."""
        model = TestModelFixtures.create_simple_cnn()
        analyzer = ModelAnalyzerAgent()

        analysis = analyzer.analyze_model(model, "test_cnn")

        assert analysis.model_size_mb > 0
        assert analysis.parameter_count > 0
        assert len(analysis.layer_distribution) > 0
        assert 0 <= analysis.redundancy_score <= 1
        assert len(analysis.recommended_strategies) > 0

    def test_mlp_analysis(self):
        """Test MLP model analysis."""
        model = TestModelFixtures.create_simple_mlp()
        analyzer = ModelAnalyzerAgent()

        analysis = analyzer.analyze_model(model, "test_mlp")

        assert analysis.model_size_mb > 0
        assert analysis.parameter_count > 0
        assert 'Linear' in analysis.layer_distribution
        assert isinstance(analysis.pruning_candidates, list)

    def test_analysis_caching(self):
        """Test analysis result caching."""
        model = TestModelFixtures.create_simple_cnn()
        analyzer = ModelAnalyzerAgent()

        # First analysis
        analysis1 = analyzer.analyze_model(model, "test_model")

        # Check cached result
        cached_analysis = analyzer.get_cached_analysis("test_model")
        assert cached_analysis is not None
        assert cached_analysis.parameter_count == analysis1.parameter_count


class TestPruningAgent:
    """Test PruningAgent."""

    def test_pruning_agent_initialization(self):
        """Test pruning agent initialization."""
        agent = PruningAgent()
        assert agent is not None
        assert hasattr(agent, 'prune_model')

    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        model = TestModelFixtures.create_simple_mlp()
        original_params = sum(p.numel() for p in model.parameters())

        agent = PruningAgent()
        config = PruningConfig(
            strategy='magnitude',
            sparsity_ratio=0.5,
            granularity='unstructured',
            schedule='oneshot'
        )

        results = agent.prune_model(model, config)

        assert results.original_params == original_params
        assert 0 <= results.actual_sparsity <= 1.0
        assert len(results.layers_pruned) > 0

    def test_structured_pruning(self):
        """Test structured pruning."""
        model = TestModelFixtures.create_simple_cnn()
        agent = PruningAgent()

        config = PruningConfig(
            strategy='magnitude',
            sparsity_ratio=0.3,
            granularity='structured',
            schedule='oneshot'
        )

        results = agent.prune_model(model, config)

        assert results.original_params > 0
        assert results.actual_sparsity > 0

    def test_gradual_pruning(self):
        """Test gradual pruning schedule."""
        model = TestModelFixtures.create_simple_mlp()
        agent = PruningAgent()

        config = PruningConfig(
            strategy='magnitude',
            sparsity_ratio=0.4,
            schedule='gradual'
        )

        results = agent.prune_model(model, config)
        assert results.actual_sparsity > 0


class TestQuantizationAgent:
    """Test QuantizationAgent."""

    def test_quantization_agent_initialization(self):
        """Test quantization agent initialization."""
        agent = QuantizationAgent()
        assert agent is not None

    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = TestModelFixtures.create_simple_mlp()
        agent = QuantizationAgent()

        config = QuantizationConfig(
            strategy='dynamic',
            bit_width=8,
            target_dtype='int8'
        )

        quantized_model, results = agent.quantize_model(model, config)

        assert results.original_size_mb > 0
        assert results.quantized_size_mb > 0
        assert results.compression_ratio >= 1.0
        assert len(results.layers_quantized) >= 0

    def test_quantization_benchmarking(self):
        """Test quantization strategy benchmarking."""
        model = TestModelFixtures.create_simple_mlp()
        agent = QuantizationAgent()

        strategies = ['dynamic']
        bit_widths = [8]

        benchmark_results = agent.benchmark_quantization_strategies(
            model, strategies, bit_widths
        )

        assert len(benchmark_results) > 0
        for result in benchmark_results.values():
            assert result.compression_ratio >= 1.0


class TestKnowledgeDistiller:
    """Test KnowledgeDistillationAgent."""

    def test_distillation_agent_initialization(self):
        """Test knowledge distillation agent initialization."""
        agent = KnowledgeDistillationAgent()
        assert agent is not None

    @pytest.mark.slow
    def test_response_distillation(self):
        """Test response-based knowledge distillation."""
        teacher = TestModelFixtures.create_simple_mlp()
        student = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        agent = KnowledgeDistillationAgent()
        train_data = TestModelFixtures.create_training_data(batch_size=8, num_batches=3, input_size=(784,))
        val_data = TestModelFixtures.create_validation_data(batch_size=8, num_batches=2, input_size=(784,))

        config = DistillationConfig(
            temperature=4.0,
            alpha=0.7,
            beta=0.3,
            distillation_type='response',
            epochs=2  # Short for testing
        )

        distilled_student, results = agent.distill_knowledge(
            teacher, student, train_data, val_data, config
        )

        assert results.teacher_performance is not None
        assert results.student_performance is not None
        assert results.compression_ratio > 1.0


class TestArchitectureOptimizer:
    """Test ArchitectureOptimizerAgent."""

    def test_architecture_optimizer_initialization(self):
        """Test architecture optimizer initialization."""
        agent = ArchitectureOptimizerAgent()
        assert agent is not None

    @pytest.mark.slow
    def test_cnn_architecture_optimization(self):
        """Test CNN architecture optimization."""
        agent = ArchitectureOptimizerAgent()

        config = ArchitectureConfig(
            search_strategy='evolutionary',
            target_params=10000,
            max_generations=5,  # Short for testing
            population_size=4   # Small for testing
        )

        results = agent.optimize_architecture(
            'cnn', config, (3, 32, 32), 10
        )

        assert results.best_architecture is not None
        assert results.best_model is not None
        assert results.best_metrics is not None
        assert len(results.optimization_history) > 0


class TestCompressionValidator:
    """Test CompressionValidatorAgent."""

    def test_compression_validator_initialization(self):
        """Test compression validator initialization."""
        agent = CompressionValidatorAgent()
        assert agent is not None

    def test_model_validation(self):
        """Test comprehensive model validation."""
        original_model = TestModelFixtures.create_simple_cnn()
        compressed_model = TestModelFixtures.create_simple_cnn()  # Same for testing
        validation_data = TestModelFixtures.create_validation_data()

        agent = CompressionValidatorAgent()
        config = ValidationConfig(
            accuracy_threshold=0.9,
            device='cpu'
        )

        results = agent.validate_compression(
            original_model, compressed_model, validation_data, config
        )

        assert results.overall_metrics is not None
        assert results.layer_wise_analysis is not None
        assert results.performance_benchmarks is not None
        assert isinstance(results.deployment_readiness, dict)


class TestDeploymentPackager:
    """Test DeploymentPackagerAgent."""

    def test_deployment_packager_initialization(self):
        """Test deployment packager initialization."""
        agent = DeploymentPackagerAgent()
        assert agent is not None

    def test_pytorch_deployment_package(self):
        """Test PyTorch deployment package creation."""
        model = TestModelFixtures.create_simple_cnn()
        agent = DeploymentPackagerAgent()

        config = DeploymentConfig(
            target_platform='cpu',
            runtime_environment='pytorch',
            optimization_level=1,
            include_preprocessing=True,
            include_postprocessing=True
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            results = agent.create_deployment_package(
                model, config, output_dir, "test_model"
            )

            assert len(results.packages) > 0
            package = results.packages[0]
            assert package.model_path.exists()
            assert package.config_path.exists()
            assert package.package_size_mb > 0


class TestPerformanceProfiler:
    """Test PerformanceProfilerAgent."""

    def test_performance_profiler_initialization(self):
        """Test performance profiler initialization."""
        agent = PerformanceProfilerAgent()
        assert agent is not None

    def test_model_profiling(self):
        """Test comprehensive model profiling."""
        model = TestModelFixtures.create_simple_cnn()
        input_tensors = [torch.randn(4, 3, 32, 32)]

        agent = PerformanceProfilerAgent()
        config = ProfilingConfig(
            device='cpu',
            batch_sizes=[1, 4],
            measurement_iterations=10  # Short for testing
        )

        results = agent.profile_model(model, input_tensors, "test_model", config)

        assert results.model_name == "test_model"
        assert len(results.latency_metrics) > 0
        assert len(results.throughput_metrics) > 0
        assert results.system_info is not None

    def test_model_comparison(self):
        """Test model performance comparison."""
        model1 = TestModelFixtures.create_simple_cnn()
        model2 = TestModelFixtures.create_simple_cnn()
        input_tensors = [torch.randn(2, 3, 32, 32)]

        agent = PerformanceProfilerAgent()
        models = {'model1': model1, 'model2': model2}

        config = ProfilingConfig(
            batch_sizes=[1, 2],
            measurement_iterations=5
        )

        comparison_results = agent.compare_models(models, input_tensors, config)

        assert len(comparison_results) == 2
        assert 'model1' in comparison_results
        assert 'model2' in comparison_results


class TestCompressionAlgorithms:
    """Test core compression algorithms."""

    def test_algorithm_factory(self):
        """Test compression algorithm factory."""
        factory = CompressionAlgorithmFactory()
        algorithms = factory.get_available_algorithms()

        assert len(algorithms) > 0
        assert 'magnitude_pruning' in algorithms
        assert 'weight_clustering' in algorithms

    def test_magnitude_pruning_algorithm(self):
        """Test magnitude pruning algorithm."""
        model = TestModelFixtures.create_simple_mlp()
        algorithm = MagnitudePruning(sparsity_ratio=0.5)

        compressed_model, metrics = algorithm.compress(model)

        assert metrics.compression_ratio >= 1.0
        assert 0 <= metrics.parameter_reduction <= 1.0
        assert compressed_model is not None

    def test_weight_clustering_algorithm(self):
        """Test weight clustering algorithm."""
        model = TestModelFixtures.create_simple_mlp()
        algorithm = WeightClustering(num_clusters=16)

        compressed_model, metrics = algorithm.compress(model)

        assert metrics.compression_ratio > 1.0
        assert compressed_model is not None

    def test_svd_compression_algorithm(self):
        """Test SVD compression algorithm."""
        model = TestModelFixtures.create_simple_mlp()
        algorithm = SVDCompression(rank_ratio=0.5)

        compressed_model, metrics = algorithm.compress(model)

        assert metrics.compression_ratio > 1.0
        assert metrics.parameter_reduction > 0
        assert compressed_model is not None


class TestHyperparameterOptimization:
    """Test hyperparameter optimization."""

    def test_optimization_config_creation(self):
        """Test optimization configuration creation."""
        objectives = [
            OptimizationObjective('compression_ratio', weight=0.5, minimize=False),
            OptimizationObjective('accuracy_retention', weight=0.5, minimize=False)
        ]

        search_space = {
            'sparsity_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9}
        }

        config = OptimizationConfig(
            objectives=objectives,
            search_space=search_space,
            n_trials=10
        )

        assert len(config.objectives) == 2
        assert len(config.search_space) == 1
        assert config.n_trials == 10

    @pytest.mark.slow
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        objectives = [
            OptimizationObjective('compression_ratio', weight=1.0, minimize=False)
        ]

        search_space = {
            'sparsity_ratio': {'type': 'float', 'low': 0.1, 'high': 0.5}
        }

        config = OptimizationConfig(
            objectives=objectives,
            search_space=search_space,
            n_trials=5  # Small for testing
        )

        optimizer = HyperparameterOptimizer(config)
        model = TestModelFixtures.create_simple_mlp()

        def compression_function(model, **params):
            algorithm = MagnitudePruning(sparsity_ratio=params.get('sparsity_ratio', 0.5))
            compressed_model, _ = algorithm.compress(copy.deepcopy(model))
            return compressed_model

        results = optimizer.optimize(compression_function, model)

        assert results.best_params is not None
        assert results.execution_time > 0
        assert len(results.optimization_history) > 0


class TestModelValidation:
    """Test model validation framework."""

    def test_validation_framework_initialization(self):
        """Test validation framework initialization."""
        thresholds = ValidationThresholds(
            min_accuracy_retention=0.9,
            max_accuracy_drop=0.1
        )

        framework = ModelValidationFramework(thresholds)
        assert framework is not None
        assert framework.thresholds.min_accuracy_retention == 0.9

    def test_model_validation_report(self):
        """Test comprehensive model validation."""
        original_model = TestModelFixtures.create_simple_cnn()
        compressed_model = TestModelFixtures.create_simple_cnn()
        validation_data = TestModelFixtures.create_validation_data()

        framework = ModelValidationFramework()
        report = framework.validate_model(
            original_model, compressed_model, validation_data, "test_model"
        )

        assert report.model_name == "test_model"
        assert report.metrics is not None
        assert report.validation_summary is not None
        assert isinstance(report.recommendations, list)

    def test_validation_report_saving(self):
        """Test validation report saving."""
        original_model = TestModelFixtures.create_simple_mlp()
        compressed_model = TestModelFixtures.create_simple_mlp()
        validation_data = TestModelFixtures.create_validation_data(input_size=(784,))

        framework = ModelValidationFramework()
        report = framework.validate_model(
            original_model, compressed_model, validation_data
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        framework.save_validation_report(report, tmp_path)
        assert tmp_path.exists()

        # Verify report content
        with open(tmp_path, 'r') as f:
            report_data = json.load(f)

        assert 'validation_summary' in report_data
        assert 'overall_score' in report_data

        # Cleanup
        tmp_path.unlink()


class TestCompressionOrchestrator:
    """Test compression orchestrator."""

    def test_orchestrator_initialization(self):
        """Test compression orchestrator initialization."""
        target = CompressionTarget(
            max_model_size_mb=50.0,
            min_accuracy_retention=0.95,
            target_platform='cpu'
        )

        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PRUNING_ONLY,
            target=target
        )

        orchestrator = CompressionOrchestrator(config)
        assert orchestrator is not None
        assert len(orchestrator.agents) > 0

    @pytest.mark.slow
    def test_pruning_pipeline(self):
        """Test pruning-only compression pipeline."""
        model = TestModelFixtures.create_simple_mlp()
        validation_data = TestModelFixtures.create_validation_data(input_size=(784,))

        target = CompressionTarget(
            min_accuracy_retention=0.8,  # Relaxed for testing
            target_platform='cpu'
        )

        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PRUNING_ONLY,
            target=target,
            save_intermediate_results=False  # Skip saving for tests
        )

        orchestrator = CompressionOrchestrator(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_directory = Path(tmp_dir)

            results = orchestrator.compress_model(
                model, validation_data, model_name="test_pipeline"
            )

            assert results.original_model is not None
            assert results.best_model is not None
            assert results.compression_ratio > 0
            assert results.execution_time > 0
            assert results.pipeline_state.current_phase.value == 'completed'

    def test_pipeline_status(self):
        """Test pipeline status monitoring."""
        target = CompressionTarget(target_platform='cpu')
        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PRUNING_ONLY,
            target=target
        )

        orchestrator = CompressionOrchestrator(config)
        status = orchestrator.get_pipeline_status()

        assert 'current_phase' in status
        assert 'completed_steps' in status
        assert 'elapsed_time' in status


class TestIntegration:
    """Integration tests for complete compression pipeline."""

    @pytest.mark.slow
    def test_end_to_end_compression(self):
        """Test complete end-to-end compression pipeline."""
        # Create a larger model for meaningful compression
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        validation_data = TestModelFixtures.create_validation_data(
            batch_size=8, num_batches=5
        )

        # Configure compression pipeline
        target = CompressionTarget(
            min_accuracy_retention=0.80,
            target_platform='cpu'
        )

        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.HYBRID_COMPRESSION,
            target=target,
            parallel_execution=False,  # Disable for simpler testing
            save_intermediate_results=False
        )

        orchestrator = CompressionOrchestrator(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_directory = Path(tmp_dir)

            # Run compression
            results = orchestrator.compress_model(
                model, validation_data, model_name="integration_test"
            )

            # Verify results
            assert results.compression_ratio > 1.0
            assert results.best_model is not None
            assert results.analysis_results is not None
            assert len(results.compressed_models) > 0

            # Verify phases completed
            completed_phases = list(results.pipeline_state.phase_times.keys())
            assert 'analysis' in completed_phases
            assert 'compression' in completed_phases

    def test_phase7_to_phase8_integration(self):
        """Test integration from Phase 7 ADAS to Phase 8 Compression."""
        # Simulate ADAS model from Phase 7
        adas_model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 classes for ADAS output
        )

        # Create validation data
        validation_data = TestModelFixtures.create_validation_data(
            batch_size=4, num_batches=3
        )

        # Configure for deployment
        target = CompressionTarget(
            max_model_size_mb=10.0,
            min_accuracy_retention=0.95,
            target_platform='cpu'
        )

        config = CompressionPipelineConfig(
            strategy=CompressionStrategy.PROGRESSIVE_COMPRESSION,
            target=target
        )

        orchestrator = CompressionOrchestrator(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_directory = Path(tmp_dir)

            results = orchestrator.compress_model(
                adas_model, validation_data, model_name="adas_compressed"
            )

            # Verify ADAS model was successfully compressed
            assert results.best_model is not None
            assert results.compression_ratio > 1.0

            # Verify deployment package was created
            if results.packaging_results and results.packaging_results.packages:
                package = results.packaging_results.packages[0]
                assert package.model_path.exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])