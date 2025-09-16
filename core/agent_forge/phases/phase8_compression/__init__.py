"""
Phase 8 Compression Module

Advanced neural network compression phase that receives models from Phase 7 ADAS
and applies comprehensive compression techniques for deployment optimization.

This module provides:
- 9 specialized compression agents
- Advanced compression algorithms
- Multi-objective optimization
- Comprehensive validation
- Deployment packaging
- Performance profiling

Architecture:
    agents/                 # 9 compression agents
    ├── model_analyzer.py           # Analyzes model characteristics
    ├── pruning_agent.py           # Neural network pruning
    ├── quantization_agent.py      # Weight/activation quantization
    ├── knowledge_distiller.py     # Knowledge distillation
    ├── architecture_optimizer.py  # Neural architecture search
    ├── compression_validator.py   # Quality validation
    ├── deployment_packager.py     # Deployment packages
    ├── performance_profiler.py    # Performance analysis
    └── compression_orchestrator.py # Pipeline coordination

    core/                   # Core compression algorithms
    ├── compression_algorithms.py  # Fundamental algorithms

    optimization/           # Advanced optimization
    ├── compression_optimizer.py   # Multi-objective optimization

    validation/             # Quality validation
    ├── model_validator.py        # Comprehensive validation

    deployment/             # Deployment support
    tests/                  # Test suites
    docs/                   # Documentation
"""

from .agents import (
    ModelAnalyzerAgent,
    PruningAgent,
    QuantizationAgent,
    KnowledgeDistillationAgent,
    ArchitectureOptimizerAgent,
    CompressionValidatorAgent,
    DeploymentPackagerAgent,
    PerformanceProfilerAgent,
    CompressionOrchestrator,
)

from .core.compression_algorithms import (
    CompressionAlgorithm,
    MagnitudePruning,
    GradientBasedPruning,
    WeightClustering,
    SVDCompression,
    HuffmanCoding,
    CompressionAlgorithmFactory,
)

from .optimization.compression_optimizer import (
    HyperparameterOptimizer,
    MultiObjectiveOptimizer,
    AdaptiveCompressionOptimizer,
)

from .validation.model_validator import (
    ModelValidationFramework,
    ValidationThresholds,
    ValidationReport,
)

__version__ = "1.0.0"
__author__ = "Phase 8 Compression Team"

__all__ = [
    # Agents
    "ModelAnalyzerAgent",
    "PruningAgent",
    "QuantizationAgent",
    "KnowledgeDistillationAgent",
    "ArchitectureOptimizerAgent",
    "CompressionValidatorAgent",
    "DeploymentPackagerAgent",
    "PerformanceProfilerAgent",
    "CompressionOrchestrator",

    # Core Algorithms
    "CompressionAlgorithm",
    "MagnitudePruning",
    "GradientBasedPruning",
    "WeightClustering",
    "SVDCompression",
    "HuffmanCoding",
    "CompressionAlgorithmFactory",

    # Optimization
    "HyperparameterOptimizer",
    "MultiObjectiveOptimizer",
    "AdaptiveCompressionOptimizer",

    # Validation
    "ModelValidationFramework",
    "ValidationThresholds",
    "ValidationReport",
]


def get_phase_info():
    """Get information about Phase 8 Compression."""
    return {
        "name": "Phase 8 - Neural Network Compression",
        "version": __version__,
        "description": "Advanced compression pipeline for neural network deployment optimization",
        "input_source": "Phase 7 ADAS models",
        "output_destination": "Deployment-ready compressed models",
        "agents": [
            "ModelAnalyzer - Model structure analysis",
            "PruningAgent - Neural network pruning",
            "QuantizationAgent - Weight/activation quantization",
            "KnowledgeDistiller - Knowledge distillation",
            "ArchitectureOptimizer - Neural architecture search",
            "CompressionValidator - Quality validation",
            "DeploymentPackager - Deployment packaging",
            "PerformanceProfiler - Performance analysis",
            "CompressionOrchestrator - Pipeline coordination"
        ],
        "key_features": [
            "Multi-technique compression pipeline",
            "Advanced optimization algorithms",
            "Comprehensive quality validation",
            "Deployment-ready packaging",
            "Performance profiling and analysis",
            "Multi-objective optimization",
            "Adaptive compression strategies"
        ],
        "compression_techniques": [
            "Magnitude-based pruning",
            "Gradient-based pruning",
            "Structured pruning",
            "Dynamic quantization",
            "Static quantization",
            "Knowledge distillation",
            "Neural architecture search",
            "Weight clustering",
            "SVD compression",
            "Huffman coding"
        ]
    }


def create_compression_pipeline(strategy="hybrid", target_platform="cpu", **kwargs):
    """
    Create a compression pipeline with specified strategy.

    Args:
        strategy: Compression strategy ('pruning', 'quantization', 'distillation',
                 'architecture_search', 'hybrid', 'progressive')
        target_platform: Target deployment platform ('cpu', 'cuda', 'mobile', 'edge')
        **kwargs: Additional configuration parameters

    Returns:
        Configured CompressionOrchestrator instance
    """
    from .agents.compression_orchestrator import (
        CompressionOrchestrator,
        CompressionPipelineConfig,
        CompressionStrategy,
        CompressionTarget
    )

    # Map string strategies to enums
    strategy_map = {
        'pruning': CompressionStrategy.PRUNING_ONLY,
        'quantization': CompressionStrategy.QUANTIZATION_ONLY,
        'distillation': CompressionStrategy.DISTILLATION_ONLY,
        'architecture_search': CompressionStrategy.ARCHITECTURE_SEARCH,
        'hybrid': CompressionStrategy.HYBRID_COMPRESSION,
        'progressive': CompressionStrategy.PROGRESSIVE_COMPRESSION,
        'custom': CompressionStrategy.CUSTOM_PIPELINE
    }

    # Create compression target
    target = CompressionTarget(
        target_platform=target_platform,
        **{k: v for k, v in kwargs.items() if k.startswith('target_')}
    )

    # Create pipeline configuration
    config = CompressionPipelineConfig(
        strategy=strategy_map.get(strategy, CompressionStrategy.HYBRID_COMPRESSION),
        target=target,
        **{k: v for k, v in kwargs.items() if not k.startswith('target_')}
    )

    return CompressionOrchestrator(config)


def get_supported_compression_techniques():
    """Get list of supported compression techniques."""
    from .core.compression_algorithms import CompressionAlgorithmFactory
    return CompressionAlgorithmFactory.get_available_algorithms()


# Phase 8 Compression Pipeline Example Usage:
"""
# Basic usage example:
from phase8_compression import create_compression_pipeline

# Create compression pipeline
pipeline = create_compression_pipeline(
    strategy="hybrid",
    target_platform="cpu",
    target_accuracy_retention=0.95,
    target_model_size_mb=50.0
)

# Compress model from Phase 7 ADAS
results = pipeline.compress_model(
    adas_model,
    validation_data,
    training_data,
    model_name="adas_compressed"
)

# Access compressed model
compressed_model = results.best_model
compression_ratio = results.compression_ratio
accuracy_retention = results.accuracy_retention

print(f"Achieved {compression_ratio:.2f}x compression")
print(f"Retained {accuracy_retention:.1%} accuracy")
"""