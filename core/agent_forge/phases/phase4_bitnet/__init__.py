"""
BitNet Performance Optimization Suite - Agent Forge Phase 4

Comprehensive 1-bit Neural Network Optimization Framework
========================================================

This module provides a complete performance optimization suite for BitNet models,
targeting 8x memory reduction and 2-4x speedup with <10% accuracy degradation.

Key Components:
- Memory optimization with advanced pooling and compression
- Inference speed optimization with custom kernels
- Training optimization for quantized weights
- Hardware-specific optimizations (CUDA, CPU, etc.)
- Comprehensive benchmarking and profiling tools
- Performance target validation framework

Author: Agent Forge Phase 4 - Performance Optimization Team
License: NASA POT10 Compliant
"""

from .optimization.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizationConfig,
    create_memory_optimizer
)

from .optimization.inference_optimizer import (
    InferenceOptimizer,
    InferenceOptimizationConfig,
    create_inference_optimizer
)

from .optimization.training_optimizer import (
    TrainingOptimizer,
    TrainingOptimizationConfig,
    create_training_optimizer
)

from .optimization.hardware_optimizer import (
    HardwareOptimizer,
    HardwareOptimizationConfig,
    create_hardware_optimizer
)

from .benchmarks.performance_suite import (
    PerformanceBenchmarkSuite,
    BenchmarkConfig,
    create_benchmark_suite
)

from .benchmarks.baseline_comparison import (
    BaselineComparisonSuite,
    ValidationResults,
    PerformanceTargets
)

from .profiling.memory_profiler import (
    MemoryProfiler,
    MemoryProfilingConfig,
    create_memory_profiler
)

from .profiling.speed_profiler import (
    SpeedProfiler,
    SpeedProfilingConfig,
    create_speed_profiler
)

from .validate_performance_targets import BitNetPerformanceValidator

__version__ = "1.0.0"
__author__ = "Agent Forge Phase 4 Team"
__license__ = "NASA POT10 Compliant"

# Performance targets for BitNet optimization
BITNET_PERFORMANCE_TARGETS = {
    "memory_reduction": 8.0,      # 8x memory reduction target
    "speedup_minimum": 2.0,       # 2x minimum speedup target
    "speedup_optimal": 4.0,       # 4x optimal speedup target
    "accuracy_limit": 0.1,        # 10% maximum accuracy degradation
    "real_time_ms": 50.0          # 50ms real-time inference threshold
}

def optimize_bitnet_model(model, device=None, optimization_level="production"):
    """
    Convenience function to apply comprehensive BitNet optimizations.

    Args:
        model: PyTorch model to optimize
        device: Target device (auto-detected if None)
        optimization_level: "development", "balanced", or "production"

    Returns:
        Optimized model and optimization statistics
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create optimizers
    memory_optimizer = create_memory_optimizer(device, optimization_level)
    inference_optimizer = create_inference_optimizer(device, optimization_level)
    training_optimizer = create_training_optimizer(device, optimization_level)
    hardware_optimizer = create_hardware_optimizer(device, optimization_level)

    # Apply optimizations
    model = model.to(device)

    # Example input for optimization
    example_input = torch.randn(1, 128, 768, device=device)

    with memory_optimizer.memory_optimization_context():
        model = memory_optimizer.optimize_model(model)

    model = inference_optimizer.optimize_model_for_inference(model, (example_input,))
    model = training_optimizer.optimize_model_for_training(model)
    model = hardware_optimizer.optimize_model_for_hardware(model)

    # Collect optimization statistics
    stats = {
        "memory_optimization": memory_optimizer.get_optimization_statistics(),
        "inference_optimization": inference_optimizer.get_optimization_statistics(),
        "training_optimization": training_optimizer.get_training_statistics(),
        "hardware_optimization": hardware_optimizer.get_optimization_statistics()
    }

    return model, stats

def validate_bitnet_performance(model, test_inputs=None, create_baseline=True):
    """
    Convenience function for comprehensive BitNet performance validation.

    Args:
        model: BitNet model to validate
        test_inputs: List of test input tensors (generated if None)
        create_baseline: Whether to create a baseline model for comparison

    Returns:
        Comprehensive validation results
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if test_inputs is None:
        test_inputs = [
            torch.randn(8, 128, 768),
            torch.randn(16, 256, 768),
            torch.randn(4, 512, 768)
        ]

    validator = BitNetPerformanceValidator(device, "comprehensive")

    return validator.validate_bitnet_model(
        model, test_inputs, "bitnet_model", create_baseline
    )

def create_comprehensive_optimizer_suite(device=None, optimization_level="production"):
    """
    Create a complete suite of BitNet optimizers.

    Args:
        device: Target device (auto-detected if None)
        optimization_level: "development", "balanced", or "production"

    Returns:
        Dictionary containing all optimizer instances
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "memory_optimizer": create_memory_optimizer(device, optimization_level),
        "inference_optimizer": create_inference_optimizer(device, optimization_level),
        "training_optimizer": create_training_optimizer(device, optimization_level),
        "hardware_optimizer": create_hardware_optimizer(device, optimization_level),
        "memory_profiler": create_memory_profiler(device, optimization_level),
        "speed_profiler": create_speed_profiler(device, optimization_level),
        "benchmark_suite": create_benchmark_suite(optimization_level),
        "baseline_comparison": BaselineComparisonSuite(device),
        "performance_validator": BitNetPerformanceValidator(device, optimization_level)
    }

def get_optimization_recommendations(model, device=None):
    """
    Get optimization recommendations for a BitNet model.

    Args:
        model: BitNet model to analyze
        device: Target device (auto-detected if None)

    Returns:
        List of optimization recommendations
    """
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Analyze model characteristics
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    recommendations = []

    # Size-based recommendations
    if model_size_mb > 1000:  # > 1GB
        recommendations.append("Large model detected. Consider aggressive memory optimization and gradient checkpointing.")

    if total_params > 100_000_000:  # > 100M parameters
        recommendations.append("High parameter count. Implement quantization-aware training and model pruning.")

    # Device-specific recommendations
    if device.type == 'cuda':
        recommendations.append("CUDA device detected. Enable tensor core optimization and custom CUDA kernels.")
    else:
        recommendations.append("CPU device detected. Focus on SIMD optimizations and memory layout optimization.")

    # General recommendations
    recommendations.extend([
        "Apply comprehensive memory optimization for 8x reduction target.",
        "Implement inference speed optimization for 2-4x speedup target.",
        "Use hardware-specific optimizations for maximum performance.",
        "Validate performance targets with comprehensive benchmarking."
    ])

    return recommendations

# Export all public APIs
__all__ = [
    # Core optimizers
    "MemoryOptimizer", "InferenceOptimizer", "TrainingOptimizer", "HardwareOptimizer",

    # Profilers
    "MemoryProfiler", "SpeedProfiler",

    # Benchmarking
    "PerformanceBenchmarkSuite", "BaselineComparisonSuite",

    # Validation
    "BitNetPerformanceValidator", "ValidationResults", "PerformanceTargets",

    # Configuration classes
    "MemoryOptimizationConfig", "InferenceOptimizationConfig",
    "TrainingOptimizationConfig", "HardwareOptimizationConfig",
    "BenchmarkConfig", "MemoryProfilingConfig", "SpeedProfilingConfig",

    # Factory functions
    "create_memory_optimizer", "create_inference_optimizer",
    "create_training_optimizer", "create_hardware_optimizer",
    "create_memory_profiler", "create_speed_profiler", "create_benchmark_suite",

    # Convenience functions
    "optimize_bitnet_model", "validate_bitnet_performance",
    "create_comprehensive_optimizer_suite", "get_optimization_recommendations",

    # Constants
    "BITNET_PERFORMANCE_TARGETS"
]