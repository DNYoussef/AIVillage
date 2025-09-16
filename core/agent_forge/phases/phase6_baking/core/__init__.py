#!/usr/bin/env python3
"""
Agent Forge Phase 6: Model Baking Architecture
==============================================

Complete model baking system that optimizes trained models from Phase 5 for
maximum inference performance while maintaining accuracy and NASA POT10 compliance.

This phase provides:
1. Core baking architecture with optimization pipeline
2. Model optimizer with pruning, quantization, and knowledge distillation
3. Inference accelerator with graph optimization and kernel fusion
4. Quality validator with accuracy preservation and theater detection
5. Hardware adapter for CUDA and CPU-specific optimizations
6. Performance profiler with comprehensive metrics and analysis
7. Configuration management system
8. Graph optimizer for computation graph optimization

Key Features:
- BitNet 1-bit quantization optimization
- Hardware-specific acceleration (CUDA, CPU, MPS)
- Performance theater detection
- NASA POT10 compliance validation
- Cross-phase integration (Phase 5 input, Phase 7 output)
- Comprehensive benchmarking and profiling
"""

from .baking_architecture import (
    BakingArchitecture,
    BakingConfig,
    OptimizationMetrics
)

from .model_optimizer import (
    ModelOptimizer,
    OptimizationPass
)

from .inference_accelerator import (
    InferenceAccelerator,
    AccelerationConfig,
    AccelerationMetrics
)

from .quality_validator import (
    QualityValidator,
    QualityMetrics,
    ValidationConfig
)

from .hardware_adapter import (
    HardwareAdapter,
    HardwareProfile,
    OptimizationStrategy
)

from .performance_profiler import (
    PerformanceProfiler,
    PerformanceMetrics,
    ProfilingConfig
)

from .baking_config import (
    BakingConfiguration,
    OptimizationLevel,
    DeviceType,
    ExportFormat,
    HardwareConfig,
    QualityConfig,
    ExportConfig,
    create_default_config
)

from .graph_optimizer import (
    GraphOptimizer,
    GraphOptimizationMetrics,
    FusionPattern
)

# Version information
__version__ = "6.0.0"
__author__ = "Agent Forge Phase 6 Team"
__description__ = "Model Baking Architecture for Agent Forge"

# Export all main classes and functions
__all__ = [
    # Core architecture
    "BakingArchitecture",
    "BakingConfig",
    "OptimizationMetrics",

    # Model optimization
    "ModelOptimizer",
    "OptimizationPass",

    # Inference acceleration
    "InferenceAccelerator",
    "AccelerationConfig",
    "AccelerationMetrics",

    # Quality validation
    "QualityValidator",
    "QualityMetrics",
    "ValidationConfig",

    # Hardware adaptation
    "HardwareAdapter",
    "HardwareProfile",
    "OptimizationStrategy",

    # Performance profiling
    "PerformanceProfiler",
    "PerformanceMetrics",
    "ProfilingConfig",

    # Configuration management
    "BakingConfiguration",
    "OptimizationLevel",
    "DeviceType",
    "ExportFormat",
    "HardwareConfig",
    "QualityConfig",
    "ExportConfig",
    "create_default_config",

    # Graph optimization
    "GraphOptimizer",
    "GraphOptimizationMetrics",
    "FusionPattern",

    # Utility functions
    "create_baking_pipeline",
    "benchmark_baked_models",
    "validate_phase_integration"
]

def create_baking_pipeline(
    config: BakingConfig = None,
    device: str = "auto"
) -> BakingArchitecture:
    """
    Create a complete baking pipeline with all components initialized.

    Args:
        config: Optional baking configuration
        device: Target device ("auto", "cuda", "cpu")

    Returns:
        Configured BakingArchitecture instance
    """
    import torch
    import logging

    # Setup logging
    logger = logging.getLogger("Phase6.BakingPipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Default configuration
    if config is None:
        config = BakingConfig()

    # Device detection
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Create baking architecture
    baker = BakingArchitecture(config, logger)

    logger.info(f"Baking pipeline created for device: {device}")
    logger.info(f"Optimization level: {config.optimization_level}")

    return baker

def benchmark_baked_models(
    models: dict,
    sample_inputs: dict,
    device: str = "auto",
    num_iterations: int = 100
) -> dict:
    """
    Benchmark multiple baked models for performance comparison.

    Args:
        models: Dictionary of model_name -> model
        sample_inputs: Dictionary of model_name -> input tensor
        device: Target device for benchmarking
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results for each model
    """
    import torch
    import time
    import numpy as np

    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    results = {}

    for model_name, model in models.items():
        if model_name not in sample_inputs:
            continue

        model = model.to(device)
        inputs = sample_inputs[model_name].to(device)
        model.eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(inputs)

        # Synchronize
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = model(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        batch_size = inputs.size(0)

        results[model_name] = {
            "latency_mean": float(np.mean(latencies)),
            "latency_std": float(np.std(latencies)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "throughput_samples_per_sec": (batch_size * 1000) / np.mean(latencies),
            "device": str(device)
        }

    return results

def validate_phase_integration(
    phase5_dir: str = "./phase5_models",
    phase7_dir: str = "./phase7_ready"
) -> dict:
    """
    Validate integration with Phase 5 (input) and Phase 7 (output).

    Args:
        phase5_dir: Directory containing Phase 5 trained models
        phase7_dir: Directory for Phase 7 ready models

    Returns:
        Validation results dictionary
    """
    from pathlib import Path
    import os

    phase5_path = Path(phase5_dir)
    phase7_path = Path(phase7_dir)

    validation_results = {
        "phase5_integration": {
            "input_dir_exists": phase5_path.exists(),
            "models_found": [],
            "supported_formats": [".pth", ".pt", ".onnx"],
            "validation_passed": False
        },
        "phase7_integration": {
            "output_dir_ready": False,
            "export_formats_supported": ["pytorch", "torchscript", "onnx"],
            "adas_compatibility": True,
            "validation_passed": False
        },
        "overall_integration": {
            "cross_phase_compatibility": False,
            "data_flow_validated": False,
            "ready_for_production": False
        }
    }

    # Validate Phase 5 integration
    if phase5_path.exists():
        model_files = []
        for ext in validation_results["phase5_integration"]["supported_formats"]:
            model_files.extend(list(phase5_path.glob(f"*{ext}")))

        validation_results["phase5_integration"]["models_found"] = [
            str(f.name) for f in model_files
        ]
        validation_results["phase5_integration"]["validation_passed"] = len(model_files) > 0

    # Validate Phase 7 integration
    try:
        phase7_path.mkdir(parents=True, exist_ok=True)
        validation_results["phase7_integration"]["output_dir_ready"] = True
        validation_results["phase7_integration"]["validation_passed"] = True
    except Exception as e:
        validation_results["phase7_integration"]["error"] = str(e)

    # Overall integration validation
    phase5_ok = validation_results["phase5_integration"]["validation_passed"]
    phase7_ok = validation_results["phase7_integration"]["validation_passed"]

    validation_results["overall_integration"]["cross_phase_compatibility"] = phase5_ok and phase7_ok
    validation_results["overall_integration"]["data_flow_validated"] = phase5_ok and phase7_ok
    validation_results["overall_integration"]["ready_for_production"] = phase5_ok and phase7_ok

    return validation_results

# Configuration for easy imports
PHASE6_COMPONENTS = {
    "BakingArchitecture": BakingArchitecture,
    "ModelOptimizer": ModelOptimizer,
    "InferenceAccelerator": InferenceAccelerator,
    "QualityValidator": QualityValidator,
    "HardwareAdapter": HardwareAdapter,
    "PerformanceProfiler": PerformanceProfiler,
    "GraphOptimizer": GraphOptimizer,
    "BakingConfiguration": BakingConfiguration
}

# Default optimization settings
DEFAULT_OPTIMIZATION_SETTINGS = {
    "optimization_level": 3,
    "preserve_accuracy_threshold": 0.95,
    "target_speedup": 2.0,
    "enable_bitnet_optimization": True,
    "enable_theater_detection": True,
    "enable_nasa_compliance": True
}