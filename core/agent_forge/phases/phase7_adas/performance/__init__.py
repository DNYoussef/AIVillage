"""
ADAS Performance Optimization Suite
===================================

Comprehensive performance monitoring and optimization for automotive ADAS systems.
Supports NVIDIA Drive, Qualcomm Snapdragon Ride, and generic automotive ECUs.

Key Features:
- Sub-10ms inference optimization
- Real-time resource management
- Comprehensive benchmarking
- Hardware-specific profiling
- Deployment validation

Target Platforms:
- NVIDIA Drive AGX/PX
- Qualcomm Snapdragon Ride
- Automotive ECUs (ARM/x86)
- Generic edge devices
"""

from .latency_optimizer import (
    LatencyOptimizer,
    OptimizationLevel,
    LatencyBenchmark,
    PipelineStage,
    create_automotive_pipeline
)

from .resource_manager import (
    ResourceManager,
    ResourceType,
    PowerMode,
    ThermalState,
    ResourceUsage,
    ResourceLimits,
    ResourceAllocation,
    ResourceError
)

from .benchmark_suite import (
    BenchmarkSuite,
    BenchmarkType,
    BenchmarkResult,
    PerformanceProfile,
    Bottleneck,
    OptimizationRecommendation as BenchmarkOptimizationRecommendation,
    Severity
)

from .edge_profiler import (
    EdgeProfiler,
    HardwarePlatform,
    OptimizationTarget,
    HardwareSpecs,
    ProfileResult,
    OptimizationRecommendation as ProfilerOptimizationRecommendation,
    DeploymentValidation
)

__version__ = "1.0.0"
__author__ = "ADAS Performance Team"

__all__ = [
    # Latency Optimizer
    "LatencyOptimizer",
    "OptimizationLevel", 
    "LatencyBenchmark",
    "PipelineStage",
    "create_automotive_pipeline",
    
    # Resource Manager
    "ResourceManager",
    "ResourceType",
    "PowerMode",
    "ThermalState",
    "ResourceUsage",
    "ResourceLimits",
    "ResourceAllocation",
    "ResourceError",
    
    # Benchmark Suite
    "BenchmarkSuite",
    "BenchmarkType",
    "BenchmarkResult",
    "PerformanceProfile",
    "Bottleneck",
    "BenchmarkOptimizationRecommendation",
    "Severity",
    
    # Edge Profiler
    "EdgeProfiler",
    "HardwarePlatform",
    "OptimizationTarget",
    "HardwareSpecs",
    "ProfileResult",
    "ProfilerOptimizationRecommendation",
    "DeploymentValidation"
]


def get_version():
    """Get package version"""
    return __version__


def get_supported_platforms():
    """Get list of supported hardware platforms"""
    return [platform.value for platform in HardwarePlatform]


def get_optimization_targets():
    """Get list of available optimization targets"""
    return [target.value for target in OptimizationTarget]