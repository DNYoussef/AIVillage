"""
Phase 8 Compression Agents

A comprehensive suite of neural network compression agents implementing
genuine compression algorithms with no theater patterns.

Agents:
- ModelAnalyzer: Analyzes model structure for compression opportunities
- PruningAgent: Implements various pruning strategies (magnitude, structured, gradual, lottery ticket)
- QuantizationAgent: Implements quantization strategies (static, dynamic, QAT, FX)
- KnowledgeDistiller: Implements knowledge distillation techniques
- ArchitectureOptimizer: Implements neural architecture search for compression
- CompressionValidator: Validates compression results with comprehensive metrics
- DeploymentPackager: Packages compressed models for deployment
- PerformanceProfiler: Profiles performance characteristics
- CompressionOrchestrator: Coordinates all compression agents

All agents integrate with Phase 8 core algorithms and provide genuine
compression techniques with real performance improvements.
"""

from .model_analyzer import ModelAnalyzer, ModelAnalysis
from .pruning_agent import PruningAgent, PruningConfig, PruningResult
from .quantization_agent import QuantizationAgent, QuantizationConfig, QuantizationResult
from .knowledge_distiller import KnowledgeDistiller, DistillationConfig, DistillationResult
from .architecture_optimizer import ArchitectureOptimizer, ArchitectureConfig, OptimizationResult
from .compression_validator import CompressionValidator, ValidationConfig, ValidationMetrics
from .deployment_packager import DeploymentPackager, DeploymentConfig, DeploymentPackage
from .performance_profiler import PerformanceProfiler, ProfilingConfig, PerformanceMetrics
from .compression_orchestrator import CompressionOrchestrator, CompressionStrategy, CompressionResults

__all__ = [
    # Core orchestrator
    'CompressionOrchestrator',
    'CompressionStrategy',
    'CompressionResults',

    # Individual agents
    'ModelAnalyzer',
    'PruningAgent',
    'QuantizationAgent',
    'KnowledgeDistiller',
    'ArchitectureOptimizer',
    'CompressionValidator',
    'DeploymentPackager',
    'PerformanceProfiler',

    # Configuration classes
    'ModelAnalysis',
    'PruningConfig',
    'PruningResult',
    'QuantizationConfig',
    'QuantizationResult',
    'DistillationConfig',
    'DistillationResult',
    'ArchitectureConfig',
    'OptimizationResult',
    'ValidationConfig',
    'ValidationMetrics',
    'DeploymentConfig',
    'DeploymentPackage',
    'ProfilingConfig',
    'PerformanceMetrics'
]

# Version info
__version__ = "1.0.0"
__author__ = "ADAS Compression Team"
__description__ = "Phase 8 Compression Agents - Genuine Neural Network Compression"