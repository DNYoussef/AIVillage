#!/usr/bin/env python3
"""
BitNet Phase 4 - Integration Package

Complete BitNet Phase 4 implementation with comprehensive integration support.
"""

from .bitnet_core import (
    BitNetQuantizer,
    BitNetLinear,
    BitNetAttention,
    BitNetTransformerBlock,
    BitNetModel,
    create_bitnet_model,
    convert_to_bitnet,
    benchmark_bitnet_performance
)

from .optimization import (
    BitNetOptimizer,
    BitNetLRScheduler,
    BitNetTrainer,
    BitNetMemoryOptimizer,
    create_bitnet_trainer,
    optimize_bitnet_model,
    EarlyStopping,
    TrainingMetrics
)

__all__ = [
    # Core components
    'BitNetQuantizer',
    'BitNetLinear',
    'BitNetAttention',
    'BitNetTransformerBlock',
    'BitNetModel',
    'create_bitnet_model',
    'convert_to_bitnet',
    'benchmark_bitnet_performance',

    # Optimization components
    'BitNetOptimizer',
    'BitNetLRScheduler',
    'BitNetTrainer',
    'BitNetMemoryOptimizer',
    'create_bitnet_trainer',
    'optimize_bitnet_model',
    'EarlyStopping',
    'TrainingMetrics'
]

__version__ = "1.0.0"