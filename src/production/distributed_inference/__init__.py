"""Distributed Inference System for Sprint 7.

This module implements distributed inference capabilities that shard models
across multiple devices in a P2P network, enabling inference on resource-
constrained devices while maintaining performance.

Key Components:
- ModelShardingEngine: Intelligent model partitioning and distribution
- AdaptiveResharding: Dynamic resharding based on network changes
- CompressionIntegration: Integration with existing compression pipeline
"""

from .adaptive_resharding import AdaptiveReshardingManager
from .compression_integration import DistributedCompressionManager
from .model_sharding_engine import ModelShardingEngine, ShardingStrategy
from .tokenomics_receipts import (
    TokenomicsConfig,
    TokenomicsReceipt,
    TokenomicsReceiptManager,
)

__all__ = [
    "AdaptiveReshardingManager",
    "DistributedCompressionManager",
    "ModelShardingEngine",
    "ShardingStrategy",
    "TokenomicsReceiptManager",
    "TokenomicsReceipt",
    "TokenomicsConfig",
]
