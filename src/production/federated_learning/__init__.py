"""Federated Learning System for Sprint 7

This module implements federated learning on the distributed infrastructure,
enabling collaborative training across devices while preserving privacy.

Key Components:
- DistributedFederatedLearning: Main coordinator for federated training
- SecureAggregation: Privacy-preserving gradient aggregation
- HierarchicalAggregation: Multi-tier aggregation for efficiency
"""

from .federated_coordinator import DistributedFederatedLearning, FederatedTrainingRound
from .hierarchical_aggregation import AggregationTier, HierarchicalAggregator
from .secure_aggregation import PrivacyConfig, SecureAggregationProtocol

__all__ = [
    "AggregationTier",
    "DistributedFederatedLearning",
    "FederatedTrainingRound",
    "HierarchicalAggregator",
    "PrivacyConfig",
    "SecureAggregationProtocol",
]
