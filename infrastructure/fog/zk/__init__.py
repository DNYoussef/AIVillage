"""
Zero-Knowledge Predicates for Privacy-Preserving Fog Computing

This module provides narrow, practical zero-knowledge predicates for:
- Network policy verification
- MIME type validation
- Model pack hash verification
- Privacy-preserving compliance checks

The predicates are designed for real-world fog computing scenarios where
privacy and verification must be balanced with practical performance.
"""

from .zk_predicates import (
    CompliancePredicate,
    MimeTypePredicate,
    ModelHashPredicate,
    NetworkPolicyPredicate,
    PredicateType,
    ProofResult,
    ZKPredicate,
    ZKPredicateEngine,
    ZKProof,
)

__all__ = [
    "ZKPredicateEngine",
    "ZKPredicate",
    "ZKProof",
    "NetworkPolicyPredicate",
    "MimeTypePredicate",
    "ModelHashPredicate",
    "CompliancePredicate",
    "PredicateType",
    "ProofResult",
]
