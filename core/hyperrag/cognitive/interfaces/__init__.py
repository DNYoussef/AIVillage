"""
GraphFixer Service Interfaces Package

Interface definitions for all GraphFixer services,
enabling loose coupling and testability.
"""

from .base_service import BaseService, ServiceConfig, AsyncServiceMixin, CacheableMixin
from .service_interfaces import (
    IGapDetectionService,
    INodeProposalService,
    IRelationshipAnalyzerService,
    IConfidenceCalculatorService,
    IGraphAnalyticsService,
    IKnowledgeValidatorService,
)

__all__ = [
    "BaseService",
    "ServiceConfig",
    "AsyncServiceMixin",
    "CacheableMixin",
    "IGapDetectionService",
    "INodeProposalService",
    "IRelationshipAnalyzerService",
    "IConfidenceCalculatorService",
    "IGraphAnalyticsService",
    "IKnowledgeValidatorService",
]
