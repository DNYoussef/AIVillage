"""
GraphFixer Service Interfaces Package

Interface definitions for all GraphFixer services,
enabling loose coupling and testability.
"""

from .base_service import AsyncServiceMixin, BaseService, CacheableMixin, ServiceConfig
from .service_interfaces import (
    IConfidenceCalculatorService,
    IGapDetectionService,
    IGraphAnalyticsService,
    IKnowledgeValidatorService,
    INodeProposalService,
    IRelationshipAnalyzerService,
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
