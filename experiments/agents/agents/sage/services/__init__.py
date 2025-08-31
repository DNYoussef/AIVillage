"""
Service layer for SageAgent dependency injection and management.
"""

from .interfaces import (
    ICognitiveService,
    IProcessingService,
    IKnowledgeService,
    ILearningService,
)
from .service_locator import SageAgentServiceLocator
from .cognitive_composite import CognitiveLayerComposite
from .processing_factory import ProcessingChainFactory
from .config import SageAgentConfig

__all__ = [
    "ICognitiveService",
    "IProcessingService", 
    "IKnowledgeService",
    "ILearningService",
    "SageAgentServiceLocator",
    "CognitiveLayerComposite",
    "ProcessingChainFactory",
    "SageAgentConfig",
]