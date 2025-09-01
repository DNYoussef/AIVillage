"""
Service factory functions for SageAgent components.
"""

import logging
from typing import Any, Dict

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.core.latent_space_activation import LatentSpaceActivation
from rag_system.error_handling.adaptive_controller import AdaptiveErrorController
from rag_system.processing.confidence_estimator import ConfidenceEstimator
from rag_system.retrieval.vector_store import VectorStore
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker

from ..collaboration import CollaborationManager
from ..research_capabilities import ResearchCapabilities
from ..user_intent_interpreter import UserIntentInterpreter

from .cognitive_composite import CognitiveLayerComposite
from .processing_factory import ProcessingChainFactory
from .config import (
    CognitiveServiceConfig,
    ProcessingServiceConfig,
    CollaborationServiceConfig,
    ResearchServiceConfig,
)

logger = logging.getLogger(__name__)


async def create_cognitive_composite(
    vector_store: VectorStore, config: CognitiveServiceConfig
) -> CognitiveLayerComposite:
    """Create and initialize cognitive layer composite."""
    try:
        composite = CognitiveLayerComposite(vector_store, config)
        await composite.initialize()
        return composite
    except Exception as e:
        logger.error(f"Failed to create cognitive composite: {e}")
        raise


async def create_processing_chain(
    rag_system: EnhancedRAGPipeline,
    latent_space_activation: LatentSpaceActivation,
    cognitive_nexus: CognitiveNexus,
    confidence_estimator: ConfidenceEstimator,
    sage_agent: Any,
    config: ProcessingServiceConfig,
) -> ProcessingChainFactory:
    """Create and initialize processing chain factory."""
    try:
        factory = ProcessingChainFactory(
            rag_system=rag_system,
            latent_space_activation=latent_space_activation,
            cognitive_nexus=cognitive_nexus,
            confidence_estimator=confidence_estimator,
            sage_agent=sage_agent,
            config=config,
        )
        await factory.initialize()
        return factory
    except Exception as e:
        logger.error(f"Failed to create processing chain: {e}")
        raise


async def create_rag_system(config: Any, knowledge_tracker: UnifiedKnowledgeTracker) -> EnhancedRAGPipeline:
    """Create RAG system."""
    try:
        return EnhancedRAGPipeline(config, knowledge_tracker)
    except Exception as e:
        logger.error(f"Failed to create RAG system: {e}")
        raise


async def create_cognitive_nexus() -> CognitiveNexus:
    """Create cognitive nexus."""
    try:
        return CognitiveNexus()
    except Exception as e:
        logger.error(f"Failed to create cognitive nexus: {e}")
        raise


async def create_latent_space_activation() -> LatentSpaceActivation:
    """Create latent space activation."""
    try:
        return LatentSpaceActivation()
    except Exception as e:
        logger.error(f"Failed to create latent space activation: {e}")
        raise


async def create_error_controller() -> AdaptiveErrorController:
    """Create adaptive error controller."""
    try:
        return AdaptiveErrorController()
    except Exception as e:
        logger.error(f"Failed to create error controller: {e}")
        raise


async def create_confidence_estimator() -> ConfidenceEstimator:
    """Create confidence estimator."""
    try:
        return ConfidenceEstimator()
    except Exception as e:
        logger.error(f"Failed to create confidence estimator: {e}")
        raise


async def create_collaboration_manager(sage_agent: Any, config: CollaborationServiceConfig) -> CollaborationManager:
    """Create collaboration manager."""
    try:
        return CollaborationManager(sage_agent)
    except Exception as e:
        logger.error(f"Failed to create collaboration manager: {e}")
        raise


async def create_research_capabilities(sage_agent: Any, config: ResearchServiceConfig) -> ResearchCapabilities:
    """Create research capabilities manager."""
    try:
        return ResearchCapabilities(sage_agent)
    except Exception as e:
        logger.error(f"Failed to create research capabilities: {e}")
        raise


async def create_user_intent_interpreter() -> UserIntentInterpreter:
    """Create user intent interpreter."""
    try:
        return UserIntentInterpreter()
    except Exception as e:
        logger.error(f"Failed to create user intent interpreter: {e}")
        raise


def create_service_factory_registry() -> Dict[str, callable]:
    """Create registry of all service factories."""
    return {
        "cognitive_composite": create_cognitive_composite,
        "processing_chain": create_processing_chain,
        "rag_system": create_rag_system,
        "cognitive_nexus": create_cognitive_nexus,
        "latent_space_activation": create_latent_space_activation,
        "error_controller": create_error_controller,
        "confidence_estimator": create_confidence_estimator,
        "collaboration_manager": create_collaboration_manager,
        "research_capabilities": create_research_capabilities,
        "user_intent_interpreter": create_user_intent_interpreter,
    }
