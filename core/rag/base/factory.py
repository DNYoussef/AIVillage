"""
RAG Factory Pattern Implementation

Provides factory methods for creating appropriate RAG system instances
based on configuration and requirements. Handles the complexity of
different RAG system setups and dependencies.
"""

from __future__ import annotations

from typing import Dict, Type, Optional, Any
import logging
from enum import Enum

from .base_rag_system import BaseRAGSystem, RAGConfiguration, RAGType, ProcessingMode


logger = logging.getLogger(__name__)


class RAGFactory:
    """
    Factory for creating RAG system instances
    
    Provides a central point for RAG system instantiation with
    proper dependency injection and configuration management.
    """
    
    _registry: Dict[RAGType, Type[BaseRAGSystem]] = {}
    
    @classmethod
    def register(cls, rag_type: RAGType, rag_class: Type[BaseRAGSystem]):
        """Register a RAG system implementation"""
        cls._registry[rag_type] = rag_class
        logger.info(f"Registered RAG system: {rag_type.value} -> {rag_class.__name__}")
    
    @classmethod
    def create(cls, config: RAGConfiguration) -> BaseRAGSystem:
        """
        Create a RAG system instance based on configuration
        
        Args:
            config: RAG system configuration
            
        Returns:
            Configured RAG system instance
            
        Raises:
            ValueError: If RAG type is not registered
        """
        rag_type = config.rag_type
        
        if rag_type not in cls._registry:
            available_types = ", ".join([t.value for t in cls._registry.keys()])
            raise ValueError(
                f"Unknown RAG type: {rag_type.value}. "
                f"Available types: {available_types}"
            )
        
        rag_class = cls._registry[rag_type]
        
        try:
            instance = rag_class(config)
            logger.info(f"Created RAG system: {rag_type.value} with ID: {config.system_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create RAG system {rag_type.value}: {e}")
            raise
    
    @classmethod
    def create_default(
        cls, 
        rag_type: RAGType,
        system_id: str,
        **kwargs
    ) -> BaseRAGSystem:
        """
        Create a RAG system with default configuration
        
        Args:
            rag_type: Type of RAG system to create
            system_id: Unique identifier for the system
            **kwargs: Additional configuration overrides
            
        Returns:
            RAG system with default configuration
        """
        config = RAGConfiguration(
            rag_type=rag_type,
            system_id=system_id,
            **kwargs
        )
        
        return cls.create(config)
    
    @classmethod
    def create_for_mode(
        cls,
        mode: ProcessingMode,
        system_id: str,
        **kwargs
    ) -> BaseRAGSystem:
        """
        Create optimal RAG system for specific processing mode
        
        Args:
            mode: Processing mode requirements
            system_id: Unique identifier for the system
            **kwargs: Additional configuration overrides
            
        Returns:
            RAG system optimized for the specified mode
        """
        # Determine optimal RAG type based on mode
        optimal_type = cls._get_optimal_rag_type(mode)
        
        # Create mode-specific configuration
        config = cls._create_mode_specific_config(
            rag_type=optimal_type,
            mode=mode,
            system_id=system_id,
            **kwargs
        )
        
        return cls.create(config)
    
    @classmethod
    def get_registered_types(cls) -> list[RAGType]:
        """Get list of registered RAG types"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, rag_type: RAGType) -> bool:
        """Check if a RAG type is registered"""
        return rag_type in cls._registry
    
    @classmethod
    def _get_optimal_rag_type(cls, mode: ProcessingMode) -> RAGType:
        """Determine optimal RAG type for processing mode"""
        mode_to_type_mapping = {
            ProcessingMode.FAST: RAGType.MINI,
            ProcessingMode.BALANCED: RAGType.HYPER,
            ProcessingMode.COMPREHENSIVE: RAGType.HYPER,
            ProcessingMode.CREATIVE: RAGType.HYPER,
            ProcessingMode.ANALYTICAL: RAGType.HYPER,
            ProcessingMode.DISTRIBUTED: RAGType.HYPER,
            ProcessingMode.EDGE_OPTIMIZED: RAGType.MINI,
            ProcessingMode.PRIVACY_FIRST: RAGType.MINI
        }
        
        optimal_type = mode_to_type_mapping.get(mode, RAGType.HYPER)
        
        # Fall back if optimal type not available
        if not cls.is_registered(optimal_type):
            # Prefer HYPER as general-purpose fallback, then MINI, then BASE
            for fallback_type in [RAGType.HYPER, RAGType.MINI, RAGType.BASE]:
                if cls.is_registered(fallback_type):
                    logger.warning(
                        f"Optimal RAG type {optimal_type.value} not available for mode {mode.value}. "
                        f"Using fallback: {fallback_type.value}"
                    )
                    return fallback_type
            
            raise RuntimeError("No RAG system types are registered")
        
        return optimal_type
    
    @classmethod
    def _create_mode_specific_config(
        cls,
        rag_type: RAGType,
        mode: ProcessingMode,
        system_id: str,
        **kwargs
    ) -> RAGConfiguration:
        """Create configuration optimized for specific mode"""
        
        # Base configuration
        config_params = {
            "rag_type": rag_type,
            "system_id": system_id,
            "default_mode": mode
        }
        
        # Mode-specific optimizations
        if mode == ProcessingMode.FAST:
            config_params.update({
                "max_results": 5,
                "similarity_threshold": 0.8,  # Higher threshold for fewer, better results
                "enable_reasoning": False,     # Disable for speed
                "enable_synthesis": False,
                "cache_ttl_seconds": 7200,     # Longer cache for speed
                "vector_backend": "simple"     # Faster backend
            })
        
        elif mode == ProcessingMode.COMPREHENSIVE:
            config_params.update({
                "max_results": 20,
                "similarity_threshold": 0.5,  # Lower threshold for more results
                "enable_reasoning": True,
                "enable_synthesis": True,
                "enable_graph_traversal": True,
                "max_concurrent_queries": 5,  # Lower concurrency for thoroughness
                "vector_backend": "faiss"      # More accurate backend
            })
        
        elif mode == ProcessingMode.CREATIVE:
            config_params.update({
                "max_results": 15,
                "similarity_threshold": 0.6,
                "enable_reasoning": True,
                "enable_synthesis": True,
                "enable_graph_traversal": True,
                "custom_config": {
                    "creativity_boost": True,
                    "diverse_results": True,
                    "cross_domain_search": True
                }
            })
        
        elif mode == ProcessingMode.ANALYTICAL:
            config_params.update({
                "max_results": 15,
                "similarity_threshold": 0.7,
                "confidence_threshold": 0.8,  # Higher confidence for analysis
                "enable_reasoning": True,
                "enable_synthesis": True,
                "custom_config": {
                    "fact_checking": True,
                    "source_validation": True,
                    "logical_consistency": True
                }
            })
        
        elif mode == ProcessingMode.EDGE_OPTIMIZED:
            config_params.update({
                "max_results": 3,
                "similarity_threshold": 0.8,
                "enable_caching": True,
                "cache_ttl_seconds": 14400,   # Long cache for offline capability
                "enable_reasoning": False,
                "enable_synthesis": False,
                "storage_backend": "memory",   # In-memory for speed
                "vector_backend": "simple",
                "max_concurrent_queries": 3   # Limited resources
            })
        
        elif mode == ProcessingMode.PRIVACY_FIRST:
            config_params.update({
                "privacy_mode": True,
                "anonymize_queries": True,
                "memory_enabled": False,       # Don't store queries
                "enable_caching": False,       # Don't cache for privacy
                "storage_backend": "memory",   # Ephemeral storage
                "custom_config": {
                    "local_processing": True,
                    "encrypted_vectors": True,
                    "no_external_calls": True
                }
            })
        
        elif mode == ProcessingMode.DISTRIBUTED:
            config_params.update({
                "max_results": 25,
                "max_concurrent_queries": 20,
                "enable_reasoning": True,
                "enable_synthesis": True,
                "custom_config": {
                    "distributed_search": True,
                    "federation_enabled": True,
                    "load_balancing": True
                }
            })
        
        # Apply any additional overrides
        config_params.update(kwargs)
        
        return RAGConfiguration(**config_params)
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get factory information and registered systems"""
        return {
            "registered_types": [t.value for t in cls._registry.keys()],
            "available_modes": [m.value for m in ProcessingMode],
            "factory_version": "1.0.0",
            "total_registered": len(cls._registry)
        }


# Auto-registration decorator for RAG implementations
def rag_system(rag_type: RAGType):
    """
    Decorator to automatically register RAG system implementations
    
    Usage:
        @rag_system(RAGType.HYPER)
        class HyperRAGSystem(BaseRAGSystem):
            ...
    """
    def decorator(cls: Type[BaseRAGSystem]):
        RAGFactory.register(rag_type, cls)
        return cls
    return decorator