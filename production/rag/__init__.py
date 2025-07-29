"""Production RAG components organized by Sprint 2."""

# Export main RAG classes
try:
    from .rag_system.core.config import RAGConfig
    from .rag_system.core.pipeline import EnhancedRAGPipeline

    # Alias for convenience
    RAGPipeline = EnhancedRAGPipeline

    __all__ = ["EnhancedRAGPipeline", "RAGConfig", "RAGPipeline"]
except ImportError:
    # Handle missing dependencies gracefully
    RAGPipeline = None
    EnhancedRAGPipeline = None
    RAGConfig = None
    __all__ = []
