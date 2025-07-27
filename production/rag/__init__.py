"""Production RAG components organized by Sprint 2."""

# Export main RAG classes
try:
    from .rag_system.core.pipeline import EnhancedRAGPipeline
    from .rag_system.core.config import RAGConfig

    # Alias for convenience
    RAGPipeline = EnhancedRAGPipeline

    __all__ = ['RAGPipeline', 'EnhancedRAGPipeline', 'RAGConfig']
except ImportError:
    # Handle missing dependencies gracefully
    RAGPipeline = None
    EnhancedRAGPipeline = None
    RAGConfig = None
    __all__ = []
