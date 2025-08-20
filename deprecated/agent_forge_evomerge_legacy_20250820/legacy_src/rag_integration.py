"""
RAG Integration Compatibility Layer

This module provides backward compatibility for Agent Forge RAG integration.
The RAG system has been consolidated into packages/rag/.

DEPRECATED: This module is deprecated. Use the new unified RAG system instead.
"""

import warnings
from typing import Any

# Issue deprecation warning
warnings.warn(
    "src.agent_forge.rag_integration is deprecated. "
    "Use the new unified RAG system at packages.rag instead. "
    "See deprecated/rag_consolidation/20250818/DEPRECATION_NOTICE.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Compatibility imports from the new unified system
try:
    from packages.rag import HyperRAG, QueryMode, RAGConfig

    # Compatibility class that wraps the new HyperRAG system
    class HyperRAGIntegration:
        """Compatibility wrapper for the old HyperRAGIntegration class."""

        def __init__(self, config: Any = None):
            warnings.warn(
                "HyperRAGIntegration is deprecated. Use HyperRAG from packages.rag instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Convert old config to new format if needed
            rag_config = RAGConfig()
            if hasattr(config, "model_path"):
                # Old config was for a specific model, new system is model-agnostic
                pass
            self._hyper_rag = HyperRAG(rag_config)

        async def initialize(self):
            return await self._hyper_rag.initialize()

        async def answer_question(self, query: str) -> tuple[str, Any]:
            result = await self._hyper_rag.query(query, mode=QueryMode.BALANCED)
            # Return format compatible with old interface
            return result.synthesized_answer.answer, None

        async def validate_rag_performance(self, test_queries: list[str]) -> dict[str, Any]:
            results = {"total_queries": len(test_queries), "successful_queries": 0}
            for query in test_queries:
                try:
                    await self.answer_question(query)
                    results["successful_queries"] += 1
                except Exception:
                    pass
            return results

    # Compatibility function for model selection
    class AgentForgeRAGSelector:
        """Compatibility wrapper for the old AgentForgeRAGSelector class."""

        def __init__(self, results_dir: str):
            warnings.warn(
                "AgentForgeRAGSelector is deprecated. The new HyperRAG system is model-agnostic.",
                DeprecationWarning,
                stacklevel=2,
            )

        async def select_best_model(self) -> dict[str, Any]:
            return {
                "selected_phase": "unified_hyper_rag",
                "model_path": "packages.rag.HyperRAG",
                "confidence": "high",
                "recommendation": "Use the new unified HyperRAG system",
            }

except ImportError:
    # Fallback if new system isn't available
    class HyperRAGIntegration:
        def __init__(self, config: Any = None):
            raise ImportError(
                "New unified RAG system not available. " "Please ensure packages/rag/ is properly installed."
            )

    class AgentForgeRAGSelector:
        def __init__(self, results_dir: str):
            raise ImportError(
                "New unified RAG system not available. " "Please ensure packages/rag/ is properly installed."
            )


# Export compatibility interface
__all__ = ["HyperRAGIntegration", "AgentForgeRAGSelector"]
