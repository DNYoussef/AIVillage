# rag_system/processing/self_referential_query_processor.py

from typing import Dict, Any, List
from ..core.pipeline import EnhancedRAGPipeline

class SelfReferentialQueryProcessor:
    def __init__(self, rag_system: EnhancedRAGPipeline, history_limit: int = 100):
        self.rag_system = rag_system
        # Keep an in-memory list of processed queries
        self.query_history: List[str] = []
        self.history_limit = history_limit

    async def process_self_query(self, query: str) -> str:
        # Record every processed query
        self.query_history.append(query)
        # Enforce history size limit
        if len(self.query_history) > self.history_limit:
            self.query_history.pop(0)

        if query.startswith("SELF:"):
            return await self._process_internal_query(query[5:])
        else:
            return await self.rag_system.process(query)

    async def _process_internal_query(self, query: str) -> str:
        # Implement logic to query system's internal state, knowledge, or history
        if query.startswith("STATUS"):
            return await self._get_system_status()
        elif query.startswith("KNOWLEDGE"):
            return await self._get_knowledge_summary()
        elif query.startswith("HISTORY"):
            # Parse optional limit, e.g., "HISTORY 5"
            parts = query.split()
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            return await self._get_query_history(limit)
        else:
            return f"Unknown self-referential query: {query}"

    async def _get_system_status(self) -> str:
        # Implement logic to return system status
        return "System is operational. Current load: 30%"

    async def _get_knowledge_summary(self) -> str:
        # Implement logic to summarize current knowledge
        vector_count = await self.rag_system.hybrid_retriever.vector_store.get_count()
        graph_count = await self.rag_system.hybrid_retriever.graph_store.get_count()
        return f"Current knowledge: {vector_count} vector entries, {graph_count} graph nodes"

    async def _get_query_history(self, n: int = 5) -> str:
        """Return a string describing the last ``n`` processed queries."""
        history = self.query_history[-n:]
        formatted = ", ".join(f"[{q}]" for q in history)
        return f"Recent queries: {formatted}" if formatted else "Recent queries:"
