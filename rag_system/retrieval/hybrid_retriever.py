# rag_system/retrieval/hybrid_retriever.py

from typing import List, Dict, Any
from ..core.interfaces import Retriever
from ..core.config import RAGConfig
from .vector_store import VectorStore
from .graph_store import GraphStore

class HybridRetriever(Retriever):
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = VectorStore(config)
        self.graph_store = GraphStore(config)

    async def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        vector_results = await self.vector_store.retrieve(query, k)
        graph_results = await self.graph_store.retrieve(query, k)
        
        combined_results = self._combine_results(vector_results, graph_results)
        return combined_results[:k]

    def _combine_results(self, vector_results: List[Dict[str, Any]], graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Normalize scores
        max_vector_score = max([r['score'] for r in vector_results]) if vector_results else 1
        max_graph_score = max([r['score'] for r in graph_results]) if graph_results else 1
        
        for r in vector_results:
            r['normalized_score'] = r['score'] / max_vector_score
        for r in graph_results:
            r['normalized_score'] = r['score'] / max_graph_score
        
        # Combine results
        combined = vector_results + graph_results
        
        # Remove duplicates, favoring higher scores
        seen = {}
        unique_combined = []
        for r in combined:
            if r['id'] not in seen or r['normalized_score'] > seen[r['id']]['normalized_score']:
                seen[r['id']] = r
                unique_combined.append(r)
        
        # Sort by normalized score
        return sorted(unique_combined, key=lambda x: x['normalized_score'], reverse=True)
