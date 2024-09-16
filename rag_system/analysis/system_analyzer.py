# rag_system/analysis/system_analyzer.py

from typing import Dict, Any
from ..core.config import RAGConfig
from ..community_detector.community_detector import CommunityDetector
from ..retrieval.graph_store import GraphStore
from ..retrieval.vector_store import VectorStore

class SystemAnalyzer:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.graph_store = GraphStore(config)
        self.vector_store = VectorStore(config)
        self.community_detector = CommunityDetector(config, self.graph_store)

    async def analyze_system_structure(self) -> Dict[str, Any]:
        try:
            communities = await self.community_detector.detect_communities()
            graph_summary = await self.graph_store.get_graph_summary()
            vector_store_stats = await self.vector_store.get_statistics()
            
            # Additional community analysis
            community_sizes = {}
            community_densities = {}
            for community_id in set(communities.values()):
                community_sizes[community_id] = await self.community_detector.get_community_size(community_id)
                community_densities[community_id] = await self.community_detector.calculate_community_density(community_id)

            return {
                "communities": communities,
                "community_sizes": community_sizes,
                "community_densities": community_densities,
                "graph_summary": graph_summary,
                "vector_store_stats": vector_store_stats
            }
        except Exception as e:
            raise Exception(f"Error analyzing system structure: {str(e)}")
