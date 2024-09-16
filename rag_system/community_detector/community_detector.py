# rag_system/community_detector/community_detector.py

import networkx as nx
from typing import List, Dict, Any
from community import community_louvain
from ..core.config import RAGConfig
from ..retrieval.graph_store import GraphStore

class CommunityDetector:
    def __init__(self, config: RAGConfig, graph_store: GraphStore):
        self.config = config
        self.graph_store = graph_store
        self.nx_graph = None
        self.communities = None

    async def detect_communities(self):
        self._update_nx_graph()
        self.communities = community_louvain.best_partition(self.nx_graph)
        return self.communities

    def _update_nx_graph(self):
      self.nx_graph = nx.Graph()
      with self.graph_store.driver.session() as session:
          # Fetch nodes
          result = session.run("MATCH (n) RETURN id(n) as id, properties(n) as props")
          for record in result:
              self.nx_graph.add_node(record["id"], **record["props"])
        
          # Fetch relationships
          result = session.run("MATCH (a)-[r]->(b) RETURN id(a) as source, id(b) as target, type(r) as type, properties(r) as props")
          for record in result:
              self.nx_graph.add_edge(record["source"], record["target"], type=record["type"], **record["props"])

    async def get_community_for_node(self, node: Any) -> int:
        if self.communities is None:
            await self.detect_communities()
        return self.communities.get(node, -1)

    async def get_community_size(self, community_id: int) -> int:
        if self.communities is None:
            await self.detect_communities()
        return sum(1 for v in self.communities.values() if v == community_id)

    async def get_inter_community_connections(self, community_id: int) -> List[tuple]:
        if self.communities is None:
            await self.detect_communities()
        
        inter_community_edges = []
        for edge in self.nx_graph.edges():
            if self.communities[edge[0]] == community_id and self.communities[edge[1]] != community_id:
                inter_community_edges.append(edge)
            elif self.communities[edge[1]] == community_id and self.communities[edge[0]] != community_id:
                inter_community_edges.append(edge)
        return inter_community_edges

    async def calculate_community_density(self, community_id: int) -> float:
        if self.communities is None:
            await self.detect_communities()
        
        community_nodes = [node for node, comm in self.communities.items() if comm == community_id]
        subgraph = self.nx_graph.subgraph(community_nodes)