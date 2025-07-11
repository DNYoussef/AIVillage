"""Real shortest-/semantic-path explainer with in-memory cache."""

import os
import logging
import time
import functools
from typing import Dict, Any, List

import networkx as nx
from .retrieval.graph_store import GraphStore


logger = logging.getLogger(__name__)

# currently default to NetworkX but optionally support Neo4j via ``GraphStore``
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx").lower()
MAX_HOPS = int(os.getenv("MAX_EXPLAIN_HOPS", "3"))


class GraphPathExplainer:
    """Explain relationships between nodes of a knowledge graph."""

    def __init__(self, graph_store: GraphStore | None = None) -> None:
        self.graph_store = graph_store or GraphStore()
        self._load_graph()

    # ------------------------------------------------------------------ #
    def _load_graph(self) -> None:
        """Loads or constructs the knowledge graph."""
        t0 = time.time()
        self.g = self.graph_store.graph
        if hasattr(self.g, "number_of_edges") and self.g.number_of_edges() == 0:
            # demo seed – in production this would load from parquet or DB
            self.g.add_edge("A", "B", relation="linked_to")
            self.g.add_edge("B", "C", relation="causes")
            self.g.add_edge("C", "D", relation="enables")
        try:
            num_nodes = self.g.number_of_nodes()
        except Exception:
            num_nodes = 0
        logger.info(
            "Graph loaded (%d nodes) in %.0f ms",
            num_nodes,
            (time.time() - t0) * 1000,
        )

    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=10_000)
    def explain_path(
        self, start: str, end: str, max_hops: int = MAX_HOPS
    ) -> Dict[str, Any]:
        """Return the reasoning path between two nodes if it exists."""
        t0 = time.time()

        nodes: List[str] = []
        edges: List[Dict[str, Any]] = []

        if GRAPH_BACKEND == "neo4j" and self.graph_store.driver is not None:
            query = (
                "MATCH (a {id: $start}), (b {id: $end}) "
                "MATCH p = shortestPath((a)-[*..$max_hops]->(b)) "
                "RETURN [n IN nodes(p) | n.id] AS nodes, "
                "[r IN relationships(p) | {source: startNode(r).id, target: endNode(r).id, props: properties(r)}] AS rels"
            )
            with self.graph_store.driver.session() as session:
                rec = session.run(
                    query, start=start, end=end, max_hops=max_hops
                ).single()
                if rec:
                    nodes = rec["nodes"]
                    edges = rec["rels"]
        else:
            if start not in self.g or end not in self.g:
                return {"nodes": [], "edges": [], "hops": 0, "found": False}
            try:
                nodes = nx.shortest_path(self.g, start, end)
            except nx.NetworkXNoPath:
                return {"nodes": [], "edges": [], "hops": 0, "found": False}

            if len(nodes) - 1 > max_hops:
                return {"nodes": [], "edges": [], "hops": 0, "found": False}

            for src, dst in zip(nodes[:-1], nodes[1:]):
                data = dict(self.g.get_edge_data(src, dst, default={}))
                edges.append({"source": src, "target": dst, **data})

        if not nodes:
            return {"nodes": [], "edges": [], "hops": 0, "found": False}

        return {
            "nodes": nodes,
            "edges": edges,
            "hops": len(edges),
            "found": True,
            "processing_ms": round((time.time() - t0) * 1000, 1),
        }


# Module-level singleton for reuse
_explainer = GraphPathExplainer()


def explain_path(start: str, end: str, max_hops: int = MAX_HOPS) -> Dict[str, Any]:
    """Public wrapper used by the API layer."""
    return _explainer.explain_path(start, end, max_hops)
