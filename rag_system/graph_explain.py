"""Real shortest-/semantic-path explainer with in-memory cache."""

import os
import logging
import time
import functools
from typing import Dict, Any, List

import networkx as nx


logger = logging.getLogger(__name__)

# currently we only support NetworkX but keep a flag for future Neo4j
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx")
MAX_HOPS = int(os.getenv("MAX_EXPLAIN_HOPS", "3"))


class GraphPathExplainer:
    """Explain relationships between nodes of a knowledge graph."""

    def __init__(self) -> None:
        self._load_graph()

    # ------------------------------------------------------------------ #
    def _load_graph(self) -> None:
        """Loads or constructs the knowledge graph."""
        t0 = time.time()
        self.g = nx.DiGraph()
        # demo seed – in production this would load from parquet or DB
        self.g.add_edge("A", "B", relation="linked_to")
        self.g.add_edge("B", "C", relation="causes")
        self.g.add_edge("C", "D", relation="enables")
        logger.info(
            "Graph loaded (%d nodes) in %.0f ms",
            self.g.number_of_nodes(),
            (time.time() - t0) * 1000,
        )

    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=10_000)
    def explain_path(self, start: str, end: str, max_hops: int = MAX_HOPS) -> Dict[str, Any]:
        """Return the reasoning path between two nodes if it exists."""
        t0 = time.time()

        if start not in self.g or end not in self.g:
            return {"path": [], "hops": 0, "found": False}

        try:
            path_nodes = nx.shortest_path(self.g, start, end)
        except nx.NetworkXNoPath:
            return {"path": [], "hops": 0, "found": False}

        if len(path_nodes) - 1 > max_hops:
            return {"path": [], "hops": 0, "found": False}

        # Build hop list with relation details
        hops: List[Dict[str, str]] = []
        for src, dst in zip(path_nodes[:-1], path_nodes[1:]):
            rel = self.g[src][dst].get("relation", "related_to")
            hops.append({"s": src, "r": rel, "t": dst})

        return {
            "path": hops,
            "hops": len(hops),
            "found": True,
            "processing_ms": round((time.time() - t0) * 1000, 1),
        }


# Module-level singleton for reuse
_explainer = GraphPathExplainer()


def explain_path(start: str, end: str) -> Dict[str, Any]:
    """Public wrapper used by the API layer."""
    return _explainer.explain_path(start, end, MAX_HOPS)

