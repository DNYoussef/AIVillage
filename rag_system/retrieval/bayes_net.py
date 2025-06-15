from typing import Dict, Any
from collections import defaultdict

class BayesNet:
    """Simple Bayesian network for storing probabilistic knowledge."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_node(self, node_id: str, content: str, probability: float = 0.5, uncertainty: float = 0.1) -> None:
        self.nodes[node_id] = {
            "content": content,
            "probability": probability,
            "uncertainty": uncertainty,
        }

    def add_edge(self, parent: str, child: str, probability: float) -> None:
        self.edges[parent][child] = probability

    def update_node(self, node_id: str, probability: float, uncertainty: float) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["probability"] = probability
            self.nodes[node_id]["uncertainty"] = uncertainty

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.nodes.get(node_id, {})

    def all_nodes(self) -> Dict[str, Dict[str, Any]]:
        return self.nodes
