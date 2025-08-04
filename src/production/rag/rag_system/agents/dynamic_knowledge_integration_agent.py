from typing import Any

from rag_system.core.agent_interface import AgentInterface
from rag_system.retrieval.graph_store import GraphStore


class DynamicKnowledgeIntegrationAgent(AgentInterface):
    """Agent responsible for updating the knowledge graph with new relations discovered during interactions."""

    def __init__(self, graph_store: GraphStore) -> None:
        super().__init__()
        self.graph_store = graph_store

    def integrate_new_knowledge(self, new_relations: dict[str, Any]) -> None:
        """Integrate new relations into the knowledge graph.

        Args:
            new_relations (Dict[str, Any]): New relations to be added to the knowledge graph.
        """
        # ``new_relations`` is expected to be a mapping describing edges to add
        # to the underlying :class:`GraphStore`.  Each entry should contain a
        # source node, a target node and a relation type.  The method gracefully
        # handles missing nodes by creating them on-the-fly.

        relations = new_relations.get("relations", [])
        for rel in relations:
            source = rel.get("source")
            target = rel.get("target")
            relation_type = rel.get("relation")
            if not source or not target:
                continue

            if not self.graph_store.graph.has_node(source):
                self.graph_store.graph.add_node(source)
            if not self.graph_store.graph.has_node(target):
                self.graph_store.graph.add_node(target)

            self.graph_store.graph.add_edge(source, target, relation=relation_type)
