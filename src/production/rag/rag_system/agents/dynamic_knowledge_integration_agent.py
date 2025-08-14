from typing import Any

from ..core.agent_interface import AgentInterface
from ..retrieval.graph_store import GraphStore
from ..utils.embedding import BERTEmbeddingModel


class DynamicKnowledgeIntegrationAgent(AgentInterface):
    """Agent responsible for updating the knowledge graph with new relations discovered during interactions."""

    def __init__(self, graph_store: GraphStore | None = None) -> None:
        super().__init__()
        self.graph_store = graph_store or GraphStore()
        self.embedding_model = BERTEmbeddingModel()

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

    async def generate(self, prompt: str) -> str:
        """Generate response for knowledge integration tasks."""
        if "integrate" in prompt.lower() or "knowledge" in prompt.lower():
            return f"Processing knowledge integration request: {prompt[:100]}..."
        elif "relation" in prompt.lower():
            return f"Analyzing relationships in: {prompt[:100]}..."
        else:
            return f"Knowledge integration analysis: {prompt[:100]}..."

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        try:
            _, embeddings = self.embedding_model.encode(text)
            mean_embedding = embeddings.mean(dim=0).detach().cpu().tolist()
            return [float(x) for x in mean_embedding]
        except Exception:
            # Fallback to deterministic random embedding
            import hashlib
            import random

            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            rng = random.Random(seed)
            return [rng.random() for _ in range(self.embedding_model.hidden_size)]

    async def rerank(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Rerank results based on knowledge graph connectivity."""
        for result in results:
            # Boost score if result mentions entities in our graph
            content = result.get("content", "")
            boost = 0.0

            if hasattr(self.graph_store.graph, "nodes"):
                for node_id in self.graph_store.graph.nodes():
                    if str(node_id) in content:
                        boost += 0.1

            result["score"] = result.get("score", 0.0) + boost

        # Sort and return top k
        return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return introspection information."""
        return {
            "type": "DynamicKnowledgeIntegrationAgent",
            "graph_nodes": await self.graph_store.get_count(),
            "embedding_model": "BERTEmbeddingModel",
            "capabilities": [
                "knowledge_integration",
                "relation_extraction",
                "graph_enhancement",
            ],
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with another agent."""
        response = await recipient.generate(f"Knowledge integration message: {message}")
        return f"Sent knowledge update: {message}, Received: {response}"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space for knowledge discovery."""
        # Analyze query for knowledge gaps
        background = f"Knowledge graph analysis for: {query}. "

        if hasattr(self.graph_store.graph, "nodes"):
            node_count = len(list(self.graph_store.graph.nodes()))
            background += f"Current graph contains {node_count} entities. "

        background += "Searching for new relations and knowledge gaps."

        refined_query = f"Enhanced query with graph context: {query}"
        return background, refined_query
