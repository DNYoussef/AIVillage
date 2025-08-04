from datetime import datetime
import math
import random
from typing import Any

try:  # networkx might not be installed in minimal test envs
    import networkx as nx
except Exception:  # pragma: no cover - handled by fallback logic
    nx = None  # type: ignore

try:  # embedding dependencies are optional
    from AIVillage.src.production.rag.rag_system.utils.embedding import (
        BERTEmbeddingModel,
    )
except Exception:  # pragma: no cover - missing torch/transformers
    BERTEmbeddingModel = None  # type: ignore

from AIVillage.src.production.rag.rag_system.core.config import UnifiedConfig
from AIVillage.src.production.rag.rag_system.core.structures import RetrievalResult


class GraphStore:
    def __init__(
        self, config: UnifiedConfig | None = None, embedding_model: Any | None = None
    ) -> None:
        """Create a GraphStore.

        Similar to :class:`VectorStore`, older code instantiated ``GraphStore``
        without providing a configuration object which caused a ``TypeError``
        after the constructor signature changed.  The configuration parameter is
        now optional and defaults to a new :class:`UnifiedConfig` instance.
        """
        self.config = config or UnifiedConfig()
        self.embedding_model = embedding_model
        if self.embedding_model is None and BERTEmbeddingModel is not None:
            try:
                self.embedding_model = BERTEmbeddingModel()
            except Exception:  # pragma: no cover - model load failed
                self.embedding_model = None
        try:
            self.graph = nx.Graph()
        except Exception:  # pragma: no cover - fallback if networkx is stubbed
            self.graph = object()
        self._nodes: dict[str, dict[str, Any]] = {}
        self.driver = None  # This should be initialized with a proper Neo4j driver
        self.causal_edges = {}
        self.llm = None  # This should be initialized with a proper language model

    def _generate_embedding(self, text: str) -> list[float]:
        """Return an embedding vector for ``text``.

        When an embedding model is available it is used; otherwise a
        deterministic random vector based on ``hash(text)`` is returned.
        """
        if self.embedding_model is not None:
            try:
                _, emb = self.embedding_model.encode(text)
                try:
                    vec = emb.mean(dim=0).detach().cpu().tolist()
                except Exception:
                    try:
                        vec = emb.tolist()
                    except Exception:
                        vec = list(emb)
                return [float(v) for v in vec]
            except Exception:
                pass

        rng = random.Random(abs(hash(text)) % (2**32))
        return [rng.random() for _ in range(64)]

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for doc in documents:
            # Ensure each document has an embedding
            if "embedding" not in doc or doc["embedding"] is None:
                doc["embedding"] = self._generate_embedding(str(doc.get("content", "")))

            if hasattr(self.graph, "add_node"):
                self.graph.add_node(doc["id"], **doc)
            else:
                self._nodes[doc["id"]] = doc

        for i, doc in enumerate(documents):
            emb_i = doc.get("embedding", [])
            for other in documents[i + 1 :]:
                emb_j = other.get("embedding", [])
                dot = sum(a * b for a, b in zip(emb_i, emb_j, strict=False))
                norm_i = math.sqrt(sum(a * a for a in emb_i))
                norm_j = math.sqrt(sum(b * b for b in emb_j))
                denom = norm_i * norm_j
                sim = float(dot / denom) if denom else 0.0
                if hasattr(self.graph, "add_edge"):
                    self.graph.add_edge(doc["id"], other["id"], weight=sim)

    async def retrieve(
        self, query: str, k: int, timestamp: datetime | None = None
    ) -> list[RetrievalResult]:
        """Return nodes that match ``query``.

        When ``self.driver`` is provided it should be an instance of
        :class:`neo4j.Driver` configured with a full-text index named
        ``"nodeContent"``.  In that case this method performs the Neo4j query
        shown below.  If ``self.driver`` is ``None`` a fallback search is
        executed over ``self.graph`` by scanning the ``content`` attribute of
        each node.
        """
        if self.driver is None:
            query_lower = query.lower()
            results: list[RetrievalResult] = []
            for node_id, data in self.graph.nodes(data=True):
                content = str(data.get("content", ""))
                if query_lower in content.lower():
                    node_ts = data.get("timestamp", datetime.min)
                    if timestamp is None or node_ts <= timestamp:
                        results.append(
                            RetrievalResult(
                                id=node_id,
                                content=content,
                                score=1.0,
                                uncertainty=data.get("uncertainty", 0.0),
                                timestamp=node_ts,
                                version=data.get("version", 0),
                            )
                        )
                if len(results) >= k:
                    break
            return results[:k]

        with self.driver.session() as session:
            if timestamp:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes("nodeContent", $query)
                    YIELD node, score
                    MATCH (node)-[:VERSION]->(v:NodeVersion)
                    WHERE v.timestamp <= $timestamp
                    WITH node, score, v
                    ORDER BY v.timestamp DESC, score DESC
                    LIMIT $k
                    RETURN id(node) as id, v.content as content, score,
                        v.uncertainty as uncertainty, v.timestamp as timestamp,
                        v.version as version
                    """,
                    query=query,
                    timestamp=timestamp,
                    k=k,
                )
            else:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes("nodeContent", $query)
                    YIELD node, score
                    MATCH (node)-[:VERSION]->(v:NodeVersion)
                    WITH node, score, v
                    ORDER BY v.timestamp DESC, score DESC
                    LIMIT $k
                    RETURN id(node) as id, v.content as content, score,
                        v.uncertainty as uncertainty, v.timestamp as timestamp,
                        v.version as version
                    """,
                    query=query,
                    k=k,
                )

        return [
            RetrievalResult(
                id=record["id"],
                content=record["content"],
                score=record["score"],
                uncertainty=record["uncertainty"],
                timestamp=record["timestamp"],
                version=record["version"],
            )
            for record in result
        ]

    def update_causal_strength(
        self, source: str, target: str, observed_probability: float
    ) -> None:
        edge = self.causal_edges.get((source, target))
        if edge:
            learning_rate = 0.1
            edge.strength = (
                1 - learning_rate
            ) * edge.strength + learning_rate * observed_probability

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    async def get_snapshot(self, timestamp: datetime) -> dict[str, Any]:
        """Return a snapshot of the graph up to ``timestamp``.

        Nodes and edges whose ``timestamp`` attribute is greater than the
        provided ``timestamp`` are omitted from the snapshot.  If a node or edge
        does not have a ``timestamp`` attribute it is assumed to always be
        present.
        """
        snapshot = nx.Graph()

        for node_id, data in self.graph.nodes(data=True):
            node_ts = data.get("timestamp")
            if node_ts is None or node_ts <= timestamp:
                snapshot.add_node(node_id, **data)

        for source, target, data in self.graph.edges(data=True):
            if not (snapshot.has_node(source) and snapshot.has_node(target)):
                continue

            edge_ts = data.get("timestamp")
            if edge_ts is None or edge_ts <= timestamp:
                snapshot.add_edge(source, target, **data)

        return {
            "nodes": list(snapshot.nodes(data=True)),
            "edges": list(snapshot.edges(data=True)),
        }

    async def beam_search(
        self, query: str, beam_width: int, max_depth: int
    ) -> list[tuple[list[str], float]]:
        initial_entities = await self.get_initial_entities(query)
        beams = [[entity] for entity in initial_entities]

        for _ in range(max_depth):
            candidates = []
            for beam in beams:
                neighbors = await self.get_neighbors(beam[-1])
                for neighbor in neighbors:
                    new_beam = [*beam, neighbor]
                    score = await self.llm.score_path(query, new_beam)
                    candidates.append((new_beam, score))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        return beams

    async def get_initial_entities(self, query: str) -> list[str]:
        """Return a list of node IDs that match the query string."""
        query_lower = query.lower()
        matches: list[str] = []

        for node_id, data in self.graph.nodes(data=True):
            content = str(data.get("content", "")).lower()
            if query_lower in content:
                matches.append(node_id)
                if len(matches) >= self.config.top_k:
                    break

        return matches

    async def get_neighbors(self, entity: str) -> list[str]:
        """Return IDs of nodes adjacent to ``entity`` in the graph."""
        if hasattr(self.graph, "has_node") and self.graph.has_node(entity):
            return list(self.graph.neighbors(entity))
        return []

    def get_document_by_id(self, doc_id: str) -> dict[str, Any]:
        if hasattr(self.graph, "has_node") and self.graph.has_node(doc_id):
            return self.graph.nodes[doc_id]
        return self._nodes.get(doc_id)

    async def get_count(self) -> int:
        """Return the number of nodes stored in the graph."""
        if hasattr(self.graph, "number_of_nodes"):
            return self.graph.number_of_nodes()
        return len(self._nodes)
