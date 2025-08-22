"""
RAG Pipeline - Core orchestration component

This module provides the main RAGPipeline class that coordinates
between vector stores, graph stores, and other RAG components.

Design follows London School TDD principles:
- Mock external services (databases, APIs) but not core business logic
- Focus on behavior and interactions between components
- Keep coupling weak and dependencies explicit
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    document_id: str
    processed_at: float
    vector_stored: bool = False
    graph_relations_added: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResult:
    """Result of query operation."""

    query: str
    results: list[dict[str, Any]]
    processing_time_ms: float
    sources_used: list[str]
    confidence_score: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Abstract interfaces for external dependencies
class VectorStore(ABC):
    """Abstract interface for vector storage systems."""

    @abstractmethod
    def add_document(self, content: str, metadata: dict[str, Any] = None) -> dict[str, Any]:
        """Add document to vector store."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents."""
        pass


class GraphStore(ABC):
    """Abstract interface for graph storage systems."""

    @abstractmethod
    def add_relations(self, content: str, doc_id: str = None) -> dict[str, Any]:
        """Extract and store relations from content."""
        pass

    @abstractmethod
    def find_related(self, query: str, depth: int = 2) -> list[dict[str, Any]]:
        """Find related entities/concepts."""
        pass


# Default implementations for standalone operation
class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing and standalone operation."""

    def __init__(self):
        self.documents: list[dict[str, Any]] = []
        self._doc_counter = 0

    def add_document(self, content: str, metadata: dict[str, Any] = None) -> dict[str, Any]:
        """Add document with simple keyword-based similarity."""
        self._doc_counter += 1
        doc_id = f"doc_{self._doc_counter}"

        doc = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "keywords": set(content.lower().split()),
            "added_at": time.time(),
        }
        self.documents.append(doc)

        return {"status": "added", "doc_id": doc_id}

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Simple keyword-based search."""
        query_keywords = set(query.lower().split())
        results = []

        for doc in self.documents:
            # Simple Jaccard similarity
            intersection = query_keywords.intersection(doc["keywords"])
            union = query_keywords.union(doc["keywords"])

            if intersection and union:
                score = len(intersection) / len(union)
                results.append(
                    {"content": doc["content"], "score": score, "doc_id": doc["id"], "metadata": doc["metadata"]}
                )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class InMemoryGraphStore(GraphStore):
    """Simple in-memory graph store for testing and standalone operation."""

    def __init__(self):
        self.relations: dict[str, list[dict[str, Any]]] = {}
        self.entities: set[str] = set()

    def add_relations(self, content: str, doc_id: str = None) -> dict[str, Any]:
        """Simple entity extraction and relation building."""
        # Extract entities (capitalized words > 2 chars)
        words = content.split()
        entities = [w.strip(".,!?()[]{}") for w in words if w[0].isupper() and len(w.strip(".,!?()[]{}")) > 2]

        relations_added = 0
        for entity in entities:
            self.entities.add(entity)
            if entity not in self.relations:
                self.relations[entity] = []

            # Simple co-occurrence relations
            for other_entity in entities:
                if other_entity != entity:
                    relation = {"target": other_entity, "type": "co_occurs_with", "doc_id": doc_id, "strength": 1.0}
                    self.relations[entity].append(relation)
                    relations_added += 1

        return {"relations_added": relations_added, "entities_found": len(entities)}

    def find_related(self, query: str, depth: int = 2) -> list[dict[str, Any]]:
        """Find related entities through graph traversal."""
        query_entities = [
            w.strip(".,!?()[]{}") for w in query.split() if w[0].isupper() and w.strip(".,!?()[]{}") in self.entities
        ]

        if not query_entities:
            return []

        related = []
        visited = set()

        for entity in query_entities:
            if entity in visited:
                continue

            visited.add(entity)

            # Direct relations
            if entity in self.relations:
                for relation in self.relations[entity][:5]:  # Limit results
                    related.append(
                        {
                            "entity": entity,
                            "relation": relation["type"],
                            "target": relation["target"],
                            "strength": relation["strength"],
                        }
                    )

        return related


class RAGPipeline:
    """
    Main RAG Pipeline orchestrator.

    Coordinates document processing and querying across vector and graph stores.
    Designed to work with dependency injection for external services while
    providing sensible defaults for standalone operation.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        enable_logging: bool = True,
    ):
        """
        Initialize RAG pipeline.

        Args:
            vector_store: Vector storage system (defaults to in-memory)
            graph_store: Graph storage system (defaults to in-memory)
            enable_logging: Whether to enable debug logging
        """
        self.vector_store = vector_store or InMemoryVectorStore()
        self.graph_store = graph_store or InMemoryGraphStore()
        self.enable_logging = enable_logging
        self._initialized = False

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            logger.info(
                "RAGPipeline initialized with %s vector store and %s graph store",
                type(self.vector_store).__name__,
                type(self.graph_store).__name__,
            )

    def initialize(self) -> bool:
        """
        Initialize the pipeline.

        For basic operation, this just sets the initialized flag.
        External implementations might connect to databases, load models, etc.
        """
        try:
            self._initialized = True
            if self.enable_logging:
                logger.info("RAGPipeline initialization completed")
            return True
        except Exception as e:
            logger.error("RAGPipeline initialization failed: %s", e)
            return False

    def process_document(self, content: str, metadata: dict[str, Any] = None) -> ProcessingResult:
        """
        Process a document through both vector and graph stores.

        Args:
            content: Document content to process
            metadata: Optional metadata for the document

        Returns:
            ProcessingResult with status of both storage operations
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Store in vector store
            vector_result = self.vector_store.add_document(content, metadata)
            vector_stored = vector_result.get("status") == "added"
            doc_id = vector_result.get("doc_id", f"unknown_{int(time.time())}")

            # Store relations in graph store
            graph_result = self.graph_store.add_relations(content, doc_id)
            relations_added = graph_result.get("relations_added", 0)

            result = ProcessingResult(
                document_id=doc_id,
                processed_at=time.time(),
                vector_stored=vector_stored,
                graph_relations_added=relations_added,
                metadata={
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "content_length": len(content),
                    "vector_result": vector_result,
                    "graph_result": graph_result,
                },
            )

            if self.enable_logging:
                logger.info("Processed document %s: vector=%s, relations=%d", doc_id, vector_stored, relations_added)

            return result

        except Exception as e:
            logger.error("Document processing failed: %s", e)
            raise

    def query(self, query: str, use_graph_expansion: bool = True, top_k: int = 5) -> QueryResult:
        """
        Query the RAG system for relevant information.

        Args:
            query: Query string
            use_graph_expansion: Whether to use graph relations for query expansion
            top_k: Maximum number of results to return

        Returns:
            QueryResult with retrieved information and metadata
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.time()
        sources_used = []
        all_results = []

        try:
            # Vector search
            vector_results = self.vector_store.search(query, top_k)
            all_results.extend(vector_results)
            sources_used.append("vector")

            # Graph expansion if enabled
            if use_graph_expansion:
                graph_relations = self.graph_store.find_related(query)
                if graph_relations:
                    sources_used.append("graph")

                    # Use graph relations to expand query
                    expanded_terms = [query]  # Original query
                    for relation in graph_relations[:3]:  # Limit expansion
                        expanded_terms.append(relation["target"])

                    expanded_query = " ".join(expanded_terms)
                    if expanded_query != query:
                        expanded_results = self.vector_store.search(expanded_query, top_k)
                        all_results.extend(expanded_results)

            # Remove duplicates and sort by score
            seen_docs = set()
            unique_results = []
            for result in all_results:
                doc_key = result.get("doc_id", result.get("content", ""))[:50]
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    unique_results.append(result)

            # Sort by score if available
            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_results = unique_results[:top_k]

            # Calculate confidence based on top score
            confidence = final_results[0].get("score", 0) if final_results else 0.0

            processing_time = (time.time() - start_time) * 1000

            result = QueryResult(
                query=query,
                results=final_results,
                processing_time_ms=processing_time,
                sources_used=sources_used,
                confidence_score=float(confidence),
                metadata={
                    "total_candidates": len(all_results),
                    "unique_results": len(unique_results),
                    "graph_expansion_used": use_graph_expansion and "graph" in sources_used,
                },
            )

            if self.enable_logging:
                logger.info(
                    "Query '%s': %d results in %.2fms (sources: %s)",
                    query,
                    len(final_results),
                    processing_time,
                    sources_used,
                )

            return result

        except Exception as e:
            logger.error("Query processing failed: %s", e)
            raise
