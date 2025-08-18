"""Local mode support for RAG pipeline."""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def is_local_mode() -> bool:
    """Check if RAG should run in local mode."""
    return os.getenv("RAG_LOCAL_MODE", "0") == "1"


def get_local_corpus() -> list[dict[str, Any]]:
    """Get a small in-memory corpus for smoke testing."""
    return [
        {
            "id": "local_doc_1",
            "content": "AIVillage is a distributed AI system with multiple specialized agents including King, Magi, Navigator, and others.",
            "metadata": {"source": "local", "type": "overview"},
        },
        {
            "id": "local_doc_2",
            "content": "The compression pipeline uses BitNet quantization and achieves significant model size reduction.",
            "metadata": {"source": "local", "type": "technical"},
        },
        {
            "id": "local_doc_3",
            "content": "The P2P network supports both BitChat (Bluetooth) and BetaNet (encrypted internet) transports.",
            "metadata": {"source": "local", "type": "networking"},
        },
        {
            "id": "local_doc_4",
            "content": "The tokenomics system uses VILLAGE credits for compute contribution and governance participation.",
            "metadata": {"source": "local", "type": "economy"},
        },
        {
            "id": "local_doc_5",
            "content": "Agent Forge implements edge-of-chaos training with 55-75% accuracy targeting for optimal learning.",
            "metadata": {"source": "local", "type": "training"},
        },
        {
            "id": "local_doc_6",
            "content": "The RAG system uses hybrid retrieval combining vector search, graph traversal, and semantic caching.",
            "metadata": {"source": "local", "type": "rag"},
        },
        {
            "id": "local_doc_7",
            "content": "Mobile optimization includes battery-aware transport selection and resource-constrained operation modes.",
            "metadata": {"source": "local", "type": "mobile"},
        },
        {
            "id": "local_doc_8",
            "content": "Security features include encrypted thought bubbles, HTX tunneling, and zero-knowledge proofs.",
            "metadata": {"source": "local", "type": "security"},
        },
    ]


class LocalModeRAG:
    """Simplified RAG for local mode operation."""

    def __init__(self):
        from .local_embedder import LocalEmbedder

        self.embedder = LocalEmbedder(dimension=128)  # Smaller for speed
        self.documents = []
        self.embeddings = []

        # Load local corpus
        corpus = get_local_corpus()
        for doc in corpus:
            self.add_document(doc)

        logger.info(f"LocalModeRAG initialized with {len(self.documents)} documents")

    def add_document(self, doc: dict[str, Any]) -> None:
        """Add document to local store."""
        self.documents.append(doc)
        embedding = self.embedder.embed_text(doc["content"])
        self.embeddings.append(embedding)

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """Add multiple documents."""
        for doc in docs:
            self.add_document(doc)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents using local embeddings."""
        if not self.documents:
            return []

        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Calculate similarities
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.embedder.cosine_similarity(query_embedding, doc_embedding)
            scores.append((i, similarity))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k documents
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx].copy()
            doc["score"] = score
            results.append(doc)

        return results

    def query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Alias for retrieve."""
        return self.retrieve(query, top_k)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Alias for retrieve."""
        return self.retrieve(query, top_k)
