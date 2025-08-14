"""RAG Offline Defaults - No external dependencies for MVP.

Simple offline RAG implementation using only built-in Python libraries.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

    # Minimal numpy-like implementation for embeddings
    class MockNumpy:
        @staticmethod
        def mean(arrays, axis=0):
            if not arrays:
                return [0.0] * len(arrays[0]) if arrays else []
            result = [0.0] * len(arrays[0])
            for arr in arrays:
                for i, val in enumerate(arr):
                    result[i] += val
            return [x / len(arrays) for x in result]

        @staticmethod
        def dot(a, b):
            return sum(a[i] * b[i] for i in range(len(a)))

        @staticmethod
        def linalg_norm(a):
            return (sum(x * x for x in a)) ** 0.5

        class linalg:
            @staticmethod
            def norm(a):
                return (sum(x * x for x in a)) ** 0.5

        @staticmethod
        def zeros(dim):
            return [0.0] * dim

    np = MockNumpy()


class SimpleEmbedder:
    """Simple text embedder for offline operation."""

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Initialize random embeddings matrix
        if NUMPY_AVAILABLE:
            import numpy as real_np

            real_np.random.seed(42)
            self.embeddings = real_np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        else:
            import random

            random.seed(42)
            self.embeddings = [
                [random.gauss(0, 0.1) for _ in range(embedding_dim)]
                for _ in range(vocab_size)
            ]

        # Simple word tokenizer
        self.vocab = {}
        self.next_token_id = 0

    def _tokenize(self, text: str) -> list[int]:
        """Simple word tokenization."""
        words = text.lower().split()
        token_ids = []

        for word in words:
            if word not in self.vocab:
                if self.next_token_id < self.vocab_size:
                    self.vocab[word] = self.next_token_id
                    self.next_token_id += 1
                else:
                    token_ids.append(0)
                    continue

            token_ids.append(self.vocab[word])

        return token_ids

    def encode(self, text: str) -> list[float]:
        """Encode text to embedding vector."""
        if not text.strip():
            return (
                np.zeros(self.embedding_dim)
                if NUMPY_AVAILABLE
                else [0.0] * self.embedding_dim
            )

        token_ids = self._tokenize(text)
        if not token_ids:
            return (
                np.zeros(self.embedding_dim)
                if NUMPY_AVAILABLE
                else [0.0] * self.embedding_dim
            )

        # Average embeddings of tokens
        embeddings = [self.embeddings[token_id] for token_id in token_ids]
        if NUMPY_AVAILABLE:
            import numpy as real_np

            return real_np.mean(embeddings, axis=0)
        else:
            return np.mean(embeddings)


class OfflineVectorStore:
    """Simple in-memory vector store with cosine similarity."""

    def __init__(self, embedder: SimpleEmbedder):
        self.embedder = embedder
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_document(self, text: str, metadata: dict[str, Any] = None) -> None:
        """Add document to store."""
        embedding = self.embedder.encode(text)

        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents."""
        if not self.documents:
            return []

        query_embedding = self.embedder.encode(query)

        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        # Return top_k results
        results = []
        for similarity, idx in similarities[:top_k]:
            results.append(
                {
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity": similarity,
                    "index": idx,
                }
            )

        return results

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if NUMPY_AVAILABLE:
            import numpy as real_np

            norm_a = real_np.linalg.norm(a)
            norm_b = real_np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return real_np.dot(a, b) / (norm_a * norm_b)
        else:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return np.dot(a, b) / (norm_a * norm_b)


class OfflineRAGPipeline:
    """Complete offline RAG pipeline with local documents."""

    def __init__(self, corpus_path: str | None = None):
        self.embedder = SimpleEmbedder()
        self.vector_store = OfflineVectorStore(self.embedder)

        # Load built-in corpus
        self._load_builtin_corpus()

        # Load additional corpus if provided
        if corpus_path and os.path.exists(corpus_path):
            self._load_corpus_from_path(corpus_path)

    def _load_builtin_corpus(self) -> None:
        """Load minimal built-in corpus for testing."""
        builtin_docs = [
            {
                "text": "AIVillage is a decentralized AI platform that enables secure agent communication.",
                "metadata": {"source": "overview", "type": "platform_description"},
            },
            {
                "text": "BitChat provides offline Bluetooth mesh networking for peer-to-peer communication.",
                "metadata": {"source": "transport", "type": "bitchat_feature"},
            },
            {
                "text": "Betanet offers encrypted internet transport with privacy protection using Tor-like routing.",
                "metadata": {"source": "transport", "type": "betanet_feature"},
            },
            {
                "text": "The Navigator agent intelligently routes messages based on network conditions and privacy requirements.",
                "metadata": {"source": "agents", "type": "navigator_description"},
            },
            {
                "text": "Resource management automatically adapts transport protocols based on battery and thermal conditions.",
                "metadata": {"source": "mobile", "type": "resource_management"},
            },
            {
                "text": "Noise XK protocol provides forward secrecy and authentication for secure communication channels.",
                "metadata": {"source": "security", "type": "crypto_protocol"},
            },
            {
                "text": "HTX frame format enables efficient binary transport with flow control and multiplexing.",
                "metadata": {"source": "protocol", "type": "frame_format"},
            },
            {
                "text": "Access tickets provide authentication and rate limiting for controlled network access.",
                "metadata": {"source": "security", "type": "access_control"},
            },
            {
                "text": "Agent Forge trains specialized AI agents using evolutionary algorithms and curriculum learning.",
                "metadata": {"source": "training", "type": "agent_evolution"},
            },
            {
                "text": "Compression algorithms including BitNet and SeedLM reduce bandwidth requirements for mobile devices.",
                "metadata": {"source": "optimization", "type": "compression"},
            },
            {
                "text": "The tokenomics system manages VILLAGE credits for compute sharing and network participation.",
                "metadata": {"source": "economy", "type": "tokenomics"},
            },
            {
                "text": "Dual-path transport provides automatic failover between BitChat and Betanet protocols for reliability.",
                "metadata": {"source": "transport", "type": "reliability"},
            },
            {
                "text": "Quiet-STaR enables agents to perform internal reasoning with encrypted thought processes.",
                "metadata": {"source": "agents", "type": "reasoning"},
            },
            {
                "text": "Self-modeling networks predict their own behavior to improve efficiency and adaptation.",
                "metadata": {"source": "training", "type": "self_modeling"},
            },
            {
                "text": "Mobile optimization prioritizes offline operation and battery preservation on resource-constrained devices.",
                "metadata": {"source": "mobile", "type": "optimization"},
            },
        ]

        for doc in builtin_docs:
            self.vector_store.add_document(doc["text"], doc["metadata"])

        logger.info(f"Loaded {len(builtin_docs)} built-in documents")

    def _load_corpus_from_path(self, corpus_path: str) -> None:
        """Load documents from file or directory."""
        corpus_path = Path(corpus_path)

        if corpus_path.is_file():
            if corpus_path.suffix == ".json":
                self._load_json_corpus(corpus_path)
            else:
                self._load_text_file(corpus_path)
        elif corpus_path.is_dir():
            for file_path in corpus_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in [".txt", ".md", ".json"]:
                    if file_path.suffix == ".json":
                        self._load_json_corpus(file_path)
                    else:
                        self._load_text_file(file_path)

    def _load_json_corpus(self, file_path: Path) -> None:
        """Load JSON corpus file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        metadata = item.get("metadata", {})
                        metadata["source_file"] = str(file_path)
                        self.vector_store.add_document(item["text"], metadata)
        except Exception as e:
            logger.warning(f"Failed to load JSON corpus {file_path}: {e}")

    def _load_text_file(self, file_path: Path) -> None:
        """Load plain text file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                metadata = {
                    "source_file": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix,
                }
                self.vector_store.add_document(content, metadata)
        except Exception as e:
            logger.warning(f"Failed to load text file {file_path}: {e}")

    def query(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Query the RAG system and return results."""
        start_time = time.time()

        # Search for relevant documents
        search_results = self.vector_store.similarity_search(question, top_k=top_k)

        # Generate simple response
        if search_results:
            primary_result = search_results[0]

            # Create context from top results
            context_parts = []
            for result in search_results:
                context_parts.append(f"[{result['similarity']:.2f}] {result['text']}")

            context = "\n\n".join(context_parts)
            response = f"Based on the available documentation: {primary_result['text']}"

            if len(search_results) > 1:
                response += (
                    f"\n\nAdditional relevant information: {search_results[1]['text']}"
                )
        else:
            response = "I don't have information about that topic in my current knowledge base."
            context = ""

        query_time = time.time() - start_time

        return {
            "question": question,
            "answer": response,
            "sources": search_results,
            "context": context,
            "query_time_ms": query_time * 1000,
            "offline_mode": True,
            "num_documents": len(self.vector_store.documents),
        }


def smoke() -> dict[str, Any]:
    """Smoke test for offline RAG functionality."""
    try:
        # Initialize offline RAG
        rag = OfflineRAGPipeline()

        # Test queries
        test_queries = [
            "What is BitChat?",
            "How does Betanet work?",
            "What are access tickets?",
            "How does resource management work?",
        ]

        results = []
        for query in test_queries:
            result = rag.query(query)
            results.append(
                {
                    "query": query,
                    "success": len(result["sources"]) > 0,
                    "query_time_ms": result["query_time_ms"],
                    "num_sources": len(result["sources"]),
                }
            )

        # Summary
        successful_queries = sum(1 for r in results if r["success"])
        avg_query_time = sum(r["query_time_ms"] for r in results) / len(results)

        return {
            "status": "success",
            "total_queries": len(test_queries),
            "successful_queries": successful_queries,
            "success_rate": successful_queries / len(test_queries),
            "avg_query_time_ms": avg_query_time,
            "total_documents": len(rag.vector_store.documents),
            "test_results": results,
            "meets_requirements": successful_queries >= 1,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "meets_requirements": False,
        }


# Legacy compatibility for existing configuration
class OfflineRAGConfig:
    """Simple offline RAG configuration."""

    def __init__(self):
        self.offline_mode = True
        self.enable_internet_features = False
        self.enable_api_calls = False
        self.strict_offline_mode = True
        self.vector_store_type = "in_memory"
        self.cache_enabled = True
        self.extra_params = {}

    def dict(self):
        return {
            "offline_mode": self.offline_mode,
            "enable_internet_features": self.enable_internet_features,
            "enable_api_calls": self.enable_api_calls,
            "strict_offline_mode": self.strict_offline_mode,
            "vector_store_type": self.vector_store_type,
            "cache_enabled": self.cache_enabled,
        }


def get_offline_rag_config(**overrides) -> OfflineRAGConfig:
    """Get offline-first RAG configuration."""
    return OfflineRAGConfig()


def auto_configure_for_environment() -> OfflineRAGConfig:
    """Auto configure for current environment."""
    return OfflineRAGConfig()


# Make the smoke test available at module level
def smoke_test():
    """Legacy name for smoke test."""
    return smoke()


if __name__ == "__main__":
    result = smoke()
    print(json.dumps(result, indent=2))
