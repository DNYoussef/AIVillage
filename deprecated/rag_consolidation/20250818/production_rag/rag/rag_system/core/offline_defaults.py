"""RAG Offline Defaults - Deterministic and dependency-light."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .embedders import SimHashEmbedder, TFIDFHelper

logger = logging.getLogger(__name__)


class OfflineVectorStore:
    """In-memory store combining SimHash and optional TF-IDF."""

    def __init__(self, embedder: SimHashEmbedder, tfidf_threshold: int = 50) -> None:
        self.embedder = embedder
        self.documents: list[str] = []
        self.embeddings: list[list[int]] = []
        self.metadata: list[dict[str, Any]] = []
        self.tfidf_threshold = tfidf_threshold
        self.tfidf: TFIDFHelper | None = None

    def add_document(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.documents.append(text)
        self.embeddings.append(self.embedder.embed(text))
        self.metadata.append(metadata or {})
        if len(self.documents) <= self.tfidf_threshold:
            self._rebuild_tfidf()

    def _rebuild_tfidf(self) -> None:
        self.tfidf = TFIDFHelper()
        self.tfidf.build(self.documents)

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.documents:
            return []
        query_emb = self.embedder.embed(query)
        simhash_scores = [
            self.embedder.cosine_similarity(query_emb, emb) for emb in self.embeddings
        ]
        combined = simhash_scores
        if self.tfidf is not None:
            q_vec, q_norm = self.tfidf.query_vector(query)
            tfidf_scores = [
                self.tfidf.cosine(q_vec, q_norm, doc_vec, doc_norm)
                for doc_vec, doc_norm in zip(
                    self.tfidf.doc_vectors, self.tfidf.doc_norms, strict=False
                )
            ]
            combined = [
                0.7 * sh + 0.3 * tf
                for sh, tf in zip(simhash_scores, tfidf_scores, strict=False)
            ]
        ranked = sorted(enumerate(combined), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            results.append(
                {
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity": score,
                    "index": idx,
                }
            )
        return results


class OfflineRAGPipeline:
    """Complete offline RAG pipeline with local documents."""

    def __init__(
        self, corpus_path: str | None = None, load_builtin: bool = True
    ) -> None:
        self.embedder = SimHashEmbedder()
        self.vector_store = OfflineVectorStore(self.embedder)
        if load_builtin:
            self._load_builtin_corpus()
        if os.environ.get("OFFLINE_RAG_SEED") == "1":
            self._load_seed_corpus()
        if corpus_path and os.path.exists(corpus_path):
            self._load_corpus_from_path(corpus_path)

    def _load_seed_corpus(self) -> None:
        docs_dir = Path("docs")
        count = 0
        for md_file in docs_dir.rglob("*.md"):
            try:
                with open(md_file, encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    self.vector_store.add_document(
                        content, {"source_file": str(md_file)}
                    )
                    count += 1
                if count >= 20:
                    break
            except Exception as exc:  # pragma: no cover - log only
                logger.warning("Failed to load seed doc %s: %s", md_file, exc)

    def _load_builtin_corpus(self) -> None:
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
                "text": "The Navigator agent intelligently routes messages based on network conditions and privacy requirements.",  # noqa: E501
                "metadata": {"source": "agents", "type": "navigator_description"},
            },
            {
                "text": "Resource management automatically adapts transport protocols based on battery and thermal conditions.",  # noqa: E501
                "metadata": {"source": "mobile", "type": "resource_management"},
            },
            {
                "text": "Noise XK protocol provides forward secrecy and authentication for secure communication channels.",  # noqa: E501
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
                "text": "Agent Forge trains specialized AI agents using evolutionary algorithms and curriculum learning.",  # noqa: E501
                "metadata": {"source": "training", "type": "agent_evolution"},
            },
            {
                "text": "Compression algorithms including BitNet and SeedLM reduce bandwidth requirements for mobile devices.",  # noqa: E501
                "metadata": {"source": "optimization", "type": "compression"},
            },
            {
                "text": "The tokenomics system manages VILLAGE credits for compute sharing and network participation.",
                "metadata": {"source": "economy", "type": "tokenomics"},
            },
            {
                "text": "Dual-path transport provides automatic failover between BitChat and Betanet protocols for reliability.",  # noqa: E501
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
                "text": "Mobile optimization prioritizes offline operation and battery preservation on resource-constrained devices.",  # noqa: E501
                "metadata": {"source": "mobile", "type": "optimization"},
            },
        ]
        for doc in builtin_docs:
            self.vector_store.add_document(doc["text"], doc["metadata"])
        logger.info("Loaded %d built-in documents", len(builtin_docs))

    def _load_corpus_from_path(self, corpus_path: str) -> None:
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
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        metadata = item.get("metadata", {})
                        metadata["source_file"] = str(file_path)
                        self.vector_store.add_document(item["text"], metadata)
        except Exception as exc:  # pragma: no cover - log only
            logger.warning("Failed to load JSON corpus %s: %s", file_path, exc)

    def _load_text_file(self, file_path: Path) -> None:
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
        except Exception as exc:  # pragma: no cover - log only
            logger.warning("Failed to load text file %s: %s", file_path, exc)

    def query(
        self, question: str, top_k: int = 3, max_tokens: int = 80
    ) -> dict[str, Any]:
        start_time = time.time()
        search_results = self.vector_store.similarity_search(question, top_k=top_k)
        if search_results:
            snippets = []
            remaining = max_tokens
            for i, result in enumerate(search_results, 1):
                words = result["text"].split()
                snippet_words = words[:remaining]
                remaining -= len(snippet_words)
                snippet = " ".join(snippet_words)
                snippets.append(f"[{i}] {snippet}")
                if remaining <= 0:
                    break
            response = f"{search_results[0]['text']}\n\nSources:\n" + "\n".join(
                snippets
            )
            context = "\n\n".join(snippets)
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
    try:
        rag = OfflineRAGPipeline()
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
        successful_queries = sum(1 for r in results if r["success"])
        avg_query_time = sum(r["query_time_ms"] for r in results) / len(results)
        return {
            "status": "success",
            "total_queries": len(test_queries),
            "successful_queries": successful_queries,
            "success_rate": successful_queries / len(test_queries),
            "average_query_time_ms": avg_query_time,
            "results": results,
        }
    except Exception as exc:  # pragma: no cover - log only
        return {"status": "error", "error": str(exc)}
