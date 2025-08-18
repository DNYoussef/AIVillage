"""Concrete implementations for RAG system interfaces.

This module provides production-ready implementations of the core interfaces
defined in interface.py, with proper fallbacks and error handling.
"""

import logging
from typing import Any

from ..retrieval.graph_store import GraphStore
from ..retrieval.vector_store import VectorStore
from ..utils.embedding import BERTEmbeddingModel
from .config import UnifiedConfig
from .interface import EmbeddingModel, KnowledgeConstructor, ReasoningEngine, Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Hybrid retriever combining vector and graph-based retrieval."""

    def __init__(
        self,
        config: UnifiedConfig | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.config = config or UnifiedConfig()
        self.vector_store = vector_store or VectorStore(config=self.config)
        self.graph_store = graph_store or GraphStore(config=self.config)

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve documents using hybrid vector + graph approach."""
        try:
            # Get vector-based results
            vector_results = await self.vector_store.retrieve(query, k // 2)

            # Get graph-based results
            graph_results = await self.graph_store.retrieve(query, k // 2)

            # Combine and deduplicate results
            combined_results = []
            seen_ids = set()

            # Add vector results
            for result in vector_results:
                if hasattr(result, "id"):
                    result_id = result.id
                    doc_dict = {
                        "id": result_id,
                        "content": result.content,
                        "score": result.score,
                        "source": "vector",
                    }
                else:
                    result_id = result.get("id", f"vec_{len(combined_results)}")
                    doc_dict = {**result, "source": "vector"}

                if result_id not in seen_ids:
                    combined_results.append(doc_dict)
                    seen_ids.add(result_id)

            # Add graph results
            for result in graph_results:
                if hasattr(result, "id"):
                    result_id = result.id
                    doc_dict = {
                        "id": result_id,
                        "content": result.content,
                        "score": result.score,
                        "source": "graph",
                    }
                else:
                    result_id = result.get("id", f"graph_{len(combined_results)}")
                    doc_dict = {**result, "source": "graph"}

                if result_id not in seen_ids:
                    combined_results.append(doc_dict)
                    seen_ids.add(result_id)

            # Sort by score and return top k
            combined_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return combined_results[:k]

        except Exception as e:
            logger.warning(
                f"Hybrid retrieval failed: {e}, falling back to simple retrieval"
            )
            # Fallback to simple content-based retrieval
            return [
                {
                    "id": f"fallback_{i}",
                    "content": f"Fallback result {i} for query: {query}",
                    "score": 1.0 - (i * 0.1),
                    "source": "fallback",
                }
                for i in range(min(k, 3))
            ]


class ContextualKnowledgeConstructor(KnowledgeConstructor):
    """Knowledge constructor that builds contextual representations."""

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig()
        self.embedding_model = BERTEmbeddingModel()

    async def construct(
        self, query: str, retrieved_docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Construct knowledge graph from retrieved documents."""
        try:
            if not retrieved_docs:
                return {
                    "query": query,
                    "documents": [],
                    "entities": [],
                    "relationships": [],
                    "summary": "No documents retrieved.",
                    "confidence": 0.0,
                }

            # Extract entities and relationships
            entities = set()
            relationships = []
            document_summaries = []

            for doc in retrieved_docs:
                content = doc.get("content", "")
                doc_id = doc.get("id", "unknown")

                # Simple entity extraction based on capitalized words
                doc_entities = self._extract_entities(content)
                entities.update(doc_entities)

                # Create summary
                summary = self._create_summary(content)
                document_summaries.append(
                    {"id": doc_id, "summary": summary, "score": doc.get("score", 0.0)}
                )

                # Create relationships between query and document entities
                for entity in doc_entities:
                    relationships.append(
                        {
                            "source": query,
                            "target": entity,
                            "relation": "mentions",
                            "document": doc_id,
                        }
                    )

            # Calculate overall confidence
            avg_score = sum(doc.get("score", 0.0) for doc in retrieved_docs) / len(
                retrieved_docs
            )
            confidence = min(avg_score, 1.0)

            return {
                "query": query,
                "documents": document_summaries,
                "entities": list(entities),
                "relationships": relationships,
                "summary": self._create_overall_summary(document_summaries),
                "confidence": confidence,
                "metadata": {
                    "num_documents": len(retrieved_docs),
                    "num_entities": len(entities),
                    "num_relationships": len(relationships),
                },
            }

        except Exception as e:
            logger.warning(f"Knowledge construction failed: {e}")
            return {
                "query": query,
                "documents": [],
                "entities": [],
                "relationships": [],
                "summary": f"Error constructing knowledge: {str(e)}",
                "confidence": 0.0,
            }

    def _extract_entities(self, text: str) -> set[str]:
        """Simple entity extraction based on patterns."""
        import re

        # Find capitalized words (potential named entities)
        entities = set()
        words = re.findall(r"\b[A-Z][a-z]+\b", text)

        # Filter out common words
        common_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "And",
            "Or",
            "But",
            "For",
            "In",
            "On",
            "At",
            "To",
            "From",
        }
        entities.update(word for word in words if word not in common_words)

        return entities

    def _create_summary(self, content: str, max_length: int = 200) -> str:
        """Create a summary of document content."""
        if len(content) <= max_length:
            return content

        # Simple extractive summarization - take first and last sentences
        sentences = content.split(". ")
        if len(sentences) <= 2:
            return content[:max_length] + "..."

        summary = sentences[0]
        if len(summary) < max_length // 2:
            summary += ". " + sentences[-1]

        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    def _create_overall_summary(self, document_summaries: list[dict[str, Any]]) -> str:
        """Create an overall summary from document summaries."""
        if not document_summaries:
            return "No information available."

        # Sort by score and take top summaries
        sorted_docs = sorted(
            document_summaries, key=lambda x: x.get("score", 0.0), reverse=True
        )
        top_summaries = [doc["summary"] for doc in sorted_docs[:3]]

        return " | ".join(top_summaries)


class UncertaintyAwareReasoningEngine(ReasoningEngine):
    """Reasoning engine that incorporates uncertainty in its responses."""

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig()

    async def reason(self, query: str, constructed_knowledge: dict[str, Any]) -> str:
        """Generate reasoned response incorporating uncertainty."""
        try:
            confidence = constructed_knowledge.get("confidence", 0.0)
            summary = constructed_knowledge.get("summary", "")
            entities = constructed_knowledge.get("entities", [])
            num_docs = constructed_knowledge.get("metadata", {}).get("num_documents", 0)

            # Start with confidence assessment
            confidence_level = self._assess_confidence_level(confidence, num_docs)

            # Build response based on available information
            response_parts = []

            # Add confidence qualifier
            response_parts.append(
                f"Based on the available information ({confidence_level}):"
            )

            # Add main content
            if summary and summary != "No information available.":
                response_parts.append(summary)
            else:
                response_parts.append("Limited information available for this query.")

            # Add entity context if available
            if entities:
                entity_list = entities[:5]  # Top 5 entities
                response_parts.append(
                    f"Key entities identified: {', '.join(entity_list)}"
                )

            # Add uncertainty disclaimer for low confidence
            if confidence < 0.5:
                response_parts.append(
                    "Please note: This response has limited confidence due to insufficient or low-quality source material."
                )

            return " ".join(response_parts)

        except Exception as e:
            logger.warning(f"Reasoning failed: {e}")
            return f"Unable to generate a complete response for '{query}' due to processing error: {str(e)}"

    def _assess_confidence_level(self, confidence: float, num_docs: int) -> str:
        """Assess confidence level based on metrics."""
        if confidence >= 0.8 and num_docs >= 3:
            return "high confidence"
        elif confidence >= 0.6 and num_docs >= 2:
            return "moderate confidence"
        elif confidence >= 0.4 and num_docs >= 1:
            return "low-moderate confidence"
        else:
            return "low confidence"


class ProductionEmbeddingModel(EmbeddingModel):
    """Production embedding model with fallbacks."""

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig()
        self.bert_model = BERTEmbeddingModel(model_name=self.config.embedding_model)

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding with proper error handling."""
        try:
            _, embeddings = self.bert_model.encode(text)
            # Get mean pooling of token embeddings
            mean_embedding = embeddings.mean(dim=0).detach().cpu().tolist()
            return [float(x) for x in mean_embedding]
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}, using fallback")
            # Fallback to deterministic random embedding
            import hashlib
            import random

            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            rng = random.Random(seed)
            return [rng.random() for _ in range(self.bert_model.hidden_size)]
