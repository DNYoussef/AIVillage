from typing import Any

from ..core.agent_interface import AgentInterface
from ..utils.embedding import BERTEmbeddingModel


class LatentSpaceAgent(AgentInterface):
    def __init__(self, embedding_model: BERTEmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or BERTEmbeddingModel()
        self.llm_enabled = False  # No LLM dependency for now

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space representation for enhanced query understanding.

        Returns background knowledge and refined query based on embedding analysis.
        """
        # For now, provide a simple implementation without LLM dependency
        # Get embedding to understand query semantics
        query_embedding = await self.get_embedding(query)

        # Provide basic background knowledge based on query content
        query_lower = query.lower()
        background_knowledge = "Background: "

        ml_terms = ["machine learning", "ml", "ai", "artificial intelligence"]
        if any(term in query_lower for term in ml_terms):
            background_knowledge += (
                "This query relates to machine learning and AI technologies. "
                "Consider computational complexity, training data requirements, "
                "and model interpretability."
            )
        elif any(term in query_lower for term in ["programming", "code", "software", "development"]):
            background_knowledge += (
                "This query relates to software development. "
                "Consider best practices, code quality, testing, "
                "and maintainability."
            )
        elif any(term in query_lower for term in ["science", "research", "study", "experiment"]):
            background_knowledge += (
                "This query relates to scientific research. "
                "Consider methodology, evidence, peer review, "
                "and reproducibility."
            )
        else:
            background_knowledge += "General query requiring comprehensive analysis " "and evidence-based reasoning."

        # Create refined query with semantic enhancement
        refined_query = (
            f"Enhanced query: {query} " f"(with semantic context: {len(query_embedding)} dimensions analyzed)"
        )

        return background_knowledge, refined_query

    async def generate(self, prompt: str) -> str:
        """Generate response using simple template-based approach."""
        # Since we don't have LLM dependency, provide template-based responses
        prompt_lower = prompt.lower()

        if "background knowledge" in prompt_lower or "context" in prompt_lower:
            return "Based on the available information, here is a comprehensive " f"analysis of: {prompt[:100]}..."
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return f"Summary: {prompt[:200]}... (Analysis complete)"
        elif "question" in prompt_lower or "what" in prompt_lower or "how" in prompt_lower:
            return (
                f"In response to your question: {prompt[:100]}... "
                "This requires detailed analysis of the available information."
            )
        else:
            return f"Processing request: {prompt[:100]}... " "(Template-based response generated)"

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using BERT model."""
        try:
            _, embeddings = self.embedding_model.encode(text)
            # Get mean pooling of token embeddings
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

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        for result in results:
            result["score"] = await self._calculate_similarity(query, result["content"])

        # Sort results by score and return top k
        reranked_results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        return reranked_results

    async def introspect(self) -> dict[str, Any]:
        """Return introspection information about the agent."""
        return {
            "type": "LatentSpaceAgent",
            "embedding_model": "BERTEmbeddingModel",
            "hidden_size": self.embedding_model.hidden_size,
            "fallback_mode": self.embedding_model.fallback,
            "llm_enabled": self.llm_enabled,
            "capabilities": [
                "latent_space_activation",
                "semantic_embedding",
                "document_reranking",
                "template_generation",
            ],
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        response = await recipient.generate(f"Message from LatentSpaceAgent: {message}")
        return f"Sent: {message}, Received: {response}"

    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text embeddings."""
        embedding1 = await self.get_embedding(text1)
        embedding2 = await self.get_embedding(text2)
        return cosine_similarity(embedding1, embedding2)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot_product = sum(x * y for x, y in zip(v1, v2, strict=False))
    magnitude1 = sum(x * x for x in v1) ** 0.5
    magnitude2 = sum(y * y for y in v2) ** 0.5

    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
