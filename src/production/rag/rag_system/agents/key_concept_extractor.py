from typing import Any

from ..core.agent_interface import AgentInterface
from ..utils.embedding import BERTEmbeddingModel


# Simple NER implementation as fallback
class NamedEntityRecognizer:
    """Simple named entity recognizer fallback."""

    def recognize(self, text: str) -> list[str]:
        """Simple pattern-based NER."""
        import re

        # Find capitalized words (potential named entities)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        # Filter common words
        common_words = {"The", "This", "That", "These", "Those", "And", "Or", "But"}
        return [entity for entity in entities if entity not in common_words]


class KeyConceptExtractorAgent(AgentInterface):
    """Agent responsible for extracting key concepts from text using advanced NLP techniques."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_model = BERTEmbeddingModel()
        self.named_entity_recognizer = NamedEntityRecognizer()

    def extract_key_concepts(self, text: str) -> dict[str, list[str]]:
        """Extract key concepts from the given text.

        Args:
            text (str): Input text from which to extract key concepts.

        Returns:
            Dict[str, List[str]]: A dictionary containing extracted entities and keywords.
        """
        entities = self.named_entity_recognizer.recognize(text)
        embeddings = self.embedding_model.encode(text)
        keywords = self._extract_keywords_from_embeddings(embeddings)
        return {"entities": entities, "keywords": keywords}

    def _extract_keywords_from_embeddings(self, embeddings) -> list[str]:
        """Derive simple keywords from token embeddings.

        The ``BERTEmbeddingModel.encode`` method returns a tuple of tokens and
        the corresponding embedding tensor.  This helper identifies tokens that
        are most similar to the mean sentence embedding and returns them as
        keywords.  Special tokens are ignored.
        """
        tokens, token_embeddings = embeddings

        if len(tokens) == 0:
            return []

        import torch.nn.functional as F

        sentence_emb = token_embeddings.mean(dim=0)
        similarities = F.cosine_similarity(token_embeddings, sentence_emb.unsqueeze(0), dim=1)
        topk = similarities.topk(min(5, len(tokens))).indices.tolist()

        keywords = []
        for idx in topk:
            token = tokens[idx]
            if token.startswith("[") and token.endswith("]"):
                continue
            if token.startswith("##"):
                token = token[2:]
            keywords.append(token)
            if len(keywords) == 5:
                break
        return keywords

    async def generate(self, prompt: str) -> str:
        """Generate response based on key concept extraction."""
        concepts = self.extract_key_concepts(prompt)
        entities = concepts.get("entities", [])
        keywords = concepts.get("keywords", [])

        response = f"Key concept analysis for: {prompt[:50]}..."
        if entities:
            response += f" Entities identified: {', '.join(entities[:3])}"
        if keywords:
            response += f" Keywords: {', '.join(keywords[:3])}"

        return response

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

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank results based on key concept relevance."""
        query_concepts = self.extract_key_concepts(query)
        query_entities = set(query_concepts.get("entities", []))
        query_keywords = set(query_concepts.get("keywords", []))

        for result in results:
            content = result.get("content", "")
            content_concepts = self.extract_key_concepts(content)
            content_entities = set(content_concepts.get("entities", []))
            content_keywords = set(content_concepts.get("keywords", []))

            # Calculate concept overlap boost
            entity_overlap = len(query_entities & content_entities) / max(len(query_entities), 1)
            keyword_overlap = len(query_keywords & content_keywords) / max(len(query_keywords), 1)
            concept_boost = (entity_overlap + keyword_overlap) * 0.2

            result["score"] = result.get("score", 0.0) + concept_boost

        return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return introspection information."""
        return {
            "type": "KeyConceptExtractorAgent",
            "embedding_model": "BERTEmbeddingModel",
            "ner_model": "SimplePatternNER",
            "capabilities": [
                "key_concept_extraction",
                "named_entity_recognition",
                "keyword_identification",
                "concept_based_reranking",
            ],
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with another agent."""
        concepts = self.extract_key_concepts(message)
        enhanced_message = f"Key concepts from {message}: {concepts}"
        response = await recipient.generate(enhanced_message)
        return f"Sent concept analysis: {concepts}, Received: {response}"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space for concept discovery."""
        concepts = self.extract_key_concepts(query)
        entities = concepts.get("entities", [])
        keywords = concepts.get("keywords", [])

        background = f"Concept analysis: Identified {len(entities)} entities and {len(keywords)} keywords. "
        background += f"Key entities: {', '.join(entities[:3])}. Key terms: {', '.join(keywords[:3])}."

        refined_query = f"Concept-enhanced query: {query} [Entities: {', '.join(entities[:2])}]"
        return background, refined_query
