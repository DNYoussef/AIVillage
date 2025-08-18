"""Local embedder for offline RAG operation."""

import hashlib

import numpy as np


class LocalEmbedder:
    """Simple hash-based embedder for local/offline mode.

    Uses deterministic hashing for reproducible embeddings without ML models.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocabulary = {}
        self.idf_weights = {}

    def embed_text(self, text: str) -> list[float]:
        """Create embedding using hash-based approach."""
        # Normalize text
        text_lower = text.lower().strip()

        # Create hash-based embedding
        hash_obj = hashlib.sha256(text_lower.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float vector
        embedding = []
        for i in range(0, self.dimension):
            # Use different parts of hash for each dimension
            byte_idx = i % len(hash_bytes)
            value = hash_bytes[byte_idx] / 255.0 - 0.5
            embedding.append(value)

        # Add text features
        features = self._extract_features(text_lower)
        for i, feat in enumerate(features[: min(len(features), self.dimension // 4)]):
            embedding[i] = (embedding[i] + feat) / 2.0

        return embedding

    def _extract_features(self, text: str) -> list[float]:
        """Extract simple text features."""
        words = text.split()
        features = []

        # Word count feature
        features.append(min(len(words) / 100.0, 1.0))

        # Average word length
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            features.append(min(avg_len / 10.0, 1.0))
        else:
            features.append(0.0)

        # Punctuation density
        punct_count = sum(1 for c in text if c in ".,!?;:")
        features.append(min(punct_count / (len(text) + 1), 1.0))

        # Numeric content
        numeric_chars = sum(1 for c in text if c.isdigit())
        features.append(min(numeric_chars / (len(text) + 1), 1.0))

        return features

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between vectors."""
        if not vec1 or not vec2:
            return 0.0

        # Ensure same dimension
        min_len = min(len(vec1), len(vec2))
        v1 = np.array(vec1[:min_len])
        v2 = np.array(vec2[:min_len])

        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed_text(text) for text in texts]
