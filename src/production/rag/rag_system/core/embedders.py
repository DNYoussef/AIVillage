from collections import Counter
import hashlib
import math


class SimHashEmbedder:
    """Deterministic SimHash embedder for offline use."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    def embed(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0] * self.dimension
        vector = [0] * self.dimension
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            for i in range(self.dimension):
                bit = 1 if h & (1 << i) else -1
                vector[i] += bit
        return [1 if v >= 0 else -1 for v in vector]

    def cosine_similarity(self, vec1: list[int], vec2: list[int]) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class TFIDFHelper:
    """Minimal TF-IDF implementation for small corpora."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    def build(self, documents: list[str]) -> None:
        tokenized = [self._tokenize(doc) for doc in documents]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1
        n_docs = len(tokenized)
        self.idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in df.items()
        }
        self.doc_vectors = []
        self.doc_norms = []
        for tokens in tokenized:
            tf = Counter(tokens)
            vec = {term: tf[term] * self.idf.get(term, 0.0) for term in tf}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

    def query_vector(self, text: str) -> tuple[dict[str, float], float]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = {term: tf[term] * self.idf.get(term, 0.0) for term in tf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return vec, norm

    @staticmethod
    def cosine(
        vec_a: dict[str, float], norm_a: float, vec_b: dict[str, float], norm_b: float
    ) -> float:
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot = sum(vec_a.get(t, 0.0) * w for t, w in vec_b.items())
        return dot / (norm_a * norm_b)
