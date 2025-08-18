"""Basic document processing utilities."""

from __future__ import annotations


class DocumentProcessor:
    """Splits text into overlapping chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Store chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += step
        return chunks
