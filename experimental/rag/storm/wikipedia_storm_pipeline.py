"""Wikipedia-based RAG pipeline using a simplified STORM methodology.

This module provides lightweight classes to process Wikipedia articles,
simulate multi-perspective conversations and prepare educational material
for offline mobile use. The implementation focuses on keeping a small
footprint so it can run on devices with as little as 2GB of RAM.

The code is intentionally minimalist – it does not attempt to download the
full Wikipedia dump at test time. Instead, tests may supply a small sample
``dataset`` to the :class:`WikipediaSTORMPipeline` constructor.

The real project would extend these stubs with fully–fledged components
handling language coverage, persistence and advanced retrieval strategies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:  # Optional dependency – faiss may not be installed on all runners
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from agent_forge.compression.vptq import VPTQCompressor
except Exception:  # pragma: no cover

    class VPTQCompressor:  # type: ignore
        """Fallback compressor used when agent_forge is unavailable."""

        def compress(self, weights: torch.Tensor) -> dict:
            return {"codebook": None, "indices": None}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Perspective:
    """Representation of a perspective discovered during STORM processing."""

    name: str
    focus: str
    questions: list[str]


@dataclass
class Lesson:
    """Container for generated educational lessons."""

    topic: str
    grade_level: int
    content: dict[str, Any]
    audio: bytes | None
    estimated_time: int
    prerequisites: list[str]


@dataclass
class ContentDatabase:
    """Simple in-memory content database used for offline packaging tests."""

    embeddings: np.ndarray
    metadata: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper/placeholder components
# ---------------------------------------------------------------------------


class EducationalContentFilter:
    """Very small heuristic filter for educational relevance."""

    def is_educational(self, article: dict[str, str]) -> bool:
        text = article.get("text", "").lower()
        return any(k in text for k in ["history", "science", "math", "education"])


class STORMProcessor:
    """Placeholder for a complex STORM reasoning engine."""

    async def generate_perspective_question(
        self, topic: str, perspective: Perspective, previous_context: str
    ) -> str:
        return f"What is the {perspective.focus} of {topic}?"

    async def answer_from_wikipedia(
        self, question: str, article: str, perspective: Perspective
    ) -> str:
        # Extremely small heuristic answer generation – real system would use LLMs
        return article.split(".")[0].strip()

    def calculate_confidence(self, answer: str, article: str) -> float:
        return min(len(answer) / max(len(article), 1), 1.0)


class ModelQuantizer:
    """Quantise embedding matrices to a lower precision."""

    def quantize_embeddings(self, embeddings: np.ndarray, bits: int = 8) -> np.ndarray:
        scale = np.max(np.abs(embeddings)) / (2 ** (bits - 1) - 1)
        if scale == 0:
            return embeddings.astype(np.int8)
        quantized = np.round(embeddings / scale).astype(np.int8)
        return quantized


# ---------------------------------------------------------------------------
# STORM RAG pipeline
# ---------------------------------------------------------------------------


class WikipediaSTORMPipeline:
    """Process Wikipedia articles and create STORM-optimised content."""

    def __init__(self, dataset: Iterable[dict[str, str]] | None = None) -> None:
        # ``dataset`` is an iterable of dictionaries with ``title`` and ``text``.
        # In production, ``load_dataset('wikipedia', '20220301.en')`` would be used.
        self.dataset = list(dataset) if dataset is not None else []
        self.educational_filter = EducationalContentFilter()
        self.storm_processor = STORMProcessor()
        self.processed_content: dict[str, dict[str, Any]] = {}

    def is_educational(self, article: dict[str, str]) -> bool:
        return self.educational_filter.is_educational(article)

    def find_related_articles(self, topic: str, k: int = 5) -> list[str]:
        """Return titles of related articles.

        The implementation is extremely small: it simply returns the titles of
        the first ``k`` items from the dataset. In a real system this would
        query a knowledge graph or use embedding similarity.
        """
        return [a["title"] for a in self.dataset[:k]]

    def generate_historical_questions(self, topic: str) -> list[str]:
        return [f"When did {topic} begin?", f"How has {topic} changed over time?"]

    def generate_scientific_questions(self, topic: str) -> list[str]:
        return [
            f"What principles govern {topic}?",
            f"What are key mechanisms of {topic}?",
        ]

    def generate_cultural_questions(self, topic: str) -> list[str]:
        return [
            f"How does {topic} influence culture?",
            f"Are there traditions about {topic}?",
        ]

    def generate_geographical_questions(self, topic: str) -> list[str]:
        return [
            f"Where is {topic} prevalent?",
            f"Which regions are known for {topic}?",
        ]

    def discover_perspectives(self, topic: str) -> list[Perspective]:
        related = self.find_related_articles(topic, k=20)
        perspectives: list[Perspective] = []
        for title in related:
            if "History" in title:
                perspectives.append(
                    Perspective(
                        name="Historical",
                        focus="evolution and timeline",
                        questions=self.generate_historical_questions(topic),
                    )
                )
            if "Science" in title:
                perspectives.append(
                    Perspective(
                        name="Scientific",
                        focus="principles and mechanisms",
                        questions=self.generate_scientific_questions(topic),
                    )
                )
            if "Culture" in title:
                perspectives.append(
                    Perspective(
                        name="Cultural",
                        focus="societal impact and traditions",
                        questions=self.generate_cultural_questions(topic),
                    )
                )
            if "Geography" in title:
                perspectives.append(
                    Perspective(
                        name="Geographical",
                        focus="spatial distribution and regions",
                        questions=self.generate_geographical_questions(topic),
                    )
                )
        return perspectives

    async def simulate_conversations(
        self, topic: str, perspectives: list[Perspective], article_content: str
    ) -> list[list[dict[str, Any]]]:
        conversations: list[list[dict[str, Any]]] = []
        for perspective in perspectives:
            conversation: list[dict[str, Any]] = []
            context = article_content
            for turn in range(5):
                question = await self.storm_processor.generate_perspective_question(
                    topic=topic, perspective=perspective, previous_context=context
                )
                answer = await self.storm_processor.answer_from_wikipedia(
                    question=question, article=article_content, perspective=perspective
                )
                context += f" {answer}"
                conversation.append(
                    {
                        "turn": turn,
                        "perspective": perspective.name,
                        "question": question,
                        "answer": answer,
                        "confidence": self.storm_processor.calculate_confidence(
                            answer, article_content
                        ),
                    }
                )
            conversations.append(conversation)
        return conversations

    def generate_educational_outline(
        self, conversations: list[list[dict[str, Any]]], grade_levels: list[int]
    ) -> dict[str, Any]:
        return {
            "grade_levels": grade_levels,
            "conversation_summaries": [
                c[0]["answer"] if c else "" for c in conversations
            ],
        }

    def store_educational_content(
        self, outline: dict[str, Any], article: dict[str, str]
    ) -> None:
        self.processed_content[article["title"]] = outline

    async def process_wikipedia_for_education(self) -> None:
        educational_articles = [a for a in self.dataset if self.is_educational(a)]
        for article in educational_articles:
            perspectives = self.discover_perspectives(article["title"])
            conversations = await self.simulate_conversations(
                topic=article["title"],
                perspectives=perspectives,
                article_content=article["text"],
            )
            outline = self.generate_educational_outline(
                conversations=conversations, grade_levels=[3, 4, 5, 6, 7, 8]
            )
            self.store_educational_content(outline, article)

    def get_processed_content(self, topic: str) -> dict[str, Any] | None:
        return self.processed_content.get(topic)


# ---------------------------------------------------------------------------
# Offline optimisation utilities
# ---------------------------------------------------------------------------


class OfflineOptimizedRAG:
    """Prepare embeddings and metadata for small offline devices."""

    def __init__(self, max_size_mb: int = 500) -> None:
        self.max_size = max_size_mb * 1024 * 1024
        self.quantizer = ModelQuantizer()

    def compress_metadata(self, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return metadata

    def create_minimal_vocabulary(self, content_db: ContentDatabase) -> list[str]:
        return sorted(
            {w for m in content_db.metadata for w in m.get("title", "").split()}
        )

    def calculate_size(self, index: Any, metadata: Any) -> int:
        index_size = getattr(index, "nbytes", 0)
        meta_size = len(str(metadata).encode("utf-8"))
        return index_size + meta_size

    def prepare_for_mobile(self, content_db: ContentDatabase) -> dict[str, Any]:
        quantized_embeddings = self.quantizer.quantize_embeddings(
            content_db.embeddings, bits=8
        )
        if faiss is not None:  # pragma: no cover - branch depends on env
            index = faiss.IndexHNSWFlat(quantized_embeddings.shape[1], 32)
            index.add(quantized_embeddings.astype(np.float32))
            weights = torch.from_numpy(index.reconstruct_n(0, index.ntotal))
        else:
            index = quantized_embeddings
            weights = torch.from_numpy(quantized_embeddings.astype(np.float32))
        compressed_index = VPTQCompressor().compress(weights)
        metadata = self.compress_metadata(content_db.metadata)
        mobile_package = {
            "index": compressed_index,
            "metadata": metadata,
            "vocabulary": self.create_minimal_vocabulary(content_db),
            "size_bytes": self.calculate_size(quantized_embeddings, metadata),
        }
        assert mobile_package["size_bytes"] < self.max_size
        return mobile_package


# ---------------------------------------------------------------------------
# Educational content generation
# ---------------------------------------------------------------------------


class EducationalContentGenerator:
    """Create grade-appropriate lessons from STORM processed content."""

    def __init__(self, storm_rag: WikipediaSTORMPipeline) -> None:
        self.storm_rag = storm_rag

    def get_reading_level(self, grade: int) -> int:
        return grade

    def simplify_for_grade(
        self, content: dict[str, Any], grade: int, reading_level: int
    ) -> dict[str, Any]:
        return {"summary": content.get("conversation_summaries", []), "grade": grade}

    def get_local_examples(
        self, cultural_context: dict[str, Any], topic: str
    ) -> list[str]:
        return [
            f"Example of {topic} in {cultural_context.get('region', 'local')} context"
        ]

    def add_cultural_context(
        self,
        content: dict[str, Any],
        culture: dict[str, Any],
        local_examples: list[str],
    ) -> dict[str, Any]:
        content = dict(content)
        content["local_examples"] = local_examples
        content["culture"] = culture
        return content

    def create_interactions(
        self, content: dict[str, Any], interaction_types: list[str]
    ) -> dict[str, Any]:
        content = dict(content)
        content["interactions"] = interaction_types
        content["narration"] = " ".join(content.get("summary", []))
        return content

    def generate_audio_narration(self, text: str, language: str, voice: str) -> bytes:
        return text.encode("utf-8")

    def estimate_completion_time(self, content: dict[str, Any]) -> int:
        return 5

    def identify_prerequisites(self, topic: str, grade_level: int) -> list[str]:
        return [f"Basic understanding of {topic}"]

    def generate_lesson(
        self, topic: str, grade_level: int, cultural_context: dict[str, Any]
    ) -> Lesson:
        storm_content = self.storm_rag.get_processed_content(topic) or {}
        simplified = self.simplify_for_grade(
            content=storm_content,
            grade=grade_level,
            reading_level=self.get_reading_level(grade_level),
        )
        localized = self.add_cultural_context(
            content=simplified,
            culture=cultural_context,
            local_examples=self.get_local_examples(cultural_context, topic),
        )
        interactive = self.create_interactions(
            content=localized, interaction_types=["quiz", "exploration", "experiment"]
        )
        audio = self.generate_audio_narration(
            text=interactive["narration"],
            language=cultural_context.get("language", "en"),
            voice="child_friendly",
        )
        lesson = Lesson(
            topic=topic,
            grade_level=grade_level,
            content=interactive,
            audio=audio,
            estimated_time=self.estimate_completion_time(interactive),
            prerequisites=self.identify_prerequisites(topic, grade_level),
        )
        return lesson


__all__ = [
    "ContentDatabase",
    "EducationalContentGenerator",
    "Lesson",
    "OfflineOptimizedRAG",
    "Perspective",
    "WikipediaSTORMPipeline",
]
