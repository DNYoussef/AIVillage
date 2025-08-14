"""Intelligent Chunking using Sliding Window Similarity Analysis.

Implements idea-aware document chunking that:
- Uses 3-sentence sliding windows with embedding similarity
- Detects topic boundaries when similarity drops significantly
- Creates chunks at natural idea boundaries with context preservation
- Handles edge cases like short paragraphs, lists, and code blocks
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional spaCy import - disabled due to pydantic compatibility issues
SPACY_AVAILABLE = False
spacy = None

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type for threshold calibration."""

    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    ACADEMIC = "academic"
    CONVERSATIONAL = "conversational"


class ContentType(Enum):
    """Content type for special handling."""

    TEXT = "text"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    FORMULA = "formula"


@dataclass
class SlidingWindow:
    """Sliding window with sentences and metadata."""

    start_idx: int
    end_idx: int
    sentences: list[str]
    embedding: np.ndarray
    content_type: ContentType = ContentType.TEXT

    @property
    def text(self) -> str:
        """Get concatenated text of window."""
        return " ".join(self.sentences)


@dataclass
class IdeaBoundary:
    """Detected idea boundary with metadata."""

    sentence_idx: int
    similarity_drop: float
    confidence: float
    boundary_type: str = "topic_shift"


@dataclass
class IntelligentChunk:
    """Semantically coherent chunk with context."""

    id: str
    start_sentence_idx: int
    end_sentence_idx: int
    sentences: list[str]
    content_type: ContentType
    summary: str = ""
    entities: list[str] = None
    topic_coherence: float = 0.0
    context_overlap: int = 1

    @property
    def text(self) -> str:
        """Get full text of chunk."""
        return " ".join(self.sentences)

    @property
    def word_count(self) -> int:
        """Get word count of chunk."""
        return len(self.text.split())


class IntelligentChunker:
    """Intelligent document chunker using sliding window similarity analysis."""

    def __init__(
        self,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        window_size: int = 3,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 20,
        context_overlap: int = 1,
        similarity_threshold: dict[DocumentType, float] | None = None,
        load_spacy: bool = True,
    ) -> None:
        """Initialize intelligent chunker.

        Args:
            embedding_model: SentenceTransformer model for embeddings
            window_size: Size of sliding window (sentences)
            min_chunk_sentences: Minimum sentences per chunk
            max_chunk_sentences: Maximum sentences per chunk
            context_overlap: Sentences to overlap between chunks
            similarity_threshold: Threshold by document type
            load_spacy: Whether to load spaCy for entity extraction
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.window_size = window_size
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.context_overlap = context_overlap

        # Default similarity thresholds based on research
        self.similarity_threshold = similarity_threshold or {
            DocumentType.TECHNICAL: 0.25,
            DocumentType.NARRATIVE: 0.35,
            DocumentType.ACADEMIC: 0.30,
            DocumentType.CONVERSATIONAL: 0.40,
        }

        # Load spaCy for entity extraction if available
        self.nlp = None
        if load_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for entity extraction")
            except (OSError, Exception):
                logger.warning("spaCy model not available, entity extraction disabled")
        elif load_spacy and not SPACY_AVAILABLE:
            logger.warning("spaCy not installed, entity extraction disabled")

        # Patterns for content type detection
        self.code_patterns = [
            r"```[\s\S]*?```",  # Markdown code blocks
            r"`[^`]+`",  # Inline code
            r"^\s*(?:def|class|import|from|if|for|while|try)\s",  # Python keywords
            r"^\s*(?:function|var|let|const|class)\s",  # JavaScript keywords
            r"^\s*(?:#include|int|void|char|double)\s",  # C/C++ keywords
        ]

        self.list_patterns = [
            r"^\s*[-*+]\s",  # Markdown lists
            r"^\s*\d+\.\s",  # Numbered lists
            r"^\s*[a-zA-Z]\)\s",  # Lettered lists
            r"^\s*[ivxlc]+\.\s",  # Roman numeral lists
        ]

        self.table_patterns = [
            r"\|.*\|",  # Markdown tables
            r"^\s*\+[-=]+\+",  # ASCII tables
        ]

        self.formula_patterns = [
            r"\$.*?\$",  # LaTeX math
            r"\\[a-zA-Z]+\{.*?\}",  # LaTeX commands
            r"^\s*[A-Za-z]\s*=\s*[^=]+$",  # Simple equations
        ]

    def extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text with improved sentence boundary detection.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Handle special cases first
        sentences = []

        # Split by common sentence endings
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip if too short (likely abbreviation)
            if len(sentence.split()) < 2:
                continue

            # Clean up sentence
            sentence = re.sub(r"\s+", " ", sentence)  # Normalize whitespace
            sentences.append(sentence)

        return sentences

    def detect_content_type(self, text: str) -> ContentType:
        """Detect content type for special handling.

        Args:
            text: Text to analyze

        Returns:
            Content type
        """
        text.lower().strip()

        # Check for code
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return ContentType.CODE

        # Check for lists
        for pattern in self.list_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return ContentType.LIST

        # Check for tables
        for pattern in self.table_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return ContentType.TABLE

        # Check for formulas
        for pattern in self.formula_patterns:
            if re.search(pattern, text):
                return ContentType.FORMULA

        return ContentType.TEXT

    def infer_document_type(self, text: str) -> DocumentType:
        """Infer document type for threshold calibration.

        Args:
            text: Document text

        Returns:
            Document type
        """
        text_lower = text.lower()

        # Technical indicators
        technical_terms = [
            "algorithm",
            "implementation",
            "function",
            "method",
            "api",
            "database",
            "server",
            "client",
            "protocol",
            "framework",
            "library",
            "module",
            "class",
            "object",
            "variable",
        ]

        # Academic indicators
        academic_terms = [
            "research",
            "study",
            "analysis",
            "hypothesis",
            "methodology",
            "literature",
            "findings",
            "conclusion",
            "references",
            "abstract",
            "experiment",
            "data",
            "results",
            "discussion",
            "significant",
        ]

        # Narrative indicators
        narrative_terms = [
            "story",
            "character",
            "plot",
            "narrative",
            "told",
            "said",
            "happened",
            "once",
            "suddenly",
            "then",
            "finally",
            "meanwhile",
        ]

        # Count occurrences
        technical_count = sum(1 for term in technical_terms if term in text_lower)
        academic_count = sum(1 for term in academic_terms if term in text_lower)
        narrative_count = sum(1 for term in narrative_terms if term in text_lower)

        # Determine type based on highest count
        max_count = max(technical_count, academic_count, narrative_count)

        if max_count == 0 or max_count < 3:
            return DocumentType.CONVERSATIONAL
        if technical_count == max_count:
            return DocumentType.TECHNICAL
        if academic_count == max_count:
            return DocumentType.ACADEMIC
        return DocumentType.NARRATIVE

    def create_sliding_windows(self, sentences: list[str]) -> list[SlidingWindow]:
        """Create sliding windows from sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of sliding windows
        """
        if len(sentences) < self.window_size:
            # Too few sentences, create single window
            content_type = self.detect_content_type(" ".join(sentences))
            embedding = self.embedding_model.encode(" ".join(sentences))

            return [
                SlidingWindow(
                    start_idx=0,
                    end_idx=len(sentences) - 1,
                    sentences=sentences,
                    embedding=embedding,
                    content_type=content_type,
                )
            ]

        windows = []

        for i in range(len(sentences) - self.window_size + 1):
            window_sentences = sentences[i : i + self.window_size]
            window_text = " ".join(window_sentences)

            # Detect content type
            content_type = self.detect_content_type(window_text)

            # Generate embedding
            embedding = self.embedding_model.encode(window_text)

            window = SlidingWindow(
                start_idx=i,
                end_idx=i + self.window_size - 1,
                sentences=window_sentences,
                embedding=embedding,
                content_type=content_type,
            )

            windows.append(window)

        return windows

    def calculate_similarity_scores(self, windows: list[SlidingWindow]) -> list[float]:
        """Calculate similarity scores between consecutive windows.

        Args:
            windows: List of sliding windows

        Returns:
            List of similarity scores
        """
        if len(windows) < 2:
            return []

        similarities = []

        for i in range(len(windows) - 1):
            # Calculate cosine similarity between consecutive windows
            sim = cosine_similarity(
                windows[i].embedding.reshape(1, -1),
                windows[i + 1].embedding.reshape(1, -1),
            )[0, 0]
            similarities.append(sim)

        return similarities

    def detect_idea_boundaries(
        self,
        similarities: list[float],
        doc_type: DocumentType,
        consecutive_windows: int = 3,
    ) -> list[IdeaBoundary]:
        """Detect idea boundaries using similarity analysis.

        Args:
            similarities: Similarity scores between windows
            doc_type: Document type for threshold selection
            consecutive_windows: Required consecutive low similarities

        Returns:
            List of detected boundaries
        """
        if len(similarities) < consecutive_windows:
            return []

        threshold = self.similarity_threshold[doc_type]
        boundaries = []

        # Find significant drops in similarity
        for i in range(len(similarities) - consecutive_windows + 1):
            # Check if we have consecutive low similarities
            consecutive_low = all(similarities[i + j] < (1.0 - threshold) for j in range(consecutive_windows))

            if consecutive_low:
                # Calculate average similarity drop
                avg_similarity = np.mean(similarities[i : i + consecutive_windows])
                similarity_drop = 1.0 - avg_similarity

                # Calculate confidence based on consistency
                std_similarity = np.std(similarities[i : i + consecutive_windows])
                confidence = min(1.0, similarity_drop / threshold * (1.0 - std_similarity))

                boundary = IdeaBoundary(
                    sentence_idx=i + self.window_size,  # Boundary after window
                    similarity_drop=similarity_drop,
                    confidence=confidence,
                    boundary_type="topic_shift",
                )

                boundaries.append(boundary)

        # Remove overlapping boundaries (keep highest confidence)
        filtered_boundaries = []
        for boundary in boundaries:
            # Check if too close to existing boundary
            too_close = any(
                abs(boundary.sentence_idx - existing.sentence_idx) < consecutive_windows
                for existing in filtered_boundaries
            )

            if not too_close:
                filtered_boundaries.append(boundary)
            elif boundary.confidence > max(
                b.confidence
                for b in filtered_boundaries
                if abs(b.sentence_idx - boundary.sentence_idx) < consecutive_windows
            ):
                # Replace lower confidence boundary
                filtered_boundaries = [
                    b for b in filtered_boundaries if abs(b.sentence_idx - boundary.sentence_idx) >= consecutive_windows
                ]
                filtered_boundaries.append(boundary)

        # Sort by sentence index
        filtered_boundaries.sort(key=lambda x: x.sentence_idx)

        return filtered_boundaries

    def handle_edge_cases(self, sentences: list[str], boundaries: list[int]) -> list[int]:
        """Handle edge cases in boundary detection.

        Args:
            sentences: Original sentences
            boundaries: Detected boundary positions

        Returns:
            Adjusted boundary positions
        """
        adjusted_boundaries = []

        for boundary in boundaries:
            # Don't split code blocks
            if boundary < len(sentences):
                boundary_text = sentences[boundary]
                if self.detect_content_type(boundary_text) in [
                    ContentType.CODE,
                    ContentType.FORMULA,
                ]:
                    continue

            # Don't split lists
            if boundary > 0 and boundary < len(sentences):
                prev_text = sentences[boundary - 1]
                curr_text = sentences[boundary]

                if (
                    self.detect_content_type(prev_text) == ContentType.LIST
                    and self.detect_content_type(curr_text) == ContentType.LIST
                ):
                    continue

            adjusted_boundaries.append(boundary)

        return adjusted_boundaries

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of entity labels
        """
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text[:1000])  # Limit text length for efficiency
            entities = [ent.label_ for ent in doc.ents]
            return list(set(entities))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def calculate_topic_coherence(self, sentences: list[str]) -> float:
        """Calculate topic coherence score for a chunk.

        Args:
            sentences: Chunk sentences

        Returns:
            Coherence score (0-1)
        """
        if len(sentences) < 2:
            return 1.0

        # Calculate pairwise similarities within chunk
        embeddings = self.embedding_model.encode(sentences)

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0]
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 1.0

    def generate_chunk_summary(self, text: str, max_length: int = 100) -> str:
        """Generate extractive summary for chunk.

        Args:
            text: Chunk text
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        sentences = self.extract_sentences(text)

        if len(sentences) <= 2:
            return text[:max_length]

        # Simple extractive summarization: take first and most central sentence
        if len(sentences) >= 3:
            # Use first sentence and middle sentence
            summary_sentences = [sentences[0], sentences[len(sentences) // 2]]
        else:
            summary_sentences = sentences[:1]

        summary = " ".join(summary_sentences)

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def chunk_document(
        self,
        text: str,
        document_id: str = "doc",
        doc_type: DocumentType | None = None,
    ) -> list[IntelligentChunk]:
        """Chunk document using intelligent sliding window analysis.

        Args:
            text: Document text
            document_id: Document identifier
            doc_type: Document type (inferred if None)

        Returns:
            List of intelligent chunks
        """
        logger.info(f"Starting intelligent chunking for document {document_id}")

        # Extract sentences
        sentences = self.extract_sentences(text)

        if len(sentences) == 0:
            logger.warning("No sentences found in document")
            return []

        logger.info(f"Extracted {len(sentences)} sentences")

        # Infer document type if not provided
        if doc_type is None:
            doc_type = self.infer_document_type(text)
            logger.info(f"Inferred document type: {doc_type.value}")

        # Handle very short documents
        if len(sentences) <= self.min_chunk_sentences:
            chunk = IntelligentChunk(
                id=f"{document_id}_chunk_0",
                start_sentence_idx=0,
                end_sentence_idx=len(sentences) - 1,
                sentences=sentences,
                content_type=self.detect_content_type(text),
                summary=self.generate_chunk_summary(text),
                entities=self.extract_entities(text),
                topic_coherence=1.0,
                context_overlap=0,
            )
            return [chunk]

        # Create sliding windows
        windows = self.create_sliding_windows(sentences)
        logger.info(f"Created {len(windows)} sliding windows")

        # Calculate similarity scores
        similarities = self.calculate_similarity_scores(windows)
        logger.info(f"Calculated {len(similarities)} similarity scores")

        # Detect idea boundaries
        boundaries = self.detect_idea_boundaries(similarities, doc_type)
        logger.info(f"Detected {len(boundaries)} potential boundaries")

        # Extract boundary positions and handle edge cases
        boundary_positions = [b.sentence_idx for b in boundaries]
        boundary_positions = self.handle_edge_cases(sentences, boundary_positions)
        logger.info(f"Final boundaries after edge case handling: {len(boundary_positions)}")

        # Create chunks based on boundaries
        chunks = []
        start_idx = 0

        for i, boundary_pos in enumerate([*boundary_positions, len(sentences)]):
            end_idx = min(boundary_pos, len(sentences))

            # Ensure minimum chunk size
            if end_idx - start_idx < self.min_chunk_sentences:
                if i == len(boundary_positions):  # Last chunk
                    if chunks:  # Merge with previous chunk
                        chunks[-1].end_sentence_idx = end_idx - 1
                        chunks[-1].sentences.extend(sentences[chunks[-1].end_sentence_idx + 1 : end_idx])
                        continue
                else:
                    continue

            # Ensure maximum chunk size
            if end_idx - start_idx > self.max_chunk_sentences:
                # Split large chunk into smaller pieces
                sub_start = start_idx
                while sub_start < end_idx:
                    sub_end = min(sub_start + self.max_chunk_sentences, end_idx)

                    chunk_sentences = sentences[sub_start:sub_end]
                    chunk_text = " ".join(chunk_sentences)

                    chunk = IntelligentChunk(
                        id=f"{document_id}_chunk_{len(chunks)}",
                        start_sentence_idx=sub_start,
                        end_sentence_idx=sub_end - 1,
                        sentences=chunk_sentences,
                        content_type=self.detect_content_type(chunk_text),
                        summary=self.generate_chunk_summary(chunk_text),
                        entities=self.extract_entities(chunk_text),
                        topic_coherence=self.calculate_topic_coherence(chunk_sentences),
                        context_overlap=self.context_overlap if len(chunks) > 0 else 0,
                    )

                    chunks.append(chunk)
                    sub_start = sub_end - self.context_overlap

            else:
                # Add context overlap from previous chunk
                actual_start = max(0, start_idx - self.context_overlap) if len(chunks) > 0 else start_idx
                chunk_sentences = sentences[actual_start:end_idx]

                chunk = IntelligentChunk(
                    id=f"{document_id}_chunk_{len(chunks)}",
                    start_sentence_idx=start_idx,
                    end_sentence_idx=end_idx - 1,
                    sentences=chunk_sentences,
                    content_type=self.detect_content_type(" ".join(chunk_sentences)),
                    summary=self.generate_chunk_summary(" ".join(chunk_sentences)),
                    entities=self.extract_entities(" ".join(chunk_sentences)),
                    topic_coherence=self.calculate_topic_coherence(chunk_sentences),
                    context_overlap=self.context_overlap if len(chunks) > 0 else 0,
                )

                chunks.append(chunk)

            start_idx = end_idx

        logger.info(f"Created {len(chunks)} intelligent chunks")

        # Log chunk statistics
        avg_coherence = np.mean([c.topic_coherence for c in chunks]) if chunks else 0
        avg_length = np.mean([c.word_count for c in chunks]) if chunks else 0

        logger.info(f"Average topic coherence: {avg_coherence:.3f}")
        logger.info(f"Average chunk length: {avg_length:.1f} words")

        return chunks

    def get_chunking_stats(self, chunks: list[IntelligentChunk]) -> dict[str, Any]:
        """Get statistics about chunking results.

        Args:
            chunks: List of chunks

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {}

        return {
            "total_chunks": len(chunks),
            "avg_sentences_per_chunk": np.mean([len(c.sentences) for c in chunks]),
            "avg_words_per_chunk": np.mean([c.word_count for c in chunks]),
            "avg_topic_coherence": np.mean([c.topic_coherence for c in chunks]),
            "content_types": {ct.value: sum(1 for c in chunks if c.content_type == ct) for ct in ContentType},
            "chunks_with_entities": sum(1 for c in chunks if c.entities),
            "total_entities": sum(len(c.entities or []) for c in chunks),
        }


# Test function to demonstrate intelligent chunking
async def test_intelligent_chunking():
    """Test intelligent chunking with sample documents."""
    print("Testing Intelligent Chunking System")
    print("=" * 50)

    # Initialize chunker
    chunker = IntelligentChunker(window_size=3, min_chunk_sentences=2, max_chunk_sentences=15, context_overlap=1)

    # Test document with clear topic boundaries
    test_document = """
    Artificial intelligence is revolutionizing many industries today. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. Deep learning has been particularly successful in computer vision and natural language processing tasks.

    However, AI development faces significant challenges. Data quality and bias remain major concerns that can lead to unfair or inaccurate results. Privacy issues arise when AI systems require access to personal information for training and operation.

    Climate change represents one of the most pressing global challenges of our time. Rising global temperatures are causing sea levels to rise and weather patterns to become more extreme. The primary cause is the emission of greenhouse gases from human activities, particularly the burning of fossil fuels.

    Renewable energy technologies offer promising solutions to reduce carbon emissions. Solar power has become increasingly cost-effective and efficient in recent years. Wind energy is another rapidly growing renewable source that can provide clean electricity at scale.

    Quantum computing represents a paradigm shift in information processing. Unlike classical computers that use bits, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This property, called superposition, allows quantum computers to perform certain calculations exponentially faster than classical computers.

    The potential applications of quantum computing are vast. Cryptography could be revolutionized as quantum computers could break many current encryption methods. Drug discovery might be accelerated through quantum simulation of molecular interactions.
    """

    # Chunk the document
    print("Chunking test document...")
    chunks = chunker.chunk_document(test_document, "test_doc")

    print(f"\nCreated {len(chunks)} intelligent chunks:")
    print("-" * 50)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}: {chunk.id}")
        print(f"Sentences: {chunk.start_sentence_idx}-{chunk.end_sentence_idx}")
        print(f"Content Type: {chunk.content_type.value}")
        print(f"Word Count: {chunk.word_count}")
        print(f"Topic Coherence: {chunk.topic_coherence:.3f}")
        print(f"Entities: {chunk.entities}")
        print(f"Summary: {chunk.summary}")
        print(f"Text: {chunk.text[:200]}...")

    # Get statistics
    stats = chunker.get_chunking_stats(chunks)
    print(f"\n{'=' * 50}")
    print("Chunking Statistics:")
    print(f"{'=' * 50}")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return chunks


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_intelligent_chunking())
