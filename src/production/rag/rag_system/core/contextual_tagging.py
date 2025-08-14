"""Two-Level Contextual Tagging System for Rich Metadata.

Implements comprehensive document and chunk-level context extraction with inheritance:
- Level 1: Document Context (Global metadata)
- Level 2: Chunk Context (Local metadata with inheritance)
- Context chain preservation for multi-level documents
- Support for books, articles, reports, and images

Based on:
- Rhetorical Structure Theory (Mann & Thompson, 1988)
- Discourse Segmentation frameworks
- Entity extraction with spaCy
- Extractive summarization techniques
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional imports with fallbacks
SPACY_AVAILABLE = False
spacy = None
try:
    import spacy

    SPACY_AVAILABLE = True
except (ImportError, TypeError, Exception):
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification."""

    BOOK = "book"
    ARTICLE = "article"
    REPORT = "report"
    RESEARCH_PAPER = "research_paper"
    WEBPAGE = "webpage"
    TECHNICAL_DOC = "technical_doc"
    IMAGE = "image"
    DIAGRAM = "diagram"


class ContentDomain(Enum):
    """Domain classification for documents."""

    SCIENCE = "science"
    TECHNOLOGY = "technology"
    HISTORY = "history"
    LITERATURE = "literature"
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTH = "health"
    ENGINEERING = "engineering"
    ARTS = "arts"
    GENERAL = "general"


class ReadingLevel(Enum):
    """Reading difficulty classification."""

    ELEMENTARY = "elementary"  # Grade 1-6
    MIDDLE_SCHOOL = "middle_school"  # Grade 7-9
    HIGH_SCHOOL = "high_school"  # Grade 10-12
    COLLEGE = "college"  # Undergraduate
    GRADUATE = "graduate"  # Graduate/Professional
    EXPERT = "expert"  # Technical/Research


class ChunkType(Enum):
    """Type of chunk content."""

    INTRODUCTION = "introduction"
    BODY = "body"
    CONCLUSION = "conclusion"
    EXAMPLE = "example"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    NARRATIVE = "narrative"
    DIALOGUE = "dialogue"
    DESCRIPTION = "description"
    ANALYSIS = "analysis"
    IMAGE_CAPTION = "image_caption"
    TABLE = "table"
    CODE = "code"
    FORMULA = "formula"


@dataclass
class ContextualEntity:
    """Rich entity with contextual information."""

    text: str
    label: str  # PERSON, ORG, GPE, etc.
    confidence: float
    start_char: int
    end_char: int
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    context_type: str = "mentioned"  # introduced, elaborated, mentioned


@dataclass
class DocumentContext:
    """Level 1: Document-level global context."""

    # Basic identification
    document_id: str
    title: str
    document_type: DocumentType

    # Content metadata
    executive_summary: str = ""
    author: str | None = None
    publication_date: str | None = None
    source_credibility_score: float = 0.7  # 0.0-1.0
    domain: ContentDomain = ContentDomain.GENERAL
    language: str = "en"
    reading_level: ReadingLevel = ReadingLevel.COLLEGE

    # Length and structure
    total_length: int = 0  # Character count
    estimated_reading_time: int = 0  # Minutes
    chapter_count: int = 0
    section_count: int = 0

    # Publication details
    isbn: str | None = None
    publisher: str | None = None
    edition: str | None = None

    # Classification
    genre: str | None = None
    subgenre: str | None = None
    target_audience: str = "general"  # academic, popular, textbook
    key_themes: list[str] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)

    # Image-specific
    image_description: str | None = None
    image_type: str | None = None  # photograph, diagram, chart
    copyright_info: str | None = None
    technical_specs: dict[str, Any] = field(default_factory=dict)

    # Extracted metadata
    document_entities: list[ContextualEntity] = field(default_factory=list)
    document_keywords: list[str] = field(default_factory=list)
    quality_indicators: dict[str, float] = field(default_factory=dict)

    # Processing metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context_version: str = "1.0"


@dataclass
class ChunkContext:
    """Level 2: Chunk-level local context with inheritance."""

    # Basic identification
    chunk_id: str
    chunk_position: int
    start_char: int
    end_char: int

    # Structural position
    chapter_number: int | None = None
    chapter_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_hierarchy: list[str] = field(default_factory=list)  # [h1, h2, h3]

    # Content analysis
    local_summary: str = ""
    chunk_type: ChunkType = ChunkType.BODY
    key_entities: list[ContextualEntity] = field(default_factory=list)
    local_keywords: list[str] = field(default_factory=list)

    # Relationships
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    related_chunk_ids: list[str] = field(default_factory=list)
    discourse_markers: list[str] = field(default_factory=list)  # "however", "furthermore"

    # Book-specific
    character_appearances: list[str] = field(default_factory=list)
    concept_type: str = "elaboration"  # introduction, elaboration, example
    narrative_position: str | None = None  # setup, conflict, resolution

    # Image-specific
    image_region: str | None = None  # "top-left", "center", etc.
    region_elements: list[str] = field(default_factory=list)
    detail_level: str = "overview"  # overview, detail, zoom

    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0

    # Inheritance tracking
    inherited_context: dict[str, Any] = field(default_factory=dict)
    context_overrides: dict[str, Any] = field(default_factory=dict)


class ContextualTagger:
    """Main contextual tagging system with two-level hierarchy."""

    def __init__(
        self,
        embedding_model: str = "paraphrase-MiniLM-L3-v2",
        enable_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        """Initialize contextual tagger."""
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize spaCy if available
        self.nlp = None
        if enable_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(f"spaCy model {spacy_model} not found, entity extraction limited")

        # Domain classification keywords
        self.domain_keywords = {
            ContentDomain.SCIENCE: [
                "research",
                "study",
                "hypothesis",
                "experiment",
                "data",
                "analysis",
                "theory",
                "methodology",
                "results",
                "conclusion",
                "peer-reviewed",
            ],
            ContentDomain.TECHNOLOGY: [
                "software",
                "algorithm",
                "programming",
                "system",
                "network",
                "database",
                "API",
                "framework",
                "architecture",
                "deployment",
                "code",
                "technical",
            ],
            ContentDomain.HISTORY: [
                "century",
                "ancient",
                "medieval",
                "war",
                "empire",
                "civilization",
                "historical",
                "timeline",
                "period",
                "era",
                "dynasty",
                "revolution",
            ],
            ContentDomain.LITERATURE: [
                "novel",
                "poetry",
                "narrative",
                "character",
                "plot",
                "theme",
                "author",
                "literary",
                "fiction",
                "genre",
                "story",
                "writing",
            ],
            ContentDomain.BUSINESS: [
                "company",
                "market",
                "revenue",
                "strategy",
                "management",
                "finance",
                "business",
                "corporate",
                "economics",
                "profit",
                "investment",
                "sales",
            ],
            ContentDomain.EDUCATION: [
                "learning",
                "teaching",
                "student",
                "curriculum",
                "education",
                "academic",
                "school",
                "university",
                "course",
                "textbook",
                "pedagogy",
                "instruction",
            ],
        }

        # Reading level indicators
        self.reading_level_patterns = {
            ReadingLevel.ELEMENTARY: {
                "avg_sentence_length": (5, 12),
                "avg_word_length": (3, 5),
                "complex_words_ratio": (0.0, 0.1),
            },
            ReadingLevel.MIDDLE_SCHOOL: {
                "avg_sentence_length": (10, 18),
                "avg_word_length": (4, 6),
                "complex_words_ratio": (0.1, 0.2),
            },
            ReadingLevel.HIGH_SCHOOL: {
                "avg_sentence_length": (15, 25),
                "avg_word_length": (5, 7),
                "complex_words_ratio": (0.2, 0.35),
            },
            ReadingLevel.COLLEGE: {
                "avg_sentence_length": (20, 30),
                "avg_word_length": (6, 8),
                "complex_words_ratio": (0.35, 0.5),
            },
            ReadingLevel.GRADUATE: {
                "avg_sentence_length": (25, 35),
                "avg_word_length": (7, 10),
                "complex_words_ratio": (0.5, 0.7),
            },
            ReadingLevel.EXPERT: {
                "avg_sentence_length": (30, 50),
                "avg_word_length": (8, 12),
                "complex_words_ratio": (0.7, 1.0),
            },
        }

        # Discourse markers for relationship detection
        self.discourse_markers = {
            "continuation": [
                "furthermore",
                "moreover",
                "additionally",
                "also",
                "likewise",
            ],
            "contrast": [
                "however",
                "nevertheless",
                "nonetheless",
                "on the other hand",
                "conversely",
            ],
            "causation": ["therefore", "consequently", "thus", "as a result", "hence"],
            "exemplification": [
                "for example",
                "for instance",
                "such as",
                "namely",
                "specifically",
            ],
            "summarization": [
                "in conclusion",
                "in summary",
                "to summarize",
                "overall",
                "in brief",
            ],
            "temporal": [
                "first",
                "next",
                "then",
                "finally",
                "meanwhile",
                "subsequently",
            ],
        }

    def extract_document_context(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentContext:
        """Extract Level 1 document-level context."""
        logger.info(f"Extracting document context for: {document_id}")

        metadata = metadata or {}

        # Basic document analysis
        doc_type = self._classify_document_type(title, content, metadata)
        domain = self._classify_domain(content)
        reading_level = self._assess_reading_level(content)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(content)

        # Extract entities and keywords
        entities = self._extract_entities(content, context_type="document")
        keywords = self._extract_keywords(content)

        # Calculate length metrics
        total_length = len(content)
        estimated_reading_time = max(1, total_length // 1000)  # ~200 words per minute

        # Structure analysis
        chapter_count, section_count = self._analyze_document_structure(content)

        # Extract themes and concepts
        key_themes = self._extract_themes(content, entities)
        key_concepts = self._extract_key_concepts(content, entities)

        # Quality assessment
        quality_indicators = self._assess_document_quality(content, entities)

        # Create document context
        doc_context = DocumentContext(
            document_id=document_id,
            title=title,
            document_type=doc_type,
            executive_summary=executive_summary,
            author=metadata.get("author"),
            publication_date=metadata.get("publication_date"),
            source_credibility_score=metadata.get("credibility_score", 0.7),
            domain=domain,
            language=metadata.get("language", "en"),
            reading_level=reading_level,
            total_length=total_length,
            estimated_reading_time=estimated_reading_time,
            chapter_count=chapter_count,
            section_count=section_count,
            isbn=metadata.get("isbn"),
            publisher=metadata.get("publisher"),
            edition=metadata.get("edition"),
            genre=metadata.get("genre"),
            subgenre=metadata.get("subgenre"),
            target_audience=metadata.get("target_audience", "general"),
            key_themes=key_themes,
            key_concepts=key_concepts,
            image_description=metadata.get("image_description"),
            image_type=metadata.get("image_type"),
            copyright_info=metadata.get("copyright_info"),
            technical_specs=metadata.get("technical_specs", {}),
            document_entities=entities,
            document_keywords=keywords,
            quality_indicators=quality_indicators,
        )

        logger.info(f"Document context extracted: {doc_type.value}, {domain.value}, {reading_level.value}")
        return doc_context

    def extract_chunk_context(
        self,
        chunk_id: str,
        chunk_text: str,
        chunk_position: int,
        start_char: int,
        end_char: int,
        document_context: DocumentContext,
        full_document_text: str = "",
        previous_chunk_context: Optional["ChunkContext"] = None,
    ) -> ChunkContext:
        """Extract Level 2 chunk-level context with inheritance."""
        logger.debug(f"Extracting chunk context for: {chunk_id}")

        # Analyze chunk structure and position
        structural_info = self._analyze_chunk_structure(chunk_text, full_document_text, start_char, end_char)

        # Generate local summary
        local_summary = self._generate_chunk_summary(chunk_text)

        # Classify chunk type
        chunk_type = self._classify_chunk_type(chunk_text, chunk_position, document_context)

        # Extract local entities and keywords
        local_entities = self._extract_entities(chunk_text, context_type="chunk")
        local_keywords = self._extract_keywords(chunk_text, limit=10)

        # Analyze relationships
        discourse_markers = self._detect_discourse_markers(chunk_text)

        # Content-specific analysis
        content_analysis = self._analyze_chunk_content(chunk_text, document_context, chunk_type)

        # Quality metrics
        coherence_score = self._calculate_chunk_coherence(chunk_text)
        relevance_score = self._calculate_chunk_relevance(chunk_text, document_context)
        completeness_score = self._assess_chunk_completeness(chunk_text)

        # Context inheritance
        inherited_context = self._create_inherited_context(document_context, previous_chunk_context)
        context_overrides = self._detect_context_overrides(chunk_text, inherited_context)

        # Create chunk context
        chunk_context = ChunkContext(
            chunk_id=chunk_id,
            chunk_position=chunk_position,
            start_char=start_char,
            end_char=end_char,
            chapter_number=structural_info.get("chapter_number"),
            chapter_title=structural_info.get("chapter_title"),
            page_start=structural_info.get("page_start"),
            page_end=structural_info.get("page_end"),
            section_hierarchy=structural_info.get("section_hierarchy", []),
            local_summary=local_summary,
            chunk_type=chunk_type,
            key_entities=local_entities,
            local_keywords=local_keywords,
            previous_chunk_id=(previous_chunk_context.chunk_id if previous_chunk_context else None),
            discourse_markers=discourse_markers,
            character_appearances=content_analysis.get("character_appearances", []),
            concept_type=content_analysis.get("concept_type", "elaboration"),
            narrative_position=content_analysis.get("narrative_position"),
            image_region=content_analysis.get("image_region"),
            region_elements=content_analysis.get("region_elements", []),
            detail_level=content_analysis.get("detail_level", "overview"),
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            inherited_context=inherited_context,
            context_overrides=context_overrides,
        )

        return chunk_context

    def _classify_document_type(self, title: str, content: str, metadata: dict[str, Any]) -> DocumentType:
        """Classify document type based on content and metadata."""
        # Check explicit metadata first
        if "document_type" in metadata:
            try:
                return DocumentType(metadata["document_type"])
            except ValueError:
                pass

        # Image detection
        if any(ext in title.lower() for ext in [".jpg", ".png", ".gif", ".svg", ".pdf"]):
            return DocumentType.IMAGE

        # Length-based classification
        content_length = len(content)

        if content_length > 100000:  # ~40+ pages
            return DocumentType.BOOK
        if content_length > 10000:  # ~4+ pages
            return DocumentType.REPORT
        return DocumentType.ARTICLE

    def _classify_domain(self, content: str) -> ContentDomain:
        """Classify content domain based on keyword analysis."""
        content_lower = content.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            domain_scores[domain] = score

        if not domain_scores or max(domain_scores.values()) < 3:
            return ContentDomain.GENERAL

        return max(domain_scores, key=domain_scores.get)

    def _assess_reading_level(self, content: str) -> ReadingLevel:
        """Assess reading difficulty level."""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ReadingLevel.COLLEGE

        words = content.split()
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Calculate complex words ratio (words with 3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_words_ratio = complex_words / len(words)

        # Find best matching reading level
        best_match = ReadingLevel.COLLEGE
        best_score = float("inf")

        for level, ranges in self.reading_level_patterns.items():
            score = 0

            # Sentence length score
            sent_min, sent_max = ranges["avg_sentence_length"]
            if avg_sentence_length < sent_min:
                score += (sent_min - avg_sentence_length) ** 2
            elif avg_sentence_length > sent_max:
                score += (avg_sentence_length - sent_max) ** 2

            # Word length score
            word_min, word_max = ranges["avg_word_length"]
            if avg_word_length < word_min:
                score += (word_min - avg_word_length) ** 2
            elif avg_word_length > word_max:
                score += (avg_word_length - word_max) ** 2

            # Complex words ratio score
            complex_min, complex_max = ranges["complex_words_ratio"]
            if complex_words_ratio < complex_min:
                score += (complex_min - complex_words_ratio) ** 2
            elif complex_words_ratio > complex_max:
                score += (complex_words_ratio - complex_max) ** 2

            if score < best_score:
                best_score = score
                best_match = level

        return best_match

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for reading level assessment."""
        word = word.lower().strip()
        if len(word) == 0:
            return 0

        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for _i, char in enumerate(word):
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False

        # Silent 'e' at the end
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _generate_executive_summary(self, content: str, max_length: int = 200) -> str:
        """Generate extractive summary of document."""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

        if len(sentences) <= 3:
            return " ".join(sentences)

        # Simple extractive approach: first sentence + middle sentences
        summary_sentences = []

        # Always include first sentence
        if sentences:
            summary_sentences.append(sentences[0])

        # Add representative middle sentences
        if len(sentences) > 10:
            mid_start = len(sentences) // 3
            mid_end = 2 * len(sentences) // 3
            middle_sentences = sentences[mid_start:mid_end]

            # Select most informative middle sentences (longer ones)
            middle_sentences.sort(key=len, reverse=True)
            summary_sentences.extend(middle_sentences[:2])

        summary = " ".join(summary_sentences)

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def _extract_entities(self, text: str, context_type: str = "document") -> list[ContextualEntity]:
        """Extract named entities with contextual information."""
        entities = []

        if self.nlp:
            try:
                # Limit text length for efficiency
                doc = self.nlp(text[:5000])

                for ent in doc.ents:
                    entity = ContextualEntity(
                        text=ent.text,
                        label=ent.label_,
                        confidence=getattr(ent, "confidence", 0.8),
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        context_type=context_type,
                    )
                    entities.append(entity)

            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

        # Fallback: simple pattern matching for common entities
        if not entities:
            entities = self._extract_entities_fallback(text, context_type)

        return entities

    def _extract_entities_fallback(self, text: str, context_type: str) -> list[ContextualEntity]:
        """Fallback entity extraction using patterns."""
        entities = []

        # Simple patterns for common entities
        patterns = {
            "DATE": r"\b\d{4}\b|\b\w+\s+\d{1,2},\s+\d{4}\b",
            "MONEY": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            "PERCENT": r"\d+(?:\.\d+)?%",
            "PERSON": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Simple name pattern
        }

        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entity = ContextualEntity(
                    text=match.group(),
                    label=label,
                    confidence=0.6,
                    start_char=match.start(),
                    end_char=match.end(),
                    context_type=context_type,
                )
                entities.append(entity)

        return entities

    def _extract_keywords(self, text: str, limit: int = 20) -> list[str]:
        """Extract key terms from text using TF-IDF approach."""
        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        # Remove common stop words
        stop_words = {
            "that",
            "this",
            "with",
            "from",
            "they",
            "been",
            "have",
            "their",
            "said",
            "each",
            "which",
            "will",
            "would",
            "there",
            "could",
            "when",
        }

        words = [word for word in words if word not in stop_words]

        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:limit]]

    def _analyze_document_structure(self, content: str) -> tuple[int, int]:
        """Analyze document structure to count chapters and sections."""
        chapter_patterns = [r"^Chapter\s+\d+", r"^\d+\.\s+[A-Z]", r"^[IVX]+\.\s+[A-Z]"]

        section_patterns = [
            r"^#{1,6}\s+",  # Markdown headers
            r"^\d+\.\d+\s+",  # Numbered sections
            r"^[A-Z][A-Z\s]+$",  # ALL CAPS headers
        ]

        lines = content.split("\n")
        chapter_count = 0
        section_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for chapters
            for pattern in chapter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    chapter_count += 1
                    break

            # Check for sections
            for pattern in section_patterns:
                if re.match(pattern, line):
                    section_count += 1
                    break

        return chapter_count, section_count

    def _extract_themes(self, content: str, entities: list[ContextualEntity]) -> list[str]:
        """Extract main themes from document content."""
        themes = []

        # Theme keywords and patterns
        theme_patterns = {
            "innovation": ["innovation", "breakthrough", "advancement", "pioneering"],
            "conflict": ["conflict", "tension", "dispute", "controversy", "debate"],
            "growth": ["growth", "development", "expansion", "progress", "evolution"],
            "change": ["change", "transformation", "shift", "transition", "revolution"],
            "collaboration": [
                "collaboration",
                "cooperation",
                "partnership",
                "teamwork",
            ],
            "sustainability": ["sustainability", "environmental", "green", "renewable"],
            "education": ["education", "learning", "knowledge", "teaching", "academic"],
            "technology": [
                "technology",
                "digital",
                "artificial",
                "automation",
                "innovation",
            ],
        }

        content_lower = content.lower()

        for theme, keywords in theme_patterns.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score >= 2:  # Minimum threshold
                themes.append(theme)

        return themes[:5]  # Limit to top 5 themes

    def _extract_key_concepts(self, content: str, entities: list[ContextualEntity]) -> list[str]:
        """Extract key concepts from document."""
        concepts = []

        # Extract concepts from entities
        concept_labels = ["ORG", "PRODUCT", "EVENT", "LAW", "LANGUAGE"]
        for entity in entities:
            if entity.label in concept_labels:
                concepts.append(entity.text)

        # Extract concepts from capitalized terms
        capitalized_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
        concepts.extend(capitalized_terms[:10])

        # Remove duplicates and return
        return list(set(concepts))[:10]

    def _assess_document_quality(self, content: str, entities: list[ContextualEntity]) -> dict[str, float]:
        """Assess various quality indicators of the document."""
        quality = {}

        # Entity density (more entities often indicate richer content)
        quality["entity_density"] = len(entities) / max(1, len(content.split()) / 100)

        # Sentence variety (different sentence lengths)
        sentences = re.split(r"[.!?]+", content)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            quality["sentence_variety"] = np.std(sentence_lengths) / np.mean(sentence_lengths)
        else:
            quality["sentence_variety"] = 0.0

        # Vocabulary richness (unique words / total words)
        words = content.lower().split()
        if words:
            quality["vocabulary_richness"] = len(set(words)) / len(words)
        else:
            quality["vocabulary_richness"] = 0.0

        # Structure quality (presence of headers, formatting)
        structure_indicators = len(re.findall(r"^[#*-]\s+|^\d+\.", content, re.MULTILINE))
        quality["structural_quality"] = min(1.0, structure_indicators / 10)

        return quality

    def _analyze_chunk_structure(
        self, chunk_text: str, full_document: str, start_char: int, end_char: int
    ) -> dict[str, Any]:
        """Analyze structural position of chunk within document."""
        structural_info = {}

        if not full_document:
            return structural_info

        # Find preceding context for structure detection
        preceding_text = full_document[:start_char]

        # Extract section hierarchy
        section_hierarchy = []

        # Look for markdown-style headers before this chunk
        header_patterns = [
            (r"^(#{1,6})\s+(.+)$", lambda m: (len(m.group(1)), m.group(2).strip())),
            (
                r"^(.+)\n[=-]+$",
                lambda m: (1 if "=" in m.group() else 2, m.group(1).strip()),
            ),
        ]

        preceding_lines = preceding_text.split("\n")[-50:]  # Look at last 50 lines

        for line in reversed(preceding_lines):
            for pattern, extractor in header_patterns:
                match = re.match(pattern, line)
                if match:
                    level, title = extractor(match)
                    if level <= len(section_hierarchy) + 1:
                        section_hierarchy = [*section_hierarchy[: level - 1], title]
                        break

        structural_info["section_hierarchy"] = section_hierarchy

        # Extract chapter information
        chapter_match = re.search(r"Chapter\s+(\d+)(?:\s*:\s*(.+?))?", preceding_text, re.IGNORECASE)
        if chapter_match:
            structural_info["chapter_number"] = int(chapter_match.group(1))
            if chapter_match.group(2):
                structural_info["chapter_title"] = chapter_match.group(2).strip()

        return structural_info

    def _generate_chunk_summary(self, chunk_text: str, max_length: int = 100) -> str:
        """Generate concise summary of chunk content."""
        sentences = re.split(r"[.!?]+", chunk_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        if len(sentences) == 1:
            summary = sentences[0]
        elif len(sentences) == 2:
            summary = " ".join(sentences)
        else:
            # Take first sentence and most informative subsequent sentence
            summary = sentences[0]
            if len(sentences) > 2:
                # Find longest sentence as potentially most informative
                longest = max(sentences[1:], key=len)
                summary += " " + longest

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def _classify_chunk_type(
        self, chunk_text: str, chunk_position: int, document_context: DocumentContext
    ) -> ChunkType:
        """Classify the type of chunk based on content and position."""
        text_lower = chunk_text.lower()

        # Position-based classification
        if chunk_position == 0:
            return ChunkType.INTRODUCTION

        # Content-based classification
        if any(word in text_lower for word in ["for example", "for instance", "such as"]):
            return ChunkType.EXAMPLE

        if any(word in text_lower for word in ["definition", "defined as", "refers to"]):
            return ChunkType.DEFINITION

        if any(word in text_lower for word in ["first", "then", "next", "finally", "step"]):
            return ChunkType.PROCEDURE

        if any(word in text_lower for word in ["said", "replied", "asked", "exclaimed"]):
            return ChunkType.DIALOGUE

        if re.search(r"```|<code>|def\s+\w+|function\s+\w+", chunk_text):
            return ChunkType.CODE

        if re.search(r"\$.*\$|\\[a-zA-Z]+", chunk_text):
            return ChunkType.FORMULA

        # Default to body content
        return ChunkType.BODY

    def _detect_discourse_markers(self, chunk_text: str) -> list[str]:
        """Detect discourse markers that indicate text relationships."""
        found_markers = []
        text_lower = chunk_text.lower()

        for marker_type, markers in self.discourse_markers.items():
            for marker in markers:
                if marker in text_lower:
                    found_markers.append(f"{marker_type}:{marker}")

        return found_markers

    def _analyze_chunk_content(
        self, chunk_text: str, document_context: DocumentContext, chunk_type: ChunkType
    ) -> dict[str, Any]:
        """Analyze chunk content for type-specific features."""
        analysis = {}

        # Book-specific analysis
        if document_context.document_type == DocumentType.BOOK:
            # Extract character mentions (simple approach)
            if document_context.genre and "fiction" in document_context.genre.lower():
                # Look for capitalized names that might be characters
                potential_characters = re.findall(r"\b[A-Z][a-z]+\b", chunk_text)
                # Filter common words
                common_words = {"The", "This", "That", "When", "Where", "How", "Why"}
                characters = [name for name in potential_characters if name not in common_words]
                analysis["character_appearances"] = list(set(characters))[:5]

            # Determine concept type
            if chunk_type == ChunkType.EXAMPLE:
                analysis["concept_type"] = "example"
            elif any(word in chunk_text.lower() for word in ["introduce", "first mention", "new concept"]):
                analysis["concept_type"] = "introduction"
            else:
                analysis["concept_type"] = "elaboration"

        # Image-specific analysis
        if document_context.document_type in [DocumentType.IMAGE, DocumentType.DIAGRAM]:
            analysis["image_region"] = "center"  # Default
            analysis["region_elements"] = []
            analysis["detail_level"] = "overview"

        return analysis

    def _calculate_chunk_coherence(self, chunk_text: str) -> float:
        """Calculate semantic coherence of chunk content."""
        sentences = re.split(r"[.!?]+", chunk_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0

        try:
            # Calculate sentence embeddings
            embeddings = self.embedder.encode(sentences)

            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)

            return float(np.mean(similarities)) if similarities else 0.5

        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            return 0.5

    def _calculate_chunk_relevance(self, chunk_text: str, document_context: DocumentContext) -> float:
        """Calculate relevance of chunk to document themes."""
        chunk_keywords = set(self._extract_keywords(chunk_text, limit=10))
        document_keywords = set(document_context.document_keywords)

        if not document_keywords:
            return 0.7  # Default relevance

        overlap = len(chunk_keywords.intersection(document_keywords))
        return min(1.0, overlap / len(document_keywords) * 2)

    def _assess_chunk_completeness(self, chunk_text: str) -> float:
        """Assess if chunk contains complete thoughts/ideas."""
        sentences = re.split(r"[.!?]+", chunk_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Check for incomplete sentences (very short or ending mid-thought)
        complete_sentences = 0
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 3 and sentence.endswith((".", "!", "?")):
                complete_sentences += 1

        return complete_sentences / len(sentences) if sentences else 0.0

    def _create_inherited_context(
        self,
        document_context: DocumentContext,
        previous_chunk_context: ChunkContext | None,
    ) -> dict[str, Any]:
        """Create inherited context from document and previous chunks."""
        inherited = {
            # From document context
            "document_title": document_context.title,
            "document_type": document_context.document_type.value,
            "domain": document_context.domain.value,
            "reading_level": document_context.reading_level.value,
            "author": document_context.author,
            "publication_date": document_context.publication_date,
            "key_themes": document_context.key_themes,
            "key_concepts": document_context.key_concepts,
            "target_audience": document_context.target_audience,
        }

        # From previous chunk context
        if previous_chunk_context:
            inherited.update(
                {
                    "previous_chunk_type": previous_chunk_context.chunk_type.value,
                    "previous_section_hierarchy": previous_chunk_context.section_hierarchy,
                    "previous_narrative_position": previous_chunk_context.narrative_position,
                    "continuing_discourse": previous_chunk_context.discourse_markers,
                }
            )

        return inherited

    def _detect_context_overrides(self, chunk_text: str, inherited_context: dict[str, Any]) -> dict[str, Any]:
        """Detect when chunk context overrides inherited context."""
        overrides = {}

        # Detect topic shifts that might override domain
        text_lower = chunk_text.lower()

        # Check for explicit domain shifts
        domain_shifts = {
            "technical": ["technical", "engineering", "system", "algorithm"],
            "historical": ["historically", "in the past", "centuries ago"],
            "scientific": ["research shows", "study found", "experiment"],
        }

        for new_domain, indicators in domain_shifts.items():
            if any(indicator in text_lower for indicator in indicators):
                if inherited_context.get("domain") != new_domain:
                    overrides["domain_shift"] = new_domain

        # Detect narrative position changes
        narrative_indicators = {
            "setup": ["beginning", "introduction", "start"],
            "conflict": ["problem", "challenge", "difficulty", "however"],
            "resolution": ["solution", "conclusion", "finally", "resolved"],
        }

        for position, indicators in narrative_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                overrides["narrative_position"] = position

        return overrides

    def create_contextual_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        chunk_position: int,
        start_char: int,
        end_char: int,
        document_context: DocumentContext,
        full_document_text: str = "",
        previous_chunk_context: ChunkContext | None = None,
    ) -> dict[str, Any]:
        """Create a complete contextual chunk with both levels of context."""
        # Extract chunk-level context
        chunk_context = self.extract_chunk_context(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            chunk_position=chunk_position,
            start_char=start_char,
            end_char=end_char,
            document_context=document_context,
            full_document_text=full_document_text,
            previous_chunk_context=previous_chunk_context,
        )

        # Create complete contextual representation
        contextual_chunk = {
            # Basic chunk data
            "chunk_id": chunk_id,
            "text": chunk_text,
            "position": chunk_position,
            "start_char": start_char,
            "end_char": end_char,
            # Level 1: Document Context (inherited)
            "document_context": {
                "title": document_context.title,
                "document_type": document_context.document_type.value,
                "author": document_context.author,
                "publication_date": document_context.publication_date,
                "domain": document_context.domain.value,
                "reading_level": document_context.reading_level.value,
                "executive_summary": document_context.executive_summary,
                "key_themes": document_context.key_themes,
                "key_concepts": document_context.key_concepts,
                "credibility_score": document_context.source_credibility_score,
                "total_length": document_context.total_length,
                "estimated_reading_time": document_context.estimated_reading_time,
            },
            # Level 2: Chunk Context (local)
            "chunk_context": {
                "local_summary": chunk_context.local_summary,
                "chunk_type": chunk_context.chunk_type.value,
                "section_hierarchy": chunk_context.section_hierarchy,
                "chapter_number": chunk_context.chapter_number,
                "chapter_title": chunk_context.chapter_title,
                "key_entities": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "confidence": e.confidence,
                        "context_type": e.context_type,
                    }
                    for e in chunk_context.key_entities
                ],
                "local_keywords": chunk_context.local_keywords,
                "discourse_markers": chunk_context.discourse_markers,
                "coherence_score": chunk_context.coherence_score,
                "relevance_score": chunk_context.relevance_score,
                "completeness_score": chunk_context.completeness_score,
            },
            # Contextual relationships
            "relationships": {
                "previous_chunk_id": chunk_context.previous_chunk_id,
                "next_chunk_id": chunk_context.next_chunk_id,
                "related_chunks": chunk_context.related_chunk_ids,
            },
            # Context inheritance and overrides
            "context_inheritance": {
                "inherited_context": chunk_context.inherited_context,
                "context_overrides": chunk_context.context_overrides,
            },
            # Quality metrics
            "quality_metrics": {
                "coherence_score": chunk_context.coherence_score,
                "relevance_score": chunk_context.relevance_score,
                "completeness_score": chunk_context.completeness_score,
                "overall_quality": (
                    chunk_context.coherence_score + chunk_context.relevance_score + chunk_context.completeness_score
                )
                / 3,
            },
        }

        return contextual_chunk


# Test function
async def test_contextual_tagging():
    """Test the contextual tagging system."""
    print("Testing Contextual Tagging System")
    print("=" * 50)

    # Initialize tagger
    tagger = ContextualTagger(enable_spacy=False)  # Disable spaCy for this test

    # Test document
    test_document = """
    # Introduction to Artificial Intelligence

    Artificial Intelligence (AI) represents one of the most transformative technologies of our time. This field encompasses machine learning, neural networks, and deep learning algorithms that enable computers to perform tasks traditionally requiring human intelligence.

    ## Machine Learning Fundamentals

    Machine learning is a subset of AI that focuses on developing algorithms that can learn from data without explicit programming. For example, a spam email filter learns to identify spam by analyzing thousands of email examples.

    ## Deep Learning Revolution

    Deep learning has revolutionized computer vision and natural language processing. Companies like Google, Facebook, and OpenAI have made significant breakthroughs using neural networks with multiple hidden layers.

    ## Applications and Future

    AI applications span healthcare, finance, autonomous vehicles, and scientific research. However, we must address ethical concerns about bias, privacy, and job displacement as these technologies advance.

    In conclusion, artificial intelligence will continue to shape our future, requiring careful consideration of both its potential benefits and risks.
    """

    # Extract document context
    print("Extracting document context...")
    doc_context = tagger.extract_document_context(
        document_id="ai_guide_001",
        title="Introduction to Artificial Intelligence",
        content=test_document,
        metadata={
            "author": "AI Researcher",
            "publication_date": "2024-01-01",
            "document_type": "article",
            "target_audience": "general",
        },
    )

    print(f"Document Type: {doc_context.document_type.value}")
    print(f"Domain: {doc_context.domain.value}")
    print(f"Reading Level: {doc_context.reading_level.value}")
    print(f"Key Themes: {doc_context.key_themes}")
    print(f"Key Concepts: {doc_context.key_concepts}")
    print(f"Executive Summary: {doc_context.executive_summary}")

    # Simulate chunking and context extraction
    print("\nExtracting chunk contexts...")

    # Simple sentence-based chunking for demo
    sentences = re.split(r"[.!?]+", test_document)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_pos = 0
    previous_chunk_context = None

    for i, sentence in enumerate(sentences[:5]):  # Process first 5 sentences
        chunk_id = f"ai_guide_001_chunk_{i}"
        start_char = test_document.find(sentence, current_pos)
        end_char = start_char + len(sentence)
        current_pos = end_char

        # Create contextual chunk
        contextual_chunk = tagger.create_contextual_chunk(
            chunk_id=chunk_id,
            chunk_text=sentence,
            chunk_position=i,
            start_char=start_char,
            end_char=end_char,
            document_context=doc_context,
            full_document_text=test_document,
            previous_chunk_context=previous_chunk_context,
        )

        chunks.append(contextual_chunk)

        # Extract chunk context for next iteration
        previous_chunk_context = tagger.extract_chunk_context(
            chunk_id=chunk_id,
            chunk_text=sentence,
            chunk_position=i,
            start_char=start_char,
            end_char=end_char,
            document_context=doc_context,
            full_document_text=test_document,
            previous_chunk_context=previous_chunk_context,
        )

        print(f"\nChunk {i + 1}: {chunk_id}")
        print(f"  Type: {contextual_chunk['chunk_context']['chunk_type']}")
        print(f"  Summary: {contextual_chunk['chunk_context']['local_summary']}")
        print(f"  Keywords: {contextual_chunk['chunk_context']['local_keywords'][:3]}")
        print(f"  Quality: {contextual_chunk['quality_metrics']['overall_quality']:.3f}")
        print(f"  Coherence: {contextual_chunk['quality_metrics']['coherence_score']:.3f}")
        print(f"  Text: {sentence[:100]}...")

    print(f"\n{'=' * 50}")
    print("Contextual Tagging Complete!")
    print(f"Processed {len(chunks)} chunks with full contextual metadata")

    return chunks


if __name__ == "__main__":
    asyncio.run(test_contextual_tagging())
