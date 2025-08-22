"""
Synthesis Interface

Defines the contract for response synthesis systems that generate,
format, and validate content from multiple sources.
Built upon the established KnowledgeRetrievalInterface patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SynthesisMode(Enum):
    """Content synthesis modes"""

    EXTRACTIVE = "extractive"  # Extract and combine existing content
    ABSTRACTIVE = "abstractive"  # Generate new content from understanding
    HYBRID = "hybrid"  # Combination of extractive and abstractive
    CREATIVE = "creative"  # Generate novel creative content
    TECHNICAL = "technical"  # Precise technical documentation


class ContentFormat(Enum):
    """Output content formats"""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    STRUCTURED = "structured"


class QualityMetric(Enum):
    """Content quality assessment metrics"""

    COHERENCE = "coherence"  # Logical flow and consistency
    RELEVANCE = "relevance"  # Relevance to query/context
    COMPLETENESS = "completeness"  # Coverage of required topics
    ACCURACY = "accuracy"  # Factual correctness
    CLARITY = "clarity"  # Readability and understanding


@dataclass
class SynthesisContext:
    """Context for content synthesis operations"""

    user_id: str | None = None
    session_id: str | None = None
    domain: str | None = None
    target_audience: str | None = None
    content_constraints: dict[str, Any] = None
    quality_requirements: dict[QualityMetric, float] = None


@dataclass
class ContentSource:
    """Source material for synthesis"""

    source_id: str
    content: str
    source_type: str
    reliability_score: float
    relevance_score: float
    metadata: dict[str, Any]


@dataclass
class QualityAssessment:
    """Quality assessment results for synthesized content"""

    overall_score: float
    metric_scores: dict[QualityMetric, float]
    issues_identified: list[str]
    improvement_suggestions: list[str]
    confidence: float


@dataclass
class SynthesisResult:
    """Result of content synthesis operation"""

    content: str
    format: ContentFormat
    sources_used: list[str]
    synthesis_mode: SynthesisMode
    quality_assessment: QualityAssessment
    alternative_versions: list[str]
    metadata: dict[str, Any]


class SynthesisInterface(ABC):
    """
    Abstract interface for content synthesis systems

    Defines the contract for systems that generate coherent responses
    by combining and synthesizing information from multiple sources.
    Follows the established patterns from KnowledgeRetrievalInterface.
    """

    @abstractmethod
    async def synthesize(
        self,
        query: str,
        sources: list[ContentSource],
        mode: SynthesisMode = SynthesisMode.HYBRID,
        target_format: ContentFormat = ContentFormat.MARKDOWN,
        context: SynthesisContext | None = None,
    ) -> SynthesisResult:
        """
        Synthesize coherent response from multiple content sources

        Args:
            query: Original query or topic for synthesis
            sources: Content sources to synthesize from
            mode: Synthesis approach to use
            target_format: Desired output format
            context: Additional synthesis context

        Returns:
            Synthesized content with quality metrics
        """
        pass

    @abstractmethod
    async def generate_outline(
        self,
        topic: str,
        sources: list[ContentSource],
        outline_depth: int = 3,
    ) -> dict[str, Any]:
        """
        Generate structured outline for content synthesis

        Args:
            topic: Main topic for outline
            sources: Available content sources
            outline_depth: Hierarchical depth of outline

        Returns:
            Structured outline with sections and subsections
        """
        pass

    @abstractmethod
    async def assess_quality(
        self,
        content: str,
        reference_sources: list[ContentSource],
        metrics: list[QualityMetric] | None = None,
    ) -> QualityAssessment:
        """
        Assess quality of synthesized content

        Args:
            content: Content to assess
            reference_sources: Original sources for comparison
            metrics: Specific metrics to evaluate

        Returns:
            Comprehensive quality assessment
        """
        pass

    @abstractmethod
    async def refine_content(
        self,
        content: str,
        quality_issues: list[str],
        target_improvements: dict[QualityMetric, float],
    ) -> str:
        """
        Refine content based on quality assessment feedback

        Args:
            content: Content to refine
            quality_issues: Specific issues to address
            target_improvements: Target scores for quality metrics

        Returns:
            Refined content addressing identified issues
        """
        pass

    @abstractmethod
    async def format_content(
        self,
        content: str,
        source_format: ContentFormat,
        target_format: ContentFormat,
        formatting_options: dict[str, Any] | None = None,
    ) -> str:
        """
        Convert content between different formats

        Args:
            content: Content to format
            source_format: Current format of content
            target_format: Desired output format
            formatting_options: Format-specific options

        Returns:
            Content converted to target format
        """
        pass

    @abstractmethod
    async def detect_inconsistencies(
        self,
        content: str,
        sources: list[ContentSource],
    ) -> list[dict[str, Any]]:
        """
        Detect factual inconsistencies between content and sources

        Args:
            content: Synthesized content to check
            sources: Original sources for fact-checking

        Returns:
            List of detected inconsistencies with locations and evidence
        """
        pass

    @abstractmethod
    async def generate_citations(
        self,
        content: str,
        sources: list[ContentSource],
        citation_style: str = "apa",
    ) -> dict[str, Any]:
        """
        Generate proper citations for synthesized content

        Args:
            content: Content to generate citations for
            sources: Sources to cite
            citation_style: Citation format (apa, mla, chicago, etc.)

        Returns:
            Content with embedded citations and bibliography
        """
        pass

    @abstractmethod
    async def get_synthesis_stats(self) -> dict[str, Any]:
        """
        Get synthesis system statistics and performance metrics

        Returns:
            Dictionary with system metrics and health info
        """
        pass
