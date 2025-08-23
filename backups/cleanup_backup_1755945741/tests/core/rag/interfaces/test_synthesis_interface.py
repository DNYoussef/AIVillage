"""
Tests for SynthesisInterface

Behavioral tests ensuring interface compliance and contract validation.
"""

from abc import ABC
from unittest.mock import AsyncMock

import pytest

from core.rag.interfaces.synthesis_interface import (
    ContentFormat,
    ContentSource,
    QualityAssessment,
    QualityMetric,
    SynthesisContext,
    SynthesisInterface,
    SynthesisMode,
    SynthesisResult,
)


class MockSynthesisInterface(SynthesisInterface):
    """Mock implementation for testing interface compliance"""

    async def synthesize(self, query, sources, mode=None, target_format=None, context=None):
        return self._synthesize_mock

    async def generate_outline(self, topic, sources, outline_depth=3):
        return self._generate_outline_mock

    async def assess_quality(self, content, reference_sources, metrics=None):
        return self._assess_quality_mock

    async def refine_content(self, content, quality_issues, target_improvements):
        return self._refine_content_mock

    async def format_content(self, content, source_format, target_format, formatting_options=None):
        return self._format_content_mock

    async def detect_inconsistencies(self, content, sources):
        return self._detect_inconsistencies_mock

    async def generate_citations(self, content, sources, citation_style="apa"):
        return self._generate_citations_mock

    async def get_synthesis_stats(self):
        return self._get_synthesis_stats_mock

    def __init__(self):
        self._synthesize_mock = None
        self._generate_outline_mock = None
        self._assess_quality_mock = None
        self._refine_content_mock = None
        self._format_content_mock = None
        self._detect_inconsistencies_mock = None
        self._generate_citations_mock = None
        self._get_synthesis_stats_mock = None


@pytest.fixture
def synthesis_interface():
    """Fixture providing mock synthesis interface"""
    return MockSynthesisInterface()


@pytest.fixture
def sample_context():
    """Sample synthesis context for testing"""
    return SynthesisContext(
        user_id="test_user",
        session_id="test_session",
        domain="technology",
        target_audience="technical",
        quality_requirements={QualityMetric.COHERENCE: 0.8, QualityMetric.ACCURACY: 0.9},
    )


@pytest.fixture
def sample_content_source():
    """Sample content source for testing"""
    return ContentSource(
        source_id="source_1",
        content="Test content from source",
        source_type="article",
        reliability_score=0.8,
        relevance_score=0.9,
        metadata={"author": "Test Author", "date": "2024-01-01"},
    )


@pytest.fixture
def sample_quality_assessment():
    """Sample quality assessment for testing"""
    return QualityAssessment(
        overall_score=0.85,
        metric_scores={QualityMetric.COHERENCE: 0.8, QualityMetric.RELEVANCE: 0.9, QualityMetric.ACCURACY: 0.8},
        issues_identified=["Minor coherence gap"],
        improvement_suggestions=["Add transitional sentences"],
        confidence=0.7,
    )


class TestSynthesisInterface:
    """Test synthesis interface contract and behavior"""

    def test_is_abstract_base_class(self):
        """Test that SynthesisInterface is properly abstract"""
        assert issubclass(SynthesisInterface, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            SynthesisInterface()

    @pytest.mark.asyncio
    async def test_synthesize_contract(
        self, synthesis_interface, sample_context, sample_content_source, sample_quality_assessment
    ):
        """Test synthesize method contract"""
        expected_result = SynthesisResult(
            content="Synthesized content",
            format=ContentFormat.MARKDOWN,
            sources_used=["source_1"],
            synthesis_mode=SynthesisMode.HYBRID,
            quality_assessment=sample_quality_assessment,
            alternative_versions=["Alternative version"],
            metadata={"processing_time": 1.5},
        )
        synthesis_interface.synthesize.return_value = expected_result

        result = await synthesis_interface.synthesize(
            query="Test query",
            sources=[sample_content_source],
            mode=SynthesisMode.HYBRID,
            target_format=ContentFormat.MARKDOWN,
            context=sample_context,
        )

        synthesis_interface.synthesize.assert_called_once()
        assert result == expected_result
        assert isinstance(result, SynthesisResult)

    @pytest.mark.asyncio
    async def test_generate_outline_contract(self, synthesis_interface, sample_content_source):
        """Test generate_outline method contract"""
        expected_outline = {
            "title": "Test Topic",
            "sections": [
                {"title": "Introduction", "subsections": []},
                {"title": "Main Content", "subsections": ["Subsection 1", "Subsection 2"]},
                {"title": "Conclusion", "subsections": []},
            ],
            "estimated_length": 1000,
        }
        synthesis_interface.generate_outline.return_value = expected_outline

        result = await synthesis_interface.generate_outline(
            topic="Test Topic", sources=[sample_content_source], outline_depth=3
        )

        synthesis_interface.generate_outline.assert_called_once()
        assert result == expected_outline
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_assess_quality_contract(self, synthesis_interface, sample_content_source, sample_quality_assessment):
        """Test assess_quality method contract"""
        synthesis_interface.assess_quality.return_value = sample_quality_assessment

        result = await synthesis_interface.assess_quality(
            content="Test content to assess",
            reference_sources=[sample_content_source],
            metrics=[QualityMetric.COHERENCE, QualityMetric.ACCURACY],
        )

        synthesis_interface.assess_quality.assert_called_once()
        assert result == sample_quality_assessment
        assert isinstance(result, QualityAssessment)

    @pytest.mark.asyncio
    async def test_refine_content_contract(self, synthesis_interface):
        """Test refine_content method contract"""
        expected_refined = "Refined content with improvements"
        synthesis_interface.refine_content.return_value = expected_refined

        result = await synthesis_interface.refine_content(
            content="Original content",
            quality_issues=["Issue 1", "Issue 2"],
            target_improvements={QualityMetric.COHERENCE: 0.9},
        )

        synthesis_interface.refine_content.assert_called_once()
        assert result == expected_refined
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_format_content_contract(self, synthesis_interface):
        """Test format_content method contract"""
        expected_formatted = "# Formatted Content\n\nThis is formatted content."
        synthesis_interface.format_content.return_value = expected_formatted

        result = await synthesis_interface.format_content(
            content="Formatted Content\n\nThis is formatted content.",
            source_format=ContentFormat.PLAIN_TEXT,
            target_format=ContentFormat.MARKDOWN,
            formatting_options={"include_toc": True},
        )

        synthesis_interface.format_content.assert_called_once()
        assert result == expected_formatted
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_detect_inconsistencies_contract(self, synthesis_interface, sample_content_source):
        """Test detect_inconsistencies method contract"""
        expected_inconsistencies = [
            {
                "location": "paragraph 2",
                "issue": "Contradicts source information",
                "severity": "high",
                "evidence": "Source states X, content states Y",
            }
        ]
        synthesis_interface.detect_inconsistencies.return_value = expected_inconsistencies

        result = await synthesis_interface.detect_inconsistencies(
            content="Content with potential inconsistencies", sources=[sample_content_source]
        )

        synthesis_interface.detect_inconsistencies.assert_called_once()
        assert result == expected_inconsistencies
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_generate_citations_contract(self, synthesis_interface, sample_content_source):
        """Test generate_citations method contract"""
        expected_citations = {
            "content_with_citations": "Content with [1] citations.",
            "bibliography": ["1. Test Author (2024). Source Title."],
        }
        synthesis_interface.generate_citations.return_value = expected_citations

        result = await synthesis_interface.generate_citations(
            content="Content with citations.", sources=[sample_content_source], citation_style="apa"
        )

        synthesis_interface.generate_citations.assert_called_once()
        assert result == expected_citations
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_synthesis_stats_contract(self, synthesis_interface):
        """Test get_synthesis_stats method contract"""
        expected_stats = {
            "total_syntheses": 500,
            "average_quality_score": 0.85,
            "processing_time_avg_ms": 1200,
            "system_health": "healthy",
        }
        synthesis_interface.get_synthesis_stats.return_value = expected_stats

        result = await synthesis_interface.get_synthesis_stats()

        synthesis_interface.get_synthesis_stats.assert_called_once()
        assert result == expected_stats
        assert isinstance(result, dict)


class TestSynthesisDataClasses:
    """Test synthesis data classes and enums"""

    def test_synthesis_mode_enum(self):
        """Test SynthesisMode enum values"""
        assert SynthesisMode.EXTRACTIVE.value == "extractive"
        assert SynthesisMode.ABSTRACTIVE.value == "abstractive"
        assert SynthesisMode.HYBRID.value == "hybrid"
        assert SynthesisMode.CREATIVE.value == "creative"
        assert SynthesisMode.TECHNICAL.value == "technical"

    def test_content_format_enum(self):
        """Test ContentFormat enum values"""
        assert ContentFormat.PLAIN_TEXT.value == "plain_text"
        assert ContentFormat.MARKDOWN.value == "markdown"
        assert ContentFormat.HTML.value == "html"
        assert ContentFormat.JSON.value == "json"
        assert ContentFormat.STRUCTURED.value == "structured"

    def test_quality_metric_enum(self):
        """Test QualityMetric enum values"""
        assert QualityMetric.COHERENCE.value == "coherence"
        assert QualityMetric.RELEVANCE.value == "relevance"
        assert QualityMetric.COMPLETENESS.value == "completeness"
        assert QualityMetric.ACCURACY.value == "accuracy"
        assert QualityMetric.CLARITY.value == "clarity"

    def test_synthesis_context_creation(self):
        """Test SynthesisContext dataclass creation"""
        context = SynthesisContext(
            user_id="test", target_audience="technical", quality_requirements={QualityMetric.ACCURACY: 0.9}
        )

        assert context.user_id == "test"
        assert context.target_audience == "technical"
        assert context.quality_requirements[QualityMetric.ACCURACY] == 0.9
        assert context.session_id is None  # Default value

    def test_content_source_creation(self, sample_content_source):
        """Test ContentSource dataclass creation"""
        source = sample_content_source

        assert source.source_id == "source_1"
        assert source.content == "Test content from source"
        assert source.source_type == "article"
        assert source.reliability_score == 0.8
        assert source.relevance_score == 0.9
        assert isinstance(source.metadata, dict)

    def test_quality_assessment_creation(self, sample_quality_assessment):
        """Test QualityAssessment dataclass creation"""
        assessment = sample_quality_assessment

        assert assessment.overall_score == 0.85
        assert isinstance(assessment.metric_scores, dict)
        assert assessment.issues_identified == ["Minor coherence gap"]
        assert assessment.improvement_suggestions == ["Add transitional sentences"]
        assert assessment.confidence == 0.7

    def test_synthesis_result_creation(self, sample_quality_assessment):
        """Test SynthesisResult dataclass creation"""
        result = SynthesisResult(
            content="Test content",
            format=ContentFormat.MARKDOWN,
            sources_used=["source1"],
            synthesis_mode=SynthesisMode.HYBRID,
            quality_assessment=sample_quality_assessment,
            alternative_versions=["Alt version"],
            metadata={"key": "value"},
        )

        assert result.content == "Test content"
        assert result.format == ContentFormat.MARKDOWN
        assert result.sources_used == ["source1"]
        assert result.synthesis_mode == SynthesisMode.HYBRID
        assert result.quality_assessment == sample_quality_assessment
        assert result.alternative_versions == ["Alt version"]
        assert result.metadata == {"key": "value"}


@pytest.mark.parametrize(
    "mode",
    [
        SynthesisMode.EXTRACTIVE,
        SynthesisMode.ABSTRACTIVE,
        SynthesisMode.HYBRID,
        SynthesisMode.CREATIVE,
        SynthesisMode.TECHNICAL,
    ],
)
def test_synthesis_modes_parametrized(mode):
    """Parametrized test for all synthesis modes"""
    assert isinstance(mode, SynthesisMode)
    assert isinstance(mode.value, str)


@pytest.mark.parametrize(
    "format_type",
    [
        ContentFormat.PLAIN_TEXT,
        ContentFormat.MARKDOWN,
        ContentFormat.HTML,
        ContentFormat.JSON,
        ContentFormat.STRUCTURED,
    ],
)
def test_content_formats_parametrized(format_type):
    """Parametrized test for all content formats"""
    assert isinstance(format_type, ContentFormat)
    assert isinstance(format_type.value, str)


@pytest.mark.parametrize(
    "metric",
    [
        QualityMetric.COHERENCE,
        QualityMetric.RELEVANCE,
        QualityMetric.COMPLETENESS,
        QualityMetric.ACCURACY,
        QualityMetric.CLARITY,
    ],
)
def test_quality_metrics_parametrized(metric):
    """Parametrized test for all quality metrics"""
    assert isinstance(metric, QualityMetric)
    assert isinstance(metric.value, str)
