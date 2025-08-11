"""Enhanced Query Processing System with Multi-Level Context-Aware Matching.

This system enhances query processing to leverage the intelligent chunking system,
contextual tagging, and Bayesian trust graph to provide sophisticated query
understanding and retrieval with respect for idea boundaries and trust weighting.

Features:
- Query decomposition with intent, complexity, and context analysis
- Multi-level matching strategy (Document -> Chunk -> Graph traversal)
- Context-aware ranking with trust scores and idea completeness
- Answer synthesis that respects idea boundaries and maintains context
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import time
from typing import Any

from bayesian_trust_graph import RelationshipType
from codex_rag_integration import RetrievalResult
from contextual_tagging import (
    ChunkContext,
    ChunkType,
    ContentDomain,
    DocumentContext,
    ReadingLevel,
)

# Import existing RAG components
from graph_enhanced_rag_pipeline import GraphEnhancedRAGPipeline

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Primary intent classification for queries."""

    FACTUAL = "factual"  # Direct fact lookup
    EXPLANATORY = "explanatory"  # How/why explanations
    COMPARATIVE = "comparative"  # Compare/contrast
    ANALYTICAL = "analytical"  # Complex analysis
    PROCEDURAL = "procedural"  # Step-by-step processes
    TEMPORAL = "temporal"  # Time-based queries
    CAUSAL = "causal"  # Cause-effect relationships
    HYPOTHETICAL = "hypothetical"  # What-if scenarios


class ContextLevel(Enum):
    """Required context depth for query response."""

    OVERVIEW = "overview"  # High-level summary
    DETAILED = "detailed"  # In-depth information
    COMPREHENSIVE = "comprehensive"  # Complete coverage
    SPECIFIC = "specific"  # Targeted details


class TemporalRequirement(Enum):
    """Temporal context requirements."""

    HISTORICAL = "historical"  # Past events/information
    CURRENT = "current"  # Present-day information
    PREDICTIVE = "predictive"  # Future trends/projections
    TIMELESS = "timeless"  # Time-independent facts


class ComplexityLevel(Enum):
    """Query complexity classification."""

    SIMPLE = "simple"  # Single concept queries
    MODERATE = "moderate"  # Multiple related concepts
    COMPLEX = "complex"  # Multi-faceted analysis
    EXPERT = "expert"  # Advanced technical content


@dataclass
class QueryDecomposition:
    """Comprehensive query analysis and decomposition."""

    # Core query attributes
    original_query: str
    normalized_query: str
    query_tokens: list[str]

    # Intent analysis
    primary_intent: QueryIntent
    secondary_intents: list[QueryIntent] = field(default_factory=list)
    intent_confidence: float = 0.0

    # Context requirements
    context_level: ContextLevel = ContextLevel.DETAILED
    temporal_requirement: TemporalRequirement = TemporalRequirement.TIMELESS
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE

    # Domain and audience targeting
    target_domains: list[ContentDomain] = field(default_factory=list)
    reading_level: ReadingLevel | None = None
    audience_type: str | None = None

    # Query structure analysis
    key_concepts: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # Processing hints
    requires_multi_hop: bool = False
    needs_synthesis: bool = False
    idea_boundary_sensitive: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_query": self.original_query,
            "primary_intent": self.primary_intent.value,
            "context_level": self.context_level.value,
            "complexity_level": self.complexity_level.value,
            "temporal_requirement": self.temporal_requirement.value,
            "target_domains": [d.value for d in self.target_domains],
            "key_concepts": self.key_concepts,
            "requires_multi_hop": self.requires_multi_hop,
            "needs_synthesis": self.needs_synthesis,
        }


@dataclass
class RetrievalStrategy:
    """Multi-level retrieval strategy configuration."""

    # Level 1: Document-level matching
    document_filters: dict[str, Any] = field(default_factory=dict)
    domain_weights: dict[ContentDomain, float] = field(default_factory=dict)
    credibility_threshold: float = 0.5

    # Level 2: Chunk-level matching
    chunk_filters: dict[str, Any] = field(default_factory=dict)
    chunk_type_preferences: dict[ChunkType, float] = field(default_factory=dict)
    idea_boundary_respect: bool = True

    # Graph traversal
    enable_graph_traversal: bool = True
    max_traversal_depth: int = 2
    relationship_weights: dict[RelationshipType, float] = field(default_factory=dict)
    trust_propagation: bool = True

    # Ranking weights
    semantic_weight: float = 0.4
    trust_weight: float = 0.3
    context_weight: float = 0.2
    recency_weight: float = 0.1


@dataclass
class RankedResult:
    """Enhanced retrieval result with comprehensive scoring."""

    # Base result
    result: RetrievalResult

    # Scoring breakdown
    semantic_score: float
    trust_score: float
    context_score: float
    recency_score: float
    idea_completeness_score: float
    final_score: float

    # Context information
    document_context: DocumentContext | None = None
    chunk_context: ChunkContext | None = None
    trust_lineage: list[str] = field(default_factory=list)
    idea_boundary_complete: bool = True

    # Source attribution
    source_citations: list[dict[str, Any]] = field(default_factory=list)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class SynthesizedAnswer:
    """Synthesized answer with context preservation and trust attribution."""

    # Answer content
    answer_text: str
    executive_summary: str
    synthesis_method: str

    # Quality metrics (required)
    overall_confidence: float
    trust_weighted_confidence: float
    completeness_score: float
    coherence_score: float
    processing_time_ms: float

    # Optional fields with defaults
    detailed_sections: list[dict[str, Any]] = field(default_factory=list)
    primary_sources: list[RankedResult] = field(default_factory=list)
    supporting_sources: list[RankedResult] = field(default_factory=list)
    conflicting_sources: list[RankedResult] = field(default_factory=list)
    preserved_idea_boundaries: list[dict[str, Any]] = field(default_factory=list)
    context_chain: list[dict[str, Any]] = field(default_factory=list)
    narrative_flow_score: float = 0.0
    query_decomposition: QueryDecomposition | None = None


class EnhancedQueryProcessor:
    """Enhanced query processing system that leverages intelligent chunking,
    contextual tagging, and Bayesian trust graphs for sophisticated
    query understanding and response generation.
    """

    def __init__(
        self,
        rag_pipeline: GraphEnhancedRAGPipeline,
        enable_query_expansion: bool = True,
        enable_intent_classification: bool = True,
        enable_multi_hop_reasoning: bool = True,
        default_result_limit: int = 10,
    ) -> None:
        """Initialize enhanced query processor."""
        self.rag_pipeline = rag_pipeline
        self.enable_query_expansion = enable_query_expansion
        self.enable_intent_classification = enable_intent_classification
        self.enable_multi_hop_reasoning = enable_multi_hop_reasoning
        self.default_result_limit = default_result_limit

        # Intent classification patterns
        self.intent_patterns = self._build_intent_patterns()

        # Context level patterns
        self.context_patterns = self._build_context_patterns()

        # Complexity indicators
        self.complexity_indicators = self._build_complexity_indicators()

        # Temporal patterns
        self.temporal_patterns = self._build_temporal_patterns()

        # Processing statistics
        self.processing_stats = {
            "queries_processed": 0,
            "avg_decomposition_time": 0.0,
            "avg_retrieval_time": 0.0,
            "avg_synthesis_time": 0.0,
            "intent_classification_accuracy": 0.0,
            "multi_hop_queries": 0,
            "synthesis_success_rate": 0.0,
        }

    async def process_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        custom_strategy: RetrievalStrategy | None = None,
    ) -> SynthesizedAnswer:
        """Process query with full enhancement pipeline.

        Args:
            query: User query string
            context: Optional context information
            custom_strategy: Custom retrieval strategy

        Returns:
            Synthesized answer with comprehensive context
        """
        start_time = time.perf_counter()
        context = context or {}

        logger.info(f"Processing enhanced query: '{query[:100]}...'")

        try:
            # Step 1: Query Decomposition
            decomposition_start = time.perf_counter()
            decomposition = await self.decompose_query(query, context)
            decomposition_time = (time.perf_counter() - decomposition_start) * 1000

            logger.debug(f"Query decomposition completed in {decomposition_time:.1f}ms")
            logger.debug(
                f"Intent: {decomposition.primary_intent.value}, " f"Complexity: {decomposition.complexity_level.value}"
            )

            # Step 2: Create Retrieval Strategy
            strategy = custom_strategy or self._create_retrieval_strategy(decomposition)

            # Step 3: Multi-Level Matching
            retrieval_start = time.perf_counter()
            ranked_results = await self.multi_level_matching(decomposition, strategy, context)
            retrieval_time = (time.perf_counter() - retrieval_start) * 1000

            logger.debug(f"Multi-level retrieval completed in {retrieval_time:.1f}ms")
            logger.debug(f"Retrieved {len(ranked_results)} ranked results")

            # Step 4: Answer Synthesis
            synthesis_start = time.perf_counter()
            synthesized_answer = await self.synthesize_answer(decomposition, ranked_results, strategy)
            synthesis_time = (time.perf_counter() - synthesis_start) * 1000

            logger.debug(f"Answer synthesis completed in {synthesis_time:.1f}ms")

            # Update processing metadata
            total_time = (time.perf_counter() - start_time) * 1000
            synthesized_answer.processing_time_ms = total_time
            synthesized_answer.query_decomposition = decomposition

            # Update statistics
            self._update_processing_stats(decomposition_time, retrieval_time, synthesis_time, decomposition)

            logger.info(f"Enhanced query processing completed in {total_time:.1f}ms")

            return synthesized_answer

        except Exception as e:
            logger.exception(f"Enhanced query processing failed: {e}")

            # Return fallback answer
            return SynthesizedAnswer(
                answer_text=f"I encountered an error processing your query: {e!s}",
                executive_summary="Error occurred during processing",
                overall_confidence=0.0,
                trust_weighted_confidence=0.0,
                completeness_score=0.0,
                coherence_score=0.0,
                synthesis_method="fallback",
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def decompose_query(self, query: str, context: dict[str, Any]) -> QueryDecomposition:
        """Analyze and decompose query into components for processing strategy.

        Args:
            query: Original query string
            context: Additional context information

        Returns:
            Comprehensive query decomposition
        """
        # Normalize query
        normalized_query = self._normalize_query(query)
        query_tokens = normalized_query.split()

        # Extract key concepts
        key_concepts = self._extract_key_concepts(normalized_query)

        # Classify intent
        primary_intent, secondary_intents, intent_confidence = self._classify_intent(normalized_query)

        # Determine context level
        context_level = self._determine_context_level(normalized_query, context)

        # Analyze temporal requirements
        temporal_requirement = self._analyze_temporal_requirements(normalized_query)

        # Assess complexity
        complexity_level = self._assess_complexity(normalized_query, key_concepts)

        # Extract entities and relationships
        entities = self._extract_entities(normalized_query)
        relationships = self._extract_relationships(normalized_query)

        # Determine domain targeting
        target_domains = self._determine_target_domains(normalized_query, context)

        # Analyze processing requirements
        requires_multi_hop = self._requires_multi_hop_reasoning(normalized_query)
        needs_synthesis = self._needs_synthesis(normalized_query, complexity_level)

        return QueryDecomposition(
            original_query=query,
            normalized_query=normalized_query,
            query_tokens=query_tokens,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            intent_confidence=intent_confidence,
            context_level=context_level,
            temporal_requirement=temporal_requirement,
            complexity_level=complexity_level,
            target_domains=target_domains,
            key_concepts=key_concepts,
            entities=entities,
            relationships=relationships,
            requires_multi_hop=requires_multi_hop,
            needs_synthesis=needs_synthesis,
        )

    async def multi_level_matching(
        self,
        decomposition: QueryDecomposition,
        strategy: RetrievalStrategy,
        context: dict[str, Any],
    ) -> list[RankedResult]:
        """Perform multi-level matching strategy with context-aware ranking.

        Args:
            decomposition: Query decomposition
            strategy: Retrieval strategy configuration
            context: Processing context

        Returns:
            List of ranked results with comprehensive scoring
        """
        logger.debug("Starting multi-level matching")

        # Level 1: Document-level filtering and matching
        document_candidates = await self._level1_document_matching(decomposition, strategy, context)

        logger.debug(f"Level 1: Found {len(document_candidates)} document candidates")

        # Level 2: Chunk-level matching with idea boundary respect
        chunk_results = await self._level2_chunk_matching(decomposition, strategy, document_candidates)

        logger.debug(f"Level 2: Found {len(chunk_results)} chunk results")

        # Level 3: Graph traversal for supporting/contrasting information
        if strategy.enable_graph_traversal and decomposition.requires_multi_hop:
            graph_results = await self._level3_graph_traversal(decomposition, strategy, chunk_results)
            chunk_results.extend(graph_results)

            logger.debug(f"Level 3: Added {len(graph_results)} graph traversal results")

        # Context-aware ranking
        ranked_results = await self._context_aware_ranking(decomposition, strategy, chunk_results)

        logger.debug(f"Final ranking: {len(ranked_results)} results")

        return ranked_results

    async def synthesize_answer(
        self,
        decomposition: QueryDecomposition,
        ranked_results: list[RankedResult],
        strategy: RetrievalStrategy,
    ) -> SynthesizedAnswer:
        """Synthesize comprehensive answer while respecting idea boundaries.

        Args:
            decomposition: Query decomposition
            strategy: Retrieval strategy
            ranked_results: Ranked retrieval results

        Returns:
            Synthesized answer with context preservation
        """
        if not ranked_results:
            return SynthesizedAnswer(
                answer_text="I couldn't find relevant information to answer your query.",
                executive_summary="No relevant results found",
                overall_confidence=0.0,
                trust_weighted_confidence=0.0,
                completeness_score=0.0,
                coherence_score=0.0,
                synthesis_method="empty_results",
                processing_time_ms=0.0,
            )

        # Separate sources by quality and relevance
        primary_sources = ranked_results[:3]  # Top 3 results
        supporting_sources = ranked_results[3:7]  # Next 4 results
        conflicting_sources = []  # TODO: Implement conflict detection

        # Generate executive summary
        executive_summary = await self._generate_executive_summary(decomposition, primary_sources)

        # Create detailed sections respecting idea boundaries
        detailed_sections = await self._create_detailed_sections(decomposition, ranked_results, strategy)

        # Synthesize main answer text
        answer_text = await self._synthesize_main_answer(decomposition, primary_sources, supporting_sources)

        # Calculate quality metrics
        quality_metrics = await self._calculate_synthesis_quality(decomposition, ranked_results, answer_text)

        # Preserve context chain and idea boundaries
        context_chain = self._build_context_chain(ranked_results)
        preserved_boundaries = self._identify_preserved_boundaries(ranked_results)

        return SynthesizedAnswer(
            answer_text=answer_text,
            executive_summary=executive_summary,
            detailed_sections=detailed_sections,
            primary_sources=primary_sources,
            supporting_sources=supporting_sources,
            conflicting_sources=conflicting_sources,
            overall_confidence=quality_metrics["overall_confidence"],
            trust_weighted_confidence=quality_metrics["trust_weighted_confidence"],
            completeness_score=quality_metrics["completeness_score"],
            coherence_score=quality_metrics["coherence_score"],
            preserved_idea_boundaries=preserved_boundaries,
            context_chain=context_chain,
            narrative_flow_score=quality_metrics["narrative_flow_score"],
            synthesis_method="multi_level_enhanced",
            processing_time_ms=0.0,  # Will be updated by caller
        )

    def _build_intent_patterns(self) -> dict[QueryIntent, list[str]]:
        """Build patterns for intent classification."""
        return {
            QueryIntent.FACTUAL: [
                r"\bwhat is\b",
                r"\bwho is\b",
                r"\bwhere is\b",
                r"\bwhen is\b",
                r"\bdefine\b",
                r"\bdefinition\b",
                r"\bmeans\b",
                r"\brefers to\b",
            ],
            QueryIntent.EXPLANATORY: [
                r"\bhow does\b",
                r"\bhow to\b",
                r"\bwhy does\b",
                r"\bexplain\b",
                r"\bdescribe\b",
                r"\bprocess\b",
                r"\bmechanism\b",
                r"\bwork\b",
            ],
            QueryIntent.COMPARATIVE: [
                r"\bcompare\b",
                r"\bversus\b",
                r"\bvs\b",
                r"\bdifference\b",
                r"\bbetter\b",
                r"\bworse\b",
                r"\bsimilar\b",
                r"\bunlike\b",
            ],
            QueryIntent.ANALYTICAL: [
                r"\banalyze\b",
                r"\bevaluate\b",
                r"\bassess\b",
                r"\bexamine\b",
                r"\bimpact\b",
                r"\beffect\b",
                r"\bimplications\b",
                r"\bconsequences\b",
            ],
            QueryIntent.PROCEDURAL: [
                r"\bsteps\b",
                r"\bprocedure\b",
                r"\bmethod\b",
                r"\bguide\b",
                r"\binstructions\b",
                r"\bhow to\b",
                r"\bprocess\b",
            ],
            QueryIntent.TEMPORAL: [
                r"\bwhen\b",
                r"\btimeline\b",
                r"\bhistory\b",
                r"\bevolution\b",
                r"\bchronology\b",
                r"\bover time\b",
                r"\btrend\b",
            ],
            QueryIntent.CAUSAL: [
                r"\bcause\b",
                r"\breason\b",
                r"\bwhy\b",
                r"\blead to\b",
                r"\bresult\b",
                r"\bdue to\b",
                r"\bbecause\b",
                r"\btrigger\b",
            ],
            QueryIntent.HYPOTHETICAL: [
                r"\bwhat if\b",
                r"\bsuppose\b",
                r"\bhypothetical\b",
                r"\bscenario\b",
                r"\bimagine\b",
                r"\bassume\b",
                r"\bwould happen\b",
            ],
        }

    def _build_context_patterns(self) -> dict[ContextLevel, list[str]]:
        """Build patterns for context level determination."""
        return {
            ContextLevel.OVERVIEW: [
                r"\boverview\b",
                r"\bsummary\b",
                r"\bbrief\b",
                r"\bgeneral\b",
                r"\bbasics\b",
                r"\bintroduction\b",
                r"\bquick\b",
            ],
            ContextLevel.DETAILED: [
                r"\bdetailed\b",
                r"\bspecific\b",
                r"\bparticular\b",
                r"\bexact\b",
                r"\bprecise\b",
                r"\bthorough\b",
                r"\bin depth\b",
            ],
            ContextLevel.COMPREHENSIVE: [
                r"\bcomprehensive\b",
                r"\bcomplete\b",
                r"\bexhaustive\b",
                r"\ball\b",
                r"\bentire\b",
                r"\bfull\b",
                r"\beverything\b",
            ],
            ContextLevel.SPECIFIC: [
                r"\bonly\b",
                r"\bjust\b",
                r"\bspecifically\b",
                r"\bparticular\b",
                r"\bfocus on\b",
                r"\blimited to\b",
                r"\bnarrow\b",
            ],
        }

    def _build_complexity_indicators(self) -> list[str]:
        """Build patterns indicating query complexity."""
        return [
            r"\band\b",
            r"\bor\b",
            r"\bbut\b",
            r"\bhowever\b",
            r"\balthough\b",
            r"\bmultiple\b",
            r"\bseveral\b",
            r"\bvarious\b",
            r"\bcomplex\b",
            r"\badvanced\b",
            r"\bsophisticated\b",
            r"\bintricate\b",
        ]

    def _build_temporal_patterns(self) -> dict[TemporalRequirement, list[str]]:
        """Build patterns for temporal requirement analysis."""
        return {
            TemporalRequirement.HISTORICAL: [
                r"\bhistory\b",
                r"\bhistorical\b",
                r"\bpast\b",
                r"\boriginal\b",
                r"\bearlier\b",
                r"\bprevious\b",
                r"\btraditional\b",
                r"\bwas\b",
            ],
            TemporalRequirement.CURRENT: [
                r"\bcurrent\b",
                r"\bnow\b",
                r"\btoday\b",
                r"\bpresent\b",
                r"\brecent\b",
                r"\blatest\b",
                r"\bmodern\b",
                r"\bcontemporary\b",
            ],
            TemporalRequirement.PREDICTIVE: [
                r"\bfuture\b",
                r"\bpredict\b",
                r"\bforecast\b",
                r"\btrend\b",
                r"\bexpected\b",
                r"\bwill\b",
                r"\bprojected\b",
                r"\bnext\b",
            ],
            TemporalRequirement.TIMELESS: [
                r"\balways\b",
                r"\bnever\b",
                r"\buniversal\b",
                r"\bgeneral\b",
                r"\bprinciple\b",
                r"\blaw\b",
                r"\brule\b",
                r"\bconstant\b",
            ],
        }

    async def _level1_document_matching(
        self,
        decomposition: QueryDecomposition,
        strategy: RetrievalStrategy,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Perform Level 1 document-level matching."""
        # Use the existing pipeline to get document-level context
        if self.rag_pipeline.document_contexts:
            # Filter documents by domain, credibility, and relevance
            candidates = []

            for doc_id, doc_context in self.rag_pipeline.document_contexts.items():
                # Domain filtering
                if decomposition.target_domains:
                    if doc_context.domain not in decomposition.target_domains:
                        continue

                # Credibility filtering
                if doc_context.source_credibility_score < strategy.credibility_threshold:
                    continue

                # Reading level matching
                if decomposition.reading_level:
                    if doc_context.reading_level != decomposition.reading_level:
                        continue

                # Temporal matching
                temporal_match = self._assess_temporal_match(doc_context, decomposition.temporal_requirement)
                if temporal_match < 0.3:  # Minimum temporal relevance
                    continue

                candidates.append(
                    {
                        "document_id": doc_id,
                        "document_context": doc_context,
                        "temporal_match": temporal_match,
                    }
                )

            logger.debug(f"Document-level filtering: {len(candidates)} candidates")
            return candidates

        return []

    async def _level2_chunk_matching(
        self,
        decomposition: QueryDecomposition,
        strategy: RetrievalStrategy,
        document_candidates: list[dict[str, Any]],
    ) -> list[RankedResult]:
        """Perform Level 2 chunk-level matching with idea boundary respect."""
        # Extract document IDs from candidates
        candidate_doc_ids = {d["document_id"] for d in document_candidates}

        # Perform retrieval with contextual analysis
        results, metrics = await self.rag_pipeline.retrieve_with_contextual_analysis(
            query=decomposition.normalized_query,
            k=self.default_result_limit * 2,  # Get extra for filtering
            domain_filter=(decomposition.target_domains[0] if decomposition.target_domains else None),
            reading_level_filter=decomposition.reading_level,
            min_credibility=strategy.credibility_threshold,
            context_similarity_boost=0.2,
        )

        # Filter to candidate documents and create ranked results
        ranked_results = []

        for result in results:
            if result.document_id not in candidate_doc_ids:
                continue

            # Get document candidate info
            doc_candidate = next(
                (d for d in document_candidates if d["document_id"] == result.document_id),
                None,
            )

            if not doc_candidate:
                continue

            # Create comprehensive ranked result
            ranked_result = RankedResult(
                result=result,
                semantic_score=result.score,
                trust_score=(result.metadata.get("trust_score", 0.5) if result.metadata else 0.5),
                context_score=doc_candidate["temporal_match"],
                recency_score=self._calculate_recency_score(result),
                idea_completeness_score=self._assess_idea_completeness(result),
                final_score=0.0,  # Will be calculated in ranking
                document_context=doc_candidate["document_context"],
                idea_boundary_complete=True,  # Assume true for intelligent chunks
            )

            ranked_results.append(ranked_result)

        return ranked_results

    async def _level3_graph_traversal(
        self,
        decomposition: QueryDecomposition,
        strategy: RetrievalStrategy,
        initial_results: list[RankedResult],
    ) -> list[RankedResult]:
        """Perform Level 3 graph traversal for supporting/contrasting information."""
        if not self.rag_pipeline.trust_graph:
            return []

        graph_results = []

        # Get top initial results as seeds for traversal
        seed_chunks = [r.result.chunk_id for r in initial_results[:3]]

        for seed_chunk_id in seed_chunks:
            if seed_chunk_id not in self.rag_pipeline.trust_graph.chunk_nodes:
                continue

            # Analyze relationships for this chunk
            analysis = self.rag_pipeline.analyze_graph_relationships(seed_chunk_id)

            if "error" in analysis:
                continue

            relationships = analysis["graph_analysis"]["relationships"]

            # Traverse outgoing relationships
            for rel in relationships["outgoing"]:
                target_chunk = rel["target_chunk"]
                rel_type = RelationshipType(rel["relationship_type"])

                # Filter by relationship relevance to query intent
                if not self._is_relationship_relevant(rel_type, decomposition.primary_intent):
                    continue

                # Get chunk information
                if target_chunk in self.rag_pipeline.trust_graph.chunk_nodes:
                    chunk_node = self.rag_pipeline.trust_graph.chunk_nodes[target_chunk]

                    # Create result from graph traversal
                    traversal_result = RetrievalResult(
                        chunk_id=chunk_node.chunk_id,
                        document_id=chunk_node.document_id,
                        text=chunk_node.text,
                        score=chunk_node.trust_score * rel["confidence"],
                        retrieval_method="graph_traversal",
                        metadata={
                            "traversal_source": seed_chunk_id,
                            "relationship_type": rel_type.value,
                            "relationship_confidence": rel["confidence"],
                            "trust_score": chunk_node.trust_score,
                            "centrality_score": chunk_node.centrality_score,
                        },
                    )

                    ranked_result = RankedResult(
                        result=traversal_result,
                        semantic_score=rel["confidence"],
                        trust_score=chunk_node.trust_score,
                        context_score=0.8,  # Graph context is high
                        recency_score=0.5,  # Neutral recency
                        idea_completeness_score=0.9,  # Graph chunks are complete
                        final_score=0.0,
                        idea_boundary_complete=True,
                    )

                    graph_results.append(ranked_result)

        return graph_results

    async def _context_aware_ranking(
        self,
        decomposition: QueryDecomposition,
        strategy: RetrievalStrategy,
        results: list[RankedResult],
    ) -> list[RankedResult]:
        """Apply context-aware ranking with trust scores and idea completeness."""
        for result in results:
            # Calculate weighted final score
            final_score = (
                strategy.semantic_weight * result.semantic_score
                + strategy.trust_weight * result.trust_score
                + strategy.context_weight * result.context_score
                + strategy.recency_weight * result.recency_score
            )

            # Apply idea completeness bonus
            if result.idea_boundary_complete:
                final_score *= 1.1  # 10% bonus for complete ideas

            # Apply complexity matching bonus
            complexity_bonus = self._calculate_complexity_bonus(decomposition, result)
            final_score += complexity_bonus

            result.final_score = final_score

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results

    # Helper methods for various processing steps
    def _normalize_query(self, query: str) -> str:
        """Normalize query for processing."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        return normalized

    def _extract_key_concepts(self, query: str) -> list[str]:
        """Extract key concepts from query."""
        # Simple extraction - remove stop words and get important terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }
        words = [w for w in query.split() if w not in stop_words and len(w) > 2]
        return words[:5]  # Return top 5 key concepts

    def _classify_intent(self, query: str) -> tuple[QueryIntent, list[QueryIntent], float]:
        """Classify query intent using pattern matching."""
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches

            if score > 0:
                intent_scores[intent] = score / len(patterns)

        if not intent_scores:
            return QueryIntent.FACTUAL, [], 0.5  # Default

        # Get primary intent (highest score)
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])

        # Get secondary intents (score > 0.3)
        secondary_intents = [
            intent for intent, score in intent_scores.items() if score > 0.3 and intent != primary_intent[0]
        ]

        return primary_intent[0], secondary_intents, primary_intent[1]

    def _determine_context_level(self, query: str, context: dict[str, Any]) -> ContextLevel:
        """Determine required context level."""
        for level, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return level

        # Default based on query length
        if len(query.split()) > 15:
            return ContextLevel.COMPREHENSIVE
        if len(query.split()) > 8:
            return ContextLevel.DETAILED
        return ContextLevel.OVERVIEW

    def _analyze_temporal_requirements(self, query: str) -> TemporalRequirement:
        """Analyze temporal requirements of query."""
        for requirement, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return requirement

        return TemporalRequirement.TIMELESS  # Default

    def _assess_complexity(self, query: str, key_concepts: list[str]) -> ComplexityLevel:
        """Assess query complexity."""
        complexity_score = 0

        # Check for complexity indicators
        for indicator in self.complexity_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                complexity_score += 1

        # Factor in concept count and query length
        word_count = len(query.split())
        concept_count = len(key_concepts)

        if complexity_score >= 3 or word_count > 20 or concept_count > 4:
            return ComplexityLevel.EXPERT
        if complexity_score >= 2 or word_count > 15 or concept_count > 3:
            return ComplexityLevel.COMPLEX
        if complexity_score >= 1 or word_count > 10 or concept_count > 2:
            return ComplexityLevel.MODERATE
        return ComplexityLevel.SIMPLE

    def _extract_entities(self, query: str) -> list[dict[str, Any]]:
        """Extract entities from query (simplified)."""
        # This would integrate with NER in production
        entities = []

        # Simple capitalized word detection
        words = re.findall(r"\b[A-Z][a-zA-Z]+\b", query)
        for word in words:
            entities.append({"text": word, "type": "PROPER_NOUN", "confidence": 0.7})

        return entities

    def _extract_relationships(self, query: str) -> list[str]:
        """Extract relationship indicators from query."""
        relationship_patterns = [
            r"caused by",
            r"leads to",
            r"results in",
            r"related to",
            r"associated with",
            r"similar to",
            r"different from",
        ]

        relationships = []
        for pattern in relationship_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                relationships.append(pattern)

        return relationships

    def _determine_target_domains(self, query: str, context: dict[str, Any]) -> list[ContentDomain]:
        """Determine target domains for query."""
        domain_keywords = {
            ContentDomain.SCIENCE: [
                "science",
                "research",
                "study",
                "experiment",
                "theory",
            ],
            ContentDomain.TECHNOLOGY: [
                "technology",
                "computer",
                "software",
                "digital",
                "AI",
            ],
            ContentDomain.GENERAL: ["general", "basic", "common", "everyday"],
        }

        detected_domains = []
        query_lower = query.lower()

        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)

        # Default to GENERAL if no specific domain detected
        return detected_domains or [ContentDomain.GENERAL]

    def _requires_multi_hop_reasoning(self, query: str) -> bool:
        """Determine if query requires multi-hop reasoning."""
        multi_hop_indicators = [
            r"relationship between",
            r"connection",
            r"impact of.*on",
            r"how.*affect",
            r"chain",
            r"sequence",
            r"leads to",
        ]

        for indicator in multi_hop_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return True

        return False

    def _needs_synthesis(self, query: str, complexity: ComplexityLevel) -> bool:
        """Determine if query needs answer synthesis."""
        synthesis_indicators = [
            r"compare",
            r"analyze",
            r"evaluate",
            r"explain",
            r"describe",
            r"summarize",
            r"overview",
        ]

        has_synthesis_terms = any(re.search(indicator, query, re.IGNORECASE) for indicator in synthesis_indicators)

        return has_synthesis_terms or complexity in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        ]

    def _create_retrieval_strategy(self, decomposition: QueryDecomposition) -> RetrievalStrategy:
        """Create retrieval strategy based on query decomposition."""
        # Base strategy
        strategy = RetrievalStrategy()

        # Adjust weights based on intent
        if decomposition.primary_intent == QueryIntent.FACTUAL:
            strategy.semantic_weight = 0.5
            strategy.trust_weight = 0.4
        elif decomposition.primary_intent == QueryIntent.EXPLANATORY:
            strategy.semantic_weight = 0.4
            strategy.context_weight = 0.3
        elif decomposition.primary_intent == QueryIntent.COMPARATIVE:
            strategy.enable_graph_traversal = True
            strategy.max_traversal_depth = 3

        # Adjust for complexity
        if decomposition.complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        ]:
            strategy.enable_graph_traversal = True
            strategy.trust_weight = 0.4  # Higher trust for complex queries

        # Adjust for temporal requirements
        if decomposition.temporal_requirement == TemporalRequirement.CURRENT:
            strategy.recency_weight = 0.2
        elif decomposition.temporal_requirement == TemporalRequirement.HISTORICAL:
            strategy.recency_weight = 0.05

        return strategy

    def _assess_temporal_match(self, doc_context: DocumentContext, requirement: TemporalRequirement) -> float:
        """Assess how well document matches temporal requirement."""
        # Simplified temporal matching
        # In production, this would analyze document dates, content temporal indicators
        return 0.7  # Default moderate match

    def _calculate_recency_score(self, result: RetrievalResult) -> float:
        """Calculate recency score for result."""
        # Simplified recency scoring
        # In production, this would analyze document dates and freshness
        return 0.5  # Default neutral recency

    def _assess_idea_completeness(self, result: RetrievalResult) -> float:
        """Assess if result contains complete ideas."""
        # Check if result was created by intelligent chunking
        if result.metadata and result.metadata.get("chunking_method") == "intelligent":
            return 0.9  # High completeness for intelligent chunks
        return 0.6  # Moderate completeness for traditional chunks

    def _is_relationship_relevant(self, rel_type: RelationshipType, intent: QueryIntent) -> bool:
        """Check if relationship type is relevant to query intent."""
        relevance_map = {
            QueryIntent.EXPLANATORY: [
                RelationshipType.ELABORATES,
                RelationshipType.DEFINES,
                RelationshipType.EXEMPLIFIES,
            ],
            QueryIntent.COMPARATIVE: [
                RelationshipType.CONTRASTS,
                RelationshipType.SUPPORTS,
            ],
            QueryIntent.CAUSAL: [RelationshipType.SUPPORTS, RelationshipType.CONTINUES],
            QueryIntent.ANALYTICAL: [
                RelationshipType.REFERENCES,
                RelationshipType.CONTEXTUALIZES,
            ],
        }

        relevant_types = relevance_map.get(intent, [])
        return rel_type in relevant_types

    def _calculate_complexity_bonus(self, decomposition: QueryDecomposition, result: RankedResult) -> float:
        """Calculate complexity matching bonus."""
        # Simple complexity bonus - would be more sophisticated in production
        if decomposition.complexity_level == ComplexityLevel.EXPERT:
            if result.document_context and result.document_context.reading_level.value == "graduate":
                return 0.1  # 10% bonus for graduate-level content
        elif decomposition.complexity_level == ComplexityLevel.SIMPLE and (
            result.document_context and result.document_context.reading_level.value == "high_school"
        ):
            return 0.05  # 5% bonus for high school level content

        return 0.0  # No bonus

    async def _generate_executive_summary(
        self, decomposition: QueryDecomposition, primary_sources: list[RankedResult]
    ) -> str:
        """Generate executive summary from primary sources."""
        if not primary_sources:
            return "No relevant information found."

        # Extract key points from top sources
        key_points = []
        for source in primary_sources[:3]:  # Top 3 sources
            text = source.result.text
            # Simple extraction - first sentence or first 100 characters
            summary = text[:100] + "..." if len(text) > 100 else text

            key_points.append(
                {
                    "text": summary,
                    "trust_score": source.trust_score,
                    "source": source.result.document_id,
                }
            )

        # Create executive summary
        if len(key_points) == 1:
            return f"Based on {key_points[0]['source']}: {key_points[0]['text']}"
        summary = f"Based on {len(key_points)} sources: "
        summary += " ".join([kp["text"] for kp in key_points])
        return summary

    async def _create_detailed_sections(
        self,
        decomposition: QueryDecomposition,
        results: list[RankedResult],
        strategy: RetrievalStrategy,
    ) -> list[dict[str, Any]]:
        """Create detailed sections respecting idea boundaries."""
        sections = []

        # Group results by document or topic
        for i, result in enumerate(results[:5]):  # Top 5 results
            section = {
                "section_id": f"section_{i}",
                "title": f"Source {i + 1}: {result.result.document_id}",
                "content": result.result.text,
                "trust_score": result.trust_score,
                "context": {
                    "document_context": (result.document_context.to_dict() if result.document_context else None),
                    "chunk_context": (result.chunk_context if result.chunk_context else None),
                },
                "idea_boundary_preserved": result.idea_boundary_complete,
                "source_citation": {
                    "document_id": result.result.document_id,
                    "chunk_id": result.result.chunk_id,
                    "retrieval_method": result.result.retrieval_method,
                },
            }

            sections.append(section)

        return sections

    async def _synthesize_main_answer(
        self,
        decomposition: QueryDecomposition,
        primary_sources: list[RankedResult],
        supporting_sources: list[RankedResult],
    ) -> str:
        """Synthesize main answer text."""
        if not primary_sources:
            return "I couldn't find sufficient information to answer your query."

        # Start with query acknowledgment
        answer_parts = [f"Based on the available information regarding '{decomposition.original_query}':"]

        # Add primary information
        for i, source in enumerate(primary_sources[:3]):
            trust_indicator = ""
            if source.trust_score > 0.8:
                trust_indicator = " (high confidence)"
            elif source.trust_score < 0.5:
                trust_indicator = " (moderate confidence)"

            answer_parts.append(f"\n\n{i + 1}. {source.result.text}{trust_indicator}")

        # Add supporting information if available
        if supporting_sources:
            answer_parts.append("\n\nAdditional supporting information:")
            for source in supporting_sources[:2]:  # Top 2 supporting sources
                answer_parts.append(f"\n• {source.result.text[:200]}...")

        return "".join(answer_parts)

    async def _calculate_synthesis_quality(
        self,
        decomposition: QueryDecomposition,
        results: list[RankedResult],
        answer_text: str,
    ) -> dict[str, float]:
        """Calculate quality metrics for synthesized answer."""
        if not results:
            return {
                "overall_confidence": 0.0,
                "trust_weighted_confidence": 0.0,
                "completeness_score": 0.0,
                "coherence_score": 0.0,
                "narrative_flow_score": 0.0,
            }

        # Calculate trust-weighted confidence
        total_trust = sum(r.trust_score for r in results[:5])  # Top 5
        avg_trust = total_trust / min(len(results), 5)

        # Overall confidence based on number and quality of sources
        source_count_factor = min(len(results) / 3, 1.0)  # Optimal at 3+ sources
        overall_confidence = avg_trust * source_count_factor

        # Completeness based on query complexity vs available information
        if decomposition.complexity_level == ComplexityLevel.SIMPLE:
            completeness_threshold = 2  # Need 2 sources
        elif decomposition.complexity_level == ComplexityLevel.MODERATE:
            completeness_threshold = 3  # Need 3 sources
        else:
            completeness_threshold = 5  # Need 5 sources

        completeness_score = min(len(results) / completeness_threshold, 1.0)

        # Coherence based on idea boundary preservation
        boundary_complete_count = sum(1 for r in results[:5] if r.idea_boundary_complete)
        coherence_score = boundary_complete_count / min(len(results), 5)

        # Narrative flow (simplified)
        narrative_flow_score = 0.8  # Assume good flow for synthesis method

        return {
            "overall_confidence": overall_confidence,
            "trust_weighted_confidence": avg_trust,
            "completeness_score": completeness_score,
            "coherence_score": coherence_score,
            "narrative_flow_score": narrative_flow_score,
        }

    def _build_context_chain(self, results: list[RankedResult]) -> list[dict[str, Any]]:
        """Build context chain showing information flow."""
        context_chain = []

        for i, result in enumerate(results[:5]):
            chain_link = {
                "step": i + 1,
                "document_id": result.result.document_id,
                "chunk_id": result.result.chunk_id,
                "trust_score": result.trust_score,
                "retrieval_method": result.result.retrieval_method,
                "context_contribution": f"Step {i + 1} information",
            }

            if result.result.metadata and "traversal_source" in result.result.metadata:
                chain_link["graph_traversal_from"] = result.result.metadata["traversal_source"]
                chain_link["relationship_type"] = result.result.metadata["relationship_type"]

            context_chain.append(chain_link)

        return context_chain

    def _identify_preserved_boundaries(self, results: list[RankedResult]) -> list[dict[str, Any]]:
        """Identify preserved idea boundaries in results."""
        boundaries = []

        for result in results:
            if result.idea_boundary_complete:
                boundary = {
                    "chunk_id": result.result.chunk_id,
                    "boundary_type": "complete_idea",
                    "confidence": 0.9,
                    "method": (
                        "intelligent_chunking"
                        if result.result.metadata and result.result.metadata.get("chunking_method") == "intelligent"
                        else "traditional_chunking"
                    ),
                }
                boundaries.append(boundary)

        return boundaries

    def _update_processing_stats(
        self,
        decomposition_time: float,
        retrieval_time: float,
        synthesis_time: float,
        decomposition: QueryDecomposition,
    ) -> None:
        """Update processing statistics."""
        self.processing_stats["queries_processed"] += 1

        # Update averages
        count = self.processing_stats["queries_processed"]

        self.processing_stats["avg_decomposition_time"] = (
            self.processing_stats["avg_decomposition_time"] * (count - 1) + decomposition_time
        ) / count

        self.processing_stats["avg_retrieval_time"] = (
            self.processing_stats["avg_retrieval_time"] * (count - 1) + retrieval_time
        ) / count

        self.processing_stats["avg_synthesis_time"] = (
            self.processing_stats["avg_synthesis_time"] * (count - 1) + synthesis_time
        ) / count

        # Track multi-hop queries
        if decomposition.requires_multi_hop:
            self.processing_stats["multi_hop_queries"] += 1

    def get_processing_statistics(self) -> dict[str, Any]:
        """Get comprehensive processing statistics."""
        multi_hop_rate = 0.0
        if self.processing_stats["queries_processed"] > 0:
            multi_hop_rate = self.processing_stats["multi_hop_queries"] / self.processing_stats["queries_processed"]

        return {
            "queries_processed": self.processing_stats["queries_processed"],
            "performance": {
                "avg_decomposition_time_ms": self.processing_stats["avg_decomposition_time"],
                "avg_retrieval_time_ms": self.processing_stats["avg_retrieval_time"],
                "avg_synthesis_time_ms": self.processing_stats["avg_synthesis_time"],
                "total_avg_time_ms": (
                    self.processing_stats["avg_decomposition_time"]
                    + self.processing_stats["avg_retrieval_time"]
                    + self.processing_stats["avg_synthesis_time"]
                ),
            },
            "query_characteristics": {
                "multi_hop_rate": multi_hop_rate,
                "intent_classification_accuracy": self.processing_stats["intent_classification_accuracy"],
                "synthesis_success_rate": self.processing_stats["synthesis_success_rate"],
            },
            "capabilities": {
                "query_expansion": self.enable_query_expansion,
                "intent_classification": self.enable_intent_classification,
                "multi_hop_reasoning": self.enable_multi_hop_reasoning,
                "graph_traversal": self.rag_pipeline.enable_trust_graph,
            },
        }
