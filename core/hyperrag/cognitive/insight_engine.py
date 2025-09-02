"""
CreativityEngine - Non-Obvious Path Discovery and Insight Generation

Advanced system for discovering creative connections, generating novel insights,
and finding non-obvious paths through knowledge graphs. Uses graph traversal,
semantic analysis, and creative reasoning to uncover hidden relationships
and generate innovative ideas.

This module provides the creativity component of the unified HyperRAG system.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import random
import time
from typing import Any
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be discovered."""

    NOVEL_CONNECTION = "novel_connection"  # Unexpected relationship
    BRIDGING_CONCEPT = "bridging_concept"  # Concept that connects disparate areas
    EMERGENT_PATTERN = "emergent_pattern"  # Pattern across multiple nodes
    ANALOGICAL_REASONING = "analogical_reasoning"  # Cross-domain analogy
    CREATIVE_SYNTHESIS = "creative_synthesis"  # Novel combination of ideas
    CONTRARIAN_VIEW = "contrarian_view"  # Alternative perspective
    HIDDEN_ASSUMPTION = "hidden_assumption"  # Implicit assumption made explicit
    SCALE_SHIFT = "scale_shift"  # Same concept at different scales


class CreativityMethod(Enum):
    """Methods for generating creative insights."""

    RANDOM_WALK = "random_walk"  # Random exploration of graph
    CONCEPT_BLENDING = "concept_blending"  # Combine distant concepts
    ANALOGICAL_MAPPING = "analogical_mapping"  # Map patterns across domains
    CONTRARIAN_ANALYSIS = "contrarian_analysis"  # Challenge assumptions
    LATERAL_THINKING = "lateral_thinking"  # Indirect problem solving
    SERENDIPITY_MINING = "serendipity_mining"  # Exploit happy accidents
    PATTERN_COMPLETION = "pattern_completion"  # Complete partial patterns
    PERSPECTIVE_SHIFT = "perspective_shift"  # Change point of view


@dataclass
class CreativeInsight:
    """A creative insight discovered by the system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: InsightType = InsightType.NOVEL_CONNECTION

    # Insight content
    title: str = ""
    description: str = ""
    explanation: str = ""  # Why this insight is valuable

    # Supporting elements
    source_concepts: list[str] = field(default_factory=list)
    target_concepts: list[str] = field(default_factory=list)
    bridging_elements: list[str] = field(default_factory=list)

    # Path through knowledge graph
    discovery_path: list[str] = field(default_factory=list)  # Node IDs in path
    path_description: str = ""  # Human-readable path

    # Quality metrics
    novelty_score: float = 0.5  # How novel/unexpected is this insight
    utility_score: float = 0.5  # How useful could this be
    confidence: float = 0.5  # How confident are we in this insight
    surprise_factor: float = 0.5  # How surprising is this connection

    # Generation metadata
    generation_method: CreativityMethod = CreativityMethod.RANDOM_WALK
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)

    # Validation and feedback
    validation_status: str = "discovered"  # discovered, reviewed, validated, dismissed
    human_feedback: str = ""
    usefulness_rating: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score for this insight."""
        return self.novelty_score * 0.3 + self.utility_score * 0.3 + self.confidence * 0.2 + self.surprise_factor * 0.2


@dataclass
class CreativeAnalogy:
    """An analogical mapping between different domains."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source and target domains
    source_domain: str = ""
    target_domain: str = ""

    # Mapped elements
    source_elements: list[str] = field(default_factory=list)
    target_elements: list[str] = field(default_factory=list)
    mappings: dict[str, str] = field(default_factory=dict)  # source -> target

    # Analogy quality
    structural_similarity: float = 0.5
    semantic_distance: float = 0.5  # Higher distance = more creative
    mapping_consistency: float = 0.5

    # Potential insights from this analogy
    predicted_relationships: list[str] = field(default_factory=list)
    novel_hypotheses: list[str] = field(default_factory=list)

    confidence: float = 0.5
    creativity_score: float = 0.5

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightDiscoveryResult:
    """Result from insight discovery process."""

    insights: list[CreativeInsight] = field(default_factory=list)
    analogies: list[CreativeAnalogy] = field(default_factory=list)

    # Discovery metadata
    discovery_time_ms: float = 0.0
    total_insights_generated: int = 0
    paths_explored: int = 0

    # Quality metrics
    avg_novelty: float = 0.0
    avg_confidence: float = 0.0
    most_surprising_insight: CreativeInsight | None = None

    # Method effectiveness
    method_performance: dict[CreativityMethod, float] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)


class CreativityEngine:
    """
    Non-Obvious Path Discovery and Creative Insight Generation

    Advanced creativity system that explores knowledge graphs to discover
    novel connections, generate unexpected insights, and find creative
    paths between seemingly unrelated concepts. Uses multiple creativity
    methods and validates insights for utility and novelty.

    Features:
    - Multiple creativity generation methods
    - Graph-based path exploration with serendipity
    - Analogical reasoning across domains
    - Novelty and surprise detection
    - Cross-domain pattern recognition
    - Creative synthesis and concept blending
    - Insight validation and feedback learning
    """

    def __init__(
        self,
        trust_graph=None,
        vector_engine=None,
        hippo_index=None,
        max_path_length: int = 6,
        exploration_randomness: float = 0.3,
        novelty_threshold: float = 0.6,
    ):
        self.trust_graph = trust_graph
        self.vector_engine = vector_engine
        self.hippo_index = hippo_index
        self.max_path_length = max_path_length
        self.exploration_randomness = exploration_randomness
        self.novelty_threshold = novelty_threshold

        # Creativity methods configuration
        self.creativity_methods = {
            CreativityMethod.RANDOM_WALK: {"weight": 1.0, "enabled": True},
            CreativityMethod.CONCEPT_BLENDING: {"weight": 0.8, "enabled": True},
            CreativityMethod.ANALOGICAL_MAPPING: {"weight": 0.9, "enabled": True},
            CreativityMethod.CONTRARIAN_ANALYSIS: {"weight": 0.6, "enabled": True},
            CreativityMethod.LATERAL_THINKING: {"weight": 0.7, "enabled": True},
            CreativityMethod.SERENDIPITY_MINING: {"weight": 0.5, "enabled": True},
        }

        # Insight storage and caching
        self.discovered_insights: dict[str, CreativeInsight] = {}
        self.validated_analogies: dict[str, CreativeAnalogy] = {}
        self.exploration_cache: dict[str, list[str]] = {}  # Cached paths

        # Learning and adaptation
        self.method_success_rates: dict[CreativityMethod, list[float]] = {}
        self.domain_connectivity: dict[str, set[str]] = {}  # Domain -> connected domains

        # Statistics
        self.stats = {
            "insights_discovered": 0,
            "analogies_created": 0,
            "paths_explored": 0,
            "discoveries_validated": 0,
            "discoveries_dismissed": 0,
            "avg_discovery_time": 0.0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the creativity engine."""
        logger.info("Initializing CreativityEngine...")

        # Initialize method success tracking
        for method in CreativityMethod:
            self.method_success_rates[method] = []

        # Set up periodic learning tasks
        asyncio.create_task(self._periodic_learning())

        self.initialized = True
        logger.info("🎨 CreativityEngine ready for insight discovery and creative exploration")

    async def discover_insights(
        self,
        query: str,
        retrieved_info: list[Any],
        focus_concepts: list[str] | None = None,
        creativity_level: float = 0.7,  # 0.0 = conservative, 1.0 = highly creative
    ) -> dict[str, Any]:
        """Discover creative insights related to a query."""
        start_time = time.time()

        try:
            insights = []
            analogies = []
            paths_explored = 0

            # Extract starting concepts from query and retrieved info
            starting_concepts = await self._extract_starting_concepts(query, retrieved_info, focus_concepts)

            # Run different creativity methods
            for method, config in self.creativity_methods.items():
                if not config["enabled"]:
                    continue

                # Adjust method weight by creativity level
                adjusted_weight = config["weight"] * (0.5 + 0.5 * creativity_level)

                # Generate insights using this method
                method_insights, method_analogies, method_paths = await self._run_creativity_method(
                    method, starting_concepts, adjusted_weight, creativity_level
                )

                insights.extend(method_insights)
                analogies.extend(method_analogies)
                paths_explored += method_paths

            # Remove duplicates and rank insights
            unique_insights = await self._deduplicate_insights(insights)
            ranked_insights = await self._rank_insights_by_creativity(unique_insights)

            # Find most surprising insight
            most_surprising = max(ranked_insights, key=lambda i: i.surprise_factor) if ranked_insights else None

            # Calculate discovery metrics
            discovery_time = (time.time() - start_time) * 1000
            avg_novelty = np.mean([i.novelty_score for i in ranked_insights]) if ranked_insights else 0.0
            avg_confidence = np.mean([i.confidence for i in ranked_insights]) if ranked_insights else 0.0

            # Store discoveries
            for insight in ranked_insights:
                self.discovered_insights[insight.id] = insight

            for analogy in analogies:
                self.validated_analogies[analogy.id] = analogy

            # Update statistics
            self.stats["insights_discovered"] += len(ranked_insights)
            self.stats["analogies_created"] += len(analogies)
            self.stats["paths_explored"] += paths_explored
            self.stats["avg_discovery_time"] = (
                self.stats["avg_discovery_time"] * (self.stats["insights_discovered"] - len(ranked_insights))
                + discovery_time
            ) / max(1, self.stats["insights_discovered"])

            # Create result
            result = {
                "insights": ranked_insights[:10],  # Return top 10
                "analogies": analogies[:5],  # Return top 5
                "insights_summary": await self._create_insights_summary(ranked_insights[:5]),
                "discovery_stats": {
                    "total_insights": len(ranked_insights),
                    "total_analogies": len(analogies),
                    "discovery_time_ms": discovery_time,
                    "paths_explored": paths_explored,
                    "avg_novelty": avg_novelty,
                    "avg_confidence": avg_confidence,
                    "most_surprising": most_surprising.title if most_surprising else None,
                },
            }

            logger.info(f"Discovered {len(ranked_insights)} creative insights in {discovery_time:.1f}ms")
            return result

        except Exception as e:
            logger.exception(f"Insight discovery failed: {e}")
            return {
                "insights": [],
                "analogies": [],
                "insights_summary": "Discovery failed due to error",
                "discovery_stats": {"error": str(e)},
            }

    async def explore_creative_connections(
        self, concept1: str, concept2: str, max_intermediate_steps: int = 4
    ) -> list[CreativeInsight]:
        """Find creative paths between two specific concepts."""
        try:
            insights = []

            if not self.trust_graph:
                return insights

            # Find nodes representing these concepts
            concept1_nodes = [
                node_id
                for node_id, node in self.trust_graph.nodes.items()
                if concept1.lower() in node.concept.lower() or concept1.lower() in node.content.lower()
            ]

            concept2_nodes = [
                node_id
                for node_id, node in self.trust_graph.nodes.items()
                if concept2.lower() in node.concept.lower() or concept2.lower() in node.content.lower()
            ]

            if not concept1_nodes or not concept2_nodes:
                return insights

            # Explore creative paths between concept nodes
            for start_node in concept1_nodes[:3]:  # Limit exploration
                for end_node in concept2_nodes[:3]:
                    creative_paths = await self._find_creative_paths(start_node, end_node, max_intermediate_steps)

                    for path in creative_paths:
                        insight = await self._create_path_insight(path, concept1, concept2)
                        if insight and insight.novelty_score > self.novelty_threshold:
                            insights.append(insight)

            # Rank and return top insights
            insights.sort(key=lambda i: i.calculate_overall_score(), reverse=True)
            return insights[:5]

        except Exception as e:
            logger.exception(f"Creative connection exploration failed: {e}")
            return []

    async def generate_analogies(
        self, source_domain: str, target_domain: str, max_analogies: int = 3
    ) -> list[CreativeAnalogy]:
        """Generate analogical mappings between domains."""
        try:
            analogies = []

            if not self.trust_graph:
                return analogies

            # Find nodes in each domain
            source_nodes = await self._find_domain_nodes(source_domain)
            target_nodes = await self._find_domain_nodes(target_domain)

            if len(source_nodes) < 2 or len(target_nodes) < 2:
                return analogies

            # Analyze structural patterns in source domain
            source_patterns = await self._extract_domain_patterns(source_nodes)
            target_patterns = await self._extract_domain_patterns(target_nodes)

            # Find structural correspondences
            for source_pattern in source_patterns[:5]:  # Limit to top patterns
                for target_pattern in target_patterns[:5]:
                    similarity = await self._calculate_pattern_similarity(source_pattern, target_pattern)

                    if similarity > 0.3:  # Sufficient structural similarity
                        analogy = await self._create_analogy(
                            source_domain, target_domain, source_pattern, target_pattern, similarity
                        )

                        if analogy:
                            analogies.append(analogy)

            # Rank by creativity (high similarity but distant domains)
            analogies.sort(key=lambda a: a.creativity_score, reverse=True)
            return analogies[:max_analogies]

        except Exception as e:
            logger.exception(f"Analogy generation failed: {e}")
            return []

    async def validate_insight(self, insight_id: str, is_useful: bool, feedback: str = "") -> bool:
        """Validate a discovered insight with human feedback."""
        try:
            if insight_id not in self.discovered_insights:
                return False

            insight = self.discovered_insights[insight_id]

            if is_useful:
                insight.validation_status = "validated"
                insight.usefulness_rating = 1.0
                self.stats["discoveries_validated"] += 1
            else:
                insight.validation_status = "dismissed"
                insight.usefulness_rating = 0.0
                self.stats["discoveries_dismissed"] += 1

            insight.human_feedback = feedback

            # Learn from validation
            await self._learn_from_validation(insight, is_useful)

            return True

        except Exception as e:
            logger.exception(f"Insight validation failed: {e}")
            return False

    async def get_creativity_stats(self) -> dict[str, Any]:
        """Get statistics about creativity and insight discovery."""
        try:
            # Calculate method success rates
            method_performance = {}
            for method, success_rates in self.method_success_rates.items():
                avg_success = np.mean(success_rates) if success_rates else 0.0
                method_performance[method.value] = {
                    "avg_success_rate": avg_success,
                    "total_attempts": len(success_rates),
                    "enabled": self.creativity_methods[method]["enabled"],
                }

            # Calculate validation rate
            total_evaluated = self.stats["discoveries_validated"] + self.stats["discoveries_dismissed"]
            validation_rate = self.stats["discoveries_validated"] / max(1, total_evaluated)

            return {
                "discovery_metrics": {
                    "total_insights": self.stats["insights_discovered"],
                    "total_analogies": self.stats["analogies_created"],
                    "avg_discovery_time_ms": self.stats["avg_discovery_time"],
                    "paths_explored": self.stats["paths_explored"],
                },
                "validation_metrics": {
                    "validation_rate": validation_rate,
                    "validated_insights": self.stats["discoveries_validated"],
                    "dismissed_insights": self.stats["discoveries_dismissed"],
                },
                "method_performance": method_performance,
                "system_health": {
                    "trust_graph_available": self.trust_graph is not None,
                    "vector_engine_available": self.vector_engine is not None,
                    "hippo_index_available": self.hippo_index is not None,
                    "cached_insights": len(self.discovered_insights),
                    "cached_analogies": len(self.validated_analogies),
                },
                "configuration": {
                    "max_path_length": self.max_path_length,
                    "exploration_randomness": self.exploration_randomness,
                    "novelty_threshold": self.novelty_threshold,
                },
            }

        except Exception as e:
            logger.exception(f"Statistics gathering failed: {e}")
            return {"error": str(e)}

    # Private implementation methods

    async def _extract_starting_concepts(
        self, query: str, retrieved_info: list[Any], focus_concepts: list[str] | None
    ) -> list[str]:
        """Extract starting concepts for creativity exploration."""
        concepts = set()

        # Add focus concepts if provided
        if focus_concepts:
            concepts.update(focus_concepts)

        # Extract from query
        query_words = query.lower().split()
        concepts.update(word for word in query_words if len(word) > 3)

        # Extract from retrieved information
        for info in retrieved_info:
            if hasattr(info, "content"):
                content_words = info.content.lower().split()
                concepts.update(word for word in content_words if len(word) > 4)

            # Extract concept if available
            if hasattr(info, "concept"):
                concepts.add(info.concept)

        return list(concepts)[:10]  # Limit starting concepts

    async def _run_creativity_method(
        self, method: CreativityMethod, starting_concepts: list[str], weight: float, creativity_level: float
    ) -> tuple[list[CreativeInsight], list[CreativeAnalogy], int]:
        """Run a specific creativity method."""
        try:
            insights = []
            analogies = []
            paths_explored = 0

            if method == CreativityMethod.RANDOM_WALK:
                insights, paths = await self._random_walk_exploration(starting_concepts, creativity_level)
                paths_explored = paths

            elif method == CreativityMethod.CONCEPT_BLENDING:
                insights = await self._concept_blending(starting_concepts, creativity_level)

            elif method == CreativityMethod.ANALOGICAL_MAPPING:
                analogies = await self._analogical_mapping(starting_concepts, creativity_level)

            elif method == CreativityMethod.CONTRARIAN_ANALYSIS:
                insights = await self._contrarian_analysis(starting_concepts, creativity_level)

            elif method == CreativityMethod.LATERAL_THINKING:
                insights = await self._lateral_thinking(starting_concepts, creativity_level)

            elif method == CreativityMethod.SERENDIPITY_MINING:
                insights, paths = await self._serendipity_mining(starting_concepts, creativity_level)
                paths_explored = paths

            # Record method performance
            success_rate = len(insights) / max(1, 5)  # Assume target of 5 insights per method
            self.method_success_rates[method].append(min(1.0, success_rate))

            return insights, analogies, paths_explored

        except Exception as e:
            logger.warning(f"Creativity method {method.value} failed: {e}")
            return [], [], 0

    async def _random_walk_exploration(
        self, starting_concepts: list[str], creativity_level: float
    ) -> tuple[list[CreativeInsight], int]:
        """Explore using random walks through the knowledge graph."""
        insights = []
        paths_explored = 0

        if not self.trust_graph:
            return insights, paths_explored

        try:
            # Start random walks from nodes related to starting concepts
            for concept in starting_concepts[:3]:  # Limit starting points
                start_nodes = [
                    node_id
                    for node_id, node in self.trust_graph.nodes.items()
                    if concept.lower() in node.concept.lower() or concept.lower() in node.content.lower()
                ]

                for start_node in start_nodes[:2]:  # Limit per concept
                    path = await self._perform_random_walk(start_node, creativity_level)
                    paths_explored += 1

                    if len(path) > 2:
                        insight = await self._analyze_path_for_insights(path, CreativityMethod.RANDOM_WALK)
                        if insight:
                            insights.append(insight)

            return insights, paths_explored

        except Exception as e:
            logger.warning(f"Random walk exploration failed: {e}")
            return insights, paths_explored

    async def _perform_random_walk(self, start_node: str, creativity_level: float) -> list[str]:
        """Perform a random walk starting from a node."""
        path = [start_node]
        current_node = start_node

        for _step in range(self.max_path_length):
            if current_node not in self.trust_graph.nodes:
                break

            node = self.trust_graph.nodes[current_node]

            # Get possible next nodes
            next_candidates = []
            for edge_id in node.outgoing_edges:
                if edge_id in self.trust_graph.edges:
                    edge = self.trust_graph.edges[edge_id]
                    next_candidates.append((edge.target_id, edge.trust_score))

            if not next_candidates:
                break

            # Choose next node (balance randomness with trust)
            if random.random() < self.exploration_randomness * creativity_level:
                # Random choice for creativity
                next_node, _ = random.choice(next_candidates)
            else:
                # Trust-weighted choice
                weights = [trust for _, trust in next_candidates]
                next_node = random.choices([node_id for node_id, _ in next_candidates], weights=weights)[0]

            if next_node in path:  # Avoid cycles
                break

            path.append(next_node)
            current_node = next_node

        return path

    async def _analyze_path_for_insights(self, path: list[str], method: CreativityMethod) -> CreativeInsight | None:
        """Analyze a path through the graph for creative insights."""
        if len(path) < 3:
            return None

        try:
            # Get concepts along the path
            concepts = []
            for node_id in path:
                if node_id in self.trust_graph.nodes:
                    node = self.trust_graph.nodes[node_id]
                    concepts.append(node.concept)

            if len(concepts) < 3:
                return None

            # Calculate novelty based on path unusualness
            path_trust_scores = []
            for i in range(len(path) - 1):
                source_id = path[i]
                target_id = path[i + 1]

                # Find edge between nodes
                source_node = self.trust_graph.nodes[source_id]
                for edge_id in source_node.outgoing_edges:
                    edge = self.trust_graph.edges[edge_id]
                    if edge.target_id == target_id:
                        path_trust_scores.append(edge.trust_score)
                        break

            # Lower trust = more novel/creative path
            avg_trust = np.mean(path_trust_scores) if path_trust_scores else 0.5
            novelty_score = 1.0 - avg_trust  # Invert trust for novelty

            # Create insight
            insight = CreativeInsight(
                insight_type=InsightType.NOVEL_CONNECTION,
                title=f"Creative connection: {concepts[0]} → {concepts[-1]}",
                description=f"Found creative path connecting {concepts[0]} to {concepts[-1]} through {len(concepts)-2} intermediate concepts",
                explanation="This path reveals an unexpected connection between seemingly unrelated concepts",
                source_concepts=[concepts[0]],
                target_concepts=[concepts[-1]],
                bridging_elements=concepts[1:-1],
                discovery_path=path,
                path_description=" → ".join(concepts),
                novelty_score=novelty_score,
                utility_score=0.6,  # Default utility
                confidence=0.7,  # Default confidence
                surprise_factor=novelty_score * 0.8,
                generation_method=method,
            )

            return insight

        except Exception as e:
            logger.warning(f"Path analysis failed: {e}")
            return None

    async def _concept_blending(self, starting_concepts: list[str], creativity_level: float) -> list[CreativeInsight]:
        """Generate insights by blending distant concepts."""
        insights = []

        try:
            # Blend pairs of concepts
            for i, concept1 in enumerate(starting_concepts):
                for concept2 in starting_concepts[i + 1 :]:
                    # Calculate semantic distance (sample implementation)
                    semantic_distance = 0.7  # Would use vector similarity in real implementation

                    if semantic_distance > 0.5:  # Sufficiently distant for creative blending
                        blended_insight = CreativeInsight(
                            insight_type=InsightType.CREATIVE_SYNTHESIS,
                            title=f"Synthesis: {concept1} + {concept2}",
                            description=f"Creative combination of {concept1} and {concept2} concepts",
                            explanation="Blending these distant concepts could reveal new perspectives",
                            source_concepts=[concept1, concept2],
                            novelty_score=semantic_distance * creativity_level,
                            utility_score=0.5,
                            confidence=0.6,
                            surprise_factor=semantic_distance,
                            generation_method=CreativityMethod.CONCEPT_BLENDING,
                        )
                        insights.append(blended_insight)

            return insights[:5]  # Limit insights

        except Exception as e:
            logger.warning(f"Concept blending failed: {e}")
            return insights

    async def _analogical_mapping(self, starting_concepts: list[str], creativity_level: float) -> list[CreativeAnalogy]:
        """Generate analogies by mapping between different domains."""
        analogies = []

        try:
            # Simple analogy generation (would be more sophisticated in production)
            for i, concept1 in enumerate(starting_concepts):
                for concept2 in starting_concepts[i + 1 :]:
                    # Create a sample analogy
                    analogy = CreativeAnalogy(
                        source_domain=concept1,
                        target_domain=concept2,
                        source_elements=[f"{concept1}_element1", f"{concept1}_element2"],
                        target_elements=[f"{concept2}_element1", f"{concept2}_element2"],
                        mappings={f"{concept1}_element1": f"{concept2}_element1"},
                        structural_similarity=0.6,
                        semantic_distance=0.7,
                        mapping_consistency=0.8,
                        confidence=0.7,
                        creativity_score=0.7 * creativity_level,
                    )
                    analogies.append(analogy)

            return analogies[:3]  # Limit analogies

        except Exception as e:
            logger.warning(f"Analogical mapping failed: {e}")
            return analogies

    async def _contrarian_analysis(
        self, starting_concepts: list[str], creativity_level: float
    ) -> list[CreativeInsight]:
        """Generate insights by challenging conventional assumptions."""
        insights = []

        try:
            for concept in starting_concepts[:3]:
                contrarian_insight = CreativeInsight(
                    insight_type=InsightType.CONTRARIAN_VIEW,
                    title=f"Alternative perspective on {concept}",
                    description=f"What if conventional understanding of {concept} is incomplete?",
                    explanation=f"Challenging assumptions about {concept} could reveal new possibilities",
                    source_concepts=[concept],
                    novelty_score=0.8 * creativity_level,
                    utility_score=0.6,
                    confidence=0.5,  # Lower confidence for contrarian views
                    surprise_factor=0.8,
                    generation_method=CreativityMethod.CONTRARIAN_ANALYSIS,
                )
                insights.append(contrarian_insight)

            return insights

        except Exception as e:
            logger.warning(f"Contrarian analysis failed: {e}")
            return insights

    async def _lateral_thinking(self, starting_concepts: list[str], creativity_level: float) -> list[CreativeInsight]:
        """Generate insights using lateral thinking approaches."""
        insights = []

        try:
            for concept in starting_concepts[:3]:
                lateral_insight = CreativeInsight(
                    insight_type=InsightType.SCALE_SHIFT,
                    title=f"Scale shift perspective on {concept}",
                    description=f"How does {concept} apply at different scales or contexts?",
                    explanation=f"Examining {concept} at different scales might reveal universal patterns",
                    source_concepts=[concept],
                    novelty_score=0.6 * creativity_level,
                    utility_score=0.7,
                    confidence=0.6,
                    surprise_factor=0.6,
                    generation_method=CreativityMethod.LATERAL_THINKING,
                )
                insights.append(lateral_insight)

            return insights

        except Exception as e:
            logger.warning(f"Lateral thinking failed: {e}")
            return insights

    async def _serendipity_mining(
        self, starting_concepts: list[str], creativity_level: float
    ) -> tuple[list[CreativeInsight], int]:
        """Mine for serendipitous discoveries."""
        insights = []
        paths_explored = 0

        try:
            # Look for unexpected patterns or connections
            for concept in starting_concepts[:2]:
                serendipitous_insight = CreativeInsight(
                    insight_type=InsightType.EMERGENT_PATTERN,
                    title=f"Serendipitous pattern involving {concept}",
                    description=f"Unexpected pattern or connection discovered around {concept}",
                    explanation="Happy accident discovery that reveals new understanding",
                    source_concepts=[concept],
                    novelty_score=0.9 * creativity_level,
                    utility_score=0.5,  # Uncertain utility for serendipitous discoveries
                    confidence=0.4,  # Lower confidence for unexpected discoveries
                    surprise_factor=0.9,
                    generation_method=CreativityMethod.SERENDIPITY_MINING,
                )
                insights.append(serendipitous_insight)
                paths_explored += 1

            return insights, paths_explored

        except Exception as e:
            logger.warning(f"Serendipity mining failed: {e}")
            return insights, paths_explored

    async def _deduplicate_insights(self, insights: list[CreativeInsight]) -> list[CreativeInsight]:
        """Remove duplicate insights."""
        seen = set()
        unique_insights = []

        for insight in insights:
            # Create a key based on insight content
            key = (insight.insight_type, insight.title.lower(), tuple(sorted(insight.source_concepts)))

            if key not in seen:
                seen.add(key)
                unique_insights.append(insight)

        return unique_insights

    async def _rank_insights_by_creativity(self, insights: list[CreativeInsight]) -> list[CreativeInsight]:
        """Rank insights by overall creativity score."""
        insights.sort(key=lambda i: i.calculate_overall_score(), reverse=True)
        return insights

    async def _create_insights_summary(self, top_insights: list[CreativeInsight]) -> str:
        """Create a human-readable summary of top insights."""
        if not top_insights:
            return "No significant creative insights discovered."

        summary_parts = []

        for i, insight in enumerate(top_insights, 1):
            summary_parts.append(f"{i}. {insight.title}: {insight.description}")

        return "\n".join(summary_parts)

    async def _learn_from_validation(self, insight: CreativeInsight, is_useful: bool):
        """Learn from validation feedback to improve future insights."""
        # Adjust method weights based on validation
        method = insight.generation_method
        current_weight = self.creativity_methods[method]["weight"]

        if is_useful:
            # Slightly increase weight for successful methods
            self.creativity_methods[method]["weight"] = min(1.0, current_weight * 1.05)
        else:
            # Slightly decrease weight for unsuccessful methods
            self.creativity_methods[method]["weight"] = max(0.1, current_weight * 0.95)

        logger.debug(f"Adjusted {method.value} weight to {self.creativity_methods[method]['weight']:.3f}")

    async def _periodic_learning(self):
        """Periodic learning and adaptation tasks."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Analyze recent insight performance
                # Adjust creativity parameters
                # Update domain connectivity maps
                # Cleanup old cache entries

                logger.debug("Performed periodic creativity learning")

            except Exception as e:
                logger.exception(f"Periodic learning failed: {e}")
                await asyncio.sleep(3600)

    # Additional helper methods would be implemented here...


if __name__ == "__main__":

    async def test_creativity_engine():
        """Test CreativityEngine functionality."""
        # Create system (would normally pass real components)
        engine = CreativityEngine(
            trust_graph=None,  # Would pass real graph
            vector_engine=None,  # Would pass real engine
            hippo_index=None,  # Would pass real index
            max_path_length=6,
            exploration_randomness=0.3,
            novelty_threshold=0.6,
        )
        await engine.initialize()

        # Test insight discovery
        sample_retrieved_info = [
            type("SampleInfo", (), {"content": "machine learning algorithms", "source": "vector"}),
            type("SampleInfo", (), {"content": "neural network architectures", "source": "graph"}),
        ]

        result = await engine.discover_insights(
            query="artificial intelligence creativity",
            retrieved_info=sample_retrieved_info,
            focus_concepts=["AI", "creativity", "innovation"],
            creativity_level=0.8,
        )

        print(f"Discovered {len(result['insights'])} creative insights")
        print(f"Summary: {result['insights_summary']}")
        print(f"Discovery stats: {result['discovery_stats']}")

        # Test creative connections
        connections = await engine.explore_creative_connections(
            "machine learning", "creativity", max_intermediate_steps=3
        )
        print(f"Found {len(connections)} creative connections")

        # Test analogies
        analogies = await engine.generate_analogies("neural networks", "biological systems", max_analogies=2)
        print(f"Generated {len(analogies)} analogies")

        # Get statistics
        stats = await engine.get_creativity_stats()
        print(f"Creativity stats: {stats}")

    import asyncio

    asyncio.run(test_creativity_engine())
