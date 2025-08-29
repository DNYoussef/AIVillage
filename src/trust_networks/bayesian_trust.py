"""
Bayesian Trust Graph Networks for Knowledge Validation

Implements probabilistic trust scoring, source credibility assessment,
and Bayesian inference for information reliability in RAG systems.

Key Features:
- Bayesian trust propagation through knowledge graphs
- Dynamic source credibility scoring
- Information quality assessment with uncertainty quantification
- Trust decay and reputation management
- Evidence accumulation and belief updating
- Multi-dimensional trust metrics (accuracy, completeness, recency, authority)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import math
import time
from typing import Any
import uuid

import numpy as np
from scipy.stats import beta

logger = logging.getLogger(__name__)


class TrustDimension(Enum):
    """Different dimensions of trust assessment."""

    ACCURACY = "accuracy"  # Factual correctness
    COMPLETENESS = "completeness"  # Information completeness
    RECENCY = "recency"  # Temporal relevance
    AUTHORITY = "authority"  # Source expertise/credibility
    CONSISTENCY = "consistency"  # Internal consistency
    VERIFIABILITY = "verifiability"  # Can be independently verified


class EvidenceType(Enum):
    """Types of evidence for trust assessment."""

    DIRECT = "direct"  # Direct user feedback/validation
    PEER = "peer"  # Peer source validation
    CITATION = "citation"  # Citation/reference validation
    USAGE = "usage"  # Usage patterns and acceptance
    TEMPORAL = "temporal"  # Time-based validation
    CROSS_REFERENCE = "cross_reference"  # Cross-validation with other sources


@dataclass
class TrustEvidence:
    """Evidence for trust assessment."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence_type: EvidenceType = EvidenceType.DIRECT
    dimension: TrustDimension = TrustDimension.ACCURACY

    # Evidence values
    positive_evidence: float = 0.0  # Supporting evidence strength
    negative_evidence: float = 0.0  # Contradicting evidence strength
    uncertainty: float = 0.0  # Uncertainty in evidence

    # Source information
    source_id: str = ""
    evaluator_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    # Temporal properties
    timestamp: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.1  # How fast evidence decays
    weight: float = 1.0  # Evidence weight/importance

    # Quality metrics
    confidence: float = 1.0  # Confidence in evidence
    reliability: float = 1.0  # Reliability of evidence source

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_current_strength(self, current_time: datetime) -> tuple[float, float]:
        """Calculate current evidence strength with temporal decay."""
        time_diff = (current_time - self.timestamp).total_seconds()
        hours_elapsed = time_diff / 3600

        # Exponential decay
        decay_factor = math.exp(-self.decay_rate * hours_elapsed / 24)  # Daily decay rate

        # Apply decay to evidence
        current_positive = self.positive_evidence * decay_factor * self.weight
        current_negative = self.negative_evidence * decay_factor * self.weight

        return current_positive, current_negative


@dataclass
class TrustScore:
    """Multi-dimensional trust score with uncertainty."""

    # Bayesian parameters (Beta distribution)
    alpha: dict[TrustDimension, float] = field(default_factory=dict)  # Positive evidence + 1
    beta: dict[TrustDimension, float] = field(default_factory=dict)  # Negative evidence + 1

    # Derived metrics
    mean_trust: dict[TrustDimension, float] = field(default_factory=dict)
    uncertainty: dict[TrustDimension, float] = field(default_factory=dict)
    confidence_interval: dict[TrustDimension, tuple[float, float]] = field(default_factory=dict)

    # Composite scores
    overall_trust: float = 0.0
    overall_uncertainty: float = 0.0

    # Temporal properties
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    # Evidence tracking
    evidence_count: dict[TrustDimension, int] = field(default_factory=dict)
    total_evidence_weight: dict[TrustDimension, float] = field(default_factory=dict)

    def update_dimension_scores(self):
        """Update derived metrics from Bayesian parameters."""
        for dimension in TrustDimension:
            if dimension in self.alpha and dimension in self.beta:
                alpha_val = self.alpha[dimension]
                beta_val = self.beta[dimension]

                # Mean of beta distribution
                self.mean_trust[dimension] = alpha_val / (alpha_val + beta_val)

                # Uncertainty (variance of beta distribution)
                variance = (alpha_val * beta_val) / ((alpha_val + beta_val) ** 2 * (alpha_val + beta_val + 1))
                self.uncertainty[dimension] = math.sqrt(variance)

                # 95% confidence interval
                beta_dist = beta(alpha_val, beta_val)
                lower = beta_dist.ppf(0.025)
                upper = beta_dist.ppf(0.975)
                self.confidence_interval[dimension] = (lower, upper)

        # Calculate overall scores
        if self.mean_trust:
            trust_values = list(self.mean_trust.values())
            uncertainty_values = list(self.uncertainty.values())

            # Weighted average (equal weights for now)
            self.overall_trust = np.mean(trust_values)
            self.overall_uncertainty = np.mean(uncertainty_values)

    def get_trust_summary(self) -> dict[str, Any]:
        """Get summary of trust scores."""
        return {
            "overall_trust": self.overall_trust,
            "overall_uncertainty": self.overall_uncertainty,
            "dimensional_trust": {dim.value: score for dim, score in self.mean_trust.items()},
            "dimensional_uncertainty": {dim.value: unc for dim, unc in self.uncertainty.items()},
            "confidence_intervals": {
                dim.value: {"lower": ci[0], "upper": ci[1]} for dim, ci in self.confidence_interval.items()
            },
            "evidence_counts": {dim.value: count for dim, count in self.evidence_count.items()},
            "last_updated": self.last_updated,
            "update_count": self.update_count,
        }


@dataclass
class TrustNode:
    """Node in trust graph representing an information source."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str = "document"  # document, author, domain, etc.
    source_identifier: str = ""  # URL, DOI, author name, etc.

    # Trust scoring
    trust_score: TrustScore = field(default_factory=TrustScore)
    evidence: list[TrustEvidence] = field(default_factory=list)

    # Graph relationships
    trusted_by: set[str] = field(default_factory=set)  # Nodes that trust this one
    trusts: set[str] = field(default_factory=set)  # Nodes this one trusts
    conflicts_with: set[str] = field(default_factory=set)  # Conflicting nodes

    # Content and context
    content: str = ""
    content_hash: str = ""
    keywords: list[str] = field(default_factory=list)
    domain: str = ""

    # Temporal properties
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Usage statistics
    access_count: int = 0
    reference_count: int = 0  # How many times referenced by other nodes
    validation_count: int = 0  # How many times validated

    # Quality metrics
    information_density: float = 1.0
    coherence_score: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, evidence: TrustEvidence):
        """Add evidence and update trust scores."""
        self.evidence.append(evidence)
        self._update_trust_from_evidence()
        self.last_updated = datetime.now()

    def _update_trust_from_evidence(self):
        """Update Bayesian trust scores from accumulated evidence."""
        # Initialize if needed
        for dimension in TrustDimension:
            if dimension not in self.trust_score.alpha:
                self.trust_score.alpha[dimension] = 1.0  # Uniform prior
                self.trust_score.beta[dimension] = 1.0
                self.trust_score.evidence_count[dimension] = 0
                self.trust_score.total_evidence_weight[dimension] = 0.0

        # Aggregate evidence by dimension
        current_time = datetime.now()
        evidence_by_dimension = {dim: [] for dim in TrustDimension}

        for evidence in self.evidence:
            evidence_by_dimension[evidence.dimension].append(evidence)

        # Update Bayesian parameters for each dimension
        for dimension, dim_evidence in evidence_by_dimension.items():
            if not dim_evidence:
                continue

            total_positive = 0.0
            total_negative = 0.0
            total_weight = 0.0

            for evidence in dim_evidence:
                pos, neg = evidence.calculate_current_strength(current_time)
                total_positive += pos * evidence.confidence * evidence.reliability
                total_negative += neg * evidence.confidence * evidence.reliability
                total_weight += evidence.weight

            # Update Bayesian parameters
            self.trust_score.alpha[dimension] = 1.0 + total_positive
            self.trust_score.beta[dimension] = 1.0 + total_negative
            self.trust_score.evidence_count[dimension] = len(dim_evidence)
            self.trust_score.total_evidence_weight[dimension] = total_weight

        # Update derived scores
        self.trust_score.update_dimension_scores()
        self.trust_score.update_count += 1
        self.trust_score.last_updated = current_time

    def calculate_trust_propagation_weight(self, target_dimension: TrustDimension) -> float:
        """Calculate weight for trust propagation to other nodes."""
        if target_dimension not in self.trust_score.mean_trust:
            return 0.0

        trust_value = self.trust_score.mean_trust[target_dimension]
        uncertainty = self.trust_score.uncertainty[target_dimension]
        evidence_count = self.trust_score.evidence_count.get(target_dimension, 0)

        # Weight based on trust, certainty, and evidence amount
        certainty = 1.0 - uncertainty
        evidence_factor = min(1.0, evidence_count / 10.0)  # Saturate at 10 pieces of evidence

        return trust_value * certainty * evidence_factor


@dataclass
class TrustPropagationResult:
    """Result of trust propagation through the network."""

    source_nodes: list[str] = field(default_factory=list)
    propagated_trust: dict[TrustDimension, float] = field(default_factory=dict)
    propagated_uncertainty: dict[TrustDimension, float] = field(default_factory=dict)

    # Path analysis
    trust_paths: list[list[str]] = field(default_factory=list)
    path_weights: list[float] = field(default_factory=list)
    propagation_depth: int = 0

    # Confidence metrics
    overall_confidence: float = 0.0
    convergence_score: float = 0.0  # How well different paths agree

    # Metadata
    computation_time_ms: float = 0.0
    nodes_evaluated: int = 0


class BayesianTrustNetwork:
    """
    Bayesian Trust Graph Network for Knowledge Validation

    Implements probabilistic trust assessment and propagation through
    a graph of information sources, using Bayesian inference to update
    trust scores based on evidence and peer validation.

    Features:
    - Multi-dimensional trust scoring (accuracy, authority, recency, etc.)
    - Bayesian evidence accumulation with uncertainty quantification
    - Trust propagation through network relationships
    - Temporal decay of evidence and trust scores
    - Conflict detection and resolution
    - Source credibility assessment
    """

    def __init__(
        self,
        max_propagation_depth: int = 3,
        trust_threshold: float = 0.6,
        uncertainty_threshold: float = 0.3,
        decay_rate: float = 0.05,  # Daily decay rate
        evidence_weight_threshold: float = 0.1,
    ):
        self.max_propagation_depth = max_propagation_depth
        self.trust_threshold = trust_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.decay_rate = decay_rate
        self.evidence_weight_threshold = evidence_weight_threshold

        # Graph storage
        self.nodes: dict[str, TrustNode] = {}
        self.edge_weights: dict[tuple[str, str], dict[TrustDimension, float]] = {}

        # Indexing for efficient retrieval
        self.content_index: dict[str, set[str]] = {}  # content hash -> node IDs
        self.domain_index: dict[str, set[str]] = {}  # domain -> node IDs
        self.keyword_index: dict[str, set[str]] = {}  # keyword -> node IDs

        # Trust assessment cache
        self.trust_cache: dict[str, TrustScore] = {}
        self.propagation_cache: dict[tuple[str, int], TrustPropagationResult] = {}

        # Statistics
        self.stats = {
            "nodes_created": 0,
            "evidence_added": 0,
            "trust_propagations": 0,
            "cache_hits": 0,
            "conflicts_detected": 0,
            "average_trust_score": 0.0,
            "total_evidence_pieces": 0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize the Bayesian trust network."""
        logger.info("Initializing BayesianTrustNetwork...")

        # Set up background maintenance tasks
        asyncio.create_task(self._background_maintenance())

        self.initialized = True
        logger.info("🔒 BayesianTrustNetwork ready for trust assessment")

    async def add_source(
        self,
        source_identifier: str,
        content: str,
        source_type: str = "document",
        domain: str = "",
        keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a new information source to the trust network."""
        try:
            # Create trust node
            node = TrustNode(
                source_type=source_type,
                source_identifier=source_identifier,
                content=content,
                content_hash=str(hash(content)),
                domain=domain,
                keywords=keywords or [],
                metadata=metadata or {},
            )

            # Initialize trust score with uniform priors
            for dimension in TrustDimension:
                node.trust_score.alpha[dimension] = 1.0
                node.trust_score.beta[dimension] = 1.0
                node.trust_score.evidence_count[dimension] = 0
                node.trust_score.total_evidence_weight[dimension] = 0.0

            node.trust_score.update_dimension_scores()

            # Store node
            self.nodes[node.id] = node

            # Update indexes
            await self._update_indexes(node)

            self.stats["nodes_created"] += 1
            logger.debug(f"Added trust node: {node.id}")

            return node.id

        except Exception as e:
            logger.error(f"Failed to add source: {e}")
            return ""

    async def add_evidence(
        self,
        node_id: str,
        evidence_type: EvidenceType,
        dimension: TrustDimension,
        positive_evidence: float = 0.0,
        negative_evidence: float = 0.0,
        uncertainty: float = 0.0,
        evaluator_id: str = "",
        confidence: float = 1.0,
        reliability: float = 1.0,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Add evidence for trust assessment."""
        try:
            if node_id not in self.nodes:
                logger.warning(f"Node {node_id} not found")
                return False

            # Create evidence
            evidence = TrustEvidence(
                evidence_type=evidence_type,
                dimension=dimension,
                positive_evidence=positive_evidence,
                negative_evidence=negative_evidence,
                uncertainty=uncertainty,
                source_id=node_id,
                evaluator_id=evaluator_id,
                confidence=confidence,
                reliability=reliability,
                context=context or {},
                decay_rate=self.decay_rate,
            )

            # Add to node
            node = self.nodes[node_id]
            node.add_evidence(evidence)

            # Clear related caches
            self._clear_related_caches(node_id)

            self.stats["evidence_added"] += 1
            self.stats["total_evidence_pieces"] += 1
            logger.debug(f"Added evidence for node {node_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to add evidence: {e}")
            return False

    async def get_trust_score(
        self, node_id: str, use_propagation: bool = True, propagation_depth: int = 2
    ) -> TrustScore | None:
        """Get trust score for a node, optionally with network propagation."""
        try:
            if node_id not in self.nodes:
                return None

            node = self.nodes[node_id]

            # Get direct trust score
            direct_trust = node.trust_score

            if not use_propagation:
                return direct_trust

            # Check propagation cache
            cache_key = (node_id, propagation_depth)
            if cache_key in self.propagation_cache:
                self.stats["cache_hits"] += 1
                return self._merge_trust_scores(direct_trust, self.propagation_cache[cache_key])

            # Perform trust propagation
            propagation_result = await self._propagate_trust(node_id, propagation_depth)

            # Cache result
            self.propagation_cache[cache_key] = propagation_result

            # Merge direct and propagated trust
            merged_trust = self._merge_trust_scores(direct_trust, propagation_result)

            self.stats["trust_propagations"] += 1

            return merged_trust

        except Exception as e:
            logger.error(f"Failed to get trust score: {e}")
            return None

    async def retrieve_with_trust_propagation(
        self,
        query: str,
        k: int = 10,
        min_trust_score: float = 0.4,
        trust_dimensions: list[TrustDimension] | None = None,
        domain_filter: str | None = None,
    ) -> list[tuple[TrustNode, float, TrustScore]]:
        """Retrieve nodes with trust-based ranking."""
        try:
            # Get candidate nodes
            candidates = await self._get_candidate_nodes(query, domain_filter)

            if not candidates:
                return []

            # Calculate trust scores for candidates
            trust_dimensions = trust_dimensions or [TrustDimension.ACCURACY, TrustDimension.AUTHORITY]

            scored_candidates = []
            for node_id in candidates:
                node = self.nodes[node_id]

                # Get trust score with propagation
                trust_score = await self.get_trust_score(node_id, use_propagation=True)

                if trust_score is None:
                    continue

                # Calculate relevance score (simplified for this implementation)
                relevance = self._calculate_query_relevance(query, node.content, node.keywords)

                # Calculate dimensional trust score
                dimensional_scores = []
                for dimension in trust_dimensions:
                    if dimension in trust_score.mean_trust:
                        dimensional_scores.append(trust_score.mean_trust[dimension])

                avg_trust = np.mean(dimensional_scores) if dimensional_scores else 0.0

                # Filter by minimum trust
                if avg_trust >= min_trust_score:
                    # Combined score: trust + relevance
                    combined_score = 0.7 * avg_trust + 0.3 * relevance
                    scored_candidates.append((node, combined_score, trust_score))

            # Sort by combined score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Return top k
            return scored_candidates[:k]

        except Exception as e:
            logger.error(f"Trust-based retrieval failed: {e}")
            return []

    async def detect_conflicts(
        self, content_similarity_threshold: float = 0.8, trust_difference_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Detect conflicts between high-trust sources."""
        try:
            conflicts = []

            # Get all high-trust nodes
            high_trust_nodes = []
            for node_id, node in self.nodes.items():
                trust_score = await self.get_trust_score(node_id)
                if trust_score and trust_score.overall_trust > self.trust_threshold:
                    high_trust_nodes.append((node_id, node, trust_score))

            # Compare pairs for conflicts
            for i, (id1, node1, trust1) in enumerate(high_trust_nodes):
                for j, (id2, node2, trust2) in enumerate(high_trust_nodes[i + 1 :], i + 1):
                    # Check content similarity (simplified)
                    content_sim = self._calculate_content_similarity(node1.content, node2.content)

                    if content_sim > content_similarity_threshold:
                        # Check for trust difference
                        trust_diff = abs(trust1.overall_trust - trust2.overall_trust)

                        if trust_diff > trust_difference_threshold:
                            conflicts.append(
                                {
                                    "type": "trust_conflict",
                                    "node1": id1,
                                    "node2": id2,
                                    "content_similarity": content_sim,
                                    "trust_difference": trust_diff,
                                    "trust1": trust1.overall_trust,
                                    "trust2": trust2.overall_trust,
                                    "severity": trust_diff / content_sim,
                                }
                            )

            self.stats["conflicts_detected"] += len(conflicts)
            return conflicts

        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []

    async def get_network_status(self) -> dict[str, Any]:
        """Get comprehensive network status."""
        try:
            # Calculate network statistics
            total_nodes = len(self.nodes)
            total_evidence = sum(len(node.evidence) for node in self.nodes.values())

            # Trust distribution
            trust_scores = []
            uncertainty_scores = []

            for node in self.nodes.values():
                trust_scores.append(node.trust_score.overall_trust)
                uncertainty_scores.append(node.trust_score.overall_uncertainty)

            avg_trust = np.mean(trust_scores) if trust_scores else 0.0
            avg_uncertainty = np.mean(uncertainty_scores) if uncertainty_scores else 0.0

            # Network connectivity
            total_relationships = sum(len(node.trusts) + len(node.trusted_by) for node in self.nodes.values())
            avg_connectivity = total_relationships / max(1, total_nodes)

            # Evidence type distribution
            evidence_by_type = {et.value: 0 for et in EvidenceType}
            evidence_by_dimension = {td.value: 0 for td in TrustDimension}

            for node in self.nodes.values():
                for evidence in node.evidence:
                    evidence_by_type[evidence.evidence_type.value] += 1
                    evidence_by_dimension[evidence.dimension.value] += 1

            return {
                "status": "healthy",
                "network_size": {
                    "total_nodes": total_nodes,
                    "total_evidence": total_evidence,
                    "total_relationships": total_relationships,
                },
                "trust_metrics": {
                    "average_trust": avg_trust,
                    "average_uncertainty": avg_uncertainty,
                    "average_connectivity": avg_connectivity,
                    "trust_threshold": self.trust_threshold,
                },
                "evidence_distribution": {
                    "by_type": evidence_by_type,
                    "by_dimension": evidence_by_dimension,
                },
                "performance": {
                    "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["trust_propagations"]),
                    "conflicts_detected": self.stats["conflicts_detected"],
                },
                "configuration": {
                    "max_propagation_depth": self.max_propagation_depth,
                    "trust_threshold": self.trust_threshold,
                    "uncertainty_threshold": self.uncertainty_threshold,
                    "decay_rate": self.decay_rate,
                },
                "statistics": self.stats.copy(),
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error", "error": str(e)}

    # Private implementation methods

    async def _update_indexes(self, node: TrustNode):
        """Update search indexes for a node."""
        try:
            # Content hash index
            if node.content_hash not in self.content_index:
                self.content_index[node.content_hash] = set()
            self.content_index[node.content_hash].add(node.id)

            # Domain index
            if node.domain:
                if node.domain not in self.domain_index:
                    self.domain_index[node.domain] = set()
                self.domain_index[node.domain].add(node.id)

            # Keyword index
            for keyword in node.keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = set()
                self.keyword_index[keyword].add(node.id)

        except Exception as e:
            logger.warning(f"Failed to update indexes: {e}")

    async def _get_candidate_nodes(self, query: str, domain_filter: str | None = None) -> list[str]:
        """Get candidate nodes for retrieval."""
        try:
            candidates = set()

            # Domain filtering
            if domain_filter and domain_filter in self.domain_index:
                candidates.update(self.domain_index[domain_filter])
            else:
                # Use all nodes if no domain filter
                candidates.update(self.nodes.keys())

            # Keyword matching (simplified)
            query_words = set(query.lower().split())
            for keyword in query_words:
                if keyword in self.keyword_index:
                    candidates.update(self.keyword_index[keyword])

            return list(candidates)

        except Exception as e:
            logger.warning(f"Candidate selection failed: {e}")
            return list(self.nodes.keys())

    def _calculate_query_relevance(self, query: str, content: str, keywords: list[str]) -> float:
        """Calculate relevance score between query and content."""
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            keyword_set = set(kw.lower() for kw in keywords)

            # Word overlap similarity
            content_overlap = len(query_words.intersection(content_words))
            content_union = len(query_words.union(content_words))
            content_sim = content_overlap / max(1, content_union)

            # Keyword match bonus
            keyword_overlap = len(query_words.intersection(keyword_set))
            keyword_bonus = keyword_overlap / max(1, len(query_words))

            # Combined relevance
            relevance = 0.7 * content_sim + 0.3 * keyword_bonus

            return min(1.0, relevance)

        except Exception:
            return 0.0

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between content strings."""
        try:
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())

            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0

            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return overlap / union if union > 0 else 0.0

        except Exception:
            return 0.0

    async def _propagate_trust(self, source_node_id: str, max_depth: int) -> TrustPropagationResult:
        """Propagate trust through the network."""
        try:
            start_time = time.time()

            result = TrustPropagationResult()
            result.source_nodes = [source_node_id]
            result.propagation_depth = max_depth

            # BFS-style trust propagation
            visited = set()
            current_level = {source_node_id}
            all_paths = []

            for depth in range(max_depth):
                if not current_level:
                    break

                next_level = set()

                for node_id in current_level:
                    if node_id in visited:
                        continue
                    visited.add(node_id)

                    node = self.nodes[node_id]

                    # Propagate to trusted nodes
                    for trusted_id in node.trusts:
                        if trusted_id in self.nodes and trusted_id not in visited:
                            next_level.add(trusted_id)
                            # Record path
                            path = [source_node_id] + [trusted_id]  # Simplified path
                            weight = node.calculate_trust_propagation_weight(TrustDimension.ACCURACY)
                            all_paths.append((path, weight))

                current_level = next_level

            # Aggregate trust scores from paths
            for dimension in TrustDimension:
                dimension_scores = []
                dimension_weights = []

                for path, weight in all_paths:
                    if len(path) > 1:
                        end_node_id = path[-1]
                        if end_node_id in self.nodes:
                            end_node = self.nodes[end_node_id]
                            if dimension in end_node.trust_score.mean_trust:
                                score = end_node.trust_score.mean_trust[dimension]
                                end_node.trust_score.uncertainty[dimension]

                                # Weight by path length and trust propagation weight
                                path_decay = 0.8 ** (len(path) - 1)  # Decay with distance
                                final_weight = weight * path_decay

                                dimension_scores.append(score * final_weight)
                                dimension_weights.append(final_weight)

                # Calculate weighted average
                if dimension_scores:
                    total_weight = sum(dimension_weights)
                    if total_weight > 0:
                        weighted_avg = sum(dimension_scores) / total_weight
                        result.propagated_trust[dimension] = weighted_avg

                        # Estimate propagated uncertainty
                        weight_normalized = [w / total_weight for w in dimension_weights]
                        weighted_variance = sum(w * 0.1 for w in weight_normalized)  # Simplified
                        result.propagated_uncertainty[dimension] = math.sqrt(weighted_variance)

            # Calculate overall confidence
            if result.propagated_trust:
                result.overall_confidence = np.mean(list(result.propagated_trust.values()))

            result.trust_paths = [path for path, _ in all_paths]
            result.path_weights = [weight for _, weight in all_paths]
            result.computation_time_ms = (time.time() - start_time) * 1000
            result.nodes_evaluated = len(visited)

            return result

        except Exception as e:
            logger.error(f"Trust propagation failed: {e}")
            return TrustPropagationResult()

    def _merge_trust_scores(self, direct_trust: TrustScore, propagation_result: TrustPropagationResult) -> TrustScore:
        """Merge direct trust with propagated trust."""
        try:
            merged = TrustScore()

            # Merge by dimension
            for dimension in TrustDimension:
                direct_score = direct_trust.mean_trust.get(dimension, 0.0)
                direct_uncertainty = direct_trust.uncertainty.get(dimension, 1.0)

                propagated_score = propagation_result.propagated_trust.get(dimension, 0.0)
                propagated_uncertainty = propagation_result.propagated_uncertainty.get(dimension, 1.0)

                # Weight by inverse uncertainty (more certain scores get higher weight)
                direct_weight = 1.0 / (direct_uncertainty + 0.1)
                propagated_weight = 1.0 / (propagated_uncertainty + 0.1)
                total_weight = direct_weight + propagated_weight

                if total_weight > 0:
                    merged_score = (direct_score * direct_weight + propagated_score * propagated_weight) / total_weight
                    merged_uncertainty = math.sqrt(
                        (direct_uncertainty**2 * direct_weight + propagated_uncertainty**2 * propagated_weight)
                        / total_weight
                    )

                    merged.mean_trust[dimension] = merged_score
                    merged.uncertainty[dimension] = merged_uncertainty

            # Update overall scores
            if merged.mean_trust:
                merged.overall_trust = np.mean(list(merged.mean_trust.values()))
                merged.overall_uncertainty = np.mean(list(merged.uncertainty.values()))

            merged.last_updated = datetime.now()

            return merged

        except Exception as e:
            logger.error(f"Trust score merging failed: {e}")
            return direct_trust

    def _clear_related_caches(self, node_id: str):
        """Clear caches related to a node."""
        try:
            # Clear trust cache
            if node_id in self.trust_cache:
                del self.trust_cache[node_id]

            # Clear propagation cache entries involving this node
            keys_to_remove = []
            for cache_key in self.propagation_cache:
                if cache_key[0] == node_id:
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                del self.propagation_cache[key]

        except Exception as e:
            logger.warning(f"Cache clearing failed: {e}")

    async def _background_maintenance(self):
        """Background task for network maintenance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Update trust scores with temporal decay
                datetime.now()
                for node in self.nodes.values():
                    node._update_trust_from_evidence()

                # Clear old cache entries
                self.trust_cache.clear()
                self.propagation_cache.clear()

                logger.info("Completed background trust network maintenance")

            except Exception as e:
                logger.error(f"Background maintenance failed: {e}")


# Factory functions for creating evidence


def create_accuracy_evidence(
    positive: float = 0.0, negative: float = 0.0, evaluator_id: str = "", confidence: float = 1.0
) -> TrustEvidence:
    """Create evidence for accuracy dimension."""
    return TrustEvidence(
        evidence_type=EvidenceType.DIRECT,
        dimension=TrustDimension.ACCURACY,
        positive_evidence=positive,
        negative_evidence=negative,
        evaluator_id=evaluator_id,
        confidence=confidence,
    )


def create_authority_evidence(
    authority_score: float, source_credentials: dict[str, Any], evaluator_id: str = ""
) -> TrustEvidence:
    """Create evidence for authority dimension."""
    return TrustEvidence(
        evidence_type=EvidenceType.PEER,
        dimension=TrustDimension.AUTHORITY,
        positive_evidence=authority_score,
        negative_evidence=0.0,
        evaluator_id=evaluator_id,
        context=source_credentials,
        confidence=0.9,
    )


def create_recency_evidence(content_age_days: float, domain: str = "") -> TrustEvidence:
    """Create evidence for recency dimension."""
    # Recency score decreases with age
    recency_score = max(0.0, 1.0 - (content_age_days / 365.0))  # Decay over 1 year

    return TrustEvidence(
        evidence_type=EvidenceType.TEMPORAL,
        dimension=TrustDimension.RECENCY,
        positive_evidence=recency_score,
        negative_evidence=1.0 - recency_score,
        context={"domain": domain, "age_days": content_age_days},
        confidence=0.8,
        decay_rate=0.02,  # Slower decay for temporal evidence
    )


if __name__ == "__main__":

    async def test_bayesian_trust():
        """Test Bayesian trust network functionality."""
        # Create trust network
        trust_network = BayesianTrustNetwork(max_propagation_depth=3, trust_threshold=0.6, uncertainty_threshold=0.3)

        await trust_network.initialize()

        # Add sources
        source1_id = await trust_network.add_source(
            source_identifier="research_paper_001",
            content="Machine learning algorithms can achieve high accuracy on structured data.",
            source_type="academic_paper",
            domain="machine_learning",
            keywords=["machine learning", "accuracy", "algorithms"],
            metadata={"author": "Dr. Smith", "year": 2023, "citations": 150},
        )

        source2_id = await trust_network.add_source(
            source_identifier="blog_post_001",
            content="AI will replace all human jobs in the next 5 years.",
            source_type="blog_post",
            domain="artificial_intelligence",
            keywords=["AI", "jobs", "automation"],
            metadata={"author": "Tech Blogger", "year": 2024, "citations": 0},
        )

        print(f"Added sources: {source1_id}, {source2_id}")

        # Add evidence
        await trust_network.add_evidence(
            source1_id,
            EvidenceType.PEER,
            TrustDimension.ACCURACY,
            positive_evidence=0.9,
            negative_evidence=0.1,
            evaluator_id="peer_reviewer_1",
            confidence=0.95,
        )

        await trust_network.add_evidence(
            source1_id,
            EvidenceType.CITATION,
            TrustDimension.AUTHORITY,
            positive_evidence=0.8,
            negative_evidence=0.0,
            evaluator_id="citation_analyzer",
            confidence=0.9,
        )

        await trust_network.add_evidence(
            source2_id,
            EvidenceType.DIRECT,
            TrustDimension.ACCURACY,
            positive_evidence=0.2,
            negative_evidence=0.7,
            evaluator_id="fact_checker",
            confidence=0.8,
        )

        # Get trust scores
        trust1 = await trust_network.get_trust_score(source1_id)
        trust2 = await trust_network.get_trust_score(source2_id)

        print(f"Trust score 1: {trust1.overall_trust:.3f} (±{trust1.overall_uncertainty:.3f})")
        print(f"Trust score 2: {trust2.overall_trust:.3f} (±{trust2.overall_uncertainty:.3f})")

        # Test retrieval with trust
        results = await trust_network.retrieve_with_trust_propagation(
            query="machine learning accuracy", k=5, min_trust_score=0.3
        )

        print(f"Retrieved {len(results)} trusted sources")
        for node, score, trust in results:
            print(f"  Source: {node.source_identifier}, Score: {score:.3f}, Trust: {trust.overall_trust:.3f}")

        # Check network status
        status = await trust_network.get_network_status()
        print(f"Network status: {status['status']}")
        print(f"Network size: {status['network_size']}")
        print(f"Trust metrics: {status['trust_metrics']}")

    import asyncio

    asyncio.run(test_bayesian_trust())
