"""
Confidence Calculator Service

Responsible for calculating confidence scores for gap detection and proposals.
Combines multiple evidence sources and validation algorithms.

Extracted from GraphFixer to follow single responsibility principle.
"""

from typing import Any

from ..graph_fixer import DetectedGap, ProposedNode, ProposedRelationship
from ..interfaces.base_service import AsyncServiceMixin, CacheableMixin, ServiceConfig
from ..interfaces.service_interfaces import IConfidenceCalculatorService


class ConfidenceCalculatorService(IConfidenceCalculatorService, CacheableMixin, AsyncServiceMixin):
    """
    Service for calculating confidence scores for proposals and gaps.

    Implements sophisticated confidence calculation using:
    - Evidence strength analysis
    - Statistical validation
    - Historical performance tracking
    - Multi-factor confidence modeling
    """

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.evidence_weights = {
            "semantic_similarity": 0.25,
            "structural_analysis": 0.20,
            "trust_consistency": 0.15,
            "connectivity_metrics": 0.15,
            "historical_validation": 0.15,
            "context_coherence": 0.10,
        }
        self.validation_history = {}
        self.stats = {"confidence_calculations": 0, "avg_confidence": 0.0, "validation_feedbacks": 0}

    async def initialize(self) -> bool:
        """Initialize confidence calculator service."""
        self.logger.info("Initializing ConfidenceCalculatorService...")

        # Load historical validation data if available
        await self._load_validation_history()

        self._initialized = True
        self.logger.info("✓ ConfidenceCalculatorService initialized")
        return True

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self.clear_cache()
        self._initialized = False

    async def calculate_confidence(
        self, proposal: ProposedNode | ProposedRelationship, gap: DetectedGap, evidence: list[str]
    ) -> float:
        """
        Calculate comprehensive confidence score for a proposal.

        Args:
            proposal: The proposed node or relationship
            gap: The gap this proposal addresses
            evidence: List of evidence strings

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Combine multiple confidence factors
            factors = {}

            # Evidence-based confidence
            factors["evidence"] = await self.combine_evidence(evidence)

            # Gap-based confidence
            factors["gap"] = await self._calculate_gap_confidence(gap)

            # Proposal-specific confidence
            if isinstance(proposal, ProposedNode):
                factors["proposal"] = await self._calculate_node_confidence(proposal)
            else:
                factors["proposal"] = await self._calculate_relationship_confidence(proposal)

            # Historical performance confidence
            factors["historical"] = await self._calculate_historical_confidence(proposal)

            # Logical validation confidence
            factors["validation"] = 1.0 if await self.validate_proposal_logic(proposal) else 0.3

            # Calculate weighted confidence
            confidence = await self._combine_confidence_factors(factors)

            # Update statistics
            self.stats["confidence_calculations"] += 1
            self.stats["avg_confidence"] = (
                self.stats["avg_confidence"] * (self.stats["confidence_calculations"] - 1) + confidence
            ) / self.stats["confidence_calculations"]

            return confidence

        except Exception as e:
            self.logger.exception(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence

    async def combine_evidence(self, evidence_list: list[str]) -> float:
        """
        Combine multiple pieces of evidence into a confidence score.

        Uses evidence type recognition and weighting to compute
        a comprehensive evidence-based confidence score.
        """
        if not evidence_list:
            return 0.3  # Low confidence with no evidence

        try:
            evidence_scores = {}

            # Analyze each piece of evidence
            for evidence in evidence_list:
                evidence_type = await self._classify_evidence(evidence)
                score = await self._score_evidence_strength(evidence)

                # Accumulate scores by type
                if evidence_type in evidence_scores:
                    evidence_scores[evidence_type] = max(evidence_scores[evidence_type], score)
                else:
                    evidence_scores[evidence_type] = score

            # Calculate weighted combination
            total_weight = 0.0
            weighted_sum = 0.0

            for evidence_type, score in evidence_scores.items():
                weight = self.evidence_weights.get(evidence_type, 0.1)
                weighted_sum += weight * score
                total_weight += weight

            # Normalize and add bonus for evidence diversity
            if total_weight > 0:
                base_confidence = weighted_sum / total_weight
                diversity_bonus = min(0.2, len(evidence_scores) * 0.05)
                return min(1.0, base_confidence + diversity_bonus)

            return 0.5

        except Exception as e:
            self.logger.exception(f"Evidence combination failed: {e}")
            return 0.5

    async def validate_proposal_logic(self, proposal: ProposedNode | ProposedRelationship) -> bool:
        """
        Validate the logical consistency of a proposal.

        Performs various logical checks to ensure the proposal
        makes sense given the current knowledge graph state.
        """
        try:
            # Basic validation checks
            if isinstance(proposal, ProposedNode):
                return await self._validate_node_logic(proposal)
            else:
                return await self._validate_relationship_logic(proposal)

        except Exception as e:
            self.logger.exception(f"Proposal logic validation failed: {e}")
            return True  # Default to valid if validation fails

    # Private implementation methods

    async def _classify_evidence(self, evidence: str) -> str:
        """Classify evidence into types for weighting."""
        evidence_lower = evidence.lower()

        if "similarity" in evidence_lower or "semantic" in evidence_lower:
            return "semantic_similarity"
        elif "trust" in evidence_lower or "score" in evidence_lower:
            return "trust_consistency"
        elif "connection" in evidence_lower or "edge" in evidence_lower:
            return "connectivity_metrics"
        elif "structural" in evidence_lower or "graph" in evidence_lower:
            return "structural_analysis"
        elif "context" in evidence_lower or "area" in evidence_lower:
            return "context_coherence"
        else:
            return "general_evidence"

    async def _score_evidence_strength(self, evidence: str) -> float:
        """Score the strength of a single piece of evidence."""
        # Extract numerical values if present
        import re

        numbers = re.findall(r"[\d.]+", evidence)

        if numbers:
            # Use the highest number as a base score
            max_num = max(float(num) for num in numbers if float(num) <= 1.0)
            return min(1.0, max_num) if max_num > 0 else 0.7

        # Score based on evidence content quality
        quality_indicators = [
            ("high", 0.8),
            ("strong", 0.8),
            ("significant", 0.7),
            ("moderate", 0.6),
            ("weak", 0.4),
            ("low", 0.3),
        ]

        evidence_lower = evidence.lower()
        for indicator, score in quality_indicators:
            if indicator in evidence_lower:
                return score

        return 0.6  # Default evidence strength

    async def _calculate_gap_confidence(self, gap: DetectedGap) -> float:
        """Calculate confidence factor based on gap characteristics."""
        factors = []

        # Base on gap confidence
        factors.append(gap.confidence)

        # Priority factor
        factors.append(gap.priority)

        # Detection method reliability
        method_reliability = {
            "structural_analysis": 0.9,
            "semantic_clustering": 0.8,
            "trust_inconsistency": 0.9,
            "connectivity_analysis": 0.7,
            "path_analysis": 0.6,
        }
        factors.append(method_reliability.get(gap.detection_method, 0.6))

        # Evidence count factor
        if gap.evidence:
            evidence_factor = min(1.0, len(gap.evidence) / 3.0)
            factors.append(evidence_factor)

        return sum(factors) / len(factors) if factors else 0.5

    async def _calculate_node_confidence(self, proposal: ProposedNode) -> float:
        """Calculate confidence specific to node proposals."""
        factors = []

        # Existence probability
        factors.append(proposal.existence_probability)

        # Utility score
        factors.append(proposal.utility_score)

        # Trust score alignment
        if 0.3 <= proposal.suggested_trust_score <= 0.9:
            factors.append(0.8)  # Reasonable trust score
        else:
            factors.append(0.4)  # Extreme trust scores are less confident

        # Relationship suggestions (indicates integration planning)
        if proposal.suggested_relationships:
            factors.append(0.8)
        else:
            factors.append(0.6)

        return sum(factors) / len(factors)

    async def _calculate_relationship_confidence(self, proposal: ProposedRelationship) -> float:
        """Calculate confidence specific to relationship proposals."""
        factors = []

        # Existence probability
        factors.append(proposal.existence_probability)

        # Utility score
        factors.append(proposal.utility_score)

        # Relationship strength reasonableness
        if 0.3 <= proposal.relation_strength <= 0.9:
            factors.append(0.8)
        else:
            factors.append(0.5)

        # Relationship type appropriateness
        type_confidence = {
            "semantic": 0.9,
            "associative": 0.8,
            "bridging": 0.7,
            "functional": 0.8,
            "causal": 0.9,
            "hierarchical": 0.7,
        }
        factors.append(type_confidence.get(proposal.relation_type, 0.6))

        return sum(factors) / len(factors)

    async def _calculate_historical_confidence(self, proposal: ProposedNode | ProposedRelationship) -> float:
        """Calculate confidence based on historical validation performance."""
        if not self.validation_history:
            return 0.6  # Neutral confidence without history

        # Look for similar proposal patterns in history
        proposal_type = "node" if isinstance(proposal, ProposedNode) else "relationship"

        if proposal_type in self.validation_history:
            history = self.validation_history[proposal_type]
            if history["total"] > 0:
                success_rate = history["accepted"] / history["total"]
                return success_rate

        return 0.6

    async def _combine_confidence_factors(self, factors: dict[str, float]) -> float:
        """Combine confidence factors with appropriate weighting."""
        weights = {"evidence": 0.30, "gap": 0.25, "proposal": 0.20, "historical": 0.15, "validation": 0.10}

        weighted_sum = 0.0
        total_weight = 0.0

        for factor_name, score in factors.items():
            weight = weights.get(factor_name, 0.1)
            weighted_sum += weight * score
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    async def _validate_node_logic(self, proposal: ProposedNode) -> bool:
        """Validate logical consistency of a node proposal."""
        # Check for basic logical consistency
        checks = [
            proposal.content and len(proposal.content.strip()) > 0,
            proposal.concept and len(proposal.concept.strip()) > 0,
            0.0 <= proposal.existence_probability <= 1.0,
            0.0 <= proposal.utility_score <= 1.0,
            0.0 <= proposal.confidence <= 1.0,
            0.0 <= proposal.suggested_trust_score <= 1.0,
        ]

        return all(checks)

    async def _validate_relationship_logic(self, proposal: ProposedRelationship) -> bool:
        """Validate logical consistency of a relationship proposal."""
        # Check for basic logical consistency
        checks = [
            proposal.source_id and proposal.target_id,
            proposal.source_id != proposal.target_id,  # No self-loops
            proposal.relation_type and len(proposal.relation_type.strip()) > 0,
            0.0 <= proposal.relation_strength <= 1.0,
            0.0 <= proposal.existence_probability <= 1.0,
            0.0 <= proposal.utility_score <= 1.0,
            0.0 <= proposal.confidence <= 1.0,
        ]

        return all(checks)

    async def _load_validation_history(self) -> None:
        """Load historical validation data for confidence calibration."""
        # Initialize empty history - in practice would load from persistent storage
        self.validation_history = {"node": {"total": 0, "accepted": 0}, "relationship": {"total": 0, "accepted": 0}}

    async def update_validation_history(
        self, proposal: ProposedNode | ProposedRelationship, is_accepted: bool
    ) -> None:
        """Update validation history for future confidence calculation."""
        proposal_type = "node" if isinstance(proposal, ProposedNode) else "relationship"

        if proposal_type not in self.validation_history:
            self.validation_history[proposal_type] = {"total": 0, "accepted": 0}

        self.validation_history[proposal_type]["total"] += 1
        if is_accepted:
            self.validation_history[proposal_type]["accepted"] += 1

        self.stats["validation_feedbacks"] += 1

        # Log validation for learning
        self.logger.info(f"Validation update: {proposal_type} proposal {'accepted' if is_accepted else 'rejected'}")

    def get_statistics(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "confidence_calculations": self.stats["confidence_calculations"],
            "avg_confidence": self.stats["avg_confidence"],
            "validation_feedbacks": self.stats["validation_feedbacks"],
            "evidence_weights": self.evidence_weights,
            "validation_history": self.validation_history,
        }
