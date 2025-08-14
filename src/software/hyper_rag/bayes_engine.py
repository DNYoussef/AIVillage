"""
Bayesian Belief Engine for Hyper RAG

Implements dynamic Bayes Net with probability ratings for all beliefs and ideas:
- Every belief/idea has a probability rating (0.0-1.0)
- Ratings update continuously as new information is acquired
- Semantic connections between beliefs influence probability propagation
- Supports belief revision and uncertainty quantification
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BeliefType(Enum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    OPINION = "opinion"
    PREDICTION = "prediction"
    CAUSAL_RELATIONSHIP = "causal_relationship"


class EvidenceStrength(Enum):
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


@dataclass
class BeliefNode:
    """A single belief/idea in the Bayes Net"""

    id: str
    content: str
    belief_type: BeliefType
    probability: float = 0.5  # Prior probability
    confidence: float = 0.5  # Confidence in this probability

    # Evidence tracking
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    evidence_weights: dict[str, float] = field(default_factory=dict)

    # Network connections
    parent_beliefs: list[str] = field(default_factory=list)  # Influences this belief
    child_beliefs: list[str] = field(default_factory=list)  # This belief influences
    semantic_connections: list[tuple[str, float]] = field(default_factory=list)  # (id, strength)

    # Metadata from RAG system
    source_documents: list[str] = field(default_factory=list)
    book_summary_tag: str = ""
    chapter_summary_tag: str = ""

    # Temporal tracking
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0


@dataclass
class Evidence:
    """Evidence that supports or contradicts beliefs"""

    id: str
    content: str
    strength: EvidenceStrength
    reliability: float  # Source reliability (0.0-1.0)

    supports: list[str] = field(default_factory=list)  # Belief IDs this supports
    contradicts: list[str] = field(default_factory=list)  # Belief IDs this contradicts

    source: str = ""
    timestamp: float = field(default_factory=time.time)


class BayesianBeliefEngine:
    """
    Bayesian Belief Engine for Hyper RAG

    Manages a dynamic Bayes Net where:
    - Every belief/idea has probability ratings
    - Probabilities update as new information arrives
    - Semantic connections enable belief propagation
    - Supports complex reasoning about uncertainty
    """

    def __init__(self):
        self.beliefs: dict[str, BeliefNode] = {}
        self.evidence: dict[str, Evidence] = {}

        # Network structure
        self.belief_network = defaultdict(list)  # id -> [connected_id, ...]
        self.semantic_graph = defaultdict(list)  # id -> [(connected_id, strength), ...]

        # Update tracking
        self.propagation_queue = []
        self.update_history = []

        # Configuration
        self.propagation_damping = 0.8  # How much influence propagates
        self.min_probability_change = 0.01  # Minimum change to trigger updates
        self.max_propagation_depth = 5

        self.initialized = False

    async def initialize(self):
        """Initialize the Bayesian Belief Engine"""
        try:
            logger.info("Initializing Bayesian Belief Engine...")

            # Start background tasks
            asyncio.create_task(self._process_propagation_queue())
            asyncio.create_task(self._periodic_belief_maintenance())

            self.initialized = True
            logger.info("✅ Bayesian Belief Engine initialized")

        except Exception as e:
            logger.error(f"❌ Bayesian Belief Engine initialization failed: {e}")
            raise

    async def add_belief(
        self,
        content: str,
        belief_type: BeliefType,
        initial_probability: float = 0.5,
        book_summary: str = "",
        chapter_summary: str = "",
        source_docs: list[str] = None,
    ) -> str:
        """Add new belief to the network"""

        belief_id = f"belief_{len(self.beliefs)}_{int(time.time())}"

        belief = BeliefNode(
            id=belief_id,
            content=content,
            belief_type=belief_type,
            probability=initial_probability,
            book_summary_tag=book_summary,
            chapter_summary_tag=chapter_summary,
            source_documents=source_docs or [],
        )

        self.beliefs[belief_id] = belief

        # Find semantic connections with existing beliefs
        await self._discover_semantic_connections(belief_id)

        logger.info(f"Added belief: {belief_id} (P={initial_probability:.3f})")
        return belief_id

    async def add_evidence(
        self,
        content: str,
        strength: EvidenceStrength,
        reliability: float,
        supports: list[str] = None,
        contradicts: list[str] = None,
        source: str = "",
    ) -> str:
        """Add evidence that affects belief probabilities"""

        evidence_id = f"evidence_{len(self.evidence)}_{int(time.time())}"

        evidence = Evidence(
            id=evidence_id,
            content=content,
            strength=strength,
            reliability=reliability,
            supports=supports or [],
            contradicts=contradicts or [],
            source=source,
        )

        self.evidence[evidence_id] = evidence

        # Update affected beliefs
        affected_beliefs = set(evidence.supports + evidence.contradicts)
        for belief_id in affected_beliefs:
            if belief_id in self.beliefs:
                await self._update_belief_from_evidence(belief_id, evidence_id)

        logger.info(f"Added evidence: {evidence_id} affecting {len(affected_beliefs)} beliefs")
        return evidence_id

    async def update_belief_probability(self, belief_id: str, new_evidence_id: str = None):
        """Update belief probability based on all evidence"""

        if belief_id not in self.beliefs:
            logger.warning(f"Belief {belief_id} not found")
            return

        belief = self.beliefs[belief_id]
        old_probability = belief.probability

        # Calculate new probability using Bayesian updating
        new_probability = await self._calculate_bayesian_probability(belief_id)

        # Update if change is significant
        if abs(new_probability - old_probability) >= self.min_probability_change:
            belief.probability = new_probability
            belief.last_updated = time.time()
            belief.update_count += 1

            # Queue for propagation to connected beliefs
            self.propagation_queue.append((belief_id, old_probability, new_probability))

            logger.info(f"Updated belief {belief_id}: {old_probability:.3f} → {new_probability:.3f}")

    async def _calculate_bayesian_probability(self, belief_id: str) -> float:
        """Calculate probability using Bayesian inference"""

        belief = self.beliefs[belief_id]

        # Start with prior
        log_odds = np.log(belief.probability / (1 - belief.probability + 1e-10))

        # Apply evidence
        for evidence_id in belief.supporting_evidence:
            if evidence_id in self.evidence:
                evidence = self.evidence[evidence_id]
                evidence_strength = evidence.strength.value * evidence.reliability
                log_odds += np.log(evidence_strength / (1 - evidence_strength + 1e-10))

        for evidence_id in belief.contradicting_evidence:
            if evidence_id in self.evidence:
                evidence = self.evidence[evidence_id]
                evidence_strength = evidence.strength.value * evidence.reliability
                log_odds -= np.log(evidence_strength / (1 - evidence_strength + 1e-10))

        # Apply influence from connected beliefs
        for parent_id in belief.parent_beliefs:
            if parent_id in self.beliefs:
                parent = self.beliefs[parent_id]
                influence_strength = 0.3  # Could be learned
                log_odds += influence_strength * np.log(parent.probability / (1 - parent.probability + 1e-10))

        # Convert back to probability
        probability = 1 / (1 + np.exp(-log_odds))

        # Ensure bounds
        return max(0.01, min(0.99, probability))

    async def _discover_semantic_connections(self, belief_id: str):
        """Find semantic connections with existing beliefs"""

        new_belief = self.beliefs[belief_id]

        # Simple semantic similarity (would use actual embeddings)
        for existing_id, existing_belief in self.beliefs.items():
            if existing_id == belief_id:
                continue

            # Calculate semantic similarity
            similarity = await self._calculate_semantic_similarity(new_belief.content, existing_belief.content)

            if similarity > 0.7:  # Strong semantic connection
                new_belief.semantic_connections.append((existing_id, similarity))
                existing_belief.semantic_connections.append((belief_id, similarity))

                # Add to network graph
                self.semantic_graph[belief_id].append((existing_id, similarity))
                self.semantic_graph[existing_id].append((belief_id, similarity))

    async def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between content (simplified)"""
        # Would use actual embedding similarity
        common_words = set(content1.lower().split()) & set(content2.lower().split())
        total_words = set(content1.lower().split()) | set(content2.lower().split())
        return len(common_words) / max(len(total_words), 1)

    async def _update_belief_from_evidence(self, belief_id: str, evidence_id: str):
        """Update a belief based on new evidence"""

        belief = self.beliefs[belief_id]
        evidence = self.evidence[evidence_id]

        # Add evidence to belief
        if evidence_id in evidence.supports and belief_id in evidence.supports:
            if evidence_id not in belief.supporting_evidence:
                belief.supporting_evidence.append(evidence_id)

        if evidence_id in evidence.contradicts and belief_id in evidence.contradicts:
            if evidence_id not in belief.contradicting_evidence:
                belief.contradicting_evidence.append(evidence_id)

        # Update probability
        await self.update_belief_probability(belief_id, evidence_id)

    async def _process_propagation_queue(self):
        """Process belief probability propagation"""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second

                if not self.propagation_queue:
                    continue

                # Process all queued updates
                updates = self.propagation_queue.copy()
                self.propagation_queue.clear()

                for belief_id, old_prob, new_prob in updates:
                    await self._propagate_belief_change(belief_id, old_prob, new_prob)

            except Exception as e:
                logger.error(f"Error in propagation queue: {e}")

    async def _propagate_belief_change(self, belief_id: str, old_prob: float, new_prob: float, depth: int = 0):
        """Propagate probability changes to connected beliefs"""

        if depth >= self.max_propagation_depth:
            return

        belief = self.beliefs[belief_id]
        probability_change = new_prob - old_prob

        # Propagate to child beliefs
        for child_id in belief.child_beliefs:
            if child_id in self.beliefs:
                child = self.beliefs[child_id]

                # Calculate influence (dampened by depth)
                influence = probability_change * (self.propagation_damping**depth)

                if abs(influence) >= self.min_probability_change:
                    old_child_prob = child.probability
                    new_child_prob = max(0.01, min(0.99, child.probability + influence))

                    child.probability = new_child_prob
                    child.last_updated = time.time()

                    # Continue propagation
                    await self._propagate_belief_change(child_id, old_child_prob, new_child_prob, depth + 1)

    async def _periodic_belief_maintenance(self):
        """Periodic maintenance of belief network"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour

                # Decay old beliefs slightly (knowledge decay)
                await self._apply_temporal_decay()

                # Strengthen frequently accessed beliefs
                await self._reinforce_active_beliefs()

                logger.info("Periodic belief maintenance completed")

            except Exception as e:
                logger.error(f"Error in belief maintenance: {e}")

    async def _apply_temporal_decay(self):
        """Apply slight decay to old, unused beliefs"""
        current_time = time.time()

        for belief in self.beliefs.values():
            age_hours = (current_time - belief.last_updated) / 3600

            if age_hours > 24:  # Beliefs older than 24 hours
                decay_factor = 0.995 ** (age_hours / 24)  # Very slight decay
                belief.confidence *= decay_factor

    async def _reinforce_active_beliefs(self):
        """Reinforce beliefs that are frequently accessed"""
        # Would track access patterns and reinforce popular beliefs
        pass

    async def query_beliefs(
        self, query: str, min_probability: float = 0.5, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Query beliefs based on content and probability"""

        results = []

        for belief in self.beliefs.values():
            # Simple content matching (would use semantic search)
            if query.lower() in belief.content.lower():
                if belief.probability >= min_probability:
                    results.append(
                        {
                            "id": belief.id,
                            "content": belief.content,
                            "probability": belief.probability,
                            "confidence": belief.confidence,
                            "type": belief.belief_type.value,
                            "book_context": belief.book_summary_tag,
                            "chapter_context": belief.chapter_summary_tag,
                            "last_updated": belief.last_updated,
                            "supporting_evidence_count": len(belief.supporting_evidence),
                            "contradicting_evidence_count": len(belief.contradicting_evidence),
                        }
                    )

        # Sort by probability * confidence
        results.sort(key=lambda x: x["probability"] * x["confidence"], reverse=True)

        return results[:max_results]

    async def get_belief_network_stats(self) -> dict[str, Any]:
        """Get statistics about the belief network"""

        if not self.beliefs:
            return {"total_beliefs": 0, "total_evidence": 0}

        # Calculate statistics
        probabilities = [b.probability for b in self.beliefs.values()]
        confidences = [b.confidence for b in self.beliefs.values()]

        return {
            "total_beliefs": len(self.beliefs),
            "total_evidence": len(self.evidence),
            "avg_probability": np.mean(probabilities),
            "avg_confidence": np.mean(confidences),
            "high_confidence_beliefs": sum(1 for c in confidences if c > 0.8),
            "uncertain_beliefs": sum(1 for p in probabilities if 0.3 <= p <= 0.7),
            "strong_beliefs": sum(1 for p in probabilities if p > 0.8 or p < 0.2),
            "total_connections": sum(len(b.semantic_connections) for b in self.beliefs.values()),
            "updates_processed": sum(b.update_count for b in self.beliefs.values()),
        }
