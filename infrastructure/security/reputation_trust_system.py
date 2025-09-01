"""
Reputation-Based Trust System for Federated Networks
===================================================

Advanced reputation and trust management system for federated learning participants.
Implements dynamic trust scoring, reputation decay, and trust-based decision making.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class TrustMetric(Enum):
    """Types of trust metrics."""

    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    HONESTY = "honesty"
    AVAILABILITY = "availability"
    CONSISTENCY = "consistency"
    COOPERATION = "cooperation"
    SECURITY = "security"


class ReputationEvent(Enum):
    """Types of reputation events."""

    SUCCESSFUL_TASK = "successful_task"
    FAILED_TASK = "failed_task"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    HONEST_BEHAVIOR = "honest_behavior"
    AVAILABILITY_HIGH = "availability_high"
    AVAILABILITY_LOW = "availability_low"
    SECURITY_VIOLATION = "security_violation"
    COOPERATION = "cooperation"
    NON_COOPERATION = "non_cooperation"


class TrustLevel(Enum):
    """Trust level categories."""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


@dataclass
class ReputationScore:
    """Individual reputation score component."""

    metric: TrustMetric
    score: float  # 0.0 to 1.0
    confidence: float  # How confident we are in this score
    last_updated: float = field(default_factory=time.time)
    sample_count: int = 0
    historical_values: List[float] = field(default_factory=list)


@dataclass
class TrustProfile:
    """Complete trust profile for a node."""

    node_id: str
    overall_trust_score: float = 0.5  # Start at neutral
    trust_level: TrustLevel = TrustLevel.MEDIUM
    reputation_scores: Dict[TrustMetric, ReputationScore] = field(default_factory=dict)
    interaction_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    trust_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, score)
    recommendations: Dict[str, float] = field(default_factory=dict)  # From other nodes
    penalties: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrustTransaction:
    """Trust-based transaction record."""

    transaction_id: str
    requester_node: str
    provider_node: str
    transaction_type: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    trust_requirement: float = 0.5
    provider_trust_score: float = 0.5
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: Optional[bool] = None
    trust_impact: float = 0.0


@dataclass
class RecommendationCredential:
    """Trust recommendation from another node."""

    recommender_id: str
    recommended_node: str
    trust_score: float
    confidence: float
    interaction_count: int
    recommendation_reason: str
    timestamp: float = field(default_factory=time.time)
    signature: Optional[bytes] = None


class ReputationTrustSystem:
    """
    Advanced reputation and trust management system.

    Features:
    - Multi-dimensional trust scoring
    - Reputation decay over time
    - Trust-based decision making
    - Collaborative filtering for recommendations
    - Anti-gaming mechanisms
    - Bootstrap trust for new nodes
    - Trust delegation and attestation
    """

    def __init__(self, node_id: str):
        """Initialize reputation trust system."""
        self.node_id = node_id

        # Trust profiles for all known nodes
        self.trust_profiles: Dict[str, TrustProfile] = {}

        # Trust transactions
        self.trust_transactions: List[TrustTransaction] = []
        self.active_transactions: Dict[str, TrustTransaction] = {}

        # Recommendation system
        self.trust_recommendations: List[RecommendationCredential] = []
        self.recommendation_network: Dict[str, Set[str]] = {}  # Who recommends whom

        # System configuration
        self.trust_config = {
            "initial_trust_score": 0.5,
            "trust_decay_rate": 0.98,  # Daily decay factor
            "min_interactions_for_trust": 5,
            "reputation_weights": {
                TrustMetric.RELIABILITY: 0.25,
                TrustMetric.PERFORMANCE: 0.2,
                TrustMetric.HONESTY: 0.2,
                TrustMetric.AVAILABILITY: 0.15,
                TrustMetric.CONSISTENCY: 0.1,
                TrustMetric.COOPERATION: 0.05,
                TrustMetric.SECURITY: 0.05,
            },
            "trust_thresholds": {
                TrustLevel.UNTRUSTED: 0.2,
                TrustLevel.LOW: 0.4,
                TrustLevel.MEDIUM: 0.6,
                TrustLevel.HIGH: 0.8,
                TrustLevel.VERIFIED: 0.9,
            },
            "recommendation_weight": 0.3,  # How much to trust recommendations
            "max_recommendation_age_days": 30,
            "penalty_decay_days": 7,
            "reward_boost_factor": 1.1,
            "byzantine_penalty": -0.2,
            "cooperation_reward": 0.05,
        }

        # Statistics
        self.trust_stats = {
            "nodes_tracked": 0,
            "total_interactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "byzantine_detections": 0,
            "trust_recommendations_given": 0,
            "trust_recommendations_received": 0,
            "trust_delegations": 0,
            "reputation_updates": 0,
        }

        logger.info(f"Reputation Trust System initialized for node {node_id}")

    async def initialize_node_trust(
        self,
        node_id: str,
        initial_reputation: Optional[Dict[TrustMetric, float]] = None,
        bootstrap_method: str = "neutral",
    ) -> TrustProfile:
        """Initialize trust profile for a new node."""

        if node_id in self.trust_profiles:
            return self.trust_profiles[node_id]

        # Create new trust profile
        profile = TrustProfile(node_id=node_id)

        # Set initial trust based on bootstrap method
        if bootstrap_method == "neutral":
            profile.overall_trust_score = self.trust_config["initial_trust_score"]
        elif bootstrap_method == "pessimistic":
            profile.overall_trust_score = 0.3
        elif bootstrap_method == "optimistic":
            profile.overall_trust_score = 0.7
        elif bootstrap_method == "zero_trust":
            profile.overall_trust_score = 0.1
            profile.trust_level = TrustLevel.UNTRUSTED

        # Initialize reputation scores
        for metric in TrustMetric:
            if initial_reputation and metric in initial_reputation:
                score_value = initial_reputation[metric]
            else:
                score_value = profile.overall_trust_score

            profile.reputation_scores[metric] = ReputationScore(
                metric=metric, score=score_value, confidence=0.1  # Low confidence initially
            )

        # Set trust level
        profile.trust_level = self._calculate_trust_level(profile.overall_trust_score)

        self.trust_profiles[node_id] = profile
        self.trust_stats["nodes_tracked"] += 1

        logger.info(f"Initialized trust profile for node {node_id} with score {profile.overall_trust_score:.3f}")
        return profile

    async def record_interaction(
        self, node_id: str, interaction_type: str, outcome: str, performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Record an interaction with a node and update trust."""

        # Ensure node has trust profile
        if node_id not in self.trust_profiles:
            await self.initialize_node_trust(node_id)

        profile = self.trust_profiles[node_id]

        # Update interaction statistics
        profile.interaction_count += 1
        profile.last_interaction = time.time()

        # Determine reputation events based on interaction outcome
        reputation_events = self._map_interaction_to_events(interaction_type, outcome)

        # Process reputation events
        for event in reputation_events:
            await self._process_reputation_event(node_id, event, performance_metrics)

        # Recalculate overall trust score
        await self._recalculate_trust_score(node_id)

        self.trust_stats["total_interactions"] += 1
        self.trust_stats["reputation_updates"] += 1

        logger.debug(f"Recorded interaction with {node_id}: {interaction_type} -> {outcome}")

    async def start_trust_transaction(
        self, provider_node: str, transaction_type: str, expected_outcome: str, min_trust_requirement: float = 0.5
    ) -> Tuple[bool, Optional[str]]:
        """Start a trust-based transaction."""

        # Check if provider meets trust requirements
        if provider_node not in self.trust_profiles:
            logger.warning(f"No trust profile found for provider {provider_node}")
            return False, None

        provider_profile = self.trust_profiles[provider_node]

        if provider_profile.overall_trust_score < min_trust_requirement:
            logger.warning(
                f"Provider {provider_node} trust score {provider_profile.overall_trust_score:.3f} "
                f"below requirement {min_trust_requirement:.3f}"
            )
            return False, None

        # Create transaction record
        transaction_id = str(uuid.uuid4())
        transaction = TrustTransaction(
            transaction_id=transaction_id,
            requester_node=self.node_id,
            provider_node=provider_node,
            transaction_type=transaction_type,
            expected_outcome=expected_outcome,
            trust_requirement=min_trust_requirement,
            provider_trust_score=provider_profile.overall_trust_score,
        )

        self.trust_transactions.append(transaction)
        self.active_transactions[transaction_id] = transaction

        logger.info(f"Started trust transaction {transaction_id} with {provider_node}")
        return True, transaction_id

    async def complete_trust_transaction(
        self, transaction_id: str, actual_outcome: str, success: bool, performance_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete a trust transaction and update trust."""

        if transaction_id not in self.active_transactions:
            logger.error(f"Transaction {transaction_id} not found")
            return

        transaction = self.active_transactions[transaction_id]
        transaction.actual_outcome = actual_outcome
        transaction.completed_at = time.time()
        transaction.success = success

        # Calculate trust impact
        if success:
            # Positive impact for successful transaction
            trust_impact = 0.02  # Base positive impact

            # Bonus for exceeding expectations
            if actual_outcome == "exceeded_expectations":
                trust_impact *= 1.5

            self.trust_stats["successful_transactions"] += 1

        else:
            # Negative impact for failed transaction
            trust_impact = -0.05  # Base negative impact

            # Higher penalty for critical failures
            if actual_outcome == "critical_failure":
                trust_impact *= 2.0

            self.trust_stats["failed_transactions"] += 1

        transaction.trust_impact = trust_impact

        # Update provider's trust based on transaction outcome
        performance_metrics = {}
        if performance_data:
            performance_metrics.update(performance_data)

        interaction_type = f"transaction:{transaction.transaction_type}"
        outcome = "success" if success else "failure"

        await self.record_interaction(transaction.provider_node, interaction_type, outcome, performance_metrics)

        # Remove from active transactions
        del self.active_transactions[transaction_id]

        logger.info(f"Completed trust transaction {transaction_id} with outcome: {actual_outcome}")

    async def report_byzantine_behavior(
        self, node_id: str, behavior_type: str, evidence: Dict[str, Any], confidence: float = 1.0
    ) -> None:
        """Report Byzantine behavior and apply trust penalty."""

        if node_id not in self.trust_profiles:
            await self.initialize_node_trust(node_id)

        profile = self.trust_profiles[node_id]

        # Apply Byzantine penalty
        penalty_amount = self.trust_config["byzantine_penalty"] * confidence

        # Record penalty
        penalty_record = {
            "type": "byzantine_behavior",
            "behavior_type": behavior_type,
            "penalty": penalty_amount,
            "evidence": evidence,
            "confidence": confidence,
            "timestamp": time.time(),
            "reporter": self.node_id,
        }

        profile.penalties.append(penalty_record)

        # Update honesty and security metrics
        if TrustMetric.HONESTY in profile.reputation_scores:
            honesty_score = profile.reputation_scores[TrustMetric.HONESTY]
            honesty_score.score = max(0.0, honesty_score.score + penalty_amount)
            honesty_score.last_updated = time.time()
            honesty_score.sample_count += 1

        if TrustMetric.SECURITY in profile.reputation_scores:
            security_score = profile.reputation_scores[TrustMetric.SECURITY]
            security_score.score = max(0.0, security_score.score + penalty_amount)
            security_score.last_updated = time.time()
            security_score.sample_count += 1

        # Recalculate overall trust
        await self._recalculate_trust_score(node_id)

        self.trust_stats["byzantine_detections"] += 1

        logger.warning(f"Reported Byzantine behavior for {node_id}: {behavior_type}")

    async def provide_recommendation(
        self, recommended_node: str, trust_score: float, interaction_count: int, reason: str
    ) -> RecommendationCredential:
        """Provide a trust recommendation for another node."""

        if recommended_node not in self.trust_profiles:
            logger.warning(f"Cannot recommend unknown node {recommended_node}")
            return None

        # Calculate confidence based on our interactions
        our_profile = self.trust_profiles.get(recommended_node)
        confidence = min(1.0, our_profile.interaction_count / 10.0) if our_profile else 0.1

        recommendation = RecommendationCredential(
            recommender_id=self.node_id,
            recommended_node=recommended_node,
            trust_score=trust_score,
            confidence=confidence,
            interaction_count=interaction_count,
            recommendation_reason=reason,
        )

        # Sign recommendation (simplified)
        recommendation_data = f"{self.node_id}:{recommended_node}:{trust_score}:{reason}"
        recommendation.signature = hashlib.sha256(recommendation_data.encode()).digest()

        self.trust_recommendations.append(recommendation)
        self.trust_stats["trust_recommendations_given"] += 1

        # Update recommendation network
        if self.node_id not in self.recommendation_network:
            self.recommendation_network[self.node_id] = set()
        self.recommendation_network[self.node_id].add(recommended_node)

        logger.info(f"Provided recommendation for {recommended_node}: {trust_score:.3f}")
        return recommendation

    async def incorporate_recommendation(self, recommendation: RecommendationCredential) -> bool:
        """Incorporate a trust recommendation from another node."""

        # Verify recommendation signature
        if not self._verify_recommendation_signature(recommendation):
            logger.warning(f"Invalid recommendation signature from {recommendation.recommender_id}")
            return False

        # Check if recommender is trusted
        recommender_trust = 0.5  # Default
        if recommendation.recommender_id in self.trust_profiles:
            recommender_trust = self.trust_profiles[recommendation.recommender_id].overall_trust_score

        # Weight recommendation by recommender's trust
        weighted_recommendation = recommendation.trust_score * recommendation.confidence * recommender_trust

        # Update recommended node's profile
        if recommendation.recommended_node not in self.trust_profiles:
            await self.initialize_node_trust(recommendation.recommended_node)

        profile = self.trust_profiles[recommendation.recommended_node]
        profile.recommendations[recommendation.recommender_id] = weighted_recommendation

        # Recalculate trust incorporating recommendations
        await self._recalculate_trust_score(recommendation.recommended_node)

        self.trust_stats["trust_recommendations_received"] += 1

        logger.info(
            f"Incorporated recommendation for {recommendation.recommended_node} "
            f"from {recommendation.recommender_id}: {weighted_recommendation:.3f}"
        )
        return True

    async def get_trust_decision(
        self,
        node_id: str,
        required_trust_level: TrustLevel = TrustLevel.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, str]:
        """Make a trust-based decision about a node."""

        if node_id not in self.trust_profiles:
            return False, 0.0, "Node not known"

        profile = self.trust_profiles[node_id]

        # Check trust level requirement
        trust_threshold = self.trust_config["trust_thresholds"][required_trust_level]

        if profile.overall_trust_score >= trust_threshold:
            decision = True
            reason = f"Trust score {profile.overall_trust_score:.3f} meets requirement"
        else:
            decision = False
            reason = f"Trust score {profile.overall_trust_score:.3f} below requirement {trust_threshold:.3f}"

        # Consider context-specific factors
        if context:
            context_adjustment = await self._calculate_context_adjustment(profile, context)
            adjusted_score = profile.overall_trust_score + context_adjustment

            if adjusted_score >= trust_threshold and not decision:
                decision = True
                reason = f"Context-adjusted score {adjusted_score:.3f} meets requirement"
            elif adjusted_score < trust_threshold and decision:
                decision = False
                reason = f"Context-adjusted score {adjusted_score:.3f} below requirement"

        # Additional checks
        if decision:
            # Check for recent penalties
            recent_penalties = [p for p in profile.penalties if time.time() - p["timestamp"] < 86400]  # Last 24 hours

            if recent_penalties:
                critical_penalties = [p for p in recent_penalties if p["type"] == "byzantine_behavior"]
                if critical_penalties:
                    decision = False
                    reason = "Recent Byzantine behavior detected"

        return decision, profile.overall_trust_score, reason

    # Private helper methods

    def _map_interaction_to_events(self, interaction_type: str, outcome: str) -> List[ReputationEvent]:
        """Map interaction outcomes to reputation events."""

        events = []

        if outcome in ["success", "completed", "good"]:
            events.append(ReputationEvent.SUCCESSFUL_TASK)
            if "cooperation" in interaction_type.lower():
                events.append(ReputationEvent.COOPERATION)

        elif outcome in ["failure", "failed", "error", "timeout"]:
            events.append(ReputationEvent.FAILED_TASK)
            if "cooperation" in interaction_type.lower():
                events.append(ReputationEvent.NON_COOPERATION)

        elif outcome in ["byzantine", "malicious", "dishonest"]:
            events.append(ReputationEvent.BYZANTINE_BEHAVIOR)

        elif outcome in ["honest", "truthful", "correct"]:
            events.append(ReputationEvent.HONEST_BEHAVIOR)

        elif outcome in ["available", "responsive"]:
            events.append(ReputationEvent.AVAILABILITY_HIGH)

        elif outcome in ["unavailable", "unresponsive", "offline"]:
            events.append(ReputationEvent.AVAILABILITY_LOW)

        return events

    async def _process_reputation_event(
        self, node_id: str, event: ReputationEvent, performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Process a reputation event and update relevant trust metrics."""

        profile = self.trust_profiles[node_id]
        updates = {}

        # Map events to trust metric updates
        if event == ReputationEvent.SUCCESSFUL_TASK:
            updates[TrustMetric.RELIABILITY] = 0.02
            updates[TrustMetric.PERFORMANCE] = 0.01

        elif event == ReputationEvent.FAILED_TASK:
            updates[TrustMetric.RELIABILITY] = -0.03
            updates[TrustMetric.PERFORMANCE] = -0.02

        elif event == ReputationEvent.BYZANTINE_BEHAVIOR:
            updates[TrustMetric.HONESTY] = -0.1
            updates[TrustMetric.SECURITY] = -0.05
            updates[TrustMetric.RELIABILITY] = -0.05

        elif event == ReputationEvent.HONEST_BEHAVIOR:
            updates[TrustMetric.HONESTY] = 0.03

        elif event == ReputationEvent.AVAILABILITY_HIGH:
            updates[TrustMetric.AVAILABILITY] = 0.02

        elif event == ReputationEvent.AVAILABILITY_LOW:
            updates[TrustMetric.AVAILABILITY] = -0.03

        elif event == ReputationEvent.COOPERATION:
            updates[TrustMetric.COOPERATION] = 0.02

        elif event == ReputationEvent.NON_COOPERATION:
            updates[TrustMetric.COOPERATION] = -0.03

        # Apply updates to reputation scores
        for metric, delta in updates.items():
            if metric in profile.reputation_scores:
                score = profile.reputation_scores[metric]

                # Apply exponential moving average
                alpha = 0.1  # Learning rate
                new_score = score.score + alpha * delta
                score.score = max(0.0, min(1.0, new_score))
                score.last_updated = time.time()
                score.sample_count += 1
                score.historical_values.append(score.score)

                # Keep only recent history
                if len(score.historical_values) > 100:
                    score.historical_values = score.historical_values[-100:]

                # Update confidence
                score.confidence = min(1.0, score.sample_count / 20.0)

        # Process performance metrics if provided
        if performance_metrics:
            await self._update_performance_metrics(node_id, performance_metrics)

    async def _update_performance_metrics(self, node_id: str, metrics: Dict[str, float]) -> None:
        """Update performance-related trust metrics."""

        profile = self.trust_profiles[node_id]

        # Map performance metrics to trust metrics
        metric_mapping = {
            "response_time": (TrustMetric.PERFORMANCE, True),  # True means lower is better
            "accuracy": (TrustMetric.PERFORMANCE, False),  # False means higher is better
            "availability": (TrustMetric.AVAILABILITY, False),
            "consistency": (TrustMetric.CONSISTENCY, False),
            "security_score": (TrustMetric.SECURITY, False),
        }

        for perf_metric, value in metrics.items():
            if perf_metric in metric_mapping:
                trust_metric, lower_is_better = metric_mapping[perf_metric]

                if trust_metric in profile.reputation_scores:
                    score = profile.reputation_scores[trust_metric]

                    # Normalize performance value to 0-1 range and convert to trust impact
                    if lower_is_better:
                        # For metrics where lower is better (e.g., response time)
                        # Convert to positive trust impact
                        normalized_impact = max(0.0, (1.0 - min(1.0, value))) * 0.05
                    else:
                        # For metrics where higher is better (e.g., accuracy)
                        normalized_impact = min(1.0, value) * 0.05 - 0.025  # Center around 0

                    # Apply update
                    alpha = 0.1
                    new_score = score.score + alpha * normalized_impact
                    score.score = max(0.0, min(1.0, new_score))
                    score.last_updated = time.time()
                    score.sample_count += 1

    async def _recalculate_trust_score(self, node_id: str) -> None:
        """Recalculate overall trust score for a node."""

        profile = self.trust_profiles[node_id]

        # Calculate weighted average of reputation scores
        total_weighted_score = 0.0
        total_weight = 0.0

        for metric, weight in self.trust_config["reputation_weights"].items():
            if metric in profile.reputation_scores:
                score = profile.reputation_scores[metric]
                confidence_weight = weight * score.confidence

                total_weighted_score += score.score * confidence_weight
                total_weight += confidence_weight

        base_score = total_weighted_score / max(0.01, total_weight)

        # Incorporate recommendations
        if profile.recommendations:
            recommendation_weight = self.trust_config["recommendation_weight"]
            avg_recommendation = np.mean(list(profile.recommendations.values()))

            # Blend base score with recommendations
            base_score = base_score * (1 - recommendation_weight) + avg_recommendation * recommendation_weight

        # Apply time decay
        time_since_last_interaction = time.time() - profile.last_interaction
        days_since_interaction = time_since_last_interaction / 86400.0

        if days_since_interaction > 1:
            decay_factor = self.trust_config["trust_decay_rate"] ** days_since_interaction
            base_score *= decay_factor

        # Apply recent penalties
        current_time = time.time()
        penalty_decay_time = self.trust_config["penalty_decay_days"] * 86400

        total_penalty = 0.0
        for penalty in profile.penalties:
            penalty_age = current_time - penalty["timestamp"]
            if penalty_age < penalty_decay_time:
                # Decay penalty over time
                decay_factor = 1.0 - (penalty_age / penalty_decay_time)
                total_penalty += penalty["penalty"] * decay_factor

        base_score += total_penalty  # Penalties are negative

        # Apply recent rewards
        reward_boost_factor = self.trust_config["reward_boost_factor"]
        for reward in profile.rewards:
            reward_age = current_time - reward["timestamp"]
            if reward_age < penalty_decay_time:
                decay_factor = 1.0 - (reward_age / penalty_decay_time)
                base_score *= 1 + reward["boost"] * decay_factor * (reward_boost_factor - 1)

        # Ensure score is within bounds
        profile.overall_trust_score = max(0.0, min(1.0, base_score))

        # Update trust level
        profile.trust_level = self._calculate_trust_level(profile.overall_trust_score)

        # Update history
        profile.trust_history.append((time.time(), profile.overall_trust_score))
        if len(profile.trust_history) > 1000:
            profile.trust_history = profile.trust_history[-1000:]

    def _calculate_trust_level(self, trust_score: float) -> TrustLevel:
        """Calculate trust level category from trust score."""

        thresholds = self.trust_config["trust_thresholds"]

        if trust_score >= thresholds[TrustLevel.VERIFIED]:
            return TrustLevel.VERIFIED
        elif trust_score >= thresholds[TrustLevel.HIGH]:
            return TrustLevel.HIGH
        elif trust_score >= thresholds[TrustLevel.MEDIUM]:
            return TrustLevel.MEDIUM
        elif trust_score >= thresholds[TrustLevel.LOW]:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED

    async def _calculate_context_adjustment(self, profile: TrustProfile, context: Dict[str, Any]) -> float:
        """Calculate context-specific trust adjustment."""

        adjustment = 0.0

        # Task complexity adjustment
        task_complexity = context.get("task_complexity", "medium")
        if task_complexity == "simple" and profile.overall_trust_score > 0.3:
            adjustment += 0.05  # Slight boost for simple tasks
        elif task_complexity == "complex" and profile.overall_trust_score < 0.7:
            adjustment -= 0.1  # Penalty for complex tasks with lower trust

        # Resource requirements adjustment
        resource_requirements = context.get("resource_requirements", "medium")
        availability_score = profile.reputation_scores.get(
            TrustMetric.AVAILABILITY, ReputationScore(TrustMetric.AVAILABILITY, 0.5)
        ).score

        if resource_requirements == "high" and availability_score < 0.6:
            adjustment -= 0.05
        elif resource_requirements == "low" and availability_score > 0.8:
            adjustment += 0.02

        # Time sensitivity adjustment
        time_sensitive = context.get("time_sensitive", False)
        reliability_score = profile.reputation_scores.get(
            TrustMetric.RELIABILITY, ReputationScore(TrustMetric.RELIABILITY, 0.5)
        ).score

        if time_sensitive and reliability_score < 0.7:
            adjustment -= 0.08

        # Security requirements adjustment
        security_critical = context.get("security_critical", False)
        security_score = profile.reputation_scores.get(
            TrustMetric.SECURITY, ReputationScore(TrustMetric.SECURITY, 0.5)
        ).score

        if security_critical and security_score < 0.8:
            adjustment -= 0.15

        return adjustment

    def _verify_recommendation_signature(self, recommendation: RecommendationCredential) -> bool:
        """Verify recommendation signature."""
        if not recommendation.signature:
            return False

        # Simplified signature verification
        recommendation_data = f"{recommendation.recommender_id}:{recommendation.recommended_node}:{recommendation.trust_score}:{recommendation.recommendation_reason}"
        expected_signature = hashlib.sha256(recommendation_data.encode()).digest()

        return recommendation.signature == expected_signature

    # Public API methods

    def get_node_trust_score(self, node_id: str) -> float:
        """Get current trust score for a node."""
        if node_id not in self.trust_profiles:
            return self.trust_config["initial_trust_score"]
        return self.trust_profiles[node_id].overall_trust_score

    def get_node_trust_level(self, node_id: str) -> TrustLevel:
        """Get current trust level for a node."""
        if node_id not in self.trust_profiles:
            return TrustLevel.MEDIUM
        return self.trust_profiles[node_id].trust_level

    def get_trusted_nodes(self, min_trust_level: TrustLevel = TrustLevel.MEDIUM) -> List[str]:
        """Get list of nodes meeting minimum trust level."""
        threshold = self.trust_config["trust_thresholds"][min_trust_level]

        return [node_id for node_id, profile in self.trust_profiles.items() if profile.overall_trust_score >= threshold]

    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get trust system statistics."""

        # Calculate trust distribution
        trust_distribution = {level.value: 0 for level in TrustLevel}
        for profile in self.trust_profiles.values():
            trust_distribution[profile.trust_level.value] += 1

        # Calculate average trust scores by metric
        avg_scores = {}
        for metric in TrustMetric:
            scores = [
                profile.reputation_scores[metric].score
                for profile in self.trust_profiles.values()
                if metric in profile.reputation_scores
            ]
            avg_scores[metric.value] = np.mean(scores) if scores else 0.0

        return {
            **self.trust_stats,
            "trust_distribution": trust_distribution,
            "average_trust_score": (
                np.mean([profile.overall_trust_score for profile in self.trust_profiles.values()])
                if self.trust_profiles
                else 0.0
            ),
            "average_scores_by_metric": avg_scores,
            "active_transactions": len(self.active_transactions),
            "total_recommendations": len(self.trust_recommendations),
        }

    def get_node_trust_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed trust information for a node."""

        if node_id not in self.trust_profiles:
            return None

        profile = self.trust_profiles[node_id]

        return {
            "node_id": node_id,
            "overall_trust_score": profile.overall_trust_score,
            "trust_level": profile.trust_level.value,
            "interaction_count": profile.interaction_count,
            "first_seen": profile.first_seen,
            "last_interaction": profile.last_interaction,
            "reputation_scores": {
                metric.value: {
                    "score": score.score,
                    "confidence": score.confidence,
                    "sample_count": score.sample_count,
                    "last_updated": score.last_updated,
                }
                for metric, score in profile.reputation_scores.items()
            },
            "recommendations_count": len(profile.recommendations),
            "penalties_count": len(profile.penalties),
            "rewards_count": len(profile.rewards),
            "trust_history_points": len(profile.trust_history),
        }

    async def export_trust_data(self) -> Dict[str, Any]:
        """Export trust system data for backup or analysis."""

        return {
            "node_id": self.node_id,
            "trust_profiles": {
                node_id: {
                    "overall_trust_score": profile.overall_trust_score,
                    "trust_level": profile.trust_level.value,
                    "interaction_count": profile.interaction_count,
                    "first_seen": profile.first_seen,
                    "last_interaction": profile.last_interaction,
                    "reputation_scores": {
                        metric.value: {
                            "score": score.score,
                            "confidence": score.confidence,
                            "sample_count": score.sample_count,
                            "last_updated": score.last_updated,
                        }
                        for metric, score in profile.reputation_scores.items()
                    },
                }
                for node_id, profile in self.trust_profiles.items()
            },
            "trust_recommendations": [
                {
                    "recommender_id": rec.recommender_id,
                    "recommended_node": rec.recommended_node,
                    "trust_score": rec.trust_score,
                    "confidence": rec.confidence,
                    "timestamp": rec.timestamp,
                }
                for rec in self.trust_recommendations
            ],
            "statistics": self.get_trust_statistics(),
            "export_timestamp": time.time(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""

        issues = []
        warnings = []

        # Check for nodes with critically low trust
        critical_trust_nodes = [
            node_id for node_id, profile in self.trust_profiles.items() if profile.overall_trust_score < 0.2
        ]

        if len(critical_trust_nodes) > len(self.trust_profiles) * 0.1:
            warnings.append(f"High number of low-trust nodes: {len(critical_trust_nodes)}")

        # Check for stale profiles
        current_time = time.time()
        stale_profiles = [
            node_id
            for node_id, profile in self.trust_profiles.items()
            if current_time - profile.last_interaction > 86400 * 7  # 1 week
        ]

        if len(stale_profiles) > len(self.trust_profiles) * 0.3:
            warnings.append(f"Many stale trust profiles: {len(stale_profiles)}")

        # Check transaction success rate
        if self.trust_stats["total_interactions"] > 0:
            success_rate = self.trust_stats["successful_transactions"] / (
                self.trust_stats["successful_transactions"] + self.trust_stats["failed_transactions"]
            )

            if success_rate < 0.7:
                issues.append(f"Low transaction success rate: {success_rate:.2%}")

        # Check for excessive Byzantine detections
        if self.trust_stats["byzantine_detections"] > self.trust_stats["nodes_tracked"] * 0.1:
            issues.append("High rate of Byzantine behavior detected")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "statistics": self.get_trust_statistics(),
        }
