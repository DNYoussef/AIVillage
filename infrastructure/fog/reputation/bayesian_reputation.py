"""
Bayesian Reputation System

Implements a sophisticated reputation management system using:
- Beta distributions for modeling reputation with uncertainty
- Time-based decay mechanisms
- Trust composition across different tiers
- Quality-aware reputation scoring
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import math
import time
from typing import Any


class EventType(Enum):
    """Types of reputation events"""

    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    QUALITY_HIGH = "quality_high"
    QUALITY_LOW = "quality_low"
    AVAILABILITY = "availability"
    UNAVAILABILITY = "unavailability"
    FRAUD_DETECTED = "fraud_detected"
    PEER_ENDORSEMENT = "peer_endorsement"
    PEER_COMPLAINT = "peer_complaint"


class ReputationTier(Enum):
    """Reputation tiers for different trust levels"""

    UNTRUSTED = 0
    BRONZE = 1
    SILVER = 2
    GOLD = 3
    PLATINUM = 4
    DIAMOND = 5


@dataclass
class ReputationEvent:
    """Individual reputation event"""

    node_id: str
    event_type: EventType
    timestamp: float
    weight: float = 1.0
    quality_score: float | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def is_positive(self) -> bool:
        """Check if event is positive for reputation"""
        positive_events = {
            EventType.TASK_SUCCESS,
            EventType.QUALITY_HIGH,
            EventType.AVAILABILITY,
            EventType.PEER_ENDORSEMENT,
        }
        return self.event_type in positive_events


@dataclass
class ReputationScore:
    """Bayesian reputation score with uncertainty"""

    node_id: str
    alpha: float  # Success count + prior
    beta: float  # Failure count + prior
    last_updated: float
    tier: ReputationTier = ReputationTier.UNTRUSTED

    @property
    def mean_score(self) -> float:
        """Expected value of Beta distribution"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Uncertainty/variance of reputation"""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    @property
    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Confidence interval for reputation score"""
        from scipy.stats import beta

        alpha_conf = (1 - confidence) / 2
        lower = beta.ppf(alpha_conf, self.alpha, self.beta)
        upper = beta.ppf(1 - alpha_conf, self.alpha, self.beta)
        return (lower, upper)

    @property
    def sample_size(self) -> int:
        """Effective sample size (total observations)"""
        return int(self.alpha + self.beta - 2)  # Remove prior

    def update_tier(self) -> ReputationTier:
        """Update reputation tier based on score and confidence"""
        score = self.mean_score
        uncertainty = self.uncertainty
        sample_size = self.sample_size

        # Higher tier requires both high score and low uncertainty
        confidence_factor = 1 / (1 + uncertainty * 10)  # Penalty for high uncertainty
        sample_factor = min(1.0, sample_size / 100)  # Require sufficient samples

        adjusted_score = score * confidence_factor * sample_factor

        if adjusted_score >= 0.95 and sample_size >= 200:
            tier = ReputationTier.DIAMOND
        elif adjusted_score >= 0.90 and sample_size >= 150:
            tier = ReputationTier.PLATINUM
        elif adjusted_score >= 0.80 and sample_size >= 100:
            tier = ReputationTier.GOLD
        elif adjusted_score >= 0.70 and sample_size >= 50:
            tier = ReputationTier.SILVER
        elif adjusted_score >= 0.60 and sample_size >= 20:
            tier = ReputationTier.BRONZE
        else:
            tier = ReputationTier.UNTRUSTED

        self.tier = tier
        return tier


@dataclass
class ReputationConfig:
    """Configuration for reputation system"""

    # Beta distribution priors
    prior_alpha: float = 1.0  # Optimistic prior
    prior_beta: float = 1.0

    # Decay parameters
    decay_rate: float = 0.001  # Per day
    min_alpha: float = 1.0
    min_beta: float = 1.0

    # Event weights
    event_weights: dict[EventType, float] = field(
        default_factory=lambda: {
            EventType.TASK_SUCCESS: 1.0,
            EventType.TASK_FAILURE: 1.0,
            EventType.QUALITY_HIGH: 1.5,
            EventType.QUALITY_LOW: 1.5,
            EventType.AVAILABILITY: 0.5,
            EventType.UNAVAILABILITY: 0.5,
            EventType.FRAUD_DETECTED: 3.0,
            EventType.PEER_ENDORSEMENT: 0.8,
            EventType.PEER_COMPLAINT: 0.8,
        }
    )

    # Quality score integration
    quality_weight_multiplier: float = 2.0  # How much quality affects weight

    # Trust composition
    trust_propagation_decay: float = 0.1  # How trust decreases with distance


class TrustComposition:
    """Handles trust composition across network tiers"""

    def __init__(self, config: ReputationConfig):
        self.config = config

    def compute_transitive_trust(
        self, source_node: str, target_node: str, trust_graph: dict[str, dict[str, float]], max_hops: int = 3
    ) -> float:
        """Compute transitive trust through network paths"""
        if source_node == target_node:
            return 1.0

        if target_node in trust_graph.get(source_node, {}):
            return trust_graph[source_node][target_node]

        # BFS to find trust paths
        visited = set()
        queue = [(source_node, 1.0, 0)]  # (node, trust, hops)
        max_trust = 0.0

        while queue:
            current_node, current_trust, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor, direct_trust in trust_graph.get(current_node, {}).items():
                if neighbor == target_node:
                    # Found path to target
                    path_trust = current_trust * direct_trust
                    # Apply decay based on path length
                    decayed_trust = path_trust * (1 - self.config.trust_propagation_decay) ** hops
                    max_trust = max(max_trust, decayed_trust)
                elif neighbor not in visited and hops < max_hops - 1:
                    # Continue exploring
                    new_trust = current_trust * direct_trust
                    queue.append((neighbor, new_trust, hops + 1))

        return max_trust

    def aggregate_tier_trust(
        self, node_scores: dict[str, ReputationScore], tier_structure: dict[ReputationTier, list[str]]
    ) -> dict[ReputationTier, float]:
        """Aggregate trust scores by tier"""
        tier_trust = {}

        for tier, nodes in tier_structure.items():
            if not nodes:
                tier_trust[tier] = 0.0
                continue

            # Weighted average by sample size and inverse uncertainty
            total_weight = 0.0
            weighted_sum = 0.0

            for node_id in nodes:
                if node_id in node_scores:
                    score = node_scores[node_id]
                    weight = score.sample_size / (1 + score.uncertainty)
                    weighted_sum += score.mean_score * weight
                    total_weight += weight

            tier_trust[tier] = weighted_sum / total_weight if total_weight > 0 else 0.0

        return tier_trust


class BayesianReputationEngine:
    """Main Bayesian reputation management system"""

    def __init__(self, config: ReputationConfig | None = None):
        self.config = config or ReputationConfig()
        self.reputation_scores: dict[str, ReputationScore] = {}
        self.event_history: dict[str, list[ReputationEvent]] = defaultdict(list)
        self.trust_composition = TrustComposition(self.config)
        self.logger = logging.getLogger(__name__)

    def record_event(self, event: ReputationEvent) -> None:
        """Record a reputation event"""
        node_id = event.node_id

        # Get or create reputation score
        if node_id not in self.reputation_scores:
            self.reputation_scores[node_id] = ReputationScore(
                node_id=node_id,
                alpha=self.config.prior_alpha,
                beta=self.config.prior_beta,
                last_updated=event.timestamp,
            )

        score = self.reputation_scores[node_id]

        # Apply time decay before updating
        self._apply_decay(score, event.timestamp)

        # Calculate event weight
        base_weight = self.config.event_weights.get(event.event_type, 1.0)

        # Adjust weight based on quality score
        if event.quality_score is not None:
            quality_factor = 1 + (event.quality_score - 0.5) * self.config.quality_weight_multiplier
            base_weight *= quality_factor

        final_weight = base_weight * event.weight

        # Update Beta distribution parameters
        if event.is_positive():
            score.alpha += final_weight
        else:
            score.beta += final_weight

        score.last_updated = event.timestamp
        score.update_tier()

        # Store event in history
        self.event_history[node_id].append(event)

        self.logger.info(
            f"Updated reputation for {node_id}: "
            f"score={score.mean_score:.3f}, "
            f"uncertainty={score.uncertainty:.3f}, "
            f"tier={score.tier.name}"
        )

    def _apply_decay(self, score: ReputationScore, current_time: float) -> None:
        """Apply time-based decay to reputation scores"""
        time_diff_days = (current_time - score.last_updated) / (24 * 3600)

        if time_diff_days > 0:
            decay_factor = math.exp(-self.config.decay_rate * time_diff_days)

            # Decay towards prior (regression to mean)
            score.alpha = (score.alpha - self.config.prior_alpha) * decay_factor + self.config.prior_alpha
            score.beta = (score.beta - self.config.prior_beta) * decay_factor + self.config.prior_beta

            # Ensure minimum values
            score.alpha = max(score.alpha, self.config.min_alpha)
            score.beta = max(score.beta, self.config.min_beta)

    def get_reputation_score(self, node_id: str, current_time: float | None = None) -> ReputationScore | None:
        """Get current reputation score for a node"""
        if node_id not in self.reputation_scores:
            return None

        score = self.reputation_scores[node_id]

        # Apply decay if current time provided
        if current_time:
            self._apply_decay(score, current_time)
            score.update_tier()

        return score

    def get_trust_score(self, node_id: str, requesting_node: str = None) -> float:
        """Get trust score, potentially including transitive trust"""
        score = self.get_reputation_score(node_id, time.time())

        if not score:
            return 0.0

        base_trust = score.mean_score

        # Adjust for uncertainty (lower trust if high uncertainty)
        uncertainty_penalty = score.uncertainty
        adjusted_trust = base_trust * (1 - uncertainty_penalty)

        # TODO: Add transitive trust if requesting_node provided
        # This would require maintaining a trust graph

        return max(0.0, min(1.0, adjusted_trust))

    def get_node_ranking(self, tier: ReputationTier | None = None) -> list[tuple[str, float, ReputationTier]]:
        """Get ranked list of nodes by reputation"""
        current_time = time.time()

        nodes_data = []
        for node_id, score in self.reputation_scores.items():
            self._apply_decay(score, current_time)
            score.update_tier()

            if tier is None or score.tier == tier:
                trust_score = self.get_trust_score(node_id)
                nodes_data.append((node_id, trust_score, score.tier))

        # Sort by trust score descending, then by tier
        return sorted(nodes_data, key=lambda x: (x[1], x[2].value), reverse=True)

    def get_tier_distribution(self) -> dict[ReputationTier, int]:
        """Get distribution of nodes across tiers"""
        current_time = time.time()
        distribution = defaultdict(int)

        for score in self.reputation_scores.values():
            self._apply_decay(score, current_time)
            score.update_tier()
            distribution[score.tier] += 1

        return dict(distribution)

    def recommend_nodes_for_task(
        self, task_requirements: dict[str, Any], min_trust: float = 0.7, max_nodes: int = 5
    ) -> list[str]:
        """Recommend trusted nodes for a task"""
        # Get all nodes above minimum trust threshold
        candidates = []

        for node_id in self.reputation_scores:
            trust_score = self.get_trust_score(node_id)
            if trust_score >= min_trust:
                candidates.append((node_id, trust_score))

        # Sort by trust score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in candidates[:max_nodes]]

    def get_reputation_insights(self, node_id: str) -> dict[str, Any]:
        """Get detailed reputation insights for a node"""
        score = self.get_reputation_score(node_id, time.time())

        if not score:
            return {"error": "Node not found"}

        events = self.event_history[node_id]

        # Event type distribution
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type.value] += 1

        # Recent activity (last 30 days)
        current_time = time.time()
        recent_threshold = current_time - (30 * 24 * 3600)
        recent_events = [e for e in events if e.timestamp > recent_threshold]

        confidence_interval = score.confidence_interval()

        return {
            "node_id": node_id,
            "reputation_score": score.mean_score,
            "uncertainty": score.uncertainty,
            "confidence_interval": confidence_interval,
            "tier": score.tier.name,
            "sample_size": score.sample_size,
            "trust_score": self.get_trust_score(node_id),
            "total_events": len(events),
            "recent_events_30d": len(recent_events),
            "event_distribution": dict(event_counts),
            "last_updated": datetime.fromtimestamp(score.last_updated).isoformat(),
        }

    def batch_update_from_metrics(self, metrics_data: list[dict[str, Any]]) -> None:
        """Batch update reputations from collected metrics"""
        current_time = time.time()

        for metric in metrics_data:
            node_id = metric.get("node_id")
            if not node_id:
                continue

            # Create events based on metrics
            events = []

            # Task completion metrics
            if "tasks_completed" in metric and "tasks_failed" in metric:
                for _ in range(metric["tasks_completed"]):
                    events.append(
                        ReputationEvent(
                            node_id=node_id,
                            event_type=EventType.TASK_SUCCESS,
                            timestamp=current_time,
                            quality_score=metric.get("avg_quality_score"),
                        )
                    )

                for _ in range(metric["tasks_failed"]):
                    events.append(
                        ReputationEvent(node_id=node_id, event_type=EventType.TASK_FAILURE, timestamp=current_time)
                    )

            # Availability metrics
            if "uptime_ratio" in metric:
                uptime = metric["uptime_ratio"]
                if uptime > 0.95:
                    events.append(
                        ReputationEvent(
                            node_id=node_id, event_type=EventType.AVAILABILITY, timestamp=current_time, weight=uptime
                        )
                    )
                elif uptime < 0.8:
                    events.append(
                        ReputationEvent(
                            node_id=node_id,
                            event_type=EventType.UNAVAILABILITY,
                            timestamp=current_time,
                            weight=1.0 - uptime,
                        )
                    )

            # Process all events
            for event in events:
                self.record_event(event)

    def export_state(self) -> dict[str, Any]:
        """Export reputation system state"""
        return {
            "config": {
                "prior_alpha": self.config.prior_alpha,
                "prior_beta": self.config.prior_beta,
                "decay_rate": self.config.decay_rate,
                "event_weights": {k.value: v for k, v in self.config.event_weights.items()},
            },
            "reputation_scores": {
                node_id: {
                    "alpha": score.alpha,
                    "beta": score.beta,
                    "last_updated": score.last_updated,
                    "tier": score.tier.value,
                }
                for node_id, score in self.reputation_scores.items()
            },
            "event_counts": {node_id: len(events) for node_id, events in self.event_history.items()},
        }

    def import_state(self, state_data: dict[str, Any]) -> None:
        """Import reputation system state"""
        if "reputation_scores" in state_data:
            for node_id, score_data in state_data["reputation_scores"].items():
                self.reputation_scores[node_id] = ReputationScore(
                    node_id=node_id,
                    alpha=score_data["alpha"],
                    beta=score_data["beta"],
                    last_updated=score_data["last_updated"],
                    tier=ReputationTier(score_data["tier"]),
                )


# Integration functions for fog system components


def integrate_with_scheduler(
    reputation_engine: BayesianReputationEngine, scheduler_config: dict[str, Any]
) -> dict[str, float]:
    """Integrate reputation with task scheduler"""
    trust_scores = {}

    for node_id in scheduler_config.get("available_nodes", []):
        trust_score = reputation_engine.get_trust_score(node_id)
        trust_scores[node_id] = trust_score

    return trust_scores


def integrate_with_pricing(
    reputation_engine: BayesianReputationEngine, base_prices: dict[str, float]
) -> dict[str, float]:
    """Adjust pricing based on reputation"""
    adjusted_prices = {}

    for node_id, base_price in base_prices.items():
        trust_score = reputation_engine.get_trust_score(node_id)

        # Higher reputation gets price premium (up to 50% bonus)
        reputation_multiplier = 1.0 + (trust_score - 0.5) * 1.0
        reputation_multiplier = max(0.5, min(1.5, reputation_multiplier))

        adjusted_prices[node_id] = base_price * reputation_multiplier

    return adjusted_prices


def create_reputation_metrics(reputation_engine: BayesianReputationEngine) -> dict[str, Any]:
    """Create metrics for monitoring reputation system"""
    tier_dist = reputation_engine.get_tier_distribution()

    # Calculate system health metrics
    total_nodes = sum(tier_dist.values())
    if total_nodes == 0:
        return {"error": "No nodes in system"}

    high_trust_nodes = (
        tier_dist.get(ReputationTier.GOLD, 0)
        + tier_dist.get(ReputationTier.PLATINUM, 0)
        + tier_dist.get(ReputationTier.DIAMOND, 0)
    )

    return {
        "total_nodes": total_nodes,
        "tier_distribution": {tier.name: count for tier, count in tier_dist.items()},
        "high_trust_ratio": high_trust_nodes / total_nodes,
        "system_trust_health": "healthy" if high_trust_nodes / total_nodes > 0.3 else "needs_attention",
    }
