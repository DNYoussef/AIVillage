"""Vote Weight Management for Betanet v1.1 Governance

Implements BN-10.1 requirements:
- Per-AS weight cap: ≤20% of total active weight
- Per-Org weight cap: ≤25% of total active weight
- Active weight calculation based on 14-day participation
- Post-aggregation cap enforcement
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VoteWeight:
    """Individual vote weight record."""

    node_id: str
    as_group: str
    organization: str
    base_weight: float
    participation_score: float  # 0.0-1.0 based on 14-day activity
    effective_weight: float
    last_active: float
    metadata: dict[str, any] = None


@dataclass
class WeightCaps:
    """Weight cap configuration and enforcement."""

    per_as_cap: float = 0.20  # 20% max per AS
    per_org_cap: float = 0.25  # 25% max per Organization
    min_participation_days: int = 14
    min_participation_threshold: float = 0.1  # Minimum activity to count


class VoteWeightManager:
    """Manages vote weights with BN-10.1 compliance."""

    def __init__(self, caps: WeightCaps | None = None):
        self.caps = caps or WeightCaps()
        self.weights: dict[str, VoteWeight] = {}
        self.as_groups: dict[str, set[str]] = {}  # AS -> node_ids
        self.organizations: dict[str, set[str]] = {}  # Org -> node_ids
        self.participation_history: dict[str, list[float]] = {}  # node_id -> timestamps

    def register_node(
        self, node_id: str, as_group: str, organization: str, base_weight: float
    ) -> bool:
        """Register a node with initial weight."""
        try:
            # Validate inputs
            if base_weight <= 0:
                logger.error(f"Invalid base weight {base_weight} for node {node_id}")
                return False

            # Create weight record
            weight_record = VoteWeight(
                node_id=node_id,
                as_group=as_group,
                organization=organization,
                base_weight=base_weight,
                participation_score=0.0,  # No history yet
                effective_weight=0.0,  # Will be calculated
                last_active=0.0,
            )

            self.weights[node_id] = weight_record

            # Update group mappings
            if as_group not in self.as_groups:
                self.as_groups[as_group] = set()
            self.as_groups[as_group].add(node_id)

            if organization not in self.organizations:
                self.organizations[organization] = set()
            self.organizations[organization].add(node_id)

            # Initialize participation history
            self.participation_history[node_id] = []

            logger.info(
                f"Registered node {node_id} in AS {as_group}, Org {organization}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return False

    def record_participation(self, node_id: str) -> bool:
        """Record participation event for weight calculation."""
        if node_id not in self.weights:
            logger.warning(f"Unknown node {node_id} attempting participation")
            return False

        try:
            now = time.time()
            self.participation_history[node_id].append(now)
            self.weights[node_id].last_active = now

            # Keep only last 14 days of history
            cutoff = now - (self.caps.min_participation_days * 24 * 3600)
            self.participation_history[node_id] = [
                ts for ts in self.participation_history[node_id] if ts >= cutoff
            ]

            # Recalculate participation score
            self._update_participation_score(node_id)
            return True

        except Exception as e:
            logger.error(f"Failed to record participation for {node_id}: {e}")
            return False

    def _update_participation_score(self, node_id: str):
        """Update participation score based on recent activity."""
        history = self.participation_history[node_id]
        weight_record = self.weights[node_id]

        if not history:
            weight_record.participation_score = 0.0
            return

        # Calculate score based on frequency and recency
        now = time.time()
        days_window = self.caps.min_participation_days

        # Score based on participation frequency (max 1 per day)
        daily_participation = min(len(history) / days_window, 1.0)

        # Recency bonus - more recent activity weighted higher
        if history:
            most_recent = max(history)
            hours_since_last = (now - most_recent) / 3600
            recency_factor = max(
                0.0, 1.0 - (hours_since_last / (24 * 7))
            )  # Decay over 7 days
        else:
            recency_factor = 0.0

        # Combined score
        weight_record.participation_score = (daily_participation * 0.7) + (
            recency_factor * 0.3
        )

    def calculate_effective_weights(self) -> dict[str, float]:
        """Calculate effective weights with cap enforcement."""
        # Step 1: Calculate base effective weights
        active_nodes = {}
        for node_id, weight_record in self.weights.items():
            # Only count nodes with minimum participation
            if (
                weight_record.participation_score
                >= self.caps.min_participation_threshold
            ):
                effective = (
                    weight_record.base_weight * weight_record.participation_score
                )
                active_nodes[node_id] = effective
                weight_record.effective_weight = effective
            else:
                weight_record.effective_weight = 0.0

        if not active_nodes:
            logger.warning("No active nodes meet participation threshold")
            return {}

        # Step 2: Calculate total weight for percentage calculations
        total_weight = sum(active_nodes.values())

        # Step 3: Enforce per-AS caps
        active_nodes = self._enforce_as_caps(active_nodes, total_weight)

        # Step 4: Enforce per-Organization caps
        active_nodes = self._enforce_org_caps(active_nodes, total_weight)

        # Step 5: Update weight records with final values
        for node_id, final_weight in active_nodes.items():
            self.weights[node_id].effective_weight = final_weight

        return active_nodes

    def _enforce_as_caps(
        self, weights: dict[str, float], total_weight: float
    ) -> dict[str, float]:
        """Enforce per-AS weight caps."""
        max_as_weight = total_weight * self.caps.per_as_cap

        for as_group, node_ids in self.as_groups.items():
            as_nodes = [nid for nid in node_ids if nid in weights]
            if not as_nodes:
                continue

            # Calculate current AS weight
            as_weight = sum(weights[nid] for nid in as_nodes)

            # Apply cap if exceeded
            if as_weight > max_as_weight:
                scale_factor = max_as_weight / as_weight
                logger.warning(
                    f"AS {as_group} exceeds cap, scaling by {scale_factor:.3f}"
                )

                for node_id in as_nodes:
                    weights[node_id] *= scale_factor

        return weights

    def _enforce_org_caps(
        self, weights: dict[str, float], total_weight: float
    ) -> dict[str, float]:
        """Enforce per-Organization weight caps."""
        max_org_weight = total_weight * self.caps.per_org_cap

        for organization, node_ids in self.organizations.items():
            org_nodes = [nid for nid in node_ids if nid in weights]
            if not org_nodes:
                continue

            # Calculate current organization weight
            org_weight = sum(weights[nid] for nid in org_nodes)

            # Apply cap if exceeded
            if org_weight > max_org_weight:
                scale_factor = max_org_weight / org_weight
                logger.warning(
                    f"Org {organization} exceeds cap, scaling by {scale_factor:.3f}"
                )

                for node_id in org_nodes:
                    weights[node_id] *= scale_factor

        return weights

    def get_weight_distribution(self) -> dict[str, any]:
        """Get current weight distribution statistics."""
        effective_weights = self.calculate_effective_weights()

        if not effective_weights:
            return {
                "total_weight": 0,
                "active_nodes": 0,
                "as_distribution": {},
                "org_distribution": {},
                "cap_violations": [],
            }

        total_weight = sum(effective_weights.values())

        # AS distribution
        as_distribution = {}
        for as_group, node_ids in self.as_groups.items():
            as_weight = sum(effective_weights.get(nid, 0) for nid in node_ids)
            if as_weight > 0:
                as_distribution[as_group] = {
                    "weight": as_weight,
                    "percentage": (as_weight / total_weight) * 100,
                    "node_count": len(
                        [nid for nid in node_ids if nid in effective_weights]
                    ),
                }

        # Organization distribution
        org_distribution = {}
        for organization, node_ids in self.organizations.items():
            org_weight = sum(effective_weights.get(nid, 0) for nid in node_ids)
            if org_weight > 0:
                org_distribution[organization] = {
                    "weight": org_weight,
                    "percentage": (org_weight / total_weight) * 100,
                    "node_count": len(
                        [nid for nid in node_ids if nid in effective_weights]
                    ),
                }

        # Check for violations
        violations = []
        for as_group, stats in as_distribution.items():
            if stats["percentage"] > self.caps.per_as_cap * 100:
                violations.append(
                    f"AS {as_group}: {stats['percentage']:.1f}% > {self.caps.per_as_cap * 100}%"
                )

        for org, stats in org_distribution.items():
            if stats["percentage"] > self.caps.per_org_cap * 100:
                violations.append(
                    f"Org {org}: {stats['percentage']:.1f}% > {self.caps.per_org_cap * 100}%"
                )

        return {
            "total_weight": total_weight,
            "active_nodes": len(effective_weights),
            "as_distribution": as_distribution,
            "org_distribution": org_distribution,
            "cap_violations": violations,
            "caps": {
                "per_as_cap": self.caps.per_as_cap * 100,
                "per_org_cap": self.caps.per_org_cap * 100,
            },
        }

    def validate_caps_compliance(self) -> tuple[bool, list[str]]:
        """Validate that current distribution meets cap requirements."""
        distribution = self.get_weight_distribution()
        violations = distribution["cap_violations"]

        return len(violations) == 0, violations
