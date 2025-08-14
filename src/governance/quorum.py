"""Quorum Validation for Betanet v1.1 Governance

Implements BN-10.1 quorum requirements:
- Quorum threshold: ≥0.67 of active weight (14-day window)
- AS diversity requirement: ≥24 distinct AS groups represented
- ISD diversity requirement: ≥3 distinct SCION ISDs represented
- Weight aggregation with proper cap enforcement
"""

import logging
import time
from dataclasses import dataclass

from .weights import VoteWeightManager

logger = logging.getLogger(__name__)


@dataclass
class QuorumRequirements:
    """Quorum validation requirements."""

    min_weight_threshold: float = 0.67  # 67% threshold
    min_as_groups: int = 24  # Minimum AS diversity
    min_isds: int = 3  # Minimum ISD diversity
    min_active_nodes: int = 10  # Minimum absolute number


@dataclass
class VoteRecord:
    """Individual vote record for quorum calculation."""

    node_id: str
    as_group: str
    isd: str
    organization: str
    vote: str  # "ACK", "NACK", "ABSTAIN"
    weight: float
    timestamp: float
    signature: bytes = b""


@dataclass
class QuorumResult:
    """Results of quorum validation."""

    is_valid: bool
    total_ack_weight: float
    total_eligible_weight: float
    weight_percentage: float
    as_group_count: int
    isd_count: int
    participating_nodes: int
    violations: list[str]
    metadata: dict[str, any] = None


class QuorumValidator:
    """Validates quorum requirements for Betanet governance."""

    def __init__(self, requirements: QuorumRequirements | None = None):
        self.requirements = requirements or QuorumRequirements()
        self.weight_manager: VoteWeightManager | None = None

    def set_weight_manager(self, weight_manager: VoteWeightManager):
        """Set the weight manager for quorum calculations."""
        self.weight_manager = weight_manager

    def validate_quorum(
        self, votes: list[VoteRecord], proposal_id: str = None
    ) -> QuorumResult:
        """Validate that votes meet quorum requirements."""
        if not self.weight_manager:
            return QuorumResult(
                is_valid=False,
                total_ack_weight=0,
                total_eligible_weight=0,
                weight_percentage=0,
                as_group_count=0,
                isd_count=0,
                participating_nodes=0,
                violations=["No weight manager configured"],
            )

        try:
            # Get current effective weights with caps applied
            effective_weights = self.weight_manager.calculate_effective_weights()
            total_eligible_weight = sum(effective_weights.values())

            if total_eligible_weight == 0:
                return QuorumResult(
                    is_valid=False,
                    total_ack_weight=0,
                    total_eligible_weight=0,
                    weight_percentage=0,
                    as_group_count=0,
                    isd_count=0,
                    participating_nodes=0,
                    violations=["No eligible voting weight available"],
                )

            # Process votes
            ack_votes = [v for v in votes if v.vote == "ACK"]

            # Calculate ACK weight (only count nodes with effective weight)
            total_ack_weight = 0
            valid_ack_votes = []

            for vote in ack_votes:
                if vote.node_id in effective_weights:
                    # Use the capped effective weight, not the raw vote weight
                    effective_weight = effective_weights[vote.node_id]
                    total_ack_weight += effective_weight

                    # Update vote record with effective weight
                    vote.weight = effective_weight
                    valid_ack_votes.append(vote)
                else:
                    logger.debug(f"Ignoring vote from inactive node {vote.node_id}")

            # Calculate diversity metrics
            as_groups = {vote.as_group for vote in valid_ack_votes}
            isds = {vote.isd for vote in valid_ack_votes}
            participating_nodes = len(valid_ack_votes)

            # Calculate weight percentage
            weight_percentage = (
                (total_ack_weight / total_eligible_weight)
                if total_eligible_weight > 0
                else 0
            )

            # Validate requirements
            violations = []
            is_valid = True

            # Weight threshold check
            if weight_percentage < self.requirements.min_weight_threshold:
                violations.append(
                    f"Insufficient weight: {weight_percentage:.3f} < {self.requirements.min_weight_threshold:.3f}"
                )
                is_valid = False

            # AS diversity check
            if len(as_groups) < self.requirements.min_as_groups:
                violations.append(
                    f"Insufficient AS diversity: {len(as_groups)} < {self.requirements.min_as_groups}"
                )
                is_valid = False

            # ISD diversity check
            if len(isds) < self.requirements.min_isds:
                violations.append(
                    f"Insufficient ISD diversity: {len(isds)} < {self.requirements.min_isds}"
                )
                is_valid = False

            # Minimum nodes check
            if participating_nodes < self.requirements.min_active_nodes:
                violations.append(
                    f"Insufficient participating nodes: {participating_nodes} < {self.requirements.min_active_nodes}"
                )
                is_valid = False

            logger.info(
                f"Quorum validation {proposal_id}: {'PASS' if is_valid else 'FAIL'} "
                f"({weight_percentage:.1%} weight, {len(as_groups)} AS, {len(isds)} ISDs)"
            )

            return QuorumResult(
                is_valid=is_valid,
                total_ack_weight=total_ack_weight,
                total_eligible_weight=total_eligible_weight,
                weight_percentage=weight_percentage,
                as_group_count=len(as_groups),
                isd_count=len(isds),
                participating_nodes=participating_nodes,
                violations=violations,
                metadata={
                    "proposal_id": proposal_id,
                    "timestamp": time.time(),
                    "as_groups": sorted(as_groups),
                    "isds": sorted(isds),
                    "requirements": {
                        "min_weight_threshold": self.requirements.min_weight_threshold,
                        "min_as_groups": self.requirements.min_as_groups,
                        "min_isds": self.requirements.min_isds,
                        "min_active_nodes": self.requirements.min_active_nodes,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Quorum validation failed for {proposal_id}: {e}")
            return QuorumResult(
                is_valid=False,
                total_ack_weight=0,
                total_eligible_weight=0,
                weight_percentage=0,
                as_group_count=0,
                isd_count=0,
                participating_nodes=0,
                violations=[f"Validation error: {str(e)}"],
            )

    def check_partition_safety(
        self, votes: list[VoteRecord]
    ) -> tuple[bool, dict[str, any]]:
        """Check for network partition safety indicators."""
        if not votes:
            return False, {"error": "No votes to analyze"}

        try:
            # Group votes by AS and ISD for partition analysis
            as_participation = {}
            isd_participation = {}

            for vote in votes:
                # AS participation
                if vote.as_group not in as_participation:
                    as_participation[vote.as_group] = {"nodes": 0, "weight": 0}
                as_participation[vote.as_group]["nodes"] += 1
                as_participation[vote.as_group]["weight"] += vote.weight

                # ISD participation
                if vote.isd not in isd_participation:
                    isd_participation[vote.isd] = {"nodes": 0, "weight": 0}
                isd_participation[vote.isd]["nodes"] += 1
                isd_participation[vote.isd]["weight"] += vote.weight

            # Calculate distribution metrics
            total_weight = sum(vote.weight for vote in votes)

            # Check for concerning concentrations
            partition_risks = []

            # Single AS dominance check (>40% indicates risk)
            max_as_weight = max(stats["weight"] for stats in as_participation.values())
            if total_weight > 0 and (max_as_weight / total_weight) > 0.40:
                partition_risks.append(
                    f"AS weight concentration: {max_as_weight / total_weight:.1%}"
                )

            # Single ISD dominance check (>50% indicates risk)
            max_isd_weight = max(
                stats["weight"] for stats in isd_participation.values()
            )
            if total_weight > 0 and (max_isd_weight / total_weight) > 0.50:
                partition_risks.append(
                    f"ISD weight concentration: {max_isd_weight / total_weight:.1%}"
                )

            # Geographic diversity check (heuristic based on AS distribution)
            unique_as_prefixes = {
                as_group.split("-")[0] for as_group in as_participation.keys()
            }
            if len(unique_as_prefixes) < 3:  # Less than 3 major regions
                partition_risks.append(
                    f"Limited geographic diversity: {len(unique_as_prefixes)} regions"
                )

            is_safe = len(partition_risks) == 0

            return is_safe, {
                "is_partition_safe": is_safe,
                "risks": partition_risks,
                "as_participation": as_participation,
                "isd_participation": isd_participation,
                "diversity_metrics": {
                    "as_count": len(as_participation),
                    "isd_count": len(isd_participation),
                    "geographic_regions": len(unique_as_prefixes),
                    "max_as_concentration": max_as_weight / total_weight
                    if total_weight > 0
                    else 0,
                    "max_isd_concentration": max_isd_weight / total_weight
                    if total_weight > 0
                    else 0,
                },
            }

        except Exception as e:
            logger.error(f"Partition safety check failed: {e}")
            return False, {"error": str(e)}

    def simulate_quorum_scenarios(self) -> dict[str, any]:
        """Simulate various quorum scenarios for testing."""
        if not self.weight_manager:
            return {"error": "No weight manager configured"}

        # Get current weight distribution
        distribution = self.weight_manager.get_weight_distribution()
        effective_weights = self.weight_manager.calculate_effective_weights()

        scenarios = {}

        # Scenario 1: Best case - all active nodes vote ACK
        all_ack_votes = []
        for node_id, weight in effective_weights.items():
            node_record = self.weight_manager.weights[node_id]
            vote = VoteRecord(
                node_id=node_id,
                as_group=node_record.as_group,
                isd=node_record.as_group.split("-")[0],  # Extract ISD from AS
                organization=node_record.organization,
                vote="ACK",
                weight=weight,
                timestamp=time.time(),
            )
            all_ack_votes.append(vote)

        scenarios["all_ack"] = self.validate_quorum(all_ack_votes, "scenario_all_ack")

        # Scenario 2: Minimum threshold - exactly 67%
        sorted_votes = sorted(all_ack_votes, key=lambda x: x.weight, reverse=True)
        cumulative_weight = 0
        target_weight = (
            sum(effective_weights.values()) * self.requirements.min_weight_threshold
        )

        min_threshold_votes = []
        for vote in sorted_votes:
            min_threshold_votes.append(vote)
            cumulative_weight += vote.weight
            if cumulative_weight >= target_weight:
                break

        scenarios["min_threshold"] = self.validate_quorum(
            min_threshold_votes, "scenario_min_threshold"
        )

        # Scenario 3: Insufficient diversity - only 2 ISDs
        limited_isd_votes = [v for v in all_ack_votes if v.isd in ["1", "2"]]
        scenarios["limited_isd"] = self.validate_quorum(
            limited_isd_votes, "scenario_limited_isd"
        )

        return {
            "scenarios": scenarios,
            "current_distribution": distribution,
            "requirements": {
                "min_weight_threshold": self.requirements.min_weight_threshold,
                "min_as_groups": self.requirements.min_as_groups,
                "min_isds": self.requirements.min_isds,
                "min_active_nodes": self.requirements.min_active_nodes,
            },
        }
