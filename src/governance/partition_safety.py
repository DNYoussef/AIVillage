"""Partition Safety Monitoring for Betanet v1.1 Governance

Implements BN-10.1 partition safety requirements:
- Network partition detection based on voting patterns
- AS/ISD diversity analysis for resilience validation
- Geographic distribution monitoring
- Split-brain scenario prevention
- Real-time partition risk assessment
"""

import logging
import time
from dataclasses import dataclass

from .quorum import VoteRecord
from .weights import VoteWeightManager

logger = logging.getLogger(__name__)


@dataclass
class PartitionRisk:
    """Individual partition risk assessment"""

    risk_type: str
    severity: float  # 0.0-1.0, higher is worse
    description: str
    affected_entities: list[str]
    mitigation_suggestions: list[str]


@dataclass
class PartitionSafetyReport:
    """Comprehensive partition safety assessment"""

    is_partition_safe: bool
    overall_risk_score: float  # 0.0-1.0
    detected_risks: list[PartitionRisk]
    diversity_metrics: dict[str, any]
    geographic_distribution: dict[str, any]
    temporal_analysis: dict[str, any]
    recommendations: list[str]
    timestamp: float


class PartitionSafetyMonitor:
    """Monitors network partition safety for Betanet governance."""

    def __init__(self, weight_manager: VoteWeightManager | None = None):
        self.weight_manager = weight_manager

        # Risk thresholds (configurable)
        self.risk_thresholds = {
            "as_concentration": 0.35,  # >35% weight in single AS
            "isd_concentration": 0.45,  # >45% weight in single ISD
            "org_concentration": 0.40,  # >40% weight in single Org
            "geographic_concentration": 0.50,  # >50% in single region
            "temporal_concentration": 0.60,  # >60% votes in short window
            "min_as_diversity": 20,  # Minimum AS groups required
            "min_isd_diversity": 3,  # Minimum ISDs required
            "min_geographic_regions": 3,  # Minimum geographic regions
        }

        # Historical vote tracking for temporal analysis
        self.vote_history: dict[str, list[VoteRecord]] = {}  # proposal_id -> votes
        self.partition_history: list[PartitionSafetyReport] = []

    def assess_partition_safety(self, votes: list[VoteRecord], proposal_id: str = None) -> PartitionSafetyReport:
        """Comprehensive partition safety assessment."""
        logger.info(f"Assessing partition safety for proposal {proposal_id}")

        if not votes:
            return PartitionSafetyReport(
                is_partition_safe=False,
                overall_risk_score=1.0,
                detected_risks=[
                    PartitionRisk(
                        risk_type="NO_VOTES",
                        severity=1.0,
                        description="No votes available for analysis",
                        affected_entities=[],
                        mitigation_suggestions=["Wait for votes to be submitted"],
                    )
                ],
                diversity_metrics={},
                geographic_distribution={},
                temporal_analysis={},
                recommendations=["Cannot assess safety without vote data"],
                timestamp=time.time(),
            )

        try:
            # Store votes for temporal analysis
            if proposal_id:
                self.vote_history[proposal_id] = votes

            # Perform all safety assessments
            risks = []

            # 1. AS concentration analysis
            as_risks = self._assess_as_concentration(votes)
            risks.extend(as_risks)

            # 2. ISD concentration analysis
            isd_risks = self._assess_isd_concentration(votes)
            risks.extend(isd_risks)

            # 3. Organization concentration analysis
            org_risks = self._assess_org_concentration(votes)
            risks.extend(org_risks)

            # 4. Geographic distribution analysis
            geo_risks = self._assess_geographic_distribution(votes)
            risks.extend(geo_risks)

            # 5. Temporal concentration analysis
            temporal_risks = self._assess_temporal_concentration(votes, proposal_id)
            risks.extend(temporal_risks)

            # 6. Diversity requirements check
            diversity_risks = self._assess_diversity_requirements(votes)
            risks.extend(diversity_risks)

            # Calculate overall metrics
            diversity_metrics = self._calculate_diversity_metrics(votes)
            geographic_distribution = self._calculate_geographic_distribution(votes)
            temporal_analysis = self._calculate_temporal_analysis(votes)

            # Determine overall safety
            overall_risk_score = self._calculate_overall_risk_score(risks)
            is_safe = overall_risk_score < 0.3  # Safety threshold

            # Generate recommendations
            recommendations = self._generate_recommendations(risks, diversity_metrics)

            report = PartitionSafetyReport(
                is_partition_safe=is_safe,
                overall_risk_score=overall_risk_score,
                detected_risks=risks,
                diversity_metrics=diversity_metrics,
                geographic_distribution=geographic_distribution,
                temporal_analysis=temporal_analysis,
                recommendations=recommendations,
                timestamp=time.time(),
            )

            # Store in history
            self.partition_history.append(report)

            # Limit history size
            if len(self.partition_history) > 100:
                self.partition_history.pop(0)

            logger.info(
                f"Partition safety assessment complete: "
                f"{'SAFE' if is_safe else 'RISK'} "
                f"(score={overall_risk_score:.3f})"
            )

            return report

        except Exception as e:
            logger.exception(f"Partition safety assessment failed: {e}")
            return PartitionSafetyReport(
                is_partition_safe=False,
                overall_risk_score=1.0,
                detected_risks=[
                    PartitionRisk(
                        risk_type="ASSESSMENT_ERROR",
                        severity=0.8,
                        description=f"Safety assessment failed: {str(e)}",
                        affected_entities=[],
                        mitigation_suggestions=["Check system health and retry assessment"],
                    )
                ],
                diversity_metrics={},
                geographic_distribution={},
                temporal_analysis={},
                recommendations=["System error - manual review required"],
                timestamp=time.time(),
            )

    def _assess_as_concentration(self, votes: list[VoteRecord]) -> list[PartitionRisk]:
        """Assess AS concentration risks."""
        risks = []

        # Group votes by AS
        as_weights = {}
        total_weight = sum(vote.weight for vote in votes)

        for vote in votes:
            as_group = vote.as_group
            as_weights[as_group] = as_weights.get(as_group, 0) + vote.weight

        if total_weight == 0:
            return risks

        # Check for concentration violations
        for as_group, weight in as_weights.items():
            concentration = weight / total_weight

            if concentration > self.risk_thresholds["as_concentration"]:
                severity = min(1.0, concentration / 0.5)  # Scale to max at 50%

                risks.append(
                    PartitionRisk(
                        risk_type="AS_CONCENTRATION",
                        severity=severity,
                        description=f"AS {as_group} controls {concentration:.1%} of voting weight "
                        f"(threshold: {self.risk_thresholds['as_concentration']:.1%})",
                        affected_entities=[as_group],
                        mitigation_suggestions=[
                            "Encourage participation from other AS groups",
                            f"Consider weight redistribution in AS {as_group}",
                            "Monitor for potential coordination attacks",
                        ],
                    )
                )

        return risks

    def _assess_isd_concentration(self, votes: list[VoteRecord]) -> list[PartitionRisk]:
        """Assess ISD concentration risks."""
        risks = []

        # Group votes by ISD
        isd_weights = {}
        total_weight = sum(vote.weight for vote in votes)

        for vote in votes:
            isd = vote.isd
            isd_weights[isd] = isd_weights.get(isd, 0) + vote.weight

        if total_weight == 0:
            return risks

        # Check for concentration violations
        for isd, weight in isd_weights.items():
            concentration = weight / total_weight

            if concentration > self.risk_thresholds["isd_concentration"]:
                severity = min(1.0, concentration / 0.6)  # Scale to max at 60%

                risks.append(
                    PartitionRisk(
                        risk_type="ISD_CONCENTRATION",
                        severity=severity,
                        description=f"ISD {isd} controls {concentration:.1%} of voting weight "
                        f"(threshold: {self.risk_thresholds['isd_concentration']:.1%})",
                        affected_entities=[isd],
                        mitigation_suggestions=[
                            "Encourage participation from other ISDs",
                            f"Monitor ISD {isd} for potential centralization",
                            "Consider ISD-based weight balancing mechanisms",
                        ],
                    )
                )

        return risks

    def _assess_org_concentration(self, votes: list[VoteRecord]) -> list[PartitionRisk]:
        """Assess organization concentration risks."""
        risks = []

        # Group votes by organization
        org_weights = {}
        total_weight = sum(vote.weight for vote in votes)

        for vote in votes:
            org = vote.organization
            org_weights[org] = org_weights.get(org, 0) + vote.weight

        if total_weight == 0:
            return risks

        # Check for concentration violations
        for org, weight in org_weights.items():
            concentration = weight / total_weight

            if concentration > self.risk_thresholds["org_concentration"]:
                severity = min(1.0, concentration / 0.5)  # Scale to max at 50%

                risks.append(
                    PartitionRisk(
                        risk_type="ORG_CONCENTRATION",
                        severity=severity,
                        description=f"Organization '{org}' controls {concentration:.1%} of voting weight "
                        f"(threshold: {self.risk_thresholds['org_concentration']:.1%})",
                        affected_entities=[org],
                        mitigation_suggestions=[
                            "Encourage participation from other organizations",
                            f"Monitor {org} for potential coordination",
                            "Consider organizational weight caps",
                        ],
                    )
                )

        return risks

    def _assess_geographic_distribution(self, votes: list[VoteRecord]) -> list[PartitionRisk]:
        """Assess geographic distribution risks."""
        risks = []

        # Extract geographic regions from AS groups (heuristic)
        region_weights = {}
        total_weight = sum(vote.weight for vote in votes)

        for vote in votes:
            # Extract region from AS group (simplified heuristic)
            region = self._extract_region_from_as(vote.as_group)
            region_weights[region] = region_weights.get(region, 0) + vote.weight

        if total_weight == 0:
            return risks

        # Check minimum region diversity
        if len(region_weights) < self.risk_thresholds["min_geographic_regions"]:
            risks.append(
                PartitionRisk(
                    risk_type="INSUFFICIENT_GEOGRAPHIC_DIVERSITY",
                    severity=0.7,
                    description=f"Only {len(region_weights)} geographic regions represented "
                    f"(minimum: {self.risk_thresholds['min_geographic_regions']})",
                    affected_entities=list(region_weights.keys()),
                    mitigation_suggestions=[
                        "Encourage participation from underrepresented regions",
                        "Consider geographic diversity requirements",
                        "Monitor for regional network partitions",
                    ],
                )
            )

        # Check for single region dominance
        for region, weight in region_weights.items():
            concentration = weight / total_weight

            if concentration > self.risk_thresholds["geographic_concentration"]:
                severity = min(1.0, concentration / 0.7)  # Scale to max at 70%

                risks.append(
                    PartitionRisk(
                        risk_type="GEOGRAPHIC_CONCENTRATION",
                        severity=severity,
                        description=f"Region '{region}' controls {concentration:.1%} of voting weight "
                        f"(threshold: {self.risk_thresholds['geographic_concentration']:.1%})",
                        affected_entities=[region],
                        mitigation_suggestions=[
                            "Encourage participation from other regions",
                            f"Monitor {region} for potential isolation",
                            "Consider regional weight balancing",
                        ],
                    )
                )

        return risks

    def _assess_temporal_concentration(self, votes: list[VoteRecord], proposal_id: str) -> list[PartitionRisk]:
        """Assess temporal concentration risks."""
        risks = []

        if len(votes) < 2:
            return risks  # Need at least 2 votes for temporal analysis

        # Sort votes by timestamp
        sorted_votes = sorted(votes, key=lambda v: v.timestamp)
        total_weight = sum(vote.weight for vote in votes)

        if total_weight == 0:
            return risks

        # Check for temporal clustering (votes in short time windows)
        window_duration = 300  # 5 minutes

        for i, anchor_vote in enumerate(sorted_votes):
            window_end = anchor_vote.timestamp + window_duration
            window_votes = [v for v in sorted_votes[i:] if v.timestamp <= window_end]
            window_weight = sum(v.weight for v in window_votes)
            window_concentration = window_weight / total_weight

            if window_concentration > self.risk_thresholds["temporal_concentration"]:
                severity = min(1.0, window_concentration / 0.8)

                risks.append(
                    PartitionRisk(
                        risk_type="TEMPORAL_CONCENTRATION",
                        severity=severity,
                        description=f"Temporal clustering detected: {window_concentration:.1%} of votes "
                        f"within {window_duration}s window (threshold: "
                        f"{self.risk_thresholds['temporal_concentration']:.1%})",
                        affected_entities=[f"window_{int(anchor_vote.timestamp)}"],
                        mitigation_suggestions=[
                            "Monitor for coordinated voting attacks",
                            "Consider temporal vote distribution requirements",
                            "Implement vote timing randomization",
                        ],
                    )
                )
                break  # Only report first significant cluster

        return risks

    def _assess_diversity_requirements(self, votes: list[VoteRecord]) -> list[PartitionRisk]:
        """Assess diversity requirement compliance."""
        risks = []

        # Count unique AS groups and ISDs
        as_groups = {vote.as_group for vote in votes}
        isds = {vote.isd for vote in votes}

        # Check minimum AS diversity
        if len(as_groups) < self.risk_thresholds["min_as_diversity"]:
            risks.append(
                PartitionRisk(
                    risk_type="INSUFFICIENT_AS_DIVERSITY",
                    severity=0.8,
                    description=f"Only {len(as_groups)} AS groups participating "
                    f"(minimum: {self.risk_thresholds['min_as_diversity']})",
                    affected_entities=list(as_groups),
                    mitigation_suggestions=[
                        "Encourage participation from more AS groups",
                        "Lower participation barriers",
                        "Improve network connectivity",
                    ],
                )
            )

        # Check minimum ISD diversity
        if len(isds) < self.risk_thresholds["min_isd_diversity"]:
            risks.append(
                PartitionRisk(
                    risk_type="INSUFFICIENT_ISD_DIVERSITY",
                    severity=0.9,  # Higher severity for ISD diversity
                    description=f"Only {len(isds)} ISDs participating "
                    f"(minimum: {self.risk_thresholds['min_isd_diversity']})",
                    affected_entities=list(isds),
                    mitigation_suggestions=[
                        "Encourage cross-ISD participation",
                        "Monitor inter-ISD connectivity",
                        "Consider ISD-based governance requirements",
                    ],
                )
            )

        return risks

    def _calculate_diversity_metrics(self, votes: list[VoteRecord]) -> dict[str, any]:
        """Calculate comprehensive diversity metrics."""
        metrics = {}

        # Basic counts
        as_groups = {vote.as_group for vote in votes}
        isds = {vote.isd for vote in votes}
        orgs = {vote.organization for vote in votes}

        metrics["as_diversity"] = {
            "unique_groups": len(as_groups),
            "groups": sorted(as_groups),
        }

        metrics["isd_diversity"] = {"unique_isds": len(isds), "isds": sorted(isds)}

        metrics["org_diversity"] = {"unique_orgs": len(orgs), "orgs": sorted(orgs)}

        # Weight distribution by entity type
        total_weight = sum(vote.weight for vote in votes) if votes else 0

        if total_weight > 0:
            # AS weight distribution
            as_weights = {}
            for vote in votes:
                as_weights[vote.as_group] = as_weights.get(vote.as_group, 0) + vote.weight
            metrics["as_weight_distribution"] = {
                as_group: weight / total_weight for as_group, weight in as_weights.items()
            }

            # ISD weight distribution
            isd_weights = {}
            for vote in votes:
                isd_weights[vote.isd] = isd_weights.get(vote.isd, 0) + vote.weight
            metrics["isd_weight_distribution"] = {isd: weight / total_weight for isd, weight in isd_weights.items()}

        return metrics

    def _calculate_geographic_distribution(self, votes: list[VoteRecord]) -> dict[str, any]:
        """Calculate geographic distribution metrics."""
        distribution = {}

        # Extract regions from AS groups
        region_counts = {}
        region_weights = {}
        total_weight = sum(vote.weight for vote in votes) if votes else 0

        for vote in votes:
            region = self._extract_region_from_as(vote.as_group)
            region_counts[region] = region_counts.get(region, 0) + 1
            region_weights[region] = region_weights.get(region, 0) + vote.weight

        distribution["region_node_counts"] = region_counts

        if total_weight > 0:
            distribution["region_weight_distribution"] = {
                region: weight / total_weight for region, weight in region_weights.items()
            }

        distribution["unique_regions"] = len(region_counts)

        return distribution

    def _calculate_temporal_analysis(self, votes: list[VoteRecord]) -> dict[str, any]:
        """Calculate temporal voting pattern analysis."""
        analysis = {}

        if not votes:
            return analysis

        timestamps = [vote.timestamp for vote in votes]
        timestamps.sort()

        analysis["vote_count"] = len(votes)
        analysis["time_span_seconds"] = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        analysis["first_vote"] = timestamps[0]
        analysis["last_vote"] = timestamps[-1]

        # Calculate vote rate over time
        if analysis["time_span_seconds"] > 0:
            analysis["avg_vote_rate_per_minute"] = (len(votes) / analysis["time_span_seconds"]) * 60

        # Time intervals between votes
        if len(timestamps) > 1:
            intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            analysis["avg_interval_seconds"] = sum(intervals) / len(intervals)
            analysis["max_interval_seconds"] = max(intervals)
            analysis["min_interval_seconds"] = min(intervals)

        return analysis

    def _extract_region_from_as(self, as_group: str) -> str:
        """Extract geographic region from AS group (simplified heuristic)."""
        # This is a simplified heuristic - in practice would use AS geolocation data
        if as_group.startswith(("1-", "2-", "3-")):
            return "North America"
        elif as_group.startswith(("4-", "5-", "6-")):
            return "Europe"
        elif as_group.startswith(("7-", "8-")):
            return "Asia"
        elif as_group.startswith(("9-",)):
            return "Global South"
        else:
            return "Unknown"

    def _calculate_overall_risk_score(self, risks: list[PartitionRisk]) -> float:
        """Calculate overall risk score from individual risks."""
        if not risks:
            return 0.0

        # Weight risks by severity and type
        risk_weights = {
            "AS_CONCENTRATION": 1.0,
            "ISD_CONCENTRATION": 1.2,  # Slightly higher weight
            "ORG_CONCENTRATION": 0.9,
            "GEOGRAPHIC_CONCENTRATION": 0.8,
            "TEMPORAL_CONCENTRATION": 0.7,
            "INSUFFICIENT_AS_DIVERSITY": 1.0,
            "INSUFFICIENT_ISD_DIVERSITY": 1.1,
            "INSUFFICIENT_GEOGRAPHIC_DIVERSITY": 0.6,
            "NO_VOTES": 1.0,
            "ASSESSMENT_ERROR": 0.8,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for risk in risks:
            weight = risk_weights.get(risk.risk_type, 0.5)
            weighted_score += risk.severity * weight
            total_weight += weight

        return min(1.0, weighted_score / total_weight) if total_weight > 0 else 0.0

    def _generate_recommendations(self, risks: list[PartitionRisk], diversity_metrics: dict[str, any]) -> list[str]:
        """Generate actionable recommendations based on risk assessment."""
        recommendations = []

        if not risks:
            recommendations.append("Network partition safety is good - continue monitoring")
            return recommendations

        # Group risks by type for targeted recommendations
        risk_types = {risk.risk_type for risk in risks}

        if "AS_CONCENTRATION" in risk_types or "ISD_CONCENTRATION" in risk_types:
            recommendations.append("Implement stronger weight distribution mechanisms")
            recommendations.append("Encourage participation from underrepresented AS/ISD groups")

        if "INSUFFICIENT_AS_DIVERSITY" in risk_types:
            recommendations.append("Lower barriers to AS group participation")
            recommendations.append("Implement AS discovery and onboarding programs")

        if "INSUFFICIENT_ISD_DIVERSITY" in risk_types:
            recommendations.append("Critical: Improve inter-ISD connectivity")
            recommendations.append("Consider emergency procedures for single-ISD scenarios")

        if "GEOGRAPHIC_CONCENTRATION" in risk_types:
            recommendations.append("Promote global participation and geographic diversity")
            recommendations.append("Monitor for regional network outages")

        if "TEMPORAL_CONCENTRATION" in risk_types:
            recommendations.append("Investigate potential coordinated voting patterns")
            recommendations.append("Implement vote timing diversity requirements")

        # Add severity-based recommendations
        high_severity_risks = [r for r in risks if r.severity > 0.7]
        if high_severity_risks:
            recommendations.append("HIGH PRIORITY: Address high-severity partition risks immediately")
            recommendations.append("Consider emergency governance procedures")

        return recommendations

    def get_historical_trends(self, days: int = 7) -> dict[str, any]:
        """Get historical partition safety trends."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_reports = [r for r in self.partition_history if r.timestamp >= cutoff_time]

        if not recent_reports:
            return {"error": "No historical data available"}

        # Calculate trends
        risk_scores = [r.overall_risk_score for r in recent_reports]
        safety_statuses = [r.is_partition_safe for r in recent_reports]

        trends = {
            "period_days": days,
            "total_assessments": len(recent_reports),
            "avg_risk_score": sum(risk_scores) / len(risk_scores),
            "safety_percentage": sum(safety_statuses) / len(safety_statuses) * 100,
            "risk_trend": "improving"
            if len(risk_scores) > 1 and risk_scores[-1] < risk_scores[0]
            else "stable_or_worsening",
            "most_common_risks": self._get_most_common_risks(recent_reports),
        }

        return trends

    def _get_most_common_risks(self, reports: list[PartitionSafetyReport]) -> dict[str, int]:
        """Get most common risk types from reports."""
        risk_counts = {}

        for report in reports:
            for risk in report.detected_risks:
                risk_counts[risk.risk_type] = risk_counts.get(risk.risk_type, 0) + 1

        return dict(sorted(risk_counts.items(), key=lambda x: x[1], reverse=True))
