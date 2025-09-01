"""
Heterogeneous Quorum Manager

Manages infrastructure diversity requirements for high-tier SLA guarantees.
Enforces disjoint infrastructure constraints for Gold tier services.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Optional

from .infrastructure_classifier import InfrastructureClassifier, InfrastructureProfile, PowerRegion, TEEVendor


class QuorumRequirement(Enum):
    """Quorum requirement levels"""

    NONE = "none"  # No diversity requirements
    BASIC = "basic"  # Basic geographic diversity
    ENHANCED = "enhanced"  # ASN + geographic diversity
    GOLD = "gold"  # Full disjoint infrastructure


@dataclass
class DiversityConstraints:
    """Infrastructure diversity constraints"""

    min_asn_diversity: int = 1  # Minimum unique ASNs
    min_tee_vendor_diversity: int = 1  # Minimum unique TEE vendors
    min_power_region_diversity: int = 1  # Minimum unique power regions
    min_geographic_diversity: int = 1  # Minimum unique countries
    max_devices_per_asn: int = 999  # Maximum devices per ASN
    max_devices_per_power_region: int = 999  # Max devices per power region
    require_tee_diversity: bool = False  # Require multiple TEE vendors
    min_confidence_score: float = 0.5  # Minimum classification confidence


@dataclass
class QuorumValidationResult:
    """Result of quorum validation"""

    is_valid: bool
    diversity_score: float
    violations: list[str]
    recommendations: list[str]
    profiles_used: list[InfrastructureProfile]
    metadata: dict


class QuorumManager:
    """Manages heterogeneous quorum requirements and validation"""

    def __init__(self, classifier: InfrastructureClassifier | None = None):
        self.classifier = classifier or InfrastructureClassifier()
        self.logger = logging.getLogger(__name__)

        # Predefined constraint templates
        self.constraint_templates = {
            QuorumRequirement.NONE: DiversityConstraints(),
            QuorumRequirement.BASIC: DiversityConstraints(min_geographic_diversity=2, min_confidence_score=0.6),
            QuorumRequirement.ENHANCED: DiversityConstraints(
                min_asn_diversity=2, min_geographic_diversity=2, max_devices_per_asn=2, min_confidence_score=0.7
            ),
            QuorumRequirement.GOLD: DiversityConstraints(
                min_asn_diversity=3,
                min_tee_vendor_diversity=2,
                min_power_region_diversity=2,
                min_geographic_diversity=2,
                max_devices_per_asn=1,
                max_devices_per_power_region=1,
                require_tee_diversity=True,
                min_confidence_score=0.8,
            ),
        }

    async def validate_quorum(
        self,
        device_candidates: list[dict],
        requirement: QuorumRequirement,
        target_size: int = 3,
        custom_constraints: DiversityConstraints | None = None,
    ) -> QuorumValidationResult:
        """
        Validate and select optimal quorum based on diversity constraints

        Args:
            device_candidates: List of device information dicts
            requirement: Quorum requirement level
            target_size: Target quorum size
            custom_constraints: Override default constraints

        Returns:
            Validation result with selected devices
        """
        constraints = custom_constraints or self.constraint_templates[requirement]

        # Classify all candidate devices
        profiles = await self._classify_devices(device_candidates)

        # Filter by confidence score
        qualified_profiles = [p for p in profiles if p.confidence_score >= constraints.min_confidence_score]

        if len(qualified_profiles) < target_size:
            return QuorumValidationResult(
                is_valid=False,
                diversity_score=0.0,
                violations=[f"Insufficient qualified devices: {len(qualified_profiles)} < {target_size}"],
                recommendations=["Add more devices with better classification confidence"],
                profiles_used=[],
                metadata={"total_candidates": len(device_candidates)},
            )

        # Select optimal quorum
        selected_profiles, selection_score = await self._select_optimal_quorum(
            qualified_profiles, constraints, target_size
        )

        # Validate constraints
        violations = self._validate_constraints(selected_profiles, constraints)
        diversity_metrics = self.classifier.get_diversity_metrics(selected_profiles)

        # Generate recommendations
        recommendations = self._generate_recommendations(selected_profiles, constraints, violations, diversity_metrics)

        return QuorumValidationResult(
            is_valid=len(violations) == 0,
            diversity_score=diversity_metrics["total_diversity_score"],
            violations=violations,
            recommendations=recommendations,
            profiles_used=selected_profiles,
            metadata={
                "selection_score": selection_score,
                "diversity_metrics": diversity_metrics,
                "constraint_level": requirement.value,
                "total_candidates": len(device_candidates),
                "qualified_candidates": len(qualified_profiles),
            },
        )

    async def _classify_devices(self, device_candidates: list[dict]) -> list[InfrastructureProfile]:
        """Classify all device candidates"""
        tasks = []

        for device in device_candidates:
            task = self.classifier.classify_device(
                device_id=device["id"],
                ip_address=device["ip_address"],
                attestation_data=device.get("attestation_data"),
                network_info=device.get("network_info"),
            )
            tasks.append(task)

        profiles = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_profiles = []
        for i, profile in enumerate(profiles):
            if isinstance(profile, Exception):
                self.logger.error(f"Classification failed for device {device_candidates[i]['id']}: {profile}")
            else:
                valid_profiles.append(profile)

        return valid_profiles

    async def _select_optimal_quorum(
        self, profiles: list[InfrastructureProfile], constraints: DiversityConstraints, target_size: int
    ) -> tuple[list[InfrastructureProfile], float]:
        """
        Select optimal quorum using diversity-aware algorithm
        """
        if len(profiles) <= target_size:
            return profiles, self._calculate_selection_score(profiles, constraints)

        # Use greedy algorithm to maximize diversity
        selected = []
        remaining = profiles.copy()

        while len(selected) < target_size and remaining:
            best_candidate = None
            best_score = -1

            for candidate in remaining:
                # Calculate diversity improvement
                test_selection = selected + [candidate]
                score = self._calculate_selection_score(test_selection, constraints)

                # Penalize constraint violations
                violations = self._validate_constraints(test_selection, constraints)
                violation_penalty = len(violations) * 0.2

                adjusted_score = score - violation_penalty

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        final_score = self._calculate_selection_score(selected, constraints)
        return selected, final_score

    def _calculate_selection_score(
        self, profiles: list[InfrastructureProfile], constraints: DiversityConstraints
    ) -> float:
        """Calculate selection quality score"""
        if not profiles:
            return 0.0

        metrics = self.classifier.get_diversity_metrics(profiles)

        # Base diversity score (0-1)
        diversity_score = metrics["total_diversity_score"]

        # Confidence bonus
        avg_confidence = sum(p.confidence_score for p in profiles) / len(profiles)
        confidence_bonus = (avg_confidence - 0.5) * 0.2  # Max 0.1 bonus

        # Size penalty if below target
        size_penalty = 0
        if len(profiles) < 3:  # Minimum quorum size
            size_penalty = (3 - len(profiles)) * 0.1

        # TEE diversity bonus for Gold tier
        tee_bonus = 0
        if constraints.require_tee_diversity and metrics["unique_tee_vendors"] >= 2:
            tee_bonus = 0.1

        total_score = diversity_score + confidence_bonus + tee_bonus - size_penalty
        return max(0.0, min(1.0, total_score))

    def _validate_constraints(
        self, profiles: list[InfrastructureProfile], constraints: DiversityConstraints
    ) -> list[str]:
        """Validate diversity constraints"""
        violations = []

        if not profiles:
            return ["No profiles provided"]

        metrics = self.classifier.get_diversity_metrics(profiles)

        # ASN diversity
        if metrics["unique_asns"] < constraints.min_asn_diversity:
            violations.append(f"Insufficient ASN diversity: {metrics['unique_asns']} < {constraints.min_asn_diversity}")

        # TEE vendor diversity
        if metrics["unique_tee_vendors"] < constraints.min_tee_vendor_diversity:
            violations.append(
                f"Insufficient TEE vendor diversity: {metrics['unique_tee_vendors']} < {constraints.min_tee_vendor_diversity}"
            )

        # Power region diversity
        if metrics["unique_power_regions"] < constraints.min_power_region_diversity:
            violations.append(
                f"Insufficient power region diversity: {metrics['unique_power_regions']} < {constraints.min_power_region_diversity}"
            )

        # Geographic diversity
        if metrics["unique_countries"] < constraints.min_geographic_diversity:
            violations.append(
                f"Insufficient geographic diversity: {metrics['unique_countries']} < {constraints.min_geographic_diversity}"
            )

        # Devices per ASN limit
        asn_counts = {}
        for profile in profiles:
            if profile.asn:
                asn_counts[profile.asn] = asn_counts.get(profile.asn, 0) + 1

        for asn, count in asn_counts.items():
            if count > constraints.max_devices_per_asn:
                violations.append(f"Too many devices in ASN {asn}: {count} > {constraints.max_devices_per_asn}")

        # Devices per power region limit
        power_counts = {}
        for profile in profiles:
            power_counts[profile.power_region] = power_counts.get(profile.power_region, 0) + 1

        for region, count in power_counts.items():
            if count > constraints.max_devices_per_power_region:
                violations.append(
                    f"Too many devices in power region {region.value}: {count} > {constraints.max_devices_per_power_region}"
                )

        # TEE diversity requirement
        if constraints.require_tee_diversity:
            tee_vendors = set(p.tee_vendor for p in profiles)
            if len(tee_vendors) < 2 or TEEVendor.UNKNOWN in tee_vendors:
                violations.append("Gold tier requires multiple known TEE vendors")

        # Confidence requirement
        low_confidence = [p for p in profiles if p.confidence_score < constraints.min_confidence_score]
        if low_confidence:
            violations.append(
                f"{len(low_confidence)} devices below confidence threshold {constraints.min_confidence_score}"
            )

        return violations

    def _generate_recommendations(
        self,
        profiles: list[InfrastructureProfile],
        constraints: DiversityConstraints,
        violations: list[str],
        metrics: dict,
    ) -> list[str]:
        """Generate recommendations for improving quorum"""
        recommendations = []

        if not violations:
            recommendations.append("Quorum meets all diversity requirements")
            return recommendations

        # ASN diversity recommendations
        if metrics["unique_asns"] < constraints.min_asn_diversity:
            needed = constraints.min_asn_diversity - metrics["unique_asns"]
            recommendations.append(f"Add {needed} more devices from different ASNs")

        # TEE vendor diversity
        if metrics["unique_tee_vendors"] < constraints.min_tee_vendor_diversity:
            tee_vendors = set(p.tee_vendor for p in profiles)
            if TEEVendor.AMD_SEV_SNP not in tee_vendors:
                recommendations.append("Add AMD SEV-SNP capable device")
            if TEEVendor.INTEL_TDX not in tee_vendors:
                recommendations.append("Add Intel TDX capable device")

        # Power region diversity
        if metrics["unique_power_regions"] < constraints.min_power_region_diversity:
            current_regions = set(p.power_region for p in profiles)
            missing_regions = set(PowerRegion) - current_regions - {PowerRegion.UNKNOWN}
            if missing_regions:
                recommendations.append(
                    f"Add devices from power regions: {[r.value for r in list(missing_regions)[:2]]}"
                )

        # Geographic diversity
        if metrics["unique_countries"] < constraints.min_geographic_diversity:
            recommendations.append("Add devices from different countries")

        # Confidence improvements
        low_confidence = [p for p in profiles if p.confidence_score < constraints.min_confidence_score]
        if low_confidence:
            recommendations.append("Improve device attestation data quality for better classification confidence")

        return recommendations

    def get_quorum_status_summary(self, profiles: list[InfrastructureProfile]) -> dict:
        """Get comprehensive quorum status summary"""
        if not profiles:
            return {"status": "empty", "devices": 0}

        metrics = self.classifier.get_diversity_metrics(profiles)

        # Determine highest satisfied requirement level
        satisfied_levels = []
        for level in QuorumRequirement:
            constraints = self.constraint_templates[level]
            violations = self._validate_constraints(profiles, constraints)
            if not violations:
                satisfied_levels.append(level)

        highest_level = max(satisfied_levels) if satisfied_levels else QuorumRequirement.NONE

        # ASN distribution
        asn_distribution = {}
        for profile in profiles:
            asn_key = f"AS{profile.asn}" if profile.asn else "Unknown"
            asn_distribution[asn_key] = asn_distribution.get(asn_key, 0) + 1

        # TEE vendor distribution
        tee_distribution = {}
        for profile in profiles:
            tee_distribution[profile.tee_vendor.value] = tee_distribution.get(profile.tee_vendor.value, 0) + 1

        # Power region distribution
        power_distribution = {}
        for profile in profiles:
            power_distribution[profile.power_region.value] = power_distribution.get(profile.power_region.value, 0) + 1

        return {
            "status": "active",
            "devices": len(profiles),
            "highest_sla_level": highest_level.value,
            "diversity_score": metrics["total_diversity_score"],
            "diversity_breakdown": {
                "asn": metrics["asn_diversity"],
                "tee_vendor": metrics["tee_vendor_diversity"],
                "power_region": metrics["power_region_diversity"],
                "geographic": metrics["geographic_diversity"],
            },
            "distributions": {
                "asn": asn_distribution,
                "tee_vendor": tee_distribution,
                "power_region": power_distribution,
            },
            "avg_confidence": sum(p.confidence_score for p in profiles) / len(profiles),
            "classification_time": datetime.utcnow().isoformat(),
        }

    async def continuously_monitor_quorum(
        self,
        device_candidates: list[dict],
        requirement: QuorumRequirement,
        callback: Optional[callable] = None,
        interval_seconds: int = 300,
    ) -> None:
        """Continuously monitor quorum diversity and alert on violations"""
        self.logger.info(f"Starting continuous quorum monitoring (interval: {interval_seconds}s)")

        while True:
            try:
                result = await self.validate_quorum(device_candidates, requirement)

                if callback:
                    await callback(result)
                elif not result.is_valid:
                    self.logger.warning(f"Quorum diversity violations: {result.violations}")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Quorum monitoring error: {e}")
                await asyncio.sleep(60)  # Error backoff
