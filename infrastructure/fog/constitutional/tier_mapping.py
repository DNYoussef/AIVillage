"""
Constitutional Tier Mapping System

Maps the existing 4-tier system (Small/Medium/Large/Enterprise) to a constitutional
Bronze/Silver/Gold tier system with integrated AI safety, governance, and harm prevention.

The constitutional tier system introduces:
- Graduated AI safety requirements
- Harm taxonomy enforcement levels  
- Constitutional constraint application
- Machine-only moderation thresholds
- Transparency and audit requirements

Architecture Decision Record:
- Bronze: Basic safety with minimal constitutional oversight
- Silver: Enhanced safety with moderate constitutional enforcement
- Gold: Maximum safety with full constitutional governance
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConstitutionalTier(str, Enum):
    """Constitutional tier levels with integrated AI safety"""

    BRONZE = "bronze"  # Basic constitutional protections
    SILVER = "silver"  # Enhanced constitutional enforcement
    GOLD = "gold"  # Maximum constitutional governance


class LegacyTier(str, Enum):
    """Legacy tier system from existing infrastructure"""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class SafetyLevel(str, Enum):
    """AI safety enforcement levels"""

    BASIC = "basic"  # Fundamental safety checks
    ENHANCED = "enhanced"  # Advanced safety monitoring
    MAXIMUM = "maximum"  # Full safety governance


class HarmPreventionLevel(str, Enum):
    """Harm prevention and taxonomy enforcement levels"""

    MINIMAL = "minimal"  # Basic harm detection
    MODERATE = "moderate"  # Enhanced harm prevention
    COMPREHENSIVE = "comprehensive"  # Full harm taxonomy enforcement


@dataclass
class ConstitutionalConstraints:
    """Constitutional constraints applied at tier level"""

    # Content and safety constraints
    content_filtering_level: str = "basic"
    harm_detection_threshold: Decimal = Decimal("0.7")
    safety_check_frequency: int = 100  # Per N requests

    # Moderation requirements
    requires_human_oversight: bool = False
    machine_only_moderation: bool = True
    moderation_response_time_sla: int = 300  # seconds

    # Transparency requirements
    audit_logging_required: bool = True
    transparency_level: str = "basic"  # basic, detailed, comprehensive
    decision_explanation_required: bool = False

    # Resource and access constraints
    max_concurrent_sensitive_workloads: int = 10
    requires_viewpoint_firewall: bool = False
    constitutional_review_required: bool = False

    # Governance integration
    governance_escalation_threshold: Decimal = Decimal("0.9")
    constitutional_appeal_process: bool = False
    community_oversight_required: bool = False


@dataclass
class TierCapabilities:
    """Capabilities and limits for each constitutional tier"""

    # Resource allocations (inherited from legacy system)
    max_cpu_cores: Decimal
    max_memory_gb: Decimal
    max_storage_gb: Decimal
    max_concurrent_jobs: int

    # Constitutional capabilities
    safety_level: SafetyLevel
    harm_prevention_level: HarmPreventionLevel
    constitutional_constraints: ConstitutionalConstraints

    # SLA guarantees
    guaranteed_uptime_percentage: Decimal = Decimal("95")
    max_latency_sla_ms: Decimal = Decimal("500")
    constitutional_compliance_sla: Decimal = Decimal("99")  # % constitutional compliance

    # Access controls
    sensitive_workload_access: bool = False
    advanced_ai_model_access: bool = False
    federated_coordination_access: bool = False

    # Pricing and economics
    constitutional_overhead_multiplier: Decimal = Decimal("1.0")
    governance_fee_percentage: Decimal = Decimal("0.0")


@dataclass
class TierMapping:
    """Maps legacy tier to constitutional tier with transition plan"""

    legacy_tier: LegacyTier
    constitutional_tier: ConstitutionalTier
    transition_date: datetime

    # Mapping rationale
    mapping_reason: str = ""
    compatibility_score: Decimal = Decimal("1.0")  # 0-1 compatibility

    # Transition details
    migration_required: bool = False
    breaking_changes: list[str] = field(default_factory=list)
    feature_additions: list[str] = field(default_factory=list)

    # Governance implications
    new_constraints_applied: list[str] = field(default_factory=list)
    safety_improvements: list[str] = field(default_factory=list)
    transparency_changes: list[str] = field(default_factory=list)


class ConstitutionalTierManager:
    """
    Manages constitutional tier system and mapping from legacy tiers

    Responsibilities:
    - Define constitutional tier capabilities and constraints
    - Map legacy tiers to constitutional equivalents
    - Enforce tier-based constitutional requirements
    - Manage tier transitions and migrations
    - Integrate with governance and safety systems
    """

    def __init__(self):
        self.tier_capabilities = self._initialize_tier_capabilities()
        self.tier_mappings = self._initialize_tier_mappings()

        # Integration points
        self.governance_engine = None  # Will be injected
        self.safety_monitor = None  # Will be injected
        self.transparency_logger = None  # Will be injected

        # State tracking
        self.active_tier_assignments: dict[str, ConstitutionalTier] = {}
        self.tier_transition_queue: list[dict[str, Any]] = []

        logger.info("Constitutional tier manager initialized")

    def _initialize_tier_capabilities(self) -> dict[ConstitutionalTier, TierCapabilities]:
        """Initialize capabilities for each constitutional tier"""

        return {
            ConstitutionalTier.BRONZE: TierCapabilities(
                # Resource limits (based on Small/Medium legacy tiers)
                max_cpu_cores=Decimal("16"),
                max_memory_gb=Decimal("32"),
                max_storage_gb=Decimal("200"),
                max_concurrent_jobs=25,
                # Constitutional features
                safety_level=SafetyLevel.BASIC,
                harm_prevention_level=HarmPreventionLevel.MINIMAL,
                constitutional_constraints=ConstitutionalConstraints(
                    content_filtering_level="basic",
                    harm_detection_threshold=Decimal("0.8"),
                    safety_check_frequency=200,  # Every 200 requests
                    requires_human_oversight=False,
                    machine_only_moderation=True,
                    audit_logging_required=True,
                    transparency_level="basic",
                    decision_explanation_required=False,
                    max_concurrent_sensitive_workloads=5,
                    requires_viewpoint_firewall=False,
                    governance_escalation_threshold=Decimal("0.95"),
                ),
                # SLAs
                guaranteed_uptime_percentage=Decimal("95"),
                max_latency_sla_ms=Decimal("800"),
                constitutional_compliance_sla=Decimal("95"),
                # Access
                sensitive_workload_access=False,
                advanced_ai_model_access=False,
                federated_coordination_access=True,
                # Economics
                constitutional_overhead_multiplier=Decimal("1.05"),  # 5% overhead
                governance_fee_percentage=Decimal("0.5"),  # 0.5% governance fee
            ),
            ConstitutionalTier.SILVER: TierCapabilities(
                # Resource limits (based on Large legacy tier)
                max_cpu_cores=Decimal("64"),
                max_memory_gb=Decimal("128"),
                max_storage_gb=Decimal("1000"),
                max_concurrent_jobs=75,
                # Constitutional features
                safety_level=SafetyLevel.ENHANCED,
                harm_prevention_level=HarmPreventionLevel.MODERATE,
                constitutional_constraints=ConstitutionalConstraints(
                    content_filtering_level="enhanced",
                    harm_detection_threshold=Decimal("0.6"),
                    safety_check_frequency=50,  # Every 50 requests
                    requires_human_oversight=False,
                    machine_only_moderation=True,
                    audit_logging_required=True,
                    transparency_level="detailed",
                    decision_explanation_required=True,
                    max_concurrent_sensitive_workloads=15,
                    requires_viewpoint_firewall=True,
                    governance_escalation_threshold=Decimal("0.85"),
                    constitutional_review_required=False,
                ),
                # SLAs
                guaranteed_uptime_percentage=Decimal("98"),
                max_latency_sla_ms=Decimal("400"),
                constitutional_compliance_sla=Decimal("98"),
                # Access
                sensitive_workload_access=True,
                advanced_ai_model_access=True,
                federated_coordination_access=True,
                # Economics
                constitutional_overhead_multiplier=Decimal("1.15"),  # 15% overhead
                governance_fee_percentage=Decimal("1.0"),  # 1% governance fee
            ),
            ConstitutionalTier.GOLD: TierCapabilities(
                # Resource limits (based on Enterprise legacy tier)
                max_cpu_cores=Decimal("256"),
                max_memory_gb=Decimal("512"),
                max_storage_gb=Decimal("5000"),
                max_concurrent_jobs=200,
                # Constitutional features
                safety_level=SafetyLevel.MAXIMUM,
                harm_prevention_level=HarmPreventionLevel.COMPREHENSIVE,
                constitutional_constraints=ConstitutionalConstraints(
                    content_filtering_level="comprehensive",
                    harm_detection_threshold=Decimal("0.4"),
                    safety_check_frequency=10,  # Every 10 requests
                    requires_human_oversight=True,  # Human oversight for sensitive decisions
                    machine_only_moderation=False,
                    audit_logging_required=True,
                    transparency_level="comprehensive",
                    decision_explanation_required=True,
                    max_concurrent_sensitive_workloads=50,
                    requires_viewpoint_firewall=True,
                    governance_escalation_threshold=Decimal("0.7"),
                    constitutional_review_required=True,
                    constitutional_appeal_process=True,
                    community_oversight_required=True,
                ),
                # SLAs
                guaranteed_uptime_percentage=Decimal("99.5"),
                max_latency_sla_ms=Decimal("200"),
                constitutional_compliance_sla=Decimal("99.9"),
                # Access
                sensitive_workload_access=True,
                advanced_ai_model_access=True,
                federated_coordination_access=True,
                # Economics
                constitutional_overhead_multiplier=Decimal("1.3"),  # 30% overhead
                governance_fee_percentage=Decimal("2.0"),  # 2% governance fee
            ),
        }

    def _initialize_tier_mappings(self) -> list[TierMapping]:
        """Initialize mappings from legacy to constitutional tiers"""

        return [
            # Small -> Bronze (Natural fit for basic users)
            TierMapping(
                legacy_tier=LegacyTier.SMALL,
                constitutional_tier=ConstitutionalTier.BRONZE,
                transition_date=datetime.now(UTC),
                mapping_reason="Small users get basic constitutional protections with minimal overhead",
                compatibility_score=Decimal("0.95"),
                migration_required=False,
                feature_additions=[
                    "Basic content filtering",
                    "Harm detection monitoring",
                    "Constitutional audit logging",
                    "Basic transparency reporting",
                ],
                safety_improvements=[
                    "Automated safety checks every 200 requests",
                    "Machine-only moderation with basic harm detection",
                    "Constitutional escalation for high-risk content",
                ],
                transparency_changes=[
                    "Basic audit logging for all operations",
                    "Transparency reports for constitutional actions",
                ],
            ),
            # Medium -> Bronze/Silver (Split based on usage patterns)
            TierMapping(
                legacy_tier=LegacyTier.MEDIUM,
                constitutional_tier=ConstitutionalTier.BRONZE,  # Default to Bronze
                transition_date=datetime.now(UTC),
                mapping_reason="Medium users start at Bronze, can upgrade to Silver based on needs",
                compatibility_score=Decimal("0.90"),
                migration_required=False,
                feature_additions=[
                    "Constitutional tier upgrade path to Silver",
                    "Enhanced safety monitoring options",
                    "Viewpoint firewall eligibility",
                ],
            ),
            # Large -> Silver (Enhanced constitutional governance)
            TierMapping(
                legacy_tier=LegacyTier.LARGE,
                constitutional_tier=ConstitutionalTier.SILVER,
                transition_date=datetime.now(UTC),
                mapping_reason="Large users require enhanced constitutional oversight and capabilities",
                compatibility_score=Decimal("0.85"),
                migration_required=True,
                breaking_changes=[
                    "Enhanced content filtering may block previously allowed content",
                    "Viewpoint firewall integration required",
                    "Detailed audit logging and transparency reporting",
                ],
                feature_additions=[
                    "Advanced AI model access",
                    "Sensitive workload processing",
                    "Enhanced harm prevention",
                    "Decision explanation system",
                    "Constitutional appeal process eligibility",
                ],
                safety_improvements=[
                    "Enhanced safety checks every 50 requests",
                    "Moderate harm prevention with expanded taxonomy",
                    "Machine-only moderation with human escalation paths",
                    "Constitutional review for sensitive decisions",
                ],
                transparency_changes=[
                    "Detailed audit logging for all operations",
                    "Decision explanations for constitutional actions",
                    "Regular transparency and compliance reporting",
                ],
            ),
            # Enterprise -> Gold (Maximum constitutional governance)
            TierMapping(
                legacy_tier=LegacyTier.ENTERPRISE,
                constitutional_tier=ConstitutionalTier.GOLD,
                transition_date=datetime.now(UTC),
                mapping_reason="Enterprise users get maximum constitutional governance and oversight",
                compatibility_score=Decimal("0.75"),
                migration_required=True,
                breaking_changes=[
                    "Comprehensive content filtering with strictest standards",
                    "Mandatory human oversight for sensitive decisions",
                    "Full constitutional review process integration",
                    "Community oversight participation required",
                ],
                feature_additions=[
                    "Full access to all AI models and capabilities",
                    "Maximum sensitive workload processing",
                    "Constitutional appeal process with community involvement",
                    "Advanced constitutional constraint customization",
                    "Priority constitutional support and consultation",
                ],
                safety_improvements=[
                    "Maximum safety checks every 10 requests",
                    "Comprehensive harm prevention with full taxonomy",
                    "Human oversight for all high-risk decisions",
                    "Real-time constitutional compliance monitoring",
                    "Advanced threat detection and prevention",
                ],
                transparency_changes=[
                    "Comprehensive audit logging with real-time monitoring",
                    "Full decision explanations for all constitutional actions",
                    "Public transparency reporting (anonymized)",
                    "Community oversight participation and reporting",
                ],
            ),
        ]

    def get_constitutional_tier(self, user_id: str, legacy_tier: LegacyTier | None = None) -> ConstitutionalTier:
        """Get constitutional tier for user (with fallback mapping)"""

        # Check if user already has constitutional tier assignment
        if user_id in self.active_tier_assignments:
            return self.active_tier_assignments[user_id]

        # Map from legacy tier if provided
        if legacy_tier:
            mapped_tier = self.map_legacy_tier_to_constitutional(legacy_tier)
            self.active_tier_assignments[user_id] = mapped_tier
            return mapped_tier

        # Default to Bronze tier for new users
        default_tier = ConstitutionalTier.BRONZE
        self.active_tier_assignments[user_id] = default_tier
        logger.info(f"Assigned default constitutional tier {default_tier.value} to user {user_id}")
        return default_tier

    def map_legacy_tier_to_constitutional(self, legacy_tier: LegacyTier) -> ConstitutionalTier:
        """Map legacy tier to constitutional equivalent"""

        # Find mapping
        mapping = next((m for m in self.tier_mappings if m.legacy_tier == legacy_tier), None)
        if not mapping:
            logger.warning(f"No mapping found for legacy tier {legacy_tier.value}, defaulting to Bronze")
            return ConstitutionalTier.BRONZE

        logger.info(
            f"Mapped legacy tier {legacy_tier.value} to constitutional tier {mapping.constitutional_tier.value} "
            f"(compatibility: {float(mapping.compatibility_score):.2f})"
        )

        return mapping.constitutional_tier

    def get_tier_capabilities(self, tier: ConstitutionalTier) -> TierCapabilities:
        """Get capabilities for constitutional tier"""
        return self.tier_capabilities[tier]

    def get_tier_constraints(self, tier: ConstitutionalTier) -> ConstitutionalConstraints:
        """Get constitutional constraints for tier"""
        return self.tier_capabilities[tier].constitutional_constraints

    async def validate_tier_compliance(
        self, user_id: str, workload_type: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that workload complies with user's constitutional tier requirements"""

        user_tier = self.get_constitutional_tier(user_id)
        capabilities = self.get_tier_capabilities(user_tier)
        constraints = capabilities.constitutional_constraints

        compliance_result = {
            "tier": user_tier.value,
            "compliant": True,
            "violations": [],
            "warnings": [],
            "constitutional_actions_required": [],
        }

        # Check content filtering requirements
        if workload_type == "sensitive" and not capabilities.sensitive_workload_access:
            compliance_result["compliant"] = False
            compliance_result["violations"].append(f"Sensitive workload access not allowed for {user_tier.value} tier")

        # Check safety requirements based on tier
        safety_check_needed = self._should_perform_safety_check(user_id, constraints.safety_check_frequency)
        if safety_check_needed:
            compliance_result["constitutional_actions_required"].append("safety_check")

        # Check if human oversight required
        if constraints.requires_human_oversight and self._is_high_risk_content(content_data):
            compliance_result["constitutional_actions_required"].append("human_oversight")

        # Check viewpoint firewall requirement
        if constraints.requires_viewpoint_firewall:
            compliance_result["constitutional_actions_required"].append("viewpoint_firewall_check")

        # Check governance escalation threshold
        risk_score = content_data.get("risk_score", 0.0)
        if risk_score > float(constraints.governance_escalation_threshold):
            compliance_result["constitutional_actions_required"].append("governance_escalation")

        return compliance_result

    def get_tier_upgrade_path(self, current_tier: ConstitutionalTier) -> ConstitutionalTier | None:
        """Get next tier in upgrade path"""

        upgrade_paths = {
            ConstitutionalTier.BRONZE: ConstitutionalTier.SILVER,
            ConstitutionalTier.SILVER: ConstitutionalTier.GOLD,
            ConstitutionalTier.GOLD: None,  # Maximum tier
        }

        return upgrade_paths.get(current_tier)

    def calculate_tier_migration_impact(self, user_id: str, target_tier: ConstitutionalTier) -> dict[str, Any]:
        """Calculate impact of migrating user to different constitutional tier"""

        current_tier = self.get_constitutional_tier(user_id)
        current_capabilities = self.get_tier_capabilities(current_tier)
        target_capabilities = self.get_tier_capabilities(target_tier)

        # Find relevant mapping
        mapping = None
        for tm in self.tier_mappings:
            if tm.constitutional_tier == target_tier:
                mapping = tm
                break

        impact = {
            "current_tier": current_tier.value,
            "target_tier": target_tier.value,
            "resource_changes": {
                "cpu_cores": float(target_capabilities.max_cpu_cores - current_capabilities.max_cpu_cores),
                "memory_gb": float(target_capabilities.max_memory_gb - current_capabilities.max_memory_gb),
                "concurrent_jobs": target_capabilities.max_concurrent_jobs - current_capabilities.max_concurrent_jobs,
            },
            "constitutional_changes": {
                "safety_level_change": f"{current_capabilities.safety_level.value} -> {target_capabilities.safety_level.value}",
                "harm_prevention_change": f"{current_capabilities.harm_prevention_level.value} -> {target_capabilities.harm_prevention_level.value}",
                "new_constraints": [],
                "removed_constraints": [],
            },
            "cost_impact": {
                "overhead_multiplier_change": float(
                    target_capabilities.constitutional_overhead_multiplier
                    - current_capabilities.constitutional_overhead_multiplier
                ),
                "governance_fee_change": float(
                    target_capabilities.governance_fee_percentage - current_capabilities.governance_fee_percentage
                ),
            },
            "migration_required": mapping.migration_required if mapping else True,
            "breaking_changes": mapping.breaking_changes if mapping else [],
            "feature_additions": mapping.feature_additions if mapping else [],
        }

        return impact

    async def process_tier_transitions(self):
        """Process queued tier transitions"""

        transitions_processed = 0

        for transition in self.tier_transition_queue[:]:
            try:
                user_id = transition["user_id"]
                target_tier = ConstitutionalTier(transition["target_tier"])

                # Validate transition is allowed
                if await self._validate_tier_transition(user_id, target_tier):
                    # Update user's tier assignment
                    self.active_tier_assignments[user_id] = target_tier

                    # Log transition
                    logger.info(f"Processed tier transition for user {user_id} to {target_tier.value}")
                    transitions_processed += 1

                    # Remove from queue
                    self.tier_transition_queue.remove(transition)

            except Exception as e:
                logger.error(f"Error processing tier transition: {e}")

        if transitions_processed > 0:
            logger.info(f"Processed {transitions_processed} constitutional tier transitions")

    def _should_perform_safety_check(self, user_id: str, frequency: int) -> bool:
        """Determine if safety check should be performed based on frequency"""
        # Simplified implementation - would track request counts in production
        return hash(user_id) % frequency == 0

    def _is_high_risk_content(self, content_data: dict[str, Any]) -> bool:
        """Determine if content is high-risk requiring human oversight"""
        risk_score = content_data.get("risk_score", 0.0)
        return risk_score > 0.8

    async def _validate_tier_transition(self, user_id: str, target_tier: ConstitutionalTier) -> bool:
        """Validate that tier transition is allowed and appropriate"""

        current_tier = self.get_constitutional_tier(user_id)

        # Can't transition to same tier
        if current_tier == target_tier:
            return False

        # Validate upgrade path (for now, only allow upgrades, not downgrades)
        if target_tier == ConstitutionalTier.SILVER and current_tier != ConstitutionalTier.BRONZE:
            return False
        if target_tier == ConstitutionalTier.GOLD and current_tier != ConstitutionalTier.SILVER:
            return False

        return True


# Module-level convenience functions


def map_legacy_tier_to_constitutional(legacy_tier: LegacyTier) -> ConstitutionalTier:
    """Convenience function to map legacy tier to constitutional tier"""

    mapping = {
        LegacyTier.SMALL: ConstitutionalTier.BRONZE,
        LegacyTier.MEDIUM: ConstitutionalTier.BRONZE,  # Default to Bronze, can upgrade
        LegacyTier.LARGE: ConstitutionalTier.SILVER,
        LegacyTier.ENTERPRISE: ConstitutionalTier.GOLD,
    }

    return mapping.get(legacy_tier, ConstitutionalTier.BRONZE)


def get_tier_requirements(tier: ConstitutionalTier) -> dict[str, Any]:
    """Get tier requirements for external systems"""

    manager = ConstitutionalTierManager()
    capabilities = manager.get_tier_capabilities(tier)
    constraints = capabilities.constitutional_constraints

    return {
        "tier": tier.value,
        "safety_level": capabilities.safety_level.value,
        "harm_prevention_level": capabilities.harm_prevention_level.value,
        "resource_limits": {
            "max_cpu_cores": float(capabilities.max_cpu_cores),
            "max_memory_gb": float(capabilities.max_memory_gb),
            "max_concurrent_jobs": capabilities.max_concurrent_jobs,
        },
        "constitutional_requirements": {
            "content_filtering_level": constraints.content_filtering_level,
            "harm_detection_threshold": float(constraints.harm_detection_threshold),
            "safety_check_frequency": constraints.safety_check_frequency,
            "requires_human_oversight": constraints.requires_human_oversight,
            "requires_viewpoint_firewall": constraints.requires_viewpoint_firewall,
            "audit_logging_required": constraints.audit_logging_required,
            "transparency_level": constraints.transparency_level,
        },
        "sla_commitments": {
            "guaranteed_uptime_percentage": float(capabilities.guaranteed_uptime_percentage),
            "max_latency_sla_ms": float(capabilities.max_latency_sla_ms),
            "constitutional_compliance_sla": float(capabilities.constitutional_compliance_sla),
        },
    }


if __name__ == "__main__":
    # Demo constitutional tier system
    import json

    async def demo_constitutional_tiers():
        manager = ConstitutionalTierManager()

        # Demo tier mapping
        legacy_tier = LegacyTier.LARGE
        constitutional_tier = manager.map_legacy_tier_to_constitutional(legacy_tier)

        print(f"Legacy tier {legacy_tier.value} maps to constitutional tier {constitutional_tier.value}")

        # Demo tier requirements
        requirements = get_tier_requirements(constitutional_tier)
        print(f"\nConstitutional tier {constitutional_tier.value} requirements:")
        print(json.dumps(requirements, indent=2))

        # Demo compliance validation
        compliance = await manager.validate_tier_compliance(
            user_id="demo_user", workload_type="sensitive", content_data={"risk_score": 0.85}
        )
        print("\nCompliance validation:")
        print(json.dumps(compliance, indent=2))

        # Demo migration impact
        impact = manager.calculate_tier_migration_impact("demo_user", ConstitutionalTier.GOLD)
        print("\nMigration impact to Gold tier:")
        print(json.dumps(impact, indent=2))

    asyncio.run(demo_constitutional_tiers())
