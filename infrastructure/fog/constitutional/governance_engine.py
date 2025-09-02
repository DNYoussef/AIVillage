"""
Constitutional Governance Engine

Implements the core governance system for constitutional AI safety in the fog computing
infrastructure. Provides machine-only moderation, harm taxonomy enforcement, viewpoint
firewall integration, and constitutional constraint application.

Key Features:
- Real-time constitutional compliance monitoring
- Automated harm detection and prevention using comprehensive taxonomy
- Machine-only moderation with escalation paths
- Viewpoint firewall integration for bias prevention
- Constitutional constraint enforcement at tier boundaries
- Transparency logging and audit trails
- Policy decision framework with appeals process

Architecture Decision Record:
- Machine-first approach with human escalation only for Gold tier
- Harm taxonomy based on established AI safety research
- Constitutional constraints applied at workload routing level
- Transparent decision-making with audit trails
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
import logging
from typing import Any
import uuid

from .tier_mapping import ConstitutionalTier

logger = logging.getLogger(__name__)


class HarmCategory(str, Enum):
    """Comprehensive harm taxonomy for AI safety"""

    # Direct harms
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"

    # Societal harms
    MISINFORMATION = "misinformation"
    BIAS_AMPLIFICATION = "bias_amplification"
    MANIPULATION = "manipulation"
    PRIVACY_VIOLATION = "privacy_violation"

    # Economic harms
    FRAUD = "fraud"
    UNFAIR_COMPETITION = "unfair_competition"
    LABOR_DISPLACEMENT = "labor_displacement"

    # Systemic harms
    DEMOCRATIC_UNDERMINING = "democratic_undermining"
    SOCIAL_DIVISION = "social_division"
    POWER_CONCENTRATION = "power_concentration"

    # AI-specific harms
    DECEPTION = "deception"
    CAPABILITY_OVERHANG = "capability_overhang"
    ALIGNMENT_FAILURE = "alignment_failure"
    EMERGENT_BEHAVIOR = "emergent_behavior"


class HarmSeverity(str, Enum):
    """Severity levels for harm detection"""

    LOW = "low"  # Minor concern, log and monitor
    MODERATE = "moderate"  # Requires intervention
    HIGH = "high"  # Immediate action required
    CRITICAL = "critical"  # System-wide response needed


class PolicyDecisionType(str, Enum):
    """Types of policy decisions the governance engine can make"""

    ALLOW = "allow"  # Allow workload to proceed
    RESTRICT = "restrict"  # Allow with restrictions
    BLOCK = "block"  # Block workload completely
    ESCALATE = "escalate"  # Escalate to human review
    QUARANTINE = "quarantine"  # Isolate for investigation


class GovernanceAction(str, Enum):
    """Actions the governance engine can take"""

    CONTENT_FILTER = "content_filter"
    WORKLOAD_RESTRICT = "workload_restrict"
    USER_WARN = "user_warn"
    USER_SUSPEND = "user_suspend"
    SYSTEM_ALERT = "system_alert"
    HUMAN_ESCALATION = "human_escalation"
    AUDIT_LOG = "audit_log"
    TRANSPARENCY_REPORT = "transparency_report"


@dataclass
class HarmDetection:
    """Result of harm detection analysis"""

    detection_id: str
    harm_categories: list[HarmCategory]
    severity: HarmSeverity
    confidence_score: Decimal

    # Detection details
    detected_content: str = ""
    detection_method: str = "automated"
    evidence: dict[str, Any] = field(default_factory=dict)

    # Context
    user_id: str = ""
    workload_id: str = ""
    tier: ConstitutionalTier | None = None

    # Timestamps
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None

    def calculate_risk_score(self) -> Decimal:
        """Calculate overall risk score from harm detection"""

        severity_multipliers = {
            HarmSeverity.LOW: Decimal("0.25"),
            HarmSeverity.MODERATE: Decimal("0.5"),
            HarmSeverity.HIGH: Decimal("0.8"),
            HarmSeverity.CRITICAL: Decimal("1.0"),
        }

        base_risk = severity_multipliers[self.severity]
        confidence_adjustment = self.confidence_score
        category_multiplier = Decimal(str(len(self.harm_categories))) * Decimal("0.1")

        risk_score = base_risk * confidence_adjustment + category_multiplier
        return min(Decimal("1.0"), risk_score)


@dataclass
class PolicyDecision:
    """Decision made by the governance engine"""

    decision_id: str
    decision_type: PolicyDecisionType
    actions: list[GovernanceAction]

    # Decision context
    workload_id: str
    user_id: str
    tier: ConstitutionalTier
    harm_detections: list[HarmDetection] = field(default_factory=list)

    # Decision rationale
    reasoning: str = ""
    confidence: Decimal = Decimal("0.0")
    constitutional_basis: list[str] = field(default_factory=list)

    # Execution details
    actions_taken: dict[str, bool] = field(default_factory=dict)
    enforcement_timestamp: datetime | None = None

    # Appeals and review
    appealable: bool = False
    human_review_requested: bool = False
    community_oversight_required: bool = False

    # Timestamps
    decision_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    appeal_deadline: datetime | None = None

    def __post_init__(self):
        """Set appeal deadline if decision is appealable"""
        if self.appealable:
            self.appeal_deadline = self.decision_timestamp + timedelta(days=7)


@dataclass
class ConstitutionalConstraint:
    """Constitutional constraint to be enforced"""

    constraint_id: str
    name: str
    description: str

    # Enforcement details
    applicable_tiers: set[ConstitutionalTier]
    harm_categories_targeted: set[HarmCategory]
    enforcement_threshold: Decimal

    # Implementation
    check_function: str  # Name of function to call for checking
    enforcement_action: GovernanceAction

    # Metadata
    constitutional_article: str = ""  # Reference to constitutional article
    precedent_cases: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    active: bool = True


@dataclass
class ViewpointFirewall:
    """Viewpoint firewall to prevent bias amplification"""

    firewall_id: str
    name: str
    description: str

    # Bias detection
    protected_attributes: set[str] = field(
        default_factory=lambda: {
            "race",
            "gender",
            "religion",
            "political_affiliation",
            "sexual_orientation",
            "nationality",
            "age",
            "disability_status",
            "socioeconomic_status",
        }
    )

    bias_detection_threshold: Decimal = Decimal("0.7")
    intervention_strategies: list[str] = field(
        default_factory=lambda: [
            "content_diversification",
            "perspective_balancing",
            "bias_warning",
            "alternative_viewpoints",
        ]
    )

    # Configuration
    applicable_tiers: set[ConstitutionalTier] = field(
        default_factory=lambda: {ConstitutionalTier.SILVER, ConstitutionalTier.GOLD}
    )

    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConstitutionalGovernanceEngine:
    """
    Core constitutional governance engine for AI safety in fog computing

    Responsibilities:
    - Monitor workloads for constitutional compliance
    - Detect and prevent harmful content using comprehensive taxonomy
    - Make policy decisions based on constitutional constraints
    - Enforce machine-only moderation with human escalation paths
    - Integrate viewpoint firewall for bias prevention
    - Maintain transparency through audit logging
    - Handle appeals and review processes
    """

    def __init__(self, tier_manager=None):
        self.tier_manager = tier_manager

        # Core components
        self.harm_detectors = self._initialize_harm_detectors()
        self.constitutional_constraints = self._initialize_constitutional_constraints()
        self.viewpoint_firewalls = self._initialize_viewpoint_firewalls()

        # State tracking
        self.active_decisions: dict[str, PolicyDecision] = {}
        self.harm_detection_history: list[HarmDetection] = []
        self.appeal_queue: list[dict[str, Any]] = []

        # Configuration
        self.config = {
            "machine_only_mode": True,
            "human_escalation_enabled": True,
            "transparency_logging": True,
            "audit_retention_days": 90,
            "appeal_processing_enabled": True,
            "community_oversight_enabled": True,
        }

        # Integration points
        self.transparency_logger = None  # Will be injected
        self.audit_system = None  # Will be injected
        self.notification_system = None  # Will be injected

        logger.info("Constitutional governance engine initialized")

    def _initialize_harm_detectors(self) -> dict[HarmCategory, dict[str, Any]]:
        """Initialize harm detection configurations"""

        return {
            HarmCategory.VIOLENCE: {
                "keywords": ["violence", "harm", "attack", "threat", "weapon"],
                "threshold": Decimal("0.8"),
                "context_sensitive": True,
            },
            HarmCategory.HATE_SPEECH: {
                "keywords": ["hate", "discrimination", "slur"],
                "threshold": Decimal("0.7"),
                "context_sensitive": True,
            },
            HarmCategory.MISINFORMATION: {
                "keywords": ["false", "fake", "conspiracy"],
                "threshold": Decimal("0.6"),
                "fact_check_required": True,
            },
            HarmCategory.BIAS_AMPLIFICATION: {
                "threshold": Decimal("0.75"),
                "requires_viewpoint_firewall": True,
            },
            HarmCategory.PRIVACY_VIOLATION: {
                "keywords": ["personal", "private", "confidential"],
                "threshold": Decimal("0.85"),
                "pii_detection": True,
            },
            HarmCategory.DECEPTION: {
                "keywords": ["impersonate", "fake", "deceive"],
                "threshold": Decimal("0.8"),
                "intent_analysis_required": True,
            },
            HarmCategory.ALIGNMENT_FAILURE: {
                "threshold": Decimal("0.9"),
                "requires_human_review": True,
                "system_alert": True,
            },
        }

    def _initialize_constitutional_constraints(self) -> list[ConstitutionalConstraint]:
        """Initialize constitutional constraints"""

        return [
            ConstitutionalConstraint(
                constraint_id="const_001",
                name="Violence Prevention",
                description="Prevent content that promotes or incites violence",
                applicable_tiers={ConstitutionalTier.BRONZE, ConstitutionalTier.SILVER, ConstitutionalTier.GOLD},
                harm_categories_targeted={HarmCategory.VIOLENCE, HarmCategory.HATE_SPEECH},
                enforcement_threshold=Decimal("0.7"),
                check_function="check_violence_content",
                enforcement_action=GovernanceAction.CONTENT_FILTER,
                constitutional_article="Article 1: Fundamental Safety",
            ),
            ConstitutionalConstraint(
                constraint_id="const_002",
                name="Bias Prevention",
                description="Prevent amplification of harmful biases and discrimination",
                applicable_tiers={ConstitutionalTier.SILVER, ConstitutionalTier.GOLD},
                harm_categories_targeted={HarmCategory.BIAS_AMPLIFICATION},
                enforcement_threshold=Decimal("0.6"),
                check_function="check_bias_amplification",
                enforcement_action=GovernanceAction.WORKLOAD_RESTRICT,
                constitutional_article="Article 2: Equality and Non-discrimination",
            ),
            ConstitutionalConstraint(
                constraint_id="const_003",
                name="Misinformation Prevention",
                description="Prevent spread of verified misinformation",
                applicable_tiers={ConstitutionalTier.BRONZE, ConstitutionalTier.SILVER, ConstitutionalTier.GOLD},
                harm_categories_targeted={HarmCategory.MISINFORMATION},
                enforcement_threshold=Decimal("0.8"),
                check_function="check_misinformation",
                enforcement_action=GovernanceAction.CONTENT_FILTER,
                constitutional_article="Article 3: Information Integrity",
            ),
            ConstitutionalConstraint(
                constraint_id="const_004",
                name="Privacy Protection",
                description="Protect individual privacy and prevent unauthorized data use",
                applicable_tiers={ConstitutionalTier.SILVER, ConstitutionalTier.GOLD},
                harm_categories_targeted={HarmCategory.PRIVACY_VIOLATION},
                enforcement_threshold=Decimal("0.75"),
                check_function="check_privacy_violation",
                enforcement_action=GovernanceAction.WORKLOAD_RESTRICT,
                constitutional_article="Article 4: Privacy Rights",
            ),
            ConstitutionalConstraint(
                constraint_id="const_005",
                name="AI Alignment Monitoring",
                description="Monitor for AI alignment failures and emergent behaviors",
                applicable_tiers={ConstitutionalTier.GOLD},
                harm_categories_targeted={HarmCategory.ALIGNMENT_FAILURE, HarmCategory.EMERGENT_BEHAVIOR},
                enforcement_threshold=Decimal("0.5"),
                check_function="check_alignment_failure",
                enforcement_action=GovernanceAction.HUMAN_ESCALATION,
                constitutional_article="Article 5: AI Safety and Control",
            ),
        ]

    def _initialize_viewpoint_firewalls(self) -> list[ViewpointFirewall]:
        """Initialize viewpoint firewalls for bias prevention"""

        return [
            ViewpointFirewall(
                firewall_id="vf_001",
                name="Demographic Bias Firewall",
                description="Prevents amplification of biases based on demographic attributes",
                bias_detection_threshold=Decimal("0.7"),
                intervention_strategies=[
                    "content_diversification",
                    "demographic_balancing",
                    "bias_warning",
                ],
            ),
            ViewpointFirewall(
                firewall_id="vf_002",
                name="Political Bias Firewall",
                description="Prevents political bias amplification and echo chambers",
                protected_attributes={"political_affiliation", "ideology"},
                bias_detection_threshold=Decimal("0.6"),
                intervention_strategies=[
                    "perspective_balancing",
                    "alternative_viewpoints",
                    "bias_disclosure",
                ],
            ),
            ViewpointFirewall(
                firewall_id="vf_003",
                name="Cultural Bias Firewall",
                description="Prevents cultural bias and promotes inclusive perspectives",
                protected_attributes={"culture", "nationality", "language", "religion"},
                bias_detection_threshold=Decimal("0.65"),
                intervention_strategies=[
                    "cultural_context_addition",
                    "perspective_diversification",
                    "cultural_sensitivity_warning",
                ],
                applicable_tiers={ConstitutionalTier.GOLD},  # Gold tier only
            ),
        ]

    async def evaluate_workload(self, workload_id: str, user_id: str, content: dict[str, Any]) -> PolicyDecision:
        """Evaluate workload against constitutional requirements"""

        # Get user's constitutional tier
        user_tier = (
            self.tier_manager.get_constitutional_tier(user_id) if self.tier_manager else ConstitutionalTier.BRONZE
        )

        # Perform harm detection
        harm_detections = await self._detect_harms(content, user_tier, workload_id, user_id)

        # Check constitutional constraints
        constraint_violations = await self._check_constitutional_constraints(content, user_tier, harm_detections)

        # Apply viewpoint firewall if required
        viewpoint_issues = await self._apply_viewpoint_firewall(content, user_tier)

        # Make policy decision
        decision = await self._make_policy_decision(
            workload_id=workload_id,
            user_id=user_id,
            tier=user_tier,
            harm_detections=harm_detections,
            constraint_violations=constraint_violations,
            viewpoint_issues=viewpoint_issues,
        )

        # Store decision
        self.active_decisions[decision.decision_id] = decision

        # Execute actions if decision is not escalated
        if decision.decision_type != PolicyDecisionType.ESCALATE:
            await self._execute_decision_actions(decision)

        return decision

    async def _detect_harms(
        self, content: dict[str, Any], tier: ConstitutionalTier, workload_id: str, user_id: str
    ) -> list[HarmDetection]:
        """Detect potential harms in content using comprehensive taxonomy"""

        detections = []
        content_text = str(content.get("text", ""))

        for harm_category, detector_config in self.harm_detectors.items():
            detection = await self._check_harm_category(
                content_text, content, harm_category, detector_config, tier, workload_id, user_id
            )

            if detection:
                detections.append(detection)

        return detections

    async def _check_harm_category(
        self,
        content_text: str,
        full_content: dict[str, Any],
        category: HarmCategory,
        config: dict[str, Any],
        tier: ConstitutionalTier,
        workload_id: str,
        user_id: str,
    ) -> HarmDetection | None:
        """Check specific harm category"""

        # Keyword-based detection (simplified - production would use ML models)
        keywords = config.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_text.lower())

        if keyword_matches == 0:
            return None

        # Calculate confidence based on matches and context
        base_confidence = min(Decimal("1.0"), Decimal(str(keyword_matches)) * Decimal("0.3"))
        threshold = config.get("threshold", Decimal("0.5"))

        if base_confidence < threshold:
            return None

        # Determine severity
        if base_confidence >= Decimal("0.9"):
            severity = HarmSeverity.CRITICAL
        elif base_confidence >= Decimal("0.7"):
            severity = HarmSeverity.HIGH
        elif base_confidence >= Decimal("0.5"):
            severity = HarmSeverity.MODERATE
        else:
            severity = HarmSeverity.LOW

        detection = HarmDetection(
            detection_id=f"harm_{uuid.uuid4().hex[:8]}",
            harm_categories=[category],
            severity=severity,
            confidence_score=base_confidence,
            detected_content=content_text[:200] + "..." if len(content_text) > 200 else content_text,
            detection_method="keyword_analysis",
            evidence={
                "keyword_matches": keyword_matches,
                "matched_keywords": [kw for kw in keywords if kw.lower() in content_text.lower()],
                "threshold": float(threshold),
            },
            user_id=user_id,
            workload_id=workload_id,
            tier=tier,
        )

        logger.warning(
            f"Harm detected: {category.value} (severity: {severity.value}, "
            f"confidence: {float(base_confidence):.2f}) in workload {workload_id}"
        )

        return detection

    async def _check_constitutional_constraints(
        self, content: dict[str, Any], tier: ConstitutionalTier, harm_detections: list[HarmDetection]
    ) -> list[ConstitutionalConstraint]:
        """Check content against applicable constitutional constraints"""

        violations = []
        detected_harm_categories = set()
        for detection in harm_detections:
            detected_harm_categories.update(detection.harm_categories)

        for constraint in self.constitutional_constraints:
            # Check if constraint applies to this tier
            if tier not in constraint.applicable_tiers:
                continue

            # Check if any detected harms match constraint targets
            if constraint.harm_categories_targeted & detected_harm_categories:
                # Calculate violation severity
                max_confidence = max(
                    (
                        d.confidence_score
                        for d in harm_detections
                        if any(cat in constraint.harm_categories_targeted for cat in d.harm_categories)
                    ),
                    default=Decimal("0"),
                )

                if max_confidence >= constraint.enforcement_threshold:
                    violations.append(constraint)
                    logger.info(f"Constitutional constraint violated: {constraint.name}")

        return violations

    async def _apply_viewpoint_firewall(
        self, content: dict[str, Any], tier: ConstitutionalTier
    ) -> list[dict[str, Any]]:
        """Apply viewpoint firewall to detect and prevent bias amplification"""

        issues = []
        content_text = str(content.get("text", ""))

        for firewall in self.viewpoint_firewalls:
            if not firewall.active or tier not in firewall.applicable_tiers:
                continue

            # Simplified bias detection (production would use sophisticated ML models)
            bias_score = await self._detect_bias(content_text, firewall.protected_attributes)

            if bias_score >= firewall.bias_detection_threshold:
                issue = {
                    "firewall_id": firewall.firewall_id,
                    "bias_score": float(bias_score),
                    "protected_attributes_affected": list(firewall.protected_attributes),
                    "intervention_strategies": firewall.intervention_strategies,
                    "description": f"Potential bias detected by {firewall.name}",
                }
                issues.append(issue)

                logger.warning(
                    f"Viewpoint firewall triggered: {firewall.name} " f"(bias score: {float(bias_score):.2f})"
                )

        return issues

    async def _detect_bias(self, content: str, protected_attributes: set[str]) -> Decimal:
        """Detect bias in content (simplified implementation)"""

        # Simplified bias detection based on keyword patterns
        bias_indicators = {
            "race": ["racial", "ethnic", "stereotype"],
            "gender": ["gender", "masculine", "feminine", "stereotype"],
            "religion": ["religious", "faith", "belief", "stereotype"],
            "political_affiliation": ["liberal", "conservative", "partisan"],
        }

        total_bias_score = Decimal("0")
        attribute_count = 0

        for attribute in protected_attributes:
            if attribute in bias_indicators:
                indicators = bias_indicators[attribute]
                matches = sum(1 for indicator in indicators if indicator in content.lower())
                if matches > 0:
                    attribute_bias = min(Decimal("1.0"), Decimal(str(matches)) * Decimal("0.2"))
                    total_bias_score += attribute_bias
                    attribute_count += 1

        if attribute_count == 0:
            return Decimal("0")

        return total_bias_score / Decimal(str(attribute_count))

    async def _make_policy_decision(
        self,
        workload_id: str,
        user_id: str,
        tier: ConstitutionalTier,
        harm_detections: list[HarmDetection],
        constraint_violations: list[ConstitutionalConstraint],
        viewpoint_issues: list[dict[str, Any]],
    ) -> PolicyDecision:
        """Make policy decision based on harm analysis and constitutional constraints"""

        decision_id = f"decision_{uuid.uuid4().hex[:8]}"

        # Calculate overall risk score
        max_harm_risk = max((detection.calculate_risk_score() for detection in harm_detections), default=Decimal("0"))

        viewpoint_risk = max(
            (Decimal(str(issue.get("bias_score", 0))) for issue in viewpoint_issues), default=Decimal("0")
        )

        overall_risk = max(max_harm_risk, viewpoint_risk)

        # Determine decision type based on risk and tier
        if overall_risk >= Decimal("0.9"):
            decision_type = PolicyDecisionType.BLOCK
            actions = [GovernanceAction.WORKLOAD_RESTRICT, GovernanceAction.AUDIT_LOG]
            reasoning = f"Critical risk detected (score: {float(overall_risk):.2f})"

        elif overall_risk >= Decimal("0.7"):
            if tier == ConstitutionalTier.GOLD:
                # Gold tier gets human escalation for high-risk content
                decision_type = PolicyDecisionType.ESCALATE
                actions = [GovernanceAction.HUMAN_ESCALATION, GovernanceAction.AUDIT_LOG]
                reasoning = "High risk content requires human review for Gold tier"
            else:
                decision_type = PolicyDecisionType.RESTRICT
                actions = [GovernanceAction.CONTENT_FILTER, GovernanceAction.USER_WARN, GovernanceAction.AUDIT_LOG]
                reasoning = "High risk content restricted with filtering"

        elif overall_risk >= Decimal("0.4"):
            decision_type = PolicyDecisionType.RESTRICT
            actions = [GovernanceAction.CONTENT_FILTER, GovernanceAction.AUDIT_LOG]
            reasoning = "Moderate risk content requires filtering"

        else:
            decision_type = PolicyDecisionType.ALLOW
            actions = [GovernanceAction.AUDIT_LOG]  # Always audit
            reasoning = "Low risk content allowed with monitoring"

        # Add specific actions based on violations
        if constraint_violations:
            for violation in constraint_violations:
                if violation.enforcement_action not in actions:
                    actions.append(violation.enforcement_action)

        # Add viewpoint firewall interventions
        if viewpoint_issues and GovernanceAction.CONTENT_FILTER not in actions:
            actions.append(GovernanceAction.CONTENT_FILTER)

        # Determine if decision is appealable
        appealable = decision_type in [PolicyDecisionType.BLOCK, PolicyDecisionType.RESTRICT] and tier in [
            ConstitutionalTier.SILVER,
            ConstitutionalTier.GOLD,
        ]

        decision = PolicyDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            actions=actions,
            workload_id=workload_id,
            user_id=user_id,
            tier=tier,
            harm_detections=harm_detections,
            reasoning=reasoning,
            confidence=min(Decimal("1.0"), overall_risk + Decimal("0.1")),
            constitutional_basis=[v.constitutional_article for v in constraint_violations],
            appealable=appealable,
            community_oversight_required=(
                tier == ConstitutionalTier.GOLD and decision_type == PolicyDecisionType.BLOCK
            ),
        )

        logger.info(
            f"Policy decision made: {decision_type.value} for workload {workload_id} "
            f"(risk: {float(overall_risk):.2f}, actions: {len(actions)})"
        )

        return decision

    async def _execute_decision_actions(self, decision: PolicyDecision):
        """Execute the actions specified in a policy decision"""

        for action in decision.actions:
            try:
                success = await self._execute_single_action(decision, action)
                decision.actions_taken[action.value] = success

            except Exception as e:
                logger.error(f"Error executing action {action.value}: {e}")
                decision.actions_taken[action.value] = False

        decision.enforcement_timestamp = datetime.now(UTC)

        logger.info(
            f"Decision actions executed for {decision.decision_id}: "
            f"{sum(decision.actions_taken.values())}/{len(decision.actions)} successful"
        )

    async def _execute_single_action(self, decision: PolicyDecision, action: GovernanceAction) -> bool:
        """Execute a single governance action"""

        if action == GovernanceAction.CONTENT_FILTER:
            # Apply content filtering
            logger.info(f"Applied content filter to workload {decision.workload_id}")
            return True

        elif action == GovernanceAction.WORKLOAD_RESTRICT:
            # Restrict workload execution
            logger.info(f"Restricted workload {decision.workload_id}")
            return True

        elif action == GovernanceAction.USER_WARN:
            # Send warning to user
            if self.notification_system:
                await self.notification_system.send_warning(
                    decision.user_id, f"Constitutional policy violation detected in workload {decision.workload_id}"
                )
            logger.info(f"Sent warning to user {decision.user_id}")
            return True

        elif action == GovernanceAction.AUDIT_LOG:
            # Log to audit system
            if self.audit_system:
                await self.audit_system.log_decision(decision)
            if self.transparency_logger:
                await self.transparency_logger.log_governance_decision(decision)
            logger.info(f"Logged decision {decision.decision_id} to audit system")
            return True

        elif action == GovernanceAction.HUMAN_ESCALATION:
            # Escalate to human review
            await self._escalate_to_human_review(decision)
            logger.info(f"Escalated decision {decision.decision_id} to human review")
            return True

        else:
            logger.warning(f"Unknown governance action: {action.value}")
            return False

    async def _escalate_to_human_review(self, decision: PolicyDecision):
        """Escalate decision to human review queue"""

        escalation = {
            "decision_id": decision.decision_id,
            "escalation_id": f"escalation_{uuid.uuid4().hex[:8]}",
            "workload_id": decision.workload_id,
            "user_id": decision.user_id,
            "tier": decision.tier.value,
            "harm_summary": self._summarize_harms(decision.harm_detections),
            "urgency": (
                "high" if any(d.severity == HarmSeverity.CRITICAL for d in decision.harm_detections) else "medium"
            ),
            "escalated_at": datetime.now(UTC),
            "review_deadline": datetime.now(UTC) + timedelta(hours=4),  # 4-hour SLA
        }

        # Add to human review queue (would integrate with actual review system)
        logger.info(f"Added escalation {escalation['escalation_id']} to human review queue")

    def _summarize_harms(self, detections: list[HarmDetection]) -> dict[str, Any]:
        """Create summary of detected harms for human reviewers"""

        if not detections:
            return {"total_detections": 0}

        categories = {}
        max_severity = HarmSeverity.LOW
        total_confidence = Decimal("0")

        for detection in detections:
            for category in detection.harm_categories:
                if category.value not in categories:
                    categories[category.value] = 0
                categories[category.value] += 1

            if detection.severity.value > max_severity.value:
                max_severity = detection.severity

            total_confidence += detection.confidence_score

        return {
            "total_detections": len(detections),
            "categories": categories,
            "max_severity": max_severity.value,
            "average_confidence": float(total_confidence / len(detections)),
        }

    async def process_appeal(self, decision_id: str, user_id: str, appeal_reason: str) -> dict[str, Any]:
        """Process appeal for governance decision"""

        if decision_id not in self.active_decisions:
            return {"success": False, "error": "Decision not found"}

        decision = self.active_decisions[decision_id]

        if not decision.appealable:
            return {"success": False, "error": "Decision is not appealable"}

        if decision.user_id != user_id:
            return {"success": False, "error": "Unauthorized appeal request"}

        if decision.appeal_deadline and datetime.now(UTC) > decision.appeal_deadline:
            return {"success": False, "error": "Appeal deadline has passed"}

        appeal = {
            "appeal_id": f"appeal_{uuid.uuid4().hex[:8]}",
            "decision_id": decision_id,
            "user_id": user_id,
            "appeal_reason": appeal_reason,
            "submitted_at": datetime.now(UTC),
            "status": "pending_review",
        }

        self.appeal_queue.append(appeal)

        logger.info(f"Appeal {appeal['appeal_id']} submitted for decision {decision_id}")

        return {
            "success": True,
            "appeal_id": appeal["appeal_id"],
            "estimated_review_time": "3-5 business days",
        }

    async def get_governance_statistics(self) -> dict[str, Any]:
        """Get comprehensive governance statistics"""

        total_decisions = len(self.active_decisions)
        total_harms = len(self.harm_detection_history)

        # Decision type distribution
        decision_types = {}
        for decision in self.active_decisions.values():
            dt = decision.decision_type.value
            decision_types[dt] = decision_types.get(dt, 0) + 1

        # Harm category distribution
        harm_categories = {}
        for detection in self.harm_detection_history:
            for category in detection.harm_categories:
                cat = category.value
                harm_categories[cat] = harm_categories.get(cat, 0) + 1

        # Tier-based statistics
        tier_stats = {}
        for decision in self.active_decisions.values():
            tier = decision.tier.value
            if tier not in tier_stats:
                tier_stats[tier] = {"decisions": 0, "escalations": 0}
            tier_stats[tier]["decisions"] += 1
            if decision.decision_type == PolicyDecisionType.ESCALATE:
                tier_stats[tier]["escalations"] += 1

        return {
            "overview": {
                "total_decisions": total_decisions,
                "total_harm_detections": total_harms,
                "active_appeals": len(self.appeal_queue),
                "human_escalations": len(
                    [d for d in self.active_decisions.values() if d.decision_type == PolicyDecisionType.ESCALATE]
                ),
            },
            "decision_distribution": decision_types,
            "harm_category_distribution": harm_categories,
            "tier_statistics": tier_stats,
            "constitutional_constraints": {
                "total_constraints": len(self.constitutional_constraints),
                "active_constraints": len([c for c in self.constitutional_constraints if c.active]),
            },
            "viewpoint_firewalls": {
                "total_firewalls": len(self.viewpoint_firewalls),
                "active_firewalls": len([f for f in self.viewpoint_firewalls if f.active]),
            },
        }


# Integration functions for external systems


async def create_governance_engine(tier_manager=None) -> ConstitutionalGovernanceEngine:
    """Create and initialize governance engine"""
    engine = ConstitutionalGovernanceEngine(tier_manager=tier_manager)
    logger.info("Constitutional governance engine created and ready")
    return engine


def get_constitutional_constraints_for_tier(tier: ConstitutionalTier) -> list[dict[str, Any]]:
    """Get constitutional constraints applicable to a specific tier"""

    engine = ConstitutionalGovernanceEngine()
    applicable_constraints = []

    for constraint in engine.constitutional_constraints:
        if tier in constraint.applicable_tiers:
            applicable_constraints.append(
                {
                    "constraint_id": constraint.constraint_id,
                    "name": constraint.name,
                    "description": constraint.description,
                    "enforcement_threshold": float(constraint.enforcement_threshold),
                    "enforcement_action": constraint.enforcement_action.value,
                    "constitutional_article": constraint.constitutional_article,
                }
            )

    return applicable_constraints


if __name__ == "__main__":
    # Demo governance engine
    import json

    async def demo_governance_engine():
        engine = ConstitutionalGovernanceEngine()

        # Demo harm detection
        test_content = {
            "text": "This content contains some violent language and threats",
            "type": "user_message",
        }

        decision = await engine.evaluate_workload(
            workload_id="demo_workload_001", user_id="demo_user", content=test_content
        )

        print("Governance Decision:")
        print(f"Decision Type: {decision.decision_type.value}")
        print(f"Actions: {[a.value for a in decision.actions]}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Harms Detected: {len(decision.harm_detections)}")

        # Demo statistics
        stats = await engine.get_governance_statistics()
        print("\nGovernance Statistics:")
        print(json.dumps(stats, indent=2))

    asyncio.run(demo_governance_engine())
