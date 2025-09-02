"""
Constitutional Policy Enforcement System
Implements tier-based constitutional policy enforcement with automated decision making
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConstitutionalPrinciple(Enum):
    """Core constitutional principles for content moderation"""

    FIRST_AMENDMENT = "first_amendment"
    VIEWPOINT_NEUTRALITY = "viewpoint_neutrality"
    DUE_PROCESS = "due_process"
    EQUAL_PROTECTION = "equal_protection"
    PRIOR_RESTRAINT = "prior_restraint"


class PolicyDecision(Enum):
    """Policy enforcement decisions"""

    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    RESTRICT = "restrict"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass
class EnforcementResult:
    """Result of policy enforcement evaluation"""

    decision: PolicyDecision
    rationale: str
    constitutional_analysis: dict[str, Any]
    tier_modifications: dict[str, Any]
    monitoring_requirements: list[str]
    confidence_score: float


class PolicyEnforcement:
    """
    Constitutional policy enforcement with tier-based responses
    Implements H0-H3 harm level responses while maintaining constitutional principles
    """

    def __init__(self):
        self.policy_version = "1.0.0"
        self.constitutional_guidelines = self._load_constitutional_guidelines()
        self.tier_policies = self._load_tier_policies()

        logger.info("Constitutional Policy Enforcement initialized")

    def _load_constitutional_guidelines(self) -> dict[str, Any]:
        """Load constitutional guidelines for content moderation"""
        return {
            ConstitutionalPrinciple.FIRST_AMENDMENT: {
                "protected_categories": [
                    "political_speech",
                    "religious_expression",
                    "artistic_expression",
                    "scientific_discourse",
                    "social_commentary",
                    "news_reporting",
                ],
                "unprotected_categories": [
                    "true_threats",
                    "incitement_to_violence",
                    "defamation",
                    "fraud",
                    "child_exploitation",
                    "copyright_infringement",
                ],
                "heightened_scrutiny": True,
                "burden_of_proof": "clear_and_present_danger",
            },
            ConstitutionalPrinciple.VIEWPOINT_NEUTRALITY: {
                "prohibited_discrimination": ["political_ideology", "religious_beliefs", "social_viewpoints"],
                "content_neutral_enforcement": True,
                "bias_detection_required": True,
                "equal_treatment_mandate": True,
            },
            ConstitutionalPrinciple.DUE_PROCESS: {
                "notice_requirements": True,
                "appeal_rights": True,
                "reasoned_decision": True,
                "evidence_based": True,
                "proportional_response": True,
            },
            ConstitutionalPrinciple.EQUAL_PROTECTION: {
                "no_class_discrimination": True,
                "equal_enforcement": True,
                "protected_class_awareness": True,
            },
            ConstitutionalPrinciple.PRIOR_RESTRAINT: {
                "presumption_against": True,
                "narrow_tailoring": True,
                "least_restrictive_means": True,
                "compelling_interest_required": True,
            },
        }

    def _load_tier_policies(self) -> dict[str, dict[str, Any]]:
        """Load tier-specific policy configurations"""
        return {
            "Bronze": {
                "automated_only": True,
                "human_escalation": False,
                "appeal_rights": "limited",
                "transparency_level": "basic",
                "constitutional_protection": "standard",
                "response_speed": "immediate",
                "monitoring_level": "automated",
            },
            "Silver": {
                "automated_primary": True,
                "human_escalation": "limited",
                "appeal_rights": "standard",
                "transparency_level": "enhanced",
                "constitutional_protection": "enhanced",
                "response_speed": "fast",
                "monitoring_level": "enhanced",
                "viewpoint_firewall": True,
            },
            "Gold": {
                "automated_primary": True,
                "human_escalation": "full",
                "appeal_rights": "comprehensive",
                "transparency_level": "full",
                "constitutional_protection": "maximum",
                "response_speed": "careful",
                "monitoring_level": "comprehensive",
                "constitutional_review": True,
                "community_oversight": True,
            },
        }

    async def evaluate_content(
        self, harm_analysis: Any, user_tier: str, context: dict[str, Any] = None
    ) -> EnforcementResult:
        """
        Evaluate content and determine policy enforcement action

        Args:
            harm_analysis: ContentAnalysis from moderation pipeline
            user_tier: User tier (Bronze, Silver, Gold)
            context: Additional context for decision making

        Returns:
            EnforcementResult with decision and rationale
        """
        try:
            logger.info(f"Evaluating content {harm_analysis.content_id} for {user_tier} tier")

            # Step 1: Constitutional analysis
            constitutional_analysis = await self._analyze_constitutional_implications(
                harm_analysis, user_tier, context or {}
            )

            # Step 2: Apply harm level policies
            harm_decision = await self._apply_harm_level_policy(harm_analysis, constitutional_analysis, user_tier)

            # Step 3: Apply tier-specific modifications
            tier_modifications = await self._apply_tier_modifications(harm_decision, constitutional_analysis, user_tier)

            # Step 4: Determine final decision
            final_decision = await self._determine_final_decision(
                harm_decision, tier_modifications, constitutional_analysis
            )

            # Step 5: Generate rationale
            rationale = await self._generate_policy_rationale(
                final_decision, harm_analysis, constitutional_analysis, user_tier
            )

            # Step 6: Determine monitoring requirements
            monitoring = await self._determine_monitoring_requirements(final_decision, harm_analysis, user_tier)

            # Step 7: Calculate confidence score
            confidence_score = await self._calculate_enforcement_confidence(
                harm_analysis, constitutional_analysis, final_decision
            )

            result = EnforcementResult(
                decision=final_decision,
                rationale=rationale,
                constitutional_analysis=constitutional_analysis,
                tier_modifications=tier_modifications,
                monitoring_requirements=monitoring,
                confidence_score=confidence_score,
            )

            logger.info(f"Content {harm_analysis.content_id} decision: {final_decision.value}")
            return result

        except Exception as e:
            logger.error(f"Policy enforcement failed for {harm_analysis.content_id}: {str(e)}")
            return await self._create_safe_enforcement_result(harm_analysis, user_tier, str(e))

    async def _analyze_constitutional_implications(
        self, harm_analysis: Any, user_tier: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze constitutional implications of content and potential restrictions"""

        analysis = {
            "protected_speech_detected": False,
            "viewpoint_discrimination_risk": False,
            "due_process_requirements": [],
            "prior_restraint_concerns": False,
            "heightened_scrutiny_required": False,
            "constitutional_balancing": {},
        }

        # First Amendment analysis
        first_amendment = await self._analyze_first_amendment_implications(harm_analysis)
        analysis.update(first_amendment)

        # Viewpoint neutrality analysis
        viewpoint_analysis = await self._analyze_viewpoint_neutrality(harm_analysis)
        analysis.update(viewpoint_analysis)

        # Due process analysis
        due_process = await self._analyze_due_process_requirements(harm_analysis, user_tier)
        analysis.update(due_process)

        # Prior restraint analysis
        prior_restraint = await self._analyze_prior_restraint_implications(harm_analysis)
        analysis.update(prior_restraint)

        return analysis

    async def _analyze_first_amendment_implications(self, harm_analysis: Any) -> dict[str, Any]:
        """Analyze First Amendment implications"""
        guidelines = self.constitutional_guidelines[ConstitutionalPrinciple.FIRST_AMENDMENT]

        # Check for protected speech categories
        protected_detected = any(
            category in guidelines["protected_categories"] for category in harm_analysis.harm_categories
        )

        # Check for unprotected speech categories
        unprotected_detected = any(
            category in guidelines["unprotected_categories"] for category in harm_analysis.harm_categories
        )

        # Determine if heightened scrutiny is required
        heightened_scrutiny = protected_detected and not unprotected_detected and harm_analysis.confidence_score < 0.9

        return {
            "first_amendment": {
                "protected_speech_detected": protected_detected,
                "unprotected_speech_detected": unprotected_detected,
                "heightened_scrutiny_required": heightened_scrutiny,
                "burden_of_proof": guidelines["burden_of_proof"] if protected_detected else "preponderance",
                "constitutional_balancing_required": protected_detected,
            }
        }

    async def _analyze_viewpoint_neutrality(self, harm_analysis: Any) -> dict[str, Any]:
        """Analyze viewpoint neutrality requirements"""

        # Check viewpoint bias score
        high_bias_risk = harm_analysis.viewpoint_bias_score > 0.3

        # Check for ideological content
        ideological_content = any(
            "political" in category.lower() or "ideolog" in category.lower()
            for category in harm_analysis.harm_categories
        )

        return {
            "viewpoint_neutrality": {
                "discrimination_risk": high_bias_risk,
                "ideological_content_detected": ideological_content,
                "bias_score": harm_analysis.viewpoint_bias_score,
                "content_neutral_enforcement_required": ideological_content,
                "equal_treatment_verification_needed": high_bias_risk,
            }
        }

    async def _analyze_due_process_requirements(self, harm_analysis: Any, user_tier: str) -> dict[str, Any]:
        """Analyze due process requirements based on content and tier"""

        requirements = []

        # Notice requirements
        if harm_analysis.harm_level in ["H2", "H3"]:
            requirements.append("clear_notice_of_violation")
            requirements.append("explanation_of_policy")

        # Appeal rights
        tier_policy = self.tier_policies[user_tier]
        if tier_policy["appeal_rights"] != "none":
            requirements.append("appeal_process_notification")

        # Evidence requirements
        if harm_analysis.confidence_score < 0.7:
            requirements.append("enhanced_evidence_review")

        # Proportionality requirements
        if harm_analysis.harm_level == "H1" and user_tier == "Gold":
            requirements.append("proportionality_analysis")

        return {
            "due_process": {
                "requirements": requirements,
                "notice_level": self._determine_notice_level(harm_analysis, user_tier),
                "appeal_eligible": len(requirements) > 0,
                "evidence_standard": self._determine_evidence_standard(harm_analysis),
                "proportionality_required": "proportionality_analysis" in requirements,
            }
        }

    async def _analyze_prior_restraint_implications(self, harm_analysis: Any) -> dict[str, Any]:
        """Analyze prior restraint constitutional implications"""

        # Prior restraint is presumptively unconstitutional
        restraint_action = harm_analysis.harm_level in ["H2", "H3"]

        # Check for compelling government interest
        compelling_interest = any(
            category in ["violence_incitement", "child_safety", "national_security"]
            for category in harm_analysis.harm_categories
        )

        # Check for narrow tailoring
        narrowly_tailored = harm_analysis.confidence_score > 0.8 and len(harm_analysis.harm_categories) <= 2

        return {
            "prior_restraint": {
                "restraint_action_proposed": restraint_action,
                "compelling_interest_present": compelling_interest,
                "narrowly_tailored": narrowly_tailored,
                "least_restrictive_means_analysis_required": restraint_action,
                "constitutional_justification_needed": restraint_action and not compelling_interest,
            }
        }

    async def _apply_harm_level_policy(
        self, harm_analysis: Any, constitutional_analysis: dict[str, Any], user_tier: str
    ) -> PolicyDecision:
        """Apply harm level specific policies"""

        harm_level = harm_analysis.harm_level
        confidence = harm_analysis.confidence_score

        # H0 - Constitutional: Full protection
        if harm_level == "H0":
            return PolicyDecision.ALLOW

        # H1 - Minor Concerns: Allow with constitutional safeguards
        elif harm_level == "H1":
            if constitutional_analysis.get("first_amendment", {}).get("protected_speech_detected"):
                return PolicyDecision.ALLOW  # Err on side of free speech
            else:
                return PolicyDecision.ALLOW_WITH_WARNING

        # H2 - Moderate Harm: Careful constitutional analysis
        elif harm_level == "H2":
            if constitutional_analysis.get("first_amendment", {}).get("heightened_scrutiny_required"):
                if confidence < 0.8:
                    return PolicyDecision.ALLOW_WITH_WARNING  # Doubt favors free speech
                else:
                    return PolicyDecision.RESTRICT
            else:
                return PolicyDecision.RESTRICT

        # H3 - Severe Harm: Strong action with constitutional protections
        elif harm_level == "H3":
            if constitutional_analysis.get("first_amendment", {}).get("protected_speech_detected"):
                if confidence < 0.9:
                    return PolicyDecision.ESCALATE  # Require human review
                else:
                    return PolicyDecision.QUARANTINE  # Preserve evidence
            else:
                return PolicyDecision.BLOCK

        # Default: Conservative approach
        return PolicyDecision.ESCALATE

    async def _apply_tier_modifications(
        self, initial_decision: PolicyDecision, constitutional_analysis: dict[str, Any], user_tier: str
    ) -> dict[str, Any]:
        """Apply tier-specific modifications to policy decision"""

        self.tier_policies[user_tier]
        modifications = {
            "original_decision": initial_decision.value,
            "tier_adjustments": [],
            "additional_protections": [],
            "monitoring_enhancements": [],
        }

        # Bronze tier: Standard automated processing
        if user_tier == "Bronze":
            modifications["tier_adjustments"].append("automated_only_processing")

        # Silver tier: Enhanced protections and viewpoint firewall
        elif user_tier == "Silver":
            if constitutional_analysis.get("viewpoint_neutrality", {}).get("discrimination_risk"):
                modifications["additional_protections"].append("viewpoint_firewall_active")
                modifications["monitoring_enhancements"].append("bias_monitoring")

            if initial_decision == PolicyDecision.BLOCK:
                # Silver tier gets quarantine instead of outright block
                modifications["tier_adjustments"].append("block_to_quarantine_conversion")

        # Gold tier: Maximum constitutional protections
        elif user_tier == "Gold":
            if constitutional_analysis.get("first_amendment", {}).get("protected_speech_detected"):
                modifications["additional_protections"].append("constitutional_review_required")
                modifications["monitoring_enhancements"].append("comprehensive_constitutional_monitoring")

            if initial_decision in [PolicyDecision.RESTRICT, PolicyDecision.QUARANTINE, PolicyDecision.BLOCK]:
                modifications["additional_protections"].append("community_oversight_notification")

            # Gold tier gets human escalation for uncertain cases
            if constitutional_analysis.get("due_process", {}).get("proportionality_required"):
                modifications["tier_adjustments"].append("human_escalation_required")

        return modifications

    async def _determine_final_decision(
        self,
        initial_decision: PolicyDecision,
        tier_modifications: dict[str, Any],
        constitutional_analysis: dict[str, Any],
    ) -> PolicyDecision:
        """Determine final policy decision after tier modifications"""

        final_decision = initial_decision

        # Apply tier adjustments
        for adjustment in tier_modifications.get("tier_adjustments", []):
            if adjustment == "block_to_quarantine_conversion":
                if final_decision == PolicyDecision.BLOCK:
                    final_decision = PolicyDecision.QUARANTINE

            elif adjustment == "human_escalation_required":
                final_decision = PolicyDecision.ESCALATE

        # Constitutional override protections
        if constitutional_analysis.get("prior_restraint", {}).get("constitutional_justification_needed"):
            if final_decision in [PolicyDecision.BLOCK, PolicyDecision.QUARANTINE]:
                final_decision = PolicyDecision.ESCALATE  # Require human constitutional analysis

        return final_decision

    async def _generate_policy_rationale(
        self, decision: PolicyDecision, harm_analysis: Any, constitutional_analysis: dict[str, Any], user_tier: str
    ) -> str:
        """Generate comprehensive policy rationale"""

        rationale_parts = []

        # Harm level rationale
        rationale_parts.append(
            f"Content classified as {harm_analysis.harm_level} " f"with {harm_analysis.confidence_score:.2f} confidence"
        )

        # Constitutional considerations
        if constitutional_analysis.get("first_amendment", {}).get("protected_speech_detected"):
            rationale_parts.append(
                "First Amendment protected speech detected - heightened constitutional scrutiny applied"
            )

        if constitutional_analysis.get("viewpoint_neutrality", {}).get("discrimination_risk"):
            rationale_parts.append(
                f"Viewpoint neutrality concern (bias score: {harm_analysis.viewpoint_bias_score:.2f})"
            )

        # Tier-specific rationale
        tier_policy = self.tier_policies[user_tier]
        rationale_parts.append(
            f"{user_tier} tier policy applied with {tier_policy['constitutional_protection']} constitutional protection"
        )

        # Decision-specific rationale
        if decision == PolicyDecision.ALLOW:
            rationale_parts.append("Content permitted under constitutional free speech protections")
        elif decision == PolicyDecision.ALLOW_WITH_WARNING:
            rationale_parts.append("Content permitted with warning - balances free speech with community standards")
        elif decision == PolicyDecision.RESTRICT:
            rationale_parts.append("Content restricted due to harm potential while preserving constitutional rights")
        elif decision == PolicyDecision.QUARANTINE:
            rationale_parts.append(
                "Content quarantined pending review - preserves evidence and constitutional protections"
            )
        elif decision == PolicyDecision.BLOCK:
            rationale_parts.append("Content blocked due to clear harm with no constitutional protection")
        elif decision == PolicyDecision.ESCALATE:
            rationale_parts.append("Human review required for constitutional compliance and due process")

        return " | ".join(rationale_parts)

    async def _determine_monitoring_requirements(
        self, decision: PolicyDecision, harm_analysis: Any, user_tier: str
    ) -> list[str]:
        """Determine ongoing monitoring requirements"""

        monitoring = []

        # Decision-based monitoring
        if decision == PolicyDecision.ALLOW_WITH_WARNING:
            monitoring.append("warning_compliance_tracking")

        if decision == PolicyDecision.RESTRICT:
            monitoring.append("restriction_effectiveness_monitoring")

        if decision == PolicyDecision.QUARANTINE:
            monitoring.extend(["quarantine_review_scheduling", "evidence_preservation"])

        # Constitutional monitoring
        if harm_analysis.viewpoint_bias_score > 0.2:
            monitoring.append("viewpoint_bias_monitoring")

        if any("political" in cat.lower() for cat in harm_analysis.harm_categories):
            monitoring.append("political_neutrality_monitoring")

        # Tier-specific monitoring
        tier_policy = self.tier_policies[user_tier]
        if tier_policy.get("viewpoint_firewall"):
            monitoring.append("viewpoint_firewall_monitoring")

        if tier_policy.get("constitutional_review"):
            monitoring.append("constitutional_compliance_monitoring")

        return monitoring

    async def _calculate_enforcement_confidence(
        self, harm_analysis: Any, constitutional_analysis: dict[str, Any], decision: PolicyDecision
    ) -> float:
        """Calculate confidence score for enforcement decision"""

        confidence_factors = []

        # Base confidence from harm analysis
        confidence_factors.append(harm_analysis.confidence_score)

        # Constitutional clarity factor
        if constitutional_analysis.get("first_amendment", {}).get("unprotected_speech_detected"):
            confidence_factors.append(0.9)  # Clear unprotected speech
        elif constitutional_analysis.get("first_amendment", {}).get("protected_speech_detected"):
            confidence_factors.append(0.6)  # Protected speech requires care
        else:
            confidence_factors.append(0.8)  # Neutral content

        # Decision consistency factor
        if decision == PolicyDecision.ESCALATE:
            confidence_factors.append(0.5)  # Low confidence requires human review
        else:
            confidence_factors.append(0.8)  # Automated decision confidence

        # Viewpoint neutrality factor
        bias_penalty = harm_analysis.viewpoint_bias_score * 0.3
        confidence_factors.append(max(0.5, 1.0 - bias_penalty))

        return sum(confidence_factors) / len(confidence_factors)

    def _determine_notice_level(self, harm_analysis: Any, user_tier: str) -> str:
        """Determine required notice level for due process"""

        if harm_analysis.harm_level == "H3":
            return "detailed"
        elif harm_analysis.harm_level == "H2":
            return "standard" if user_tier != "Gold" else "detailed"
        elif harm_analysis.harm_level == "H1":
            return "basic" if user_tier == "Bronze" else "standard"
        else:
            return "none"

    def _determine_evidence_standard(self, harm_analysis: Any) -> str:
        """Determine evidence standard required"""

        if harm_analysis.confidence_score > 0.9:
            return "clear_and_convincing"
        elif harm_analysis.confidence_score > 0.7:
            return "preponderance_of_evidence"
        else:
            return "reasonable_suspicion"

    async def _create_safe_enforcement_result(
        self, harm_analysis: Any, user_tier: str, error_message: str
    ) -> EnforcementResult:
        """Create safe enforcement result for error cases"""

        return EnforcementResult(
            decision=PolicyDecision.ESCALATE,
            rationale=f"Policy enforcement error requires human review: {error_message}",
            constitutional_analysis={"error": True, "requires_human_review": True},
            tier_modifications={"error_handling": True},
            monitoring_requirements=["error_investigation"],
            confidence_score=0.0,
        )

    def get_policy_version(self) -> str:
        """Get current policy version"""
        return self.policy_version

    def get_constitutional_guidelines(self) -> dict[str, Any]:
        """Get current constitutional guidelines"""
        return self.constitutional_guidelines.copy()

    def get_tier_policies(self) -> dict[str, dict[str, Any]]:
        """Get current tier policies"""
        return self.tier_policies.copy()
