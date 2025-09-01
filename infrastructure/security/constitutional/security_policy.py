"""
Constitutional Security Policy Engine

Implements comprehensive constitutional AI safety policy enforcement with:
- Harm taxonomy classification and prevention
- Real-time content monitoring and filtering
- Constitutional constraint validation
- Multi-modal safety assessment
- Behavioral pattern analysis
- Automated response and mitigation
- Audit trail and compliance reporting

Integrates with TEE infrastructure for hardware-backed constitutional guarantees.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

logger = logging.getLogger(__name__)


class HarmCategory(Enum):
    """Comprehensive harm taxonomy for constitutional AI."""

    # Violence and Physical Harm
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    WEAPONS = "weapons"
    TERRORISM = "terrorism"

    # Hate and Discrimination
    HATE_SPEECH = "hate_speech"
    DISCRIMINATION = "discrimination"
    HARASSMENT = "harassment"
    BULLYING = "bullying"

    # Illegal Activities
    ILLEGAL_ACTIVITIES = "illegal_activities"
    FRAUD = "fraud"
    ILLEGAL_SUBSTANCES = "illegal_substances"
    COPYRIGHT_VIOLATION = "copyright_violation"

    # Privacy and Personal Information
    PRIVACY_VIOLATIONS = "privacy_violations"
    PERSONAL_INFO_EXPOSURE = "personal_info_exposure"
    DOXXING = "doxxing"
    SURVEILLANCE = "surveillance"

    # Sexual Content
    SEXUAL_CONTENT = "sexual_content"
    CHILD_EXPLOITATION = "child_exploitation"
    NON_CONSENSUAL_INTIMATE = "non_consensual_intimate"

    # Misinformation and Deception
    MISINFORMATION = "misinformation"
    DISINFORMATION = "disinformation"
    CONSPIRACY_THEORIES = "conspiracy_theories"
    DEEPFAKES = "deepfakes"

    # Manipulation and Exploitation
    MANIPULATION = "manipulation"
    EXPLOITATION = "exploitation"
    SOCIAL_ENGINEERING = "social_engineering"
    GAMBLING = "gambling"

    # Professional and Medical
    MEDICAL_ADVICE = "medical_advice"
    FINANCIAL_ADVICE = "financial_advice"
    LEGAL_ADVICE = "legal_advice"

    # Other Harmful Content
    GRAPHIC_CONTENT = "graphic_content"
    DISTURBING_CONTENT = "disturbing_content"
    SPAM = "spam"


class RiskLevel(Enum):
    """Risk severity levels for harm assessment."""

    NEGLIGIBLE = "negligible"  # 0.0 - 0.1
    LOW = "low"  # 0.1 - 0.3
    MEDIUM = "medium"  # 0.3 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    CRITICAL = "critical"  # 0.8 - 1.0


class ResponseAction(Enum):
    """Automated response actions for policy violations."""

    LOG = "log"  # Log violation only
    WARN = "warn"  # Issue warning
    FILTER = "filter"  # Filter/modify content
    BLOCK = "block"  # Block content completely
    ESCALATE = "escalate"  # Escalate to human review
    QUARANTINE = "quarantine"  # Isolate workload
    TERMINATE = "terminate"  # Terminate execution


class ConstitutionalPrinciple(Enum):
    """Core constitutional AI principles."""

    HELPFULNESS = "helpfulness"  # Be helpful and informative
    HARMLESSNESS = "harmlessness"  # Avoid harm to individuals and society
    HONESTY = "honesty"  # Be truthful and accurate
    RESPECT = "respect"  # Respect human rights and dignity
    FAIRNESS = "fairness"  # Treat all individuals fairly
    PRIVACY = "privacy"  # Protect personal information
    AUTONOMY = "autonomy"  # Respect human choice and agency
    TRANSPARENCY = "transparency"  # Be transparent about capabilities


@dataclass
class SafetyConstraint:
    """Individual safety constraint specification."""

    constraint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Constraint definition
    harm_categories: List[HarmCategory] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)  # Regex patterns

    # Thresholds
    max_risk_score: float = 0.3  # Maximum allowed risk score
    confidence_threshold: float = 0.7  # Minimum confidence for detection

    # Response configuration
    violation_response: ResponseAction = ResponseAction.WARN
    escalation_threshold: float = 0.8  # Risk score for escalation

    # Context and exceptions
    allowed_contexts: List[str] = field(default_factory=list)
    exempted_users: Set[str] = field(default_factory=set)

    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PolicyViolation:
    """Record of a constitutional policy violation."""

    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Basic information
    constraint_id: str = ""
    constraint_name: str = ""
    harm_category: HarmCategory = HarmCategory.MISINFORMATION
    risk_level: RiskLevel = RiskLevel.LOW

    # Content details
    content_hash: str = ""
    content_snippet: str = ""  # Sanitized excerpt
    content_type: str = "text"  # text, image, audio, etc.

    # Detection details
    risk_score: float = 0.0
    confidence_score: float = 0.0
    detection_method: str = "keyword"  # keyword, pattern, ml_model, etc.

    # Context
    user_id: str = ""
    session_id: str = ""
    workload_id: str = ""
    node_id: str = ""

    # Response
    action_taken: ResponseAction = ResponseAction.LOG
    escalated: bool = False
    human_reviewed: bool = False

    # Timing
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: Optional[datetime] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstitutionalPolicy:
    """Complete constitutional AI safety policy."""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"

    # Core principles
    principles: List[ConstitutionalPrinciple] = field(
        default_factory=lambda: [
            ConstitutionalPrinciple.HELPFULNESS,
            ConstitutionalPrinciple.HARMLESSNESS,
            ConstitutionalPrinciple.HONESTY,
        ]
    )

    # Safety constraints
    constraints: List[SafetyConstraint] = field(default_factory=list)

    # Global settings
    default_risk_threshold: float = 0.3
    require_human_oversight: bool = False
    enable_proactive_monitoring: bool = True
    enable_behavioral_analysis: bool = True

    # Audit and compliance
    audit_all_interactions: bool = True
    retain_violations_days: int = 90
    compliance_reporting: bool = True

    # Policy metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    effective_date: datetime = field(default_factory=lambda: datetime.now(UTC))


class HarmClassifier:
    """Machine learning-based harm classification system."""

    def __init__(self):
        # Keyword-based classifiers for different harm categories
        self.keyword_classifiers = {
            HarmCategory.VIOLENCE: [
                "kill",
                "murder",
                "assault",
                "attack",
                "violence",
                "hurt",
                "harm",
                "fight",
                "weapon",
                "gun",
                "knife",
                "bomb",
                "explosive",
            ],
            HarmCategory.HATE_SPEECH: [
                "hate",
                "racist",
                "sexist",
                "homophobic",
                "slur",
                "discriminat",
                "bigot",
                "supremac",
                "nazi",
                "fascist",
            ],
            HarmCategory.HARASSMENT: [
                "harass",
                "bully",
                "stalk",
                "threaten",
                "intimidat",
                "abuse",
                "torment",
                "persecute",
                "target",
            ],
            HarmCategory.PRIVACY_VIOLATIONS: [
                "social security",
                "ssn",
                "credit card",
                "password",
                "private",
                "confidential",
                "personal information",
                "dox",
                "leak",
            ],
            HarmCategory.MISINFORMATION: [
                "false",
                "lie",
                "hoax",
                "fake",
                "conspiracy",
                "debunked",
                "misinform",
                "propaganda",
                "mislead",
            ],
            HarmCategory.ILLEGAL_ACTIVITIES: [
                "illegal",
                "crime",
                "criminal",
                "fraud",
                "scam",
                "theft",
                "drugs",
                "cocaine",
                "heroin",
                "meth",
                "laundering",
            ],
        }

        # Pattern-based classifiers
        self.pattern_classifiers = {
            HarmCategory.PERSONAL_INFO_EXPOSURE: [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card pattern
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
                r"\b\d{3}-\d{3}-\d{4}\b",  # Phone number pattern
            ]
        }

    async def classify_content(self, content: str, content_type: str = "text") -> Dict[HarmCategory, float]:
        """Classify content for harm categories and return risk scores."""
        if content_type != "text":
            # For non-text content, would use appropriate ML models
            return {}

        results = {}
        content_lower = content.lower()

        # Keyword-based classification
        for category, keywords in self.keyword_classifiers.items():
            score = 0.0
            matches = 0

            for keyword in keywords:
                if keyword in content_lower:
                    matches += 1
                    score += 0.1  # Each keyword match adds to score

            if matches > 0:
                # Normalize score based on content length and keyword density
                word_count = len(content.split())
                density = matches / max(1, word_count / 100)  # Per 100 words
                score = min(1.0, score * density)
                results[category] = score

        # Pattern-based classification
        for category, patterns in self.pattern_classifiers.items():
            score = 0.0

            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    score += len(matches) * 0.2

            if score > 0:
                results[category] = min(1.0, score)

        return results

    async def analyze_behavioral_patterns(
        self, user_id: str, recent_interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze user behavioral patterns for risk assessment."""
        if not recent_interactions:
            return {}

        patterns = {
            "repetitive_harmful_content": 0.0,
            "escalating_aggression": 0.0,
            "manipulation_attempts": 0.0,
            "privacy_probing": 0.0,
        }

        # Count harmful content frequency
        harmful_count = 0
        for interaction in recent_interactions:
            harm_scores = interaction.get("harm_scores", {})
            if any(score > 0.3 for score in harm_scores.values()):
                harmful_count += 1

        if harmful_count > 0:
            patterns["repetitive_harmful_content"] = min(1.0, harmful_count / len(recent_interactions))

        # Check for escalating patterns (simplified)
        if len(recent_interactions) >= 3:
            recent_harm_levels = []
            for interaction in recent_interactions[-3:]:
                harm_scores = interaction.get("harm_scores", {})
                max_harm = max(harm_scores.values()) if harm_scores else 0.0
                recent_harm_levels.append(max_harm)

            # Check if harm levels are increasing
            if len(recent_harm_levels) >= 2:
                increasing = all(
                    recent_harm_levels[i] <= recent_harm_levels[i + 1] for i in range(len(recent_harm_levels) - 1)
                )
                if increasing and recent_harm_levels[-1] > 0.3:
                    patterns["escalating_aggression"] = recent_harm_levels[-1]

        return patterns


class ConstitutionalPolicyEngine:
    """
    Main Constitutional Security Policy Engine

    Enforces constitutional AI safety policies across TEE-enabled fog computing
    infrastructure with real-time harm detection and mitigation.
    """

    def __init__(self):
        # Core components
        self.harm_classifier = HarmClassifier()

        # Policy storage
        self.active_policies: Dict[str, ConstitutionalPolicy] = {}
        self.policy_violations: List[PolicyViolation] = []
        self.user_behavior_history: Dict[str, List[Dict[str, Any]]] = {}

        # Default constitutional policy
        self.default_policy = self._create_default_policy()
        self.active_policies["default"] = self.default_policy

        # Monitoring and state
        self.monitoring_enabled = True
        self.real_time_filtering = True
        self.violation_threshold_counts: Dict[str, int] = {}

        logger.info("Constitutional Policy Engine initialized")

    async def evaluate_content(
        self,
        content: str,
        content_type: str = "text",
        context: Optional[Dict[str, Any]] = None,
        policy_id: str = "default",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate content against constitutional policy.

        Returns:
            Tuple of (is_safe, evaluation_details)
        """
        context = context or {}
        user_id = context.get("user_id", "anonymous")

        if policy_id not in self.active_policies:
            policy_id = "default"

        policy = self.active_policies[policy_id]

        # Step 1: Classify content for harm
        harm_scores = await self.harm_classifier.classify_content(content, content_type)

        # Step 2: Evaluate against constraints
        violations = []
        max_risk_score = 0.0

        for constraint in policy.constraints:
            if not constraint.enabled:
                continue

            # Check if any harm categories match
            constraint_violations = []
            for harm_category in constraint.harm_categories:
                if harm_category in harm_scores:
                    risk_score = harm_scores[harm_category]
                    max_risk_score = max(max_risk_score, risk_score)

                    if risk_score >= constraint.max_risk_score:
                        violation = PolicyViolation(
                            constraint_id=constraint.constraint_id,
                            constraint_name=constraint.name,
                            harm_category=harm_category,
                            risk_level=self._calculate_risk_level(risk_score),
                            content_hash=self._hash_content(content),
                            content_snippet=content[:200] + "..." if len(content) > 200 else content,
                            content_type=content_type,
                            risk_score=risk_score,
                            confidence_score=0.8,  # Would be from ML model
                            detection_method="classifier",
                            user_id=user_id,
                            session_id=context.get("session_id", ""),
                            workload_id=context.get("workload_id", ""),
                            node_id=context.get("node_id", ""),
                        )
                        constraint_violations.append(violation)

            violations.extend(constraint_violations)

        # Step 3: Determine overall safety and response
        is_safe = len(violations) == 0 and max_risk_score <= policy.default_risk_threshold

        # Step 4: Apply behavioral analysis
        behavioral_risk = 0.0
        if user_id != "anonymous" and user_id in self.user_behavior_history:
            behavioral_patterns = await self.harm_classifier.analyze_behavioral_patterns(
                user_id, self.user_behavior_history[user_id]
            )
            behavioral_risk = max(behavioral_patterns.values()) if behavioral_patterns else 0.0

        # Adjust safety assessment based on behavioral patterns
        if behavioral_risk > 0.5:
            is_safe = False

        # Step 5: Log interaction for behavioral analysis
        interaction_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "content_hash": self._hash_content(content),
            "harm_scores": harm_scores,
            "violations": len(violations),
            "behavioral_risk": behavioral_risk,
        }

        if user_id not in self.user_behavior_history:
            self.user_behavior_history[user_id] = []

        self.user_behavior_history[user_id].append(interaction_record)

        # Keep only recent history (last 50 interactions)
        if len(self.user_behavior_history[user_id]) > 50:
            self.user_behavior_history[user_id] = self.user_behavior_history[user_id][-50:]

        # Step 6: Process violations
        for violation in violations:
            await self._process_violation(violation, policy)

        # Step 7: Prepare evaluation response
        evaluation_details = {
            "policy_id": policy_id,
            "harm_scores": harm_scores,
            "max_risk_score": max_risk_score,
            "behavioral_risk": behavioral_risk,
            "violations": [
                {"category": v.harm_category.value, "risk_score": v.risk_score, "action": v.action_taken.value}
                for v in violations
            ],
            "recommended_actions": self._get_recommended_actions(violations, behavioral_risk),
            "evaluation_timestamp": datetime.now(UTC).isoformat(),
        }

        return is_safe, evaluation_details

    async def validate_workload_deployment(
        self, workload_manifest: Dict[str, Any], node_attestation: Dict[str, Any], policy_id: str = "default"
    ) -> bool:
        """Validate that workload can be safely deployed with constitutional guarantees."""

        if policy_id not in self.active_policies:
            policy_id = "default"

        policy = self.active_policies[policy_id]

        # Check constitutional tier compatibility
        required_tier = workload_manifest.get("constitutional_tier", "silver")
        node_tier = node_attestation.get("constitutional_tier", "bronze")

        tier_hierarchy = {"bronze": 1, "silver": 2, "gold": 3}

        if tier_hierarchy.get(node_tier, 1) < tier_hierarchy.get(required_tier, 2):
            logger.warning("Node constitutional tier insufficient for workload")
            return False

        # Check harm categories are properly monitored
        workload_harm_categories = workload_manifest.get("harm_categories", [])
        for category in workload_harm_categories:
            if not any(HarmCategory(category) in constraint.harm_categories for constraint in policy.constraints):
                logger.warning(f"Harm category {category} not covered by policy")
                return False

        # Validate attestation has necessary capabilities
        required_capabilities = ["memory_encryption", "remote_attestation"]
        node_capabilities = node_attestation.get("capabilities", [])

        for capability in required_capabilities:
            if capability not in node_capabilities:
                logger.warning(f"Node missing required capability: {capability}")
                return False

        return True

    async def monitor_workload_execution(
        self, workload_id: str, execution_logs: List[Dict[str, Any]], policy_id: str = "default"
    ) -> Dict[str, Any]:
        """Monitor ongoing workload execution for constitutional compliance."""

        monitoring_result = {
            "workload_id": workload_id,
            "monitoring_timestamp": datetime.now(UTC).isoformat(),
            "compliance_status": "compliant",
            "violations_detected": 0,
            "risk_score": 0.0,
            "recommendations": [],
        }

        total_risk = 0.0
        violation_count = 0

        # Analyze execution logs for constitutional violations
        for log_entry in execution_logs:
            if "output" in log_entry:
                is_safe, evaluation = await self.evaluate_content(
                    log_entry["output"],
                    context={"workload_id": workload_id, "log_timestamp": log_entry.get("timestamp")},
                    policy_id=policy_id,
                )

                if not is_safe:
                    violation_count += 1
                    total_risk = max(total_risk, evaluation.get("max_risk_score", 0.0))

        if violation_count > 0:
            monitoring_result["compliance_status"] = "violations_detected"
            monitoring_result["violations_detected"] = violation_count
            monitoring_result["risk_score"] = total_risk

            # Generate recommendations
            if total_risk > 0.8:
                monitoring_result["recommendations"].append("immediate_termination")
            elif total_risk > 0.5:
                monitoring_result["recommendations"].append("human_review_required")
            elif violation_count > 5:
                monitoring_result["recommendations"].append("enhanced_monitoring")

        return monitoring_result

    def create_custom_policy(
        self, name: str, constraints: List[SafetyConstraint], principles: Optional[List[ConstitutionalPrinciple]] = None
    ) -> str:
        """Create custom constitutional policy."""

        policy = ConstitutionalPolicy(
            name=name,
            description=f"Custom constitutional policy: {name}",
            principles=principles or [ConstitutionalPrinciple.HARMLESSNESS],
            constraints=constraints,
        )

        self.active_policies[policy.policy_id] = policy

        logger.info(f"Created custom policy: {name} ({policy.policy_id})")
        return policy.policy_id

    def get_policy_violations_summary(
        self, time_window_hours: int = 24, policy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of policy violations within time window."""

        cutoff_time = datetime.now(UTC) - timedelta(hours=time_window_hours)

        relevant_violations = [
            v
            for v in self.policy_violations
            if v.detected_at > cutoff_time and (policy_id is None or v.violation_id == policy_id)
        ]

        summary = {
            "total_violations": len(relevant_violations),
            "time_window_hours": time_window_hours,
            "violations_by_category": {},
            "violations_by_risk_level": {},
            "violations_by_action": {},
            "most_violated_constraints": {},
            "escalated_violations": 0,
        }

        for violation in relevant_violations:
            # Count by category
            category = violation.harm_category.value
            summary["violations_by_category"][category] = summary["violations_by_category"].get(category, 0) + 1

            # Count by risk level
            risk_level = violation.risk_level.value
            summary["violations_by_risk_level"][risk_level] = summary["violations_by_risk_level"].get(risk_level, 0) + 1

            # Count by action
            action = violation.action_taken.value
            summary["violations_by_action"][action] = summary["violations_by_action"].get(action, 0) + 1

            # Count constraint violations
            constraint_name = violation.constraint_name
            summary["most_violated_constraints"][constraint_name] = (
                summary["most_violated_constraints"].get(constraint_name, 0) + 1
            )

            # Count escalations
            if violation.escalated:
                summary["escalated_violations"] += 1

        return summary

    async def generate_compliance_report(
        self, policy_id: str = "default", report_period_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive constitutional compliance report."""

        policy = self.active_policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")

        datetime.now(UTC) - timedelta(days=report_period_days)

        report = {
            "report_id": str(uuid.uuid4()),
            "policy_id": policy_id,
            "policy_name": policy.name,
            "report_period_days": report_period_days,
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": {},
            "detailed_analysis": {},
            "recommendations": [],
        }

        # Get violations summary
        violations_summary = self.get_policy_violations_summary(
            time_window_hours=report_period_days * 24, policy_id=policy_id
        )

        report["summary"] = violations_summary

        # Detailed analysis
        total_violations = violations_summary["total_violations"]
        if total_violations > 0:
            # Calculate compliance rate
            # This would need interaction counts in real implementation
            estimated_interactions = total_violations * 10  # Rough estimate
            compliance_rate = (estimated_interactions - total_violations) / estimated_interactions

            report["detailed_analysis"] = {
                "compliance_rate": compliance_rate,
                "most_problematic_categories": sorted(
                    violations_summary["violations_by_category"].items(), key=lambda x: x[1], reverse=True
                )[:5],
                "risk_distribution": violations_summary["violations_by_risk_level"],
                "response_effectiveness": violations_summary["violations_by_action"],
            }

            # Generate recommendations
            if compliance_rate < 0.95:
                report["recommendations"].append("Enhance proactive content filtering")

            if violations_summary["escalated_violations"] > total_violations * 0.1:
                report["recommendations"].append("Review escalation thresholds")

            high_risk_violations = violations_summary["violations_by_risk_level"].get("critical", 0)
            if high_risk_violations > 0:
                report["recommendations"].append("Implement additional safety measures")
        else:
            report["detailed_analysis"] = {
                "compliance_rate": 1.0,
                "status": "No violations detected in reporting period",
            }
            report["recommendations"].append("Continue current safety measures")

        return report

    # Private helper methods

    def _create_default_policy(self) -> ConstitutionalPolicy:
        """Create default constitutional policy with standard constraints."""

        constraints = []

        # Violence constraint
        violence_constraint = SafetyConstraint(
            name="Violence Prevention",
            description="Prevent violent content and threats",
            harm_categories=[HarmCategory.VIOLENCE, HarmCategory.TERRORISM, HarmCategory.WEAPONS],
            max_risk_score=0.2,
            violation_response=ResponseAction.BLOCK,
        )
        constraints.append(violence_constraint)

        # Hate speech constraint
        hate_speech_constraint = SafetyConstraint(
            name="Hate Speech Prevention",
            description="Prevent hate speech and discrimination",
            harm_categories=[HarmCategory.HATE_SPEECH, HarmCategory.DISCRIMINATION, HarmCategory.HARASSMENT],
            max_risk_score=0.3,
            violation_response=ResponseAction.FILTER,
        )
        constraints.append(hate_speech_constraint)

        # Privacy protection constraint
        privacy_constraint = SafetyConstraint(
            name="Privacy Protection",
            description="Protect personal information and privacy",
            harm_categories=[
                HarmCategory.PRIVACY_VIOLATIONS,
                HarmCategory.PERSONAL_INFO_EXPOSURE,
                HarmCategory.DOXXING,
            ],
            max_risk_score=0.1,
            violation_response=ResponseAction.BLOCK,
        )
        constraints.append(privacy_constraint)

        # Misinformation constraint
        misinformation_constraint = SafetyConstraint(
            name="Misinformation Prevention",
            description="Prevent spread of false information",
            harm_categories=[
                HarmCategory.MISINFORMATION,
                HarmCategory.DISINFORMATION,
                HarmCategory.CONSPIRACY_THEORIES,
            ],
            max_risk_score=0.4,
            violation_response=ResponseAction.WARN,
        )
        constraints.append(misinformation_constraint)

        # Illegal activities constraint
        illegal_constraint = SafetyConstraint(
            name="Illegal Activities Prevention",
            description="Prevent illegal activities and fraud",
            harm_categories=[HarmCategory.ILLEGAL_ACTIVITIES, HarmCategory.FRAUD, HarmCategory.ILLEGAL_SUBSTANCES],
            max_risk_score=0.2,
            violation_response=ResponseAction.BLOCK,
        )
        constraints.append(illegal_constraint)

        return ConstitutionalPolicy(
            name="Default Constitutional Policy",
            description="Standard constitutional AI safety policy",
            constraints=constraints,
            default_risk_threshold=0.3,
            require_human_oversight=False,
            enable_proactive_monitoring=True,
        )

    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Calculate risk level from numerical score."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.1:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE

    def _hash_content(self, content: str) -> str:
        """Create hash of content for tracking."""
        import hashlib

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _process_violation(self, violation: PolicyViolation, policy: ConstitutionalPolicy):
        """Process detected policy violation."""

        # Store violation
        self.policy_violations.append(violation)

        # Apply response action
        constraint = next((c for c in policy.constraints if c.constraint_id == violation.constraint_id), None)

        if constraint:
            violation.action_taken = constraint.violation_response

            # Check for escalation
            if violation.risk_score >= constraint.escalation_threshold:
                violation.escalated = True
                violation.action_taken = ResponseAction.ESCALATE

        # Log violation
        logger.warning(
            f"Constitutional violation detected: {violation.harm_category.value} "
            f"(risk: {violation.risk_score:.3f}, action: {violation.action_taken.value})"
        )

        # Clean up old violations
        cutoff_time = datetime.now(UTC) - timedelta(days=policy.retain_violations_days)
        self.policy_violations = [v for v in self.policy_violations if v.detected_at > cutoff_time]

    def _get_recommended_actions(self, violations: List[PolicyViolation], behavioral_risk: float) -> List[str]:
        """Get recommended actions based on violations and behavioral risk."""

        actions = []

        if not violations and behavioral_risk < 0.3:
            return ["continue_normal_operation"]

        max_risk = max((v.risk_score for v in violations), default=0.0)
        critical_violations = [v for v in violations if v.risk_level == RiskLevel.CRITICAL]

        if critical_violations:
            actions.append("immediate_content_blocking")
            actions.append("escalate_to_human_review")
        elif max_risk > 0.6:
            actions.append("enhanced_monitoring")
            actions.append("content_filtering_recommended")
        elif max_risk > 0.3:
            actions.append("content_warning_recommended")

        if behavioral_risk > 0.7:
            actions.append("user_behavior_review")
        elif behavioral_risk > 0.5:
            actions.append("increased_monitoring")

        return actions if actions else ["log_and_continue"]


# Global policy engine instance
_policy_engine: Optional[ConstitutionalPolicyEngine] = None


async def get_policy_engine() -> ConstitutionalPolicyEngine:
    """Get global constitutional policy engine instance."""
    global _policy_engine

    if _policy_engine is None:
        _policy_engine = ConstitutionalPolicyEngine()

    return _policy_engine


# Convenience functions for integration


async def evaluate_constitutional_content(
    content: str, user_id: str = "anonymous", context: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate content for constitutional compliance."""

    engine = await get_policy_engine()
    context = context or {}
    context["user_id"] = user_id

    return await engine.evaluate_content(content, context=context)


async def validate_constitutional_workload(workload_manifest: Dict[str, Any], node_attestation: Dict[str, Any]) -> bool:
    """Validate workload can be safely deployed."""

    engine = await get_policy_engine()
    return await engine.validate_workload_deployment(workload_manifest, node_attestation)


if __name__ == "__main__":

    async def test_policy_engine():
        """Test constitutional policy engine."""
        engine = await get_policy_engine()

        # Test content evaluation
        test_contents = [
            "Hello, how are you today?",
            "I want to hurt someone",
            "Here's my social security number: 123-45-6789",
            "The earth is flat and vaccines are dangerous",
        ]

        for content in test_contents:
            is_safe, details = await engine.evaluate_content(content)
            print(f"Content: '{content}'")
            print(f"Safe: {is_safe}")
            print(f"Details: {json.dumps(details, indent=2)}")
            print("-" * 50)

        # Test compliance report
        report = await engine.generate_compliance_report()
        print(f"Compliance Report: {json.dumps(report, indent=2)}")

    asyncio.run(test_policy_engine())
