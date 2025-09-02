"""
SLO Recovery Router - Escalation Management
Human intervention procedures and automated escalation triggers
Target: Minimize escalations while ensuring critical issues get attention
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from .breach_classifier import BreachClassification, BreachSeverity
from .parallel_coordinator import CoordinationPlan
from .strategy_selector import StrategySelection


class EscalationLevel(Enum):
    NONE = "none"
    AUTOMATIC_RETRY = "automatic_retry"
    TEAM_NOTIFICATION = "team_notification"
    SENIOR_REVIEW = "senior_review"
    EMERGENCY_RESPONSE = "emergency_response"
    EXECUTIVE_ALERT = "executive_alert"


class EscalationTrigger(Enum):
    TIME_EXCEEDED = "time_exceeded"
    REPEATED_FAILURES = "repeated_failures"
    CRITICAL_SYSTEM_DOWN = "critical_system_down"
    SECURITY_BREACH = "security_breach"
    DATA_LOSS_RISK = "data_loss_risk"
    CASCADING_FAILURES = "cascading_failures"
    UNKNOWN_ERROR_PATTERN = "unknown_error_pattern"
    CONFIDENCE_TOO_LOW = "confidence_too_low"


@dataclass
class EscalationRule:
    rule_id: str
    name: str
    trigger: EscalationTrigger
    conditions: list[str]
    escalation_level: EscalationLevel
    time_threshold_minutes: int | None
    failure_count_threshold: int | None
    confidence_threshold: float | None
    auto_actions: list[str]
    notification_targets: list[str]
    escalation_message_template: str


@dataclass
class EscalationEvent:
    event_id: str
    trigger: EscalationTrigger
    escalation_level: EscalationLevel
    breach_classification: BreachClassification
    strategy_selection: StrategySelection | None
    coordination_plan: CoordinationPlan | None
    trigger_time: datetime
    escalation_data: dict
    resolution_actions: list[str]
    resolution_time: datetime | None = None
    human_assignee: str | None = None
    status: str = "open"


class EscalationManager:
    """
    Escalation management system for SLO recovery with human intervention procedures
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.escalation_rules = self._initialize_escalation_rules()
        self.active_escalations = {}
        self.escalation_history = []
        self.notification_handlers = {}

    def _initialize_escalation_rules(self) -> list[EscalationRule]:
        """Initialize escalation rules with thresholds and actions"""

        return [
            # Critical Security Escalations
            EscalationRule(
                rule_id="SEC_ESC_001",
                name="critical_security_breach",
                trigger=EscalationTrigger.SECURITY_BREACH,
                conditions=[
                    "breach_severity == 'critical'",
                    "security_baseline_failure == True",
                    "production_system == True",
                ],
                escalation_level=EscalationLevel.EMERGENCY_RESPONSE,
                time_threshold_minutes=5,
                failure_count_threshold=None,
                confidence_threshold=None,
                auto_actions=[
                    "isolate_affected_systems",
                    "enable_security_monitoring",
                    "create_incident_ticket",
                    "notify_security_team",
                ],
                notification_targets=["security-team@company.com", "incident-response@company.com"],
                escalation_message_template="CRITICAL: Security breach detected in production system. Immediate response required.",
            ),
            # Time-based Escalations
            EscalationRule(
                rule_id="TIME_ESC_001",
                name="mttr_threshold_exceeded",
                trigger=EscalationTrigger.TIME_EXCEEDED,
                conditions=["recovery_time_minutes > mttr_threshold", "coordination_status != 'completed'"],
                escalation_level=EscalationLevel.SENIOR_REVIEW,
                time_threshold_minutes=30,  # MTTR threshold
                failure_count_threshold=None,
                confidence_threshold=None,
                auto_actions=["create_escalation_ticket", "gather_diagnostic_data", "notify_senior_engineer"],
                notification_targets=["senior-engineers@company.com", "slo-alerts@company.com"],
                escalation_message_template="SLO BREACH: Recovery time exceeded 30 minutes. Senior review required.",
            ),
            # Repeated Failure Escalations
            EscalationRule(
                rule_id="FAIL_ESC_001",
                name="repeated_recovery_failures",
                trigger=EscalationTrigger.REPEATED_FAILURES,
                conditions=["failure_count >= failure_threshold", "time_window_hours <= 24"],
                escalation_level=EscalationLevel.TEAM_NOTIFICATION,
                time_threshold_minutes=None,
                failure_count_threshold=3,
                confidence_threshold=None,
                auto_actions=["analyze_failure_patterns", "create_investigation_ticket", "notify_development_team"],
                notification_targets=["dev-team@company.com", "reliability-team@company.com"],
                escalation_message_template="Pattern Alert: Multiple recovery failures detected. Investigation required.",
            ),
            # Low Confidence Escalations
            EscalationRule(
                rule_id="CONF_ESC_001",
                name="low_confidence_classification",
                trigger=EscalationTrigger.CONFIDENCE_TOO_LOW,
                conditions=[
                    "classification_confidence < confidence_threshold",
                    "breach_severity in ['critical', 'high']",
                ],
                escalation_level=EscalationLevel.SENIOR_REVIEW,
                time_threshold_minutes=15,
                failure_count_threshold=None,
                confidence_threshold=0.6,
                auto_actions=["request_human_classification", "gather_additional_context", "pause_automated_recovery"],
                notification_targets=["senior-sre@company.com"],
                escalation_message_template="Classification Uncertainty: Low confidence in automated classification. Human review requested.",
            ),
            # System Stability Escalations
            EscalationRule(
                rule_id="STAB_ESC_001",
                name="cascading_system_failures",
                trigger=EscalationTrigger.CASCADING_FAILURES,
                conditions=["affected_systems_count > 3", "failure_propagation_detected == True"],
                escalation_level=EscalationLevel.EMERGENCY_RESPONSE,
                time_threshold_minutes=10,
                failure_count_threshold=None,
                confidence_threshold=None,
                auto_actions=[
                    "activate_incident_response",
                    "isolate_failing_systems",
                    "enable_circuit_breakers",
                    "notify_all_teams",
                ],
                notification_targets=["incident-commander@company.com", "all-engineers@company.com"],
                escalation_message_template="EMERGENCY: Cascading system failures detected. All hands response activated.",
            ),
            # Unknown Pattern Escalations
            EscalationRule(
                rule_id="UNK_ESC_001",
                name="unknown_error_pattern",
                trigger=EscalationTrigger.UNKNOWN_ERROR_PATTERN,
                conditions=[
                    "pattern_recognition_failed == True",
                    "error_signature_unknown == True",
                    "breach_severity in ['critical', 'high']",
                ],
                escalation_level=EscalationLevel.TEAM_NOTIFICATION,
                time_threshold_minutes=20,
                failure_count_threshold=None,
                confidence_threshold=0.5,
                auto_actions=["collect_comprehensive_logs", "create_research_ticket", "notify_architecture_team"],
                notification_targets=["architecture-team@company.com", "platform-team@company.com"],
                escalation_message_template="Unknown Pattern: Novel error pattern detected. Expert analysis required.",
            ),
            # Data Safety Escalations
            EscalationRule(
                rule_id="DATA_ESC_001",
                name="data_loss_risk_detected",
                trigger=EscalationTrigger.DATA_LOSS_RISK,
                conditions=["data_integrity_at_risk == True", "backup_systems_affected == True"],
                escalation_level=EscalationLevel.EXECUTIVE_ALERT,
                time_threshold_minutes=2,
                failure_count_threshold=None,
                confidence_threshold=None,
                auto_actions=[
                    "immediate_data_backup",
                    "enable_data_protection_mode",
                    "alert_executive_team",
                    "create_crisis_ticket",
                ],
                notification_targets=["cto@company.com", "data-protection@company.com"],
                escalation_message_template="EXECUTIVE ALERT: Data loss risk detected. Immediate executive attention required.",
            ),
            # Automatic Retry (Non-escalation)
            EscalationRule(
                rule_id="RETRY_001",
                name="automatic_retry_eligible",
                trigger=EscalationTrigger.REPEATED_FAILURES,
                conditions=["failure_count <= 2", "breach_severity in ['medium', 'low']", "retry_eligible == True"],
                escalation_level=EscalationLevel.AUTOMATIC_RETRY,
                time_threshold_minutes=None,
                failure_count_threshold=2,
                confidence_threshold=None,
                auto_actions=["schedule_automatic_retry", "adjust_strategy_parameters", "log_retry_attempt"],
                notification_targets=[],
                escalation_message_template="Auto-retry scheduled for recoverable failure.",
            ),
        ]

    def evaluate_escalation_triggers(
        self,
        breach_classification: BreachClassification,
        strategy_selection: StrategySelection | None = None,
        coordination_plan: CoordinationPlan | None = None,
        execution_context: dict | None = None,
    ) -> list[EscalationEvent]:
        """
        Evaluate all escalation rules and create events for triggered rules
        """

        triggered_escalations = []

        # Build evaluation context
        context = self._build_evaluation_context(
            breach_classification, strategy_selection, coordination_plan, execution_context
        )

        for rule in self.escalation_rules:
            if self._evaluate_rule_conditions(rule, context):
                # Create escalation event
                event = self._create_escalation_event(
                    rule, breach_classification, strategy_selection, coordination_plan, context
                )
                triggered_escalations.append(event)

                # Execute auto-actions
                self._execute_auto_actions(event)

                # Send notifications
                self._send_notifications(event)

        return triggered_escalations

    def _build_evaluation_context(
        self,
        breach_classification: BreachClassification,
        strategy_selection: StrategySelection | None,
        coordination_plan: CoordinationPlan | None,
        execution_context: dict | None,
    ) -> dict:
        """Build context for rule evaluation"""

        context = {
            # Breach classification context
            "breach_severity": breach_classification.severity.value,
            "breach_category": breach_classification.category.value,
            "classification_confidence": breach_classification.confidence_score,
            "priority_score": breach_classification.priority_score,
            "escalation_required": breach_classification.escalation_required,
            # Security context
            "security_baseline_failure": "security" in breach_classification.category.value,
            "production_system": True,  # Assuming production context
            # Time context
            "breach_age_minutes": (datetime.now() - breach_classification.timestamp).total_seconds() / 60,
            "mttr_threshold": 30,
            # Failure context
            "failure_count": execution_context.get("failure_count", 1) if execution_context else 1,
            "failure_threshold": 3,
            "time_window_hours": 24,
            "retry_eligible": breach_classification.severity in [BreachSeverity.MEDIUM, BreachSeverity.LOW],
            # Pattern context
            "pattern_recognition_failed": breach_classification.confidence_score < 0.5,
            "error_signature_unknown": breach_classification.confidence_score < 0.4,
            # System context
            "affected_systems_count": execution_context.get("affected_systems", 1) if execution_context else 1,
            "failure_propagation_detected": False,  # Would be set by monitoring
            "data_integrity_at_risk": "data" in " ".join(breach_classification.indicators_matched),
            "backup_systems_affected": False,  # Would be set by monitoring
            # Coordination context
            "coordination_status": coordination_plan.coordination_status.value if coordination_plan else "unknown",
            "recovery_time_minutes": 0,  # Would be calculated from plan timing
        }

        # Add strategy context if available
        if strategy_selection:
            context.update(
                {
                    "selected_strategy": strategy_selection.selected_strategy.name,
                    "strategy_confidence": strategy_selection.confidence_score,
                }
            )

        # Add coordination timing context
        if coordination_plan:
            elapsed = datetime.now() - coordination_plan.start_time
            context["recovery_time_minutes"] = elapsed.total_seconds() / 60

        return context

    def _evaluate_rule_conditions(self, rule: EscalationRule, context: dict) -> bool:
        """Evaluate whether rule conditions are met"""

        try:
            # Check threshold conditions first
            if rule.time_threshold_minutes and context.get("recovery_time_minutes", 0) <= rule.time_threshold_minutes:
                return False

            if rule.failure_count_threshold and context.get("failure_count", 0) < rule.failure_count_threshold:
                return False

            if rule.confidence_threshold and context.get("classification_confidence", 1.0) >= rule.confidence_threshold:
                return False

            # Evaluate condition expressions
            for condition in rule.conditions:
                if not self._evaluate_condition_expression(condition, context):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating escalation rule {rule.rule_id}: {e}")
            return False

    def _evaluate_condition_expression(self, condition: str, context: dict) -> bool:
        """Evaluate a single condition expression safely"""

        try:
            # Replace context variables in condition
            for var, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(var, f"'{value}'")
                else:
                    condition = condition.replace(var, str(value))

            # Evaluate boolean expression
            return eval(condition)

        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _create_escalation_event(
        self,
        rule: EscalationRule,
        breach_classification: BreachClassification,
        strategy_selection: StrategySelection | None,
        coordination_plan: CoordinationPlan | None,
        context: dict,
    ) -> EscalationEvent:
        """Create escalation event from triggered rule"""

        event_id = f"ESC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.rule_id}"

        event = EscalationEvent(
            event_id=event_id,
            trigger=rule.trigger,
            escalation_level=rule.escalation_level,
            breach_classification=breach_classification,
            strategy_selection=strategy_selection,
            coordination_plan=coordination_plan,
            trigger_time=datetime.now(),
            escalation_data={
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "trigger_context": context,
                "auto_actions": rule.auto_actions,
                "notification_targets": rule.notification_targets,
                "escalation_message": rule.escalation_message_template,
            },
            resolution_actions=[],
        )

        self.active_escalations[event_id] = event
        return event

    def _execute_auto_actions(self, event: EscalationEvent):
        """Execute automatic actions for escalation event"""

        actions_executed = []

        for action in event.escalation_data["auto_actions"]:
            try:
                if action == "isolate_affected_systems":
                    self._isolate_systems(event)
                elif action == "enable_security_monitoring":
                    self._enable_security_monitoring(event)
                elif action == "create_incident_ticket":
                    self._create_incident_ticket(event)
                elif action == "schedule_automatic_retry":
                    self._schedule_automatic_retry(event)
                elif action == "gather_diagnostic_data":
                    self._gather_diagnostic_data(event)
                elif action == "pause_automated_recovery":
                    self._pause_automated_recovery(event)
                # Add more action handlers as needed

                actions_executed.append(action)

            except Exception as e:
                self.logger.error(f"Failed to execute auto-action '{action}': {e}")

        event.resolution_actions.extend(actions_executed)

    def _send_notifications(self, event: EscalationEvent):
        """Send notifications for escalation event"""

        message = event.escalation_data["escalation_message"]
        targets = event.escalation_data["notification_targets"]

        {
            "event_id": event.event_id,
            "escalation_level": event.escalation_level.value,
            "breach_id": event.breach_classification.breach_id,
            "severity": event.breach_classification.severity.value,
            "category": event.breach_classification.category.value,
            "message": message,
            "timestamp": event.trigger_time.isoformat(),
        }

        for target in targets:
            try:
                # This would integrate with actual notification systems
                self.logger.info(f"Sending escalation notification to {target}: {message}")
                # self.notification_handlers[target](notification_data)

            except Exception as e:
                self.logger.error(f"Failed to send notification to {target}: {e}")

    def _isolate_systems(self, event: EscalationEvent):
        """Isolate affected systems (placeholder)"""
        self.logger.info(f"Isolating affected systems for event {event.event_id}")

    def _enable_security_monitoring(self, event: EscalationEvent):
        """Enable enhanced security monitoring (placeholder)"""
        self.logger.info(f"Enabling security monitoring for event {event.event_id}")

    def _create_incident_ticket(self, event: EscalationEvent):
        """Create incident ticket (placeholder)"""
        self.logger.info(f"Creating incident ticket for event {event.event_id}")

    def _schedule_automatic_retry(self, event: EscalationEvent):
        """Schedule automatic retry (placeholder)"""
        self.logger.info(f"Scheduling automatic retry for event {event.event_id}")

    def _gather_diagnostic_data(self, event: EscalationEvent):
        """Gather diagnostic data (placeholder)"""
        self.logger.info(f"Gathering diagnostic data for event {event.event_id}")

    def _pause_automated_recovery(self, event: EscalationEvent):
        """Pause automated recovery (placeholder)"""
        self.logger.info(f"Pausing automated recovery for event {event.event_id}")

    def generate_escalation_procedures(self) -> dict:
        """Generate escalation procedures for output"""

        procedures = {
            "escalation_levels": {},
            "trigger_conditions": {},
            "automatic_actions": {},
            "notification_routing": {},
            "resolution_procedures": {},
        }

        # Escalation levels
        for level in EscalationLevel:
            procedures["escalation_levels"][level.value] = {
                "description": self._get_level_description(level),
                "response_time_sla": self._get_level_sla(level),
                "required_approvals": self._get_level_approvals(level),
            }

        # Trigger conditions
        for rule in self.escalation_rules:
            procedures["trigger_conditions"][rule.rule_id] = {
                "name": rule.name,
                "trigger": rule.trigger.value,
                "conditions": rule.conditions,
                "thresholds": {
                    "time_minutes": rule.time_threshold_minutes,
                    "failure_count": rule.failure_count_threshold,
                    "confidence": rule.confidence_threshold,
                },
            }

        # Automatic actions
        all_actions = set()
        for rule in self.escalation_rules:
            all_actions.update(rule.auto_actions)

        for action in all_actions:
            procedures["automatic_actions"][action] = {
                "description": self._get_action_description(action),
                "execution_method": "automated",
                "estimated_duration": self._get_action_duration(action),
            }

        # Notification routing
        all_targets = set()
        for rule in self.escalation_rules:
            all_targets.update(rule.notification_targets)

        for target in all_targets:
            procedures["notification_routing"][target] = {
                "contact_method": "email",  # Would be configured
                "escalation_path": self._get_escalation_path(target),
                "response_sla_minutes": self._get_response_sla(target),
            }

        # Resolution procedures
        procedures["resolution_procedures"] = {
            "human_intervention_required": [
                "critical_security_breach",
                "data_loss_risk_detected",
                "cascading_system_failures",
            ],
            "automated_resolution_eligible": ["automatic_retry_eligible", "low_severity_config_drift"],
            "escalation_closure_criteria": [
                "issue_resolved_successfully",
                "root_cause_identified_and_fixed",
                "preventive_measures_implemented",
            ],
        }

        return procedures

    def _get_level_description(self, level: EscalationLevel) -> str:
        descriptions = {
            EscalationLevel.NONE: "No escalation required",
            EscalationLevel.AUTOMATIC_RETRY: "Automatic retry with adjusted parameters",
            EscalationLevel.TEAM_NOTIFICATION: "Notify development team for awareness",
            EscalationLevel.SENIOR_REVIEW: "Senior engineer review and guidance required",
            EscalationLevel.EMERGENCY_RESPONSE: "Immediate emergency response activation",
            EscalationLevel.EXECUTIVE_ALERT: "Executive team notification required",
        }
        return descriptions.get(level, "Unknown escalation level")

    def _get_level_sla(self, level: EscalationLevel) -> int:
        """Get response time SLA in minutes for escalation level"""
        slas = {
            EscalationLevel.NONE: 0,
            EscalationLevel.AUTOMATIC_RETRY: 5,
            EscalationLevel.TEAM_NOTIFICATION: 60,
            EscalationLevel.SENIOR_REVIEW: 30,
            EscalationLevel.EMERGENCY_RESPONSE: 5,
            EscalationLevel.EXECUTIVE_ALERT: 15,
        }
        return slas.get(level, 60)

    def _get_level_approvals(self, level: EscalationLevel) -> list[str]:
        """Get required approvals for escalation level"""
        approvals = {
            EscalationLevel.NONE: [],
            EscalationLevel.AUTOMATIC_RETRY: [],
            EscalationLevel.TEAM_NOTIFICATION: ["team_lead"],
            EscalationLevel.SENIOR_REVIEW: ["senior_engineer"],
            EscalationLevel.EMERGENCY_RESPONSE: ["incident_commander"],
            EscalationLevel.EXECUTIVE_ALERT: ["cto", "vp_engineering"],
        }
        return approvals.get(level, [])

    def _get_action_description(self, action: str) -> str:
        descriptions = {
            "isolate_affected_systems": "Isolate systems to prevent failure propagation",
            "enable_security_monitoring": "Enable enhanced security monitoring and logging",
            "create_incident_ticket": "Create high-priority incident ticket with context",
            "schedule_automatic_retry": "Schedule automatic retry with exponential backoff",
            "gather_diagnostic_data": "Collect comprehensive diagnostic and log data",
            "pause_automated_recovery": "Pause automated recovery to await human review",
        }
        return descriptions.get(action, f"Execute action: {action}")

    def _get_action_duration(self, action: str) -> int:
        """Get estimated duration in minutes for action"""
        durations = {
            "isolate_affected_systems": 2,
            "enable_security_monitoring": 1,
            "create_incident_ticket": 1,
            "schedule_automatic_retry": 5,
            "gather_diagnostic_data": 3,
            "pause_automated_recovery": 1,
        }
        return durations.get(action, 2)

    def _get_escalation_path(self, target: str) -> list[str]:
        """Get escalation path for notification target"""
        # This would be configured based on organizational structure
        return ["primary_contact", "backup_contact", "manager"]

    def _get_response_sla(self, target: str) -> int:
        """Get response SLA in minutes for notification target"""
        # This would be configured based on team SLAs
        return 30


# Export for use by other components
__all__ = ["EscalationManager", "EscalationEvent", "EscalationLevel", "EscalationTrigger"]
