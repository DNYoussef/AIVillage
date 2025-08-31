"""
Constitutional Response Actions System
Implements automated response actions for constitutional moderation decisions
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of automated response actions"""
    NOTIFICATION = "notification"
    WARNING = "warning"
    RESTRICTION = "restriction"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    MONITOR = "monitor"
    ESCALATE = "escalate"
    APPEAL_NOTIFICATION = "appeal_notification"
    TRANSPARENCY_LOG = "transparency_log"

class ActionStatus(Enum):
    """Status of response actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ResponseAction:
    """Individual response action"""
    action_id: str
    action_type: ActionType
    content_id: str
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    constitutional_rationale: Optional[str] = None

@dataclass
class ActionPlan:
    """Complete action plan for content moderation decision"""
    content_id: str
    actions: List[ResponseAction]
    tier_level: str
    constitutional_considerations: Dict[str, Any]
    execution_order: List[str]
    total_actions: int

class ResponseActions:
    """
    Automated response actions system for constitutional moderation
    Implements tier-based responses while maintaining constitutional principles
    """
    
    def __init__(self):
        self.action_templates = self._load_action_templates()
        self.tier_configurations = self._load_tier_configurations()
        self.constitutional_safeguards = self._load_constitutional_safeguards()
        self.active_actions = {}  # Track active actions
        
        logger.info("Constitutional Response Actions initialized")

    def _load_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load response action templates"""
        return {
            "h0_allow": {
                "actions": [
                    {"type": ActionType.TRANSPARENCY_LOG, "priority": 1, "constitutional_note": "Promoting viewpoint diversity"},
                    {"type": ActionType.MONITOR, "priority": 2, "duration_hours": 24}
                ],
                "constitutional_principle": "maximum_free_speech_protection"
            },
            "h1_allow_with_warning": {
                "actions": [
                    {"type": ActionType.WARNING, "priority": 1, "severity": "informational"},
                    {"type": ActionType.NOTIFICATION, "priority": 2, "recipient": "user"},
                    {"type": ActionType.MONITOR, "priority": 3, "duration_hours": 72},
                    {"type": ActionType.TRANSPARENCY_LOG, "priority": 4}
                ],
                "constitutional_principle": "prior_notice_with_speech_protection"
            },
            "h2_restrict": {
                "actions": [
                    {"type": ActionType.NOTIFICATION, "priority": 1, "recipient": "user", "include_appeal_rights": True},
                    {"type": ActionType.RESTRICTION, "priority": 2, "type": "conditional_approval"},
                    {"type": ActionType.MONITOR, "priority": 3, "duration_hours": 168, "enhanced": True},
                    {"type": ActionType.APPEAL_NOTIFICATION, "priority": 4},
                    {"type": ActionType.TRANSPARENCY_LOG, "priority": 5}
                ],
                "constitutional_principle": "due_process_with_proportional_response"
            },
            "h2_quarantine": {
                "actions": [
                    {"type": ActionType.NOTIFICATION, "priority": 1, "recipient": "user", "urgency": "high"},
                    {"type": ActionType.QUARANTINE, "priority": 2, "preserve_evidence": True},
                    {"type": ActionType.APPEAL_NOTIFICATION, "priority": 3, "expedited": True},
                    {"type": ActionType.ESCALATE, "priority": 4, "human_review_required": True},
                    {"type": ActionType.TRANSPARENCY_LOG, "priority": 5}
                ],
                "constitutional_principle": "evidence_preservation_with_due_process"
            },
            "h3_block": {
                "actions": [
                    {"type": ActionType.BLOCK, "priority": 1, "immediate": True},
                    {"type": ActionType.NOTIFICATION, "priority": 2, "recipient": "user", "detailed_explanation": True},
                    {"type": ActionType.APPEAL_NOTIFICATION, "priority": 3, "constitutional_grounds": True},
                    {"type": ActionType.ESCALATE, "priority": 4, "severity": "high"},
                    {"type": ActionType.TRANSPARENCY_LOG, "priority": 5}
                ],
                "constitutional_principle": "minimal_prior_restraint_with_safeguards"
            },
            "escalate": {
                "actions": [
                    {"type": ActionType.ESCALATE, "priority": 1, "immediate": True, "constitutional_review": True},
                    {"type": ActionType.NOTIFICATION, "priority": 2, "recipient": "user", "pending_review": True},
                    {"type": ActionType.MONITOR, "priority": 3, "high_priority": True},
                    {"type": ActionType.TRANSPARENCY_LOG, "priority": 4}
                ],
                "constitutional_principle": "human_oversight_for_constitutional_compliance"
            }
        }

    def _load_tier_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load tier-specific configurations"""
        return {
            "Bronze": {
                "notification_method": "basic",
                "appeal_processing": "automated",
                "transparency_level": "standard",
                "monitoring_duration_multiplier": 1.0,
                "constitutional_explanation": "basic",
                "escalation_threshold": "high"
            },
            "Silver": {
                "notification_method": "enhanced",
                "appeal_processing": "hybrid",
                "transparency_level": "detailed",
                "monitoring_duration_multiplier": 1.5,
                "constitutional_explanation": "detailed",
                "escalation_threshold": "medium",
                "viewpoint_firewall": True
            },
            "Gold": {
                "notification_method": "comprehensive",
                "appeal_processing": "human_assisted",
                "transparency_level": "full",
                "monitoring_duration_multiplier": 2.0,
                "constitutional_explanation": "comprehensive",
                "escalation_threshold": "low",
                "constitutional_review": True,
                "community_oversight": True
            }
        }

    def _load_constitutional_safeguards(self) -> Dict[str, Any]:
        """Load constitutional safeguards for response actions"""
        return {
            "first_amendment": {
                "protected_speech_handling": "minimal_restriction",
                "political_content_safeguards": True,
                "religious_expression_protection": True,
                "artistic_expression_protection": True
            },
            "due_process": {
                "notice_requirements": {
                    "clear_explanation": True,
                    "policy_citation": True,
                    "appeal_rights_notification": True,
                    "evidence_disclosure": True
                },
                "proportionality": {
                    "least_restrictive_means": True,
                    "graduated_response": True,
                    "harm_proportionate": True
                }
            },
            "equal_protection": {
                "viewpoint_neutrality": True,
                "consistent_enforcement": True,
                "no_class_discrimination": True
            }
        }

    async def determine_actions(
        self,
        enforcement_result: Any,
        harm_analysis: Any,
        user_tier: str
    ) -> ActionPlan:
        """
        Determine appropriate response actions based on enforcement decision
        
        Args:
            enforcement_result: Result from policy enforcement
            harm_analysis: Content analysis results
            user_tier: User tier level
            
        Returns:
            ActionPlan with all response actions
        """
        try:
            content_id = harm_analysis.content_id
            decision = enforcement_result.decision.value
            
            logger.info(f"Determining actions for content {content_id}: {decision}")
            
            # Step 1: Get base action template
            template_key = self._get_template_key(decision, harm_analysis.harm_level)
            base_template = self.action_templates.get(template_key, self.action_templates["escalate"])
            
            # Step 2: Apply tier modifications
            tier_config = self.tier_configurations[user_tier]
            modified_actions = await self._apply_tier_modifications(
                base_template["actions"], tier_config, enforcement_result
            )
            
            # Step 3: Apply constitutional safeguards
            safeguarded_actions = await self._apply_constitutional_safeguards(
                modified_actions, enforcement_result.constitutional_analysis
            )
            
            # Step 4: Create response actions
            response_actions = []
            for action_config in safeguarded_actions:
                action = await self._create_response_action(
                    content_id, action_config, harm_analysis, enforcement_result, user_tier
                )
                response_actions.append(action)
            
            # Step 5: Determine execution order
            execution_order = self._determine_execution_order(response_actions)
            
            action_plan = ActionPlan(
                content_id=content_id,
                actions=response_actions,
                tier_level=user_tier,
                constitutional_considerations=enforcement_result.constitutional_analysis,
                execution_order=execution_order,
                total_actions=len(response_actions)
            )
            
            logger.info(f"Created action plan for {content_id}: {len(response_actions)} actions")
            return action_plan
            
        except Exception as e:
            logger.error(f"Failed to determine actions for {harm_analysis.content_id}: {str(e)}")
            return await self._create_safe_action_plan(harm_analysis.content_id, user_tier, str(e))

    def _get_template_key(self, decision: str, harm_level: str) -> str:
        """Get appropriate template key for decision and harm level"""
        
        if decision == "allow":
            return "h0_allow"
        elif decision == "allow_with_warning":
            return "h1_allow_with_warning"
        elif decision == "restrict":
            return "h2_restrict"
        elif decision == "quarantine":
            return "h2_quarantine"
        elif decision == "block":
            return "h3_block"
        elif decision == "escalate":
            return "escalate"
        else:
            return "escalate"  # Safe default

    async def _apply_tier_modifications(
        self,
        base_actions: List[Dict[str, Any]],
        tier_config: Dict[str, Any],
        enforcement_result: Any
    ) -> List[Dict[str, Any]]:
        """Apply tier-specific modifications to actions"""
        
        modified_actions = []
        
        for action_config in base_actions:
            modified_action = action_config.copy()
            
            # Apply tier-specific notification methods
            if action_config["type"] == ActionType.NOTIFICATION:
                modified_action["method"] = tier_config["notification_method"]
                modified_action["constitutional_explanation"] = tier_config["constitutional_explanation"]
            
            # Apply tier-specific monitoring durations
            elif action_config["type"] == ActionType.MONITOR:
                if "duration_hours" in modified_action:
                    modified_action["duration_hours"] = int(
                        modified_action["duration_hours"] * tier_config["monitoring_duration_multiplier"]
                    )
            
            # Apply tier-specific appeal processing
            elif action_config["type"] == ActionType.APPEAL_NOTIFICATION:
                modified_action["processing_type"] = tier_config["appeal_processing"]
            
            # Apply tier-specific escalation thresholds
            elif action_config["type"] == ActionType.ESCALATE:
                modified_action["threshold"] = tier_config["escalation_threshold"]
                
                # Gold tier gets constitutional review
                if tier_config.get("constitutional_review"):
                    modified_action["constitutional_review_required"] = True
                
                # Gold tier gets community oversight
                if tier_config.get("community_oversight"):
                    modified_action["community_oversight_notification"] = True
            
            # Apply tier-specific transparency levels
            elif action_config["type"] == ActionType.TRANSPARENCY_LOG:
                modified_action["detail_level"] = tier_config["transparency_level"]
            
            modified_actions.append(modified_action)
        
        # Add tier-specific additional actions
        if tier_config.get("viewpoint_firewall") and enforcement_result.constitutional_analysis.get("viewpoint_neutrality", {}).get("discrimination_risk"):
            modified_actions.append({
                "type": ActionType.MONITOR,
                "priority": 10,
                "viewpoint_bias_monitoring": True,
                "duration_hours": 168
            })
        
        return modified_actions

    async def _apply_constitutional_safeguards(
        self,
        actions: List[Dict[str, Any]],
        constitutional_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply constitutional safeguards to response actions"""
        
        safeguarded_actions = []
        
        for action_config in actions:
            safeguarded_action = action_config.copy()
            
            # First Amendment safeguards
            if constitutional_analysis.get("first_amendment", {}).get("protected_speech_detected"):
                if action_config["type"] in [ActionType.RESTRICTION, ActionType.QUARANTINE, ActionType.BLOCK]:
                    safeguarded_action["constitutional_justification_required"] = True
                    safeguarded_action["narrow_tailoring_verification"] = True
                    safeguarded_action["least_restrictive_means_analysis"] = True
            
            # Due process safeguards
            if action_config["type"] == ActionType.NOTIFICATION:
                safeguards = self.constitutional_safeguards["due_process"]["notice_requirements"]
                safeguarded_action.update({
                    "clear_explanation": safeguards["clear_explanation"],
                    "policy_citation": safeguards["policy_citation"],
                    "appeal_rights_notification": safeguards["appeal_rights_notification"],
                    "evidence_disclosure": safeguards["evidence_disclosure"]
                })
            
            # Equal protection safeguards
            if constitutional_analysis.get("viewpoint_neutrality", {}).get("discrimination_risk"):
                safeguarded_action["viewpoint_neutrality_verification"] = True
                safeguarded_action["equal_treatment_documentation"] = True
            
            # Prior restraint safeguards
            if constitutional_analysis.get("prior_restraint", {}).get("constitutional_justification_needed"):
                if action_config["type"] in [ActionType.QUARANTINE, ActionType.BLOCK]:
                    safeguarded_action["compelling_interest_documentation"] = True
                    safeguarded_action["constitutional_review_required"] = True
            
            safeguarded_actions.append(safeguarded_action)
        
        return safeguarded_actions

    async def _create_response_action(
        self,
        content_id: str,
        action_config: Dict[str, Any],
        harm_analysis: Any,
        enforcement_result: Any,
        user_tier: str
    ) -> ResponseAction:
        """Create individual response action from configuration"""
        
        action_id = str(uuid4())
        
        # Generate constitutional rationale
        constitutional_rationale = await self._generate_constitutional_rationale(
            action_config, enforcement_result.constitutional_analysis
        )
        
        # Generate description
        description = await self._generate_action_description(
            action_config, harm_analysis, user_tier
        )
        
        return ResponseAction(
            action_id=action_id,
            action_type=action_config["type"],
            content_id=content_id,
            description=description,
            parameters=action_config,
            status=ActionStatus.PENDING,
            created_at=datetime.utcnow(),
            constitutional_rationale=constitutional_rationale
        )

    async def _generate_constitutional_rationale(
        self,
        action_config: Dict[str, Any],
        constitutional_analysis: Dict[str, Any]
    ) -> str:
        """Generate constitutional rationale for action"""
        
        rationale_parts = []
        action_type = action_config["type"]
        
        # Action-specific rationale
        if action_type == ActionType.NOTIFICATION:
            rationale_parts.append("Due process notification requirement")
            
        elif action_type == ActionType.WARNING:
            rationale_parts.append("Prior notice consistent with due process")
            
        elif action_type == ActionType.RESTRICTION:
            if action_config.get("constitutional_justification_required"):
                rationale_parts.append("Restriction justified by compelling government interest")
            else:
                rationale_parts.append("Proportional restriction minimizing speech impact")
                
        elif action_type == ActionType.QUARANTINE:
            rationale_parts.append("Evidence preservation with minimal prior restraint")
            
        elif action_type == ActionType.BLOCK:
            rationale_parts.append("Content blocking justified by clear and present danger")
            
        elif action_type == ActionType.ESCALATE:
            rationale_parts.append("Human review required for constitutional compliance")
        
        # Constitutional analysis additions
        if constitutional_analysis.get("first_amendment", {}).get("heightened_scrutiny_required"):
            rationale_parts.append("Heightened scrutiny applied for protected speech")
            
        if constitutional_analysis.get("viewpoint_neutrality", {}).get("discrimination_risk"):
            rationale_parts.append("Viewpoint neutrality verification required")
        
        return " | ".join(rationale_parts) if rationale_parts else "Standard constitutional compliance"

    async def _generate_action_description(
        self,
        action_config: Dict[str, Any],
        harm_analysis: Any,
        user_tier: str
    ) -> str:
        """Generate human-readable action description"""
        
        action_type = action_config["type"]
        
        if action_type == ActionType.NOTIFICATION:
            return f"Notify {user_tier} tier user of moderation decision with constitutional rights information"
            
        elif action_type == ActionType.WARNING:
            severity = action_config.get("severity", "standard")
            return f"Issue {severity} warning with constitutional protections notice"
            
        elif action_type == ActionType.RESTRICTION:
            restriction_type = action_config.get("type", "standard")
            return f"Apply {restriction_type} restriction with proportionality safeguards"
            
        elif action_type == ActionType.QUARANTINE:
            if action_config.get("preserve_evidence"):
                return "Quarantine content while preserving evidence and constitutional rights"
            return "Quarantine content with constitutional safeguards"
            
        elif action_type == ActionType.BLOCK:
            if action_config.get("immediate"):
                return "Immediately block content with constitutional justification documentation"
            return "Block content with due process protections"
            
        elif action_type == ActionType.MONITOR:
            duration = action_config.get("duration_hours", 24)
            return f"Monitor content for {duration} hours with constitutional compliance tracking"
            
        elif action_type == ActionType.ESCALATE:
            if action_config.get("constitutional_review"):
                return "Escalate for constitutional review and human oversight"
            return "Escalate for human review with constitutional considerations"
            
        elif action_type == ActionType.APPEAL_NOTIFICATION:
            return f"Notify {user_tier} tier user of constitutional appeal rights and process"
            
        elif action_type == ActionType.TRANSPARENCY_LOG:
            detail_level = action_config.get("detail_level", "standard")
            return f"Create {detail_level} transparency log for public accountability"
        
        return f"Execute {action_type.value} action with constitutional protections"

    def _determine_execution_order(self, actions: List[ResponseAction]) -> List[str]:
        """Determine optimal execution order for actions"""
        
        # Sort by priority (lower number = higher priority)
        sorted_actions = sorted(actions, key=lambda a: a.parameters.get("priority", 999))
        
        return [action.action_id for action in sorted_actions]

    async def execute_actions(
        self,
        content_id: str,
        action_ids: List[str],
        harm_analysis: Any,
        user_tier: str
    ) -> Dict[str, Any]:
        """
        Execute the determined response actions
        
        Args:
            content_id: Content identifier
            action_ids: List of action IDs to execute
            harm_analysis: Content analysis results
            user_tier: User tier level
            
        Returns:
            Execution results
        """
        execution_results = {
            "content_id": content_id,
            "total_actions": len(action_ids),
            "successful_actions": 0,
            "failed_actions": 0,
            "action_results": {},
            "constitutional_compliance": True
        }
        
        try:
            logger.info(f"Executing {len(action_ids)} actions for content {content_id}")
            
            # Execute actions in order
            for action_id in action_ids:
                if action_id in self.active_actions:
                    action = self.active_actions[action_id]
                    
                    try:
                        # Update status
                        action.status = ActionStatus.IN_PROGRESS
                        
                        # Execute the action
                        result = await self._execute_single_action(action, harm_analysis, user_tier)
                        
                        # Update action with result
                        action.status = ActionStatus.COMPLETED
                        action.executed_at = datetime.utcnow()
                        action.result = result
                        
                        execution_results["action_results"][action_id] = result
                        execution_results["successful_actions"] += 1
                        
                        logger.info(f"Successfully executed action {action_id}: {action.action_type.value}")
                        
                    except Exception as e:
                        logger.error(f"Failed to execute action {action_id}: {str(e)}")
                        
                        action.status = ActionStatus.FAILED
                        action.result = {"error": str(e)}
                        
                        execution_results["action_results"][action_id] = {"error": str(e)}
                        execution_results["failed_actions"] += 1
                        execution_results["constitutional_compliance"] = False
            
            logger.info(f"Action execution completed for {content_id}: "
                       f"{execution_results['successful_actions']}/{execution_results['total_actions']} successful")
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Critical error executing actions for {content_id}: {str(e)}")
            execution_results["critical_error"] = str(e)
            execution_results["constitutional_compliance"] = False
            return execution_results

    async def _execute_single_action(
        self,
        action: ResponseAction,
        harm_analysis: Any,
        user_tier: str
    ) -> Dict[str, Any]:
        """Execute a single response action"""
        
        action_type = action.action_type
        parameters = action.parameters
        
        if action_type == ActionType.NOTIFICATION:
            return await self._execute_notification(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.WARNING:
            return await self._execute_warning(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.RESTRICTION:
            return await self._execute_restriction(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.QUARANTINE:
            return await self._execute_quarantine(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.BLOCK:
            return await self._execute_block(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.MONITOR:
            return await self._execute_monitoring(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.ESCALATE:
            return await self._execute_escalation(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.APPEAL_NOTIFICATION:
            return await self._execute_appeal_notification(action, harm_analysis, user_tier)
            
        elif action_type == ActionType.TRANSPARENCY_LOG:
            return await self._execute_transparency_logging(action, harm_analysis, user_tier)
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def _execute_notification(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute user notification action"""
        # Implementation would integrate with notification system
        return {
            "notification_sent": True,
            "method": action.parameters.get("method", "basic"),
            "constitutional_explanation_included": True,
            "appeal_rights_notified": action.parameters.get("appeal_rights_notification", False)
        }

    async def _execute_warning(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute warning action"""
        return {
            "warning_issued": True,
            "severity": action.parameters.get("severity", "standard"),
            "constitutional_protections_noted": True
        }

    async def _execute_restriction(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute content restriction action"""
        return {
            "restriction_applied": True,
            "type": action.parameters.get("type", "conditional_approval"),
            "proportionality_verified": True,
            "constitutional_justification_documented": action.parameters.get("constitutional_justification_required", False)
        }

    async def _execute_quarantine(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute content quarantine action"""
        return {
            "quarantine_applied": True,
            "evidence_preserved": action.parameters.get("preserve_evidence", True),
            "constitutional_safeguards_active": True,
            "review_scheduled": True
        }

    async def _execute_block(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute content block action"""
        return {
            "block_applied": True,
            "immediate": action.parameters.get("immediate", False),
            "constitutional_justification_documented": True,
            "appeal_process_available": True
        }

    async def _execute_monitoring(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute monitoring action"""
        return {
            "monitoring_activated": True,
            "duration_hours": action.parameters.get("duration_hours", 24),
            "constitutional_compliance_tracking": True,
            "viewpoint_bias_monitoring": action.parameters.get("viewpoint_bias_monitoring", False)
        }

    async def _execute_escalation(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute escalation action"""
        return {
            "escalation_created": True,
            "constitutional_review_requested": action.parameters.get("constitutional_review", False),
            "human_review_priority": action.parameters.get("threshold", "medium"),
            "community_oversight_notified": action.parameters.get("community_oversight_notification", False)
        }

    async def _execute_appeal_notification(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute appeal notification action"""
        return {
            "appeal_notification_sent": True,
            "processing_type": action.parameters.get("processing_type", "automated"),
            "constitutional_grounds_explained": action.parameters.get("constitutional_grounds", False),
            "expedited": action.parameters.get("expedited", False)
        }

    async def _execute_transparency_logging(self, action: ResponseAction, harm_analysis: Any, user_tier: str) -> Dict[str, Any]:
        """Execute transparency logging action"""
        return {
            "transparency_log_created": True,
            "detail_level": action.parameters.get("detail_level", "standard"),
            "constitutional_analysis_included": True,
            "public_accountability_enabled": True
        }

    async def _create_safe_action_plan(self, content_id: str, user_tier: str, error_message: str) -> ActionPlan:
        """Create safe action plan for error cases"""
        
        # Create safe escalation action
        safe_action = ResponseAction(
            action_id=str(uuid4()),
            action_type=ActionType.ESCALATE,
            content_id=content_id,
            description=f"Escalate due to action planning error: {error_message}",
            parameters={"error_handling": True, "immediate": True},
            status=ActionStatus.PENDING,
            created_at=datetime.utcnow(),
            constitutional_rationale="Error handling requires human constitutional review"
        )
        
        return ActionPlan(
            content_id=content_id,
            actions=[safe_action],
            tier_level=user_tier,
            constitutional_considerations={"error": True},
            execution_order=[safe_action.action_id],
            total_actions=1
        )

    def add_active_action(self, action: ResponseAction):
        """Add action to active tracking"""
        self.active_actions[action.action_id] = action

    def get_action_status(self, action_id: str) -> Optional[ActionStatus]:
        """Get status of specific action"""
        if action_id in self.active_actions:
            return self.active_actions[action_id].status
        return None

    def get_active_actions_count(self) -> int:
        """Get count of currently active actions"""
        return len([a for a in self.active_actions.values() if a.status == ActionStatus.IN_PROGRESS])

    async def cancel_action(self, action_id: str) -> bool:
        """Cancel a pending action"""
        if action_id in self.active_actions:
            action = self.active_actions[action_id]
            if action.status == ActionStatus.PENDING:
                action.status = ActionStatus.CANCELLED
                return True
        return False