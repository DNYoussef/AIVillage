"""
Constitutional Human Escalation System
Manages human escalation for constitutional moderation decisions requiring expert review
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
import json

logger = logging.getLogger(__name__)

class EscalationPriority(Enum):
    """Escalation priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CONSTITUTIONAL = "constitutional"

class EscalationStatus(Enum):
    """Status of escalation cases"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    APPEALED = "appealed"
    CLOSED = "closed"

class ReviewerSpecialty(Enum):
    """Reviewer specialties"""
    CONSTITUTIONAL_LAW = "constitutional_law"
    FIRST_AMENDMENT = "first_amendment"
    DUE_PROCESS = "due_process"
    VIEWPOINT_NEUTRALITY = "viewpoint_neutrality"
    CONTENT_POLICY = "content_policy"
    COMMUNITY_STANDARDS = "community_standards"
    TECHNICAL_ANALYSIS = "technical_analysis"

@dataclass
class EscalationCase:
    """Individual escalation case"""
    case_id: str
    content_id: str
    moderation_result: Dict[str, Any]
    priority: EscalationPriority
    constitutional_concerns: Dict[str, Any]
    user_tier: str
    created_at: datetime
    
    # Assignment and review
    assigned_reviewer: Optional[str] = None
    reviewer_specialty: Optional[ReviewerSpecialty] = None
    assigned_at: Optional[datetime] = None
    
    # Review process
    status: EscalationStatus = EscalationStatus.PENDING
    review_notes: List[Dict[str, Any]] = field(default_factory=list)
    constitutional_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Resolution
    final_decision: Optional[str] = None
    resolution_rationale: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    sla_deadline: Optional[datetime] = None
    escalation_source: str = "automated"
    community_oversight_required: bool = False

@dataclass
class ReviewerWorkload:
    """Reviewer workload tracking"""
    reviewer_id: str
    specialties: List[ReviewerSpecialty]
    current_cases: int
    max_capacity: int
    average_resolution_time: timedelta
    constitutional_expertise_score: float
    availability_status: str

class EscalationManager:
    """
    Manages human escalation for constitutional moderation decisions
    Ensures constitutional compliance through expert human review
    """
    
    def __init__(self):
        self.active_cases: Dict[str, EscalationCase] = {}
        self.reviewer_pool: Dict[str, ReviewerWorkload] = {}
        self.escalation_queue: List[str] = []
        self.priority_queue: Dict[EscalationPriority, List[str]] = {
            priority: [] for priority in EscalationPriority
        }
        
        # SLA requirements by priority
        self.sla_requirements = {
            EscalationPriority.CRITICAL: timedelta(hours=2),
            EscalationPriority.CONSTITUTIONAL: timedelta(hours=4),
            EscalationPriority.HIGH: timedelta(hours=12),
            EscalationPriority.MEDIUM: timedelta(hours=24),
            EscalationPriority.LOW: timedelta(hours=72)
        }
        
        # Constitutional expertise requirements
        self.constitutional_requirements = {
            "first_amendment_protected_speech": [ReviewerSpecialty.FIRST_AMENDMENT, ReviewerSpecialty.CONSTITUTIONAL_LAW],
            "due_process_concerns": [ReviewerSpecialty.DUE_PROCESS, ReviewerSpecialty.CONSTITUTIONAL_LAW],
            "viewpoint_discrimination": [ReviewerSpecialty.VIEWPOINT_NEUTRALITY, ReviewerSpecialty.FIRST_AMENDMENT],
            "prior_restraint_issues": [ReviewerSpecialty.FIRST_AMENDMENT, ReviewerSpecialty.CONSTITUTIONAL_LAW],
            "equal_protection_concerns": [ReviewerSpecialty.CONSTITUTIONAL_LAW, ReviewerSpecialty.DUE_PROCESS]
        }
        
        self._initialize_reviewer_pool()
        logger.info("Constitutional Escalation Manager initialized")

    def _initialize_reviewer_pool(self):
        """Initialize reviewer pool with constitutional experts"""
        # This would be populated from a database in production
        sample_reviewers = [
            {
                "reviewer_id": "constitutional_expert_1",
                "specialties": [ReviewerSpecialty.CONSTITUTIONAL_LAW, ReviewerSpecialty.FIRST_AMENDMENT],
                "max_capacity": 10,
                "expertise_score": 0.95
            },
            {
                "reviewer_id": "first_amendment_specialist",
                "specialties": [ReviewerSpecialty.FIRST_AMENDMENT, ReviewerSpecialty.VIEWPOINT_NEUTRALITY],
                "max_capacity": 8,
                "expertise_score": 0.90
            },
            {
                "reviewer_id": "due_process_expert", 
                "specialties": [ReviewerSpecialty.DUE_PROCESS, ReviewerSpecialty.CONSTITUTIONAL_LAW],
                "max_capacity": 12,
                "expertise_score": 0.88
            },
            {
                "reviewer_id": "content_policy_specialist",
                "specialties": [ReviewerSpecialty.CONTENT_POLICY, ReviewerSpecialty.COMMUNITY_STANDARDS],
                "max_capacity": 15,
                "expertise_score": 0.85
            }
        ]
        
        for reviewer_data in sample_reviewers:
            workload = ReviewerWorkload(
                reviewer_id=reviewer_data["reviewer_id"],
                specialties=reviewer_data["specialties"],
                current_cases=0,
                max_capacity=reviewer_data["max_capacity"],
                average_resolution_time=timedelta(hours=6),
                constitutional_expertise_score=reviewer_data["expertise_score"],
                availability_status="available"
            )
            self.reviewer_pool[reviewer_data["reviewer_id"]] = workload

    async def escalate_content(self, moderation_result: Any) -> EscalationCase:
        """
        Escalate content for human constitutional review
        
        Args:
            moderation_result: ModerationResult requiring escalation
            
        Returns:
            EscalationCase created for the content
        """
        try:
            case_id = str(uuid4())
            
            logger.info(f"Escalating content {moderation_result.content_id} as case {case_id}")
            
            # Analyze constitutional concerns for proper prioritization
            constitutional_concerns = await self._analyze_constitutional_urgency(moderation_result)
            
            # Determine escalation priority
            priority = await self._determine_escalation_priority(
                moderation_result, constitutional_concerns
            )
            
            # Create escalation case
            escalation_case = EscalationCase(
                case_id=case_id,
                content_id=moderation_result.content_id,
                moderation_result=self._serialize_moderation_result(moderation_result),
                priority=priority,
                constitutional_concerns=constitutional_concerns,
                user_tier=moderation_result.tier_level,
                created_at=datetime.utcnow(),
                sla_deadline=datetime.utcnow() + self.sla_requirements[priority],
                community_oversight_required=self._requires_community_oversight(moderation_result)
            )
            
            # Add to active cases and queues
            self.active_cases[case_id] = escalation_case
            self.escalation_queue.append(case_id)
            self.priority_queue[priority].append(case_id)
            
            # Attempt immediate assignment for critical cases
            if priority in [EscalationPriority.CRITICAL, EscalationPriority.CONSTITUTIONAL]:
                await self._attempt_immediate_assignment(escalation_case)
            
            # Schedule assignment for other cases
            else:
                await self._schedule_assignment(escalation_case)
            
            logger.info(f"Escalation case {case_id} created with priority {priority.value}")
            return escalation_case
            
        except Exception as e:
            logger.error(f"Failed to escalate content {moderation_result.content_id}: {str(e)}")
            raise

    async def _analyze_constitutional_urgency(self, moderation_result: Any) -> Dict[str, Any]:
        """Analyze constitutional urgency factors"""
        concerns = {
            "urgency_score": 0.0,
            "constitutional_violations": [],
            "protected_speech_risk": False,
            "viewpoint_discrimination_risk": False,
            "due_process_deficiencies": [],
            "prior_restraint_concerns": [],
            "time_sensitive_factors": []
        }
        
        harm_analysis = moderation_result.harm_analysis
        constitutional_analysis = harm_analysis.constitutional_concerns
        
        # Check for First Amendment protected speech
        if constitutional_analysis.get("first_amendment", {}).get("requires_heightened_scrutiny"):
            concerns["protected_speech_risk"] = True
            concerns["urgency_score"] += 0.3
            concerns["constitutional_violations"].append("potential_first_amendment_violation")
        
        # Check for viewpoint discrimination
        if harm_analysis.viewpoint_bias_score > 0.4:
            concerns["viewpoint_discrimination_risk"] = True
            concerns["urgency_score"] += 0.25
            concerns["constitutional_violations"].append("viewpoint_discrimination_risk")
        
        # Check for due process concerns
        if constitutional_analysis.get("due_process", {}).get("low_confidence"):
            concerns["due_process_deficiencies"].append("low_confidence_decision")
            concerns["urgency_score"] += 0.2
        
        # Check for prior restraint
        if moderation_result.decision.value in ["quarantine", "block"]:
            if not constitutional_analysis.get("compelling_interest"):
                concerns["prior_restraint_concerns"].append("insufficient_justification")
                concerns["urgency_score"] += 0.35
        
        # Time sensitive factors
        if moderation_result.tier_level == "Gold":
            concerns["time_sensitive_factors"].append("gold_tier_user")
            concerns["urgency_score"] += 0.1
        
        if any("political" in cat.lower() for cat in harm_analysis.harm_categories):
            concerns["time_sensitive_factors"].append("political_content")
            concerns["urgency_score"] += 0.15
        
        return concerns

    async def _determine_escalation_priority(
        self, 
        moderation_result: Any, 
        constitutional_concerns: Dict[str, Any]
    ) -> EscalationPriority:
        """Determine appropriate escalation priority"""
        
        urgency_score = constitutional_concerns["urgency_score"]
        
        # Constitutional priority for clear constitutional issues
        if (constitutional_concerns["protected_speech_risk"] and 
            moderation_result.decision.value in ["block", "quarantine"]):
            return EscalationPriority.CONSTITUTIONAL
        
        # Critical for urgent constitutional violations
        if urgency_score >= 0.7:
            return EscalationPriority.CRITICAL
        
        # High for significant constitutional concerns
        elif urgency_score >= 0.5:
            return EscalationPriority.HIGH
        
        # Medium for moderate concerns
        elif urgency_score >= 0.3:
            return EscalationPriority.MEDIUM
        
        # Low for routine escalations
        else:
            return EscalationPriority.LOW

    def _serialize_moderation_result(self, moderation_result: Any) -> Dict[str, Any]:
        """Serialize moderation result for storage"""
        return {
            "content_id": moderation_result.content_id,
            "decision": moderation_result.decision.value,
            "harm_analysis": {
                "harm_level": moderation_result.harm_analysis.harm_level,
                "harm_categories": moderation_result.harm_analysis.harm_categories,
                "confidence_score": moderation_result.harm_analysis.confidence_score,
                "constitutional_concerns": moderation_result.harm_analysis.constitutional_concerns,
                "viewpoint_bias_score": moderation_result.harm_analysis.viewpoint_bias_score
            },
            "policy_rationale": moderation_result.policy_rationale,
            "response_actions": moderation_result.response_actions,
            "tier_level": moderation_result.tier_level,
            "audit_trail": moderation_result.audit_trail,
            "transparency_score": moderation_result.transparency_score
        }

    def _requires_community_oversight(self, moderation_result: Any) -> bool:
        """Determine if case requires community oversight"""
        
        # Gold tier constitutional decisions require community oversight
        if (moderation_result.tier_level == "Gold" and 
            moderation_result.harm_analysis.constitutional_concerns.get("first_amendment")):
            return True
        
        # High-profile or politically sensitive content
        if any("political" in cat.lower() for cat in moderation_result.harm_analysis.harm_categories):
            return True
        
        # High viewpoint bias scores
        if moderation_result.harm_analysis.viewpoint_bias_score > 0.5:
            return True
        
        return False

    async def _attempt_immediate_assignment(self, escalation_case: EscalationCase) -> bool:
        """Attempt immediate assignment for critical cases"""
        
        # Find best available reviewer
        best_reviewer = await self._find_best_reviewer(escalation_case)
        
        if best_reviewer:
            await self._assign_case_to_reviewer(escalation_case, best_reviewer)
            return True
        
        # If no reviewer available, flag for urgent assignment
        escalation_case.escalation_source = "urgent_queue"
        return False

    async def _schedule_assignment(self, escalation_case: EscalationCase):
        """Schedule case for assignment based on priority and SLA"""
        
        # This would integrate with a task scheduler in production
        # For now, we'll mark it as scheduled
        escalation_case.escalation_source = "scheduled_assignment"
        
        logger.info(f"Case {escalation_case.case_id} scheduled for assignment "
                   f"with SLA deadline {escalation_case.sla_deadline}")

    async def _find_best_reviewer(self, escalation_case: EscalationCase) -> Optional[str]:
        """Find the best available reviewer for the case"""
        
        constitutional_concerns = escalation_case.constitutional_concerns
        required_specialties = set()
        
        # Determine required specialties based on constitutional concerns
        if constitutional_concerns.get("protected_speech_risk"):
            required_specialties.update(self.constitutional_requirements["first_amendment_protected_speech"])
        
        if constitutional_concerns.get("viewpoint_discrimination_risk"):
            required_specialties.update(self.constitutional_requirements["viewpoint_discrimination"])
        
        if constitutional_concerns.get("due_process_deficiencies"):
            required_specialties.update(self.constitutional_requirements["due_process_concerns"])
        
        if constitutional_concerns.get("prior_restraint_concerns"):
            required_specialties.update(self.constitutional_requirements["prior_restraint_issues"])
        
        # Default to constitutional law if no specific requirements
        if not required_specialties:
            required_specialties.add(ReviewerSpecialty.CONSTITUTIONAL_LAW)
        
        # Find available reviewers with matching specialties
        candidate_reviewers = []
        
        for reviewer_id, workload in self.reviewer_pool.items():
            if workload.availability_status != "available":
                continue
                
            if workload.current_cases >= workload.max_capacity:
                continue
            
            # Check specialty match
            specialty_match = len(set(workload.specialties) & required_specialties)
            if specialty_match == 0:
                continue
            
            candidate_reviewers.append({
                "reviewer_id": reviewer_id,
                "workload": workload,
                "specialty_match_score": specialty_match / len(required_specialties),
                "capacity_score": (workload.max_capacity - workload.current_cases) / workload.max_capacity,
                "expertise_score": workload.constitutional_expertise_score
            })
        
        if not candidate_reviewers:
            return None
        
        # Score and rank candidates
        for candidate in candidate_reviewers:
            candidate["total_score"] = (
                candidate["specialty_match_score"] * 0.4 +
                candidate["capacity_score"] * 0.3 +
                candidate["expertise_score"] * 0.3
            )
        
        # Return best candidate
        best_candidate = max(candidate_reviewers, key=lambda x: x["total_score"])
        return best_candidate["reviewer_id"]

    async def _assign_case_to_reviewer(self, escalation_case: EscalationCase, reviewer_id: str):
        """Assign case to specific reviewer"""
        
        escalation_case.assigned_reviewer = reviewer_id
        escalation_case.assigned_at = datetime.utcnow()
        escalation_case.status = EscalationStatus.ASSIGNED
        
        # Update reviewer workload
        workload = self.reviewer_pool[reviewer_id]
        workload.current_cases += 1
        
        # Determine reviewer specialty for this case
        constitutional_concerns = escalation_case.constitutional_concerns
        reviewer_specialties = workload.specialties
        
        if ReviewerSpecialty.FIRST_AMENDMENT in reviewer_specialties and constitutional_concerns.get("protected_speech_risk"):
            escalation_case.reviewer_specialty = ReviewerSpecialty.FIRST_AMENDMENT
        elif ReviewerSpecialty.DUE_PROCESS in reviewer_specialties and constitutional_concerns.get("due_process_deficiencies"):
            escalation_case.reviewer_specialty = ReviewerSpecialty.DUE_PROCESS
        elif ReviewerSpecialty.VIEWPOINT_NEUTRALITY in reviewer_specialties and constitutional_concerns.get("viewpoint_discrimination_risk"):
            escalation_case.reviewer_specialty = ReviewerSpecialty.VIEWPOINT_NEUTRALITY
        else:
            escalation_case.reviewer_specialty = reviewer_specialties[0]  # Default to first specialty
        
        logger.info(f"Case {escalation_case.case_id} assigned to {reviewer_id} "
                   f"with specialty {escalation_case.reviewer_specialty.value}")

    async def process_reviewer_decision(
        self, 
        case_id: str, 
        reviewer_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process reviewer decision for escalated case
        
        Args:
            case_id: Escalation case ID
            reviewer_decision: Decision from human reviewer
            
        Returns:
            Processing result
        """
        if case_id not in self.active_cases:
            raise ValueError(f"Case {case_id} not found")
        
        escalation_case = self.active_cases[case_id]
        
        try:
            # Update case with decision
            escalation_case.status = EscalationStatus.RESOLVED
            escalation_case.final_decision = reviewer_decision.get("decision")
            escalation_case.resolution_rationale = reviewer_decision.get("rationale")
            escalation_case.resolved_at = datetime.utcnow()
            
            # Store constitutional analysis
            escalation_case.constitutional_analysis = reviewer_decision.get("constitutional_analysis", {})
            
            # Add review notes
            review_note = {
                "timestamp": datetime.utcnow().isoformat(),
                "reviewer": escalation_case.assigned_reviewer,
                "specialty": escalation_case.reviewer_specialty.value if escalation_case.reviewer_specialty else "general",
                "decision": reviewer_decision.get("decision"),
                "rationale": reviewer_decision.get("rationale"),
                "constitutional_findings": reviewer_decision.get("constitutional_analysis", {})
            }
            escalation_case.review_notes.append(review_note)
            
            # Update reviewer workload
            if escalation_case.assigned_reviewer:
                workload = self.reviewer_pool[escalation_case.assigned_reviewer]
                workload.current_cases = max(0, workload.current_cases - 1)
                
                # Update average resolution time
                resolution_time = escalation_case.resolved_at - escalation_case.assigned_at
                current_avg = workload.average_resolution_time
                workload.average_resolution_time = (current_avg + resolution_time) / 2
            
            # Remove from queues
            self._remove_from_queues(case_id)
            
            result = {
                "case_id": case_id,
                "resolved": True,
                "final_decision": escalation_case.final_decision,
                "constitutional_compliance": self._verify_constitutional_compliance(escalation_case),
                "resolution_time": (escalation_case.resolved_at - escalation_case.created_at).total_seconds() / 3600,
                "sla_met": escalation_case.resolved_at <= escalation_case.sla_deadline,
                "community_oversight_required": escalation_case.community_oversight_required
            }
            
            logger.info(f"Case {case_id} resolved: {escalation_case.final_decision}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process reviewer decision for case {case_id}: {str(e)}")
            escalation_case.status = EscalationStatus.PENDING  # Reset status
            raise

    def _verify_constitutional_compliance(self, escalation_case: EscalationCase) -> bool:
        """Verify constitutional compliance of reviewer decision"""
        
        constitutional_analysis = escalation_case.constitutional_analysis
        
        # Check for required constitutional considerations
        required_checks = []
        
        if escalation_case.constitutional_concerns.get("protected_speech_risk"):
            required_checks.append("first_amendment_analysis")
        
        if escalation_case.constitutional_concerns.get("viewpoint_discrimination_risk"):
            required_checks.append("viewpoint_neutrality_analysis")
        
        if escalation_case.constitutional_concerns.get("due_process_deficiencies"):
            required_checks.append("due_process_analysis")
        
        if escalation_case.constitutional_concerns.get("prior_restraint_concerns"):
            required_checks.append("prior_restraint_analysis")
        
        # Verify all required analyses are present
        for check in required_checks:
            if not constitutional_analysis.get(check):
                logger.warning(f"Missing {check} in case {escalation_case.case_id}")
                return False
        
        return True

    def _remove_from_queues(self, case_id: str):
        """Remove case from all queues"""
        
        # Remove from main queue
        if case_id in self.escalation_queue:
            self.escalation_queue.remove(case_id)
        
        # Remove from priority queues
        for priority_list in self.priority_queue.values():
            if case_id in priority_list:
                priority_list.remove(case_id)

    async def get_queue_size(self) -> Dict[str, int]:
        """Get current queue sizes"""
        return {
            "total_queue": len(self.escalation_queue),
            "priority_queues": {
                priority.value: len(queue) 
                for priority, queue in self.priority_queue.items()
            },
            "active_cases": len([
                case for case in self.active_cases.values() 
                if case.status in [EscalationStatus.PENDING, EscalationStatus.ASSIGNED, EscalationStatus.IN_REVIEW]
            ])
        }

    async def get_case_status(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific case"""
        if case_id not in self.active_cases:
            return None
        
        escalation_case = self.active_cases[case_id]
        
        return {
            "case_id": case_id,
            "content_id": escalation_case.content_id,
            "status": escalation_case.status.value,
            "priority": escalation_case.priority.value,
            "created_at": escalation_case.created_at.isoformat(),
            "assigned_reviewer": escalation_case.assigned_reviewer,
            "reviewer_specialty": escalation_case.reviewer_specialty.value if escalation_case.reviewer_specialty else None,
            "sla_deadline": escalation_case.sla_deadline.isoformat() if escalation_case.sla_deadline else None,
            "constitutional_concerns": escalation_case.constitutional_concerns,
            "community_oversight_required": escalation_case.community_oversight_required
        }

    async def get_reviewer_workloads(self) -> Dict[str, Any]:
        """Get current reviewer workloads"""
        return {
            reviewer_id: {
                "specialties": [s.value for s in workload.specialties],
                "current_cases": workload.current_cases,
                "max_capacity": workload.max_capacity,
                "utilization": workload.current_cases / workload.max_capacity,
                "average_resolution_hours": workload.average_resolution_time.total_seconds() / 3600,
                "expertise_score": workload.constitutional_expertise_score,
                "availability_status": workload.availability_status
            }
            for reviewer_id, workload in self.reviewer_pool.items()
        }

    async def get_escalation_metrics(self) -> Dict[str, Any]:
        """Get escalation system metrics"""
        
        # Calculate metrics from active cases
        all_cases = list(self.active_cases.values())
        total_cases = len(all_cases)
        
        if total_cases == 0:
            return {"no_cases": True}
        
        # Status distribution
        status_distribution = {}
        for status in EscalationStatus:
            status_distribution[status.value] = len([
                case for case in all_cases if case.status == status
            ])
        
        # Priority distribution  
        priority_distribution = {}
        for priority in EscalationPriority:
            priority_distribution[priority.value] = len([
                case for case in all_cases if case.priority == priority
            ])
        
        # Resolution metrics
        resolved_cases = [case for case in all_cases if case.status == EscalationStatus.RESOLVED]
        if resolved_cases:
            resolution_times = [
                (case.resolved_at - case.created_at).total_seconds() / 3600
                for case in resolved_cases
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            sla_met = len([case for case in resolved_cases if case.resolved_at <= case.sla_deadline])
            sla_compliance = sla_met / len(resolved_cases)
        else:
            avg_resolution_time = 0
            sla_compliance = 0
        
        # Constitutional compliance
        constitutional_compliant = len([
            case for case in resolved_cases 
            if self._verify_constitutional_compliance(case)
        ])
        constitutional_compliance = constitutional_compliant / len(resolved_cases) if resolved_cases else 0
        
        return {
            "total_cases": total_cases,
            "status_distribution": status_distribution,
            "priority_distribution": priority_distribution,
            "resolution_metrics": {
                "average_resolution_time_hours": avg_resolution_time,
                "sla_compliance_rate": sla_compliance,
                "constitutional_compliance_rate": constitutional_compliance
            },
            "queue_health": {
                "pending_cases": status_distribution.get("pending", 0),
                "overdue_cases": len([
                    case for case in all_cases 
                    if case.sla_deadline and datetime.utcnow() > case.sla_deadline
                ])
            }
        }

    async def shutdown(self):
        """Gracefully shutdown escalation manager"""
        logger.info("Shutting down Constitutional Escalation Manager")
        
        # Save active cases and metrics
        pending_cases = len([
            case for case in self.active_cases.values() 
            if case.status != EscalationStatus.RESOLVED
        ])
        
        if pending_cases > 0:
            logger.warning(f"Shutting down with {pending_cases} pending escalation cases")
        
        # Clear queues
        self.escalation_queue.clear()
        for queue in self.priority_queue.values():
            queue.clear()