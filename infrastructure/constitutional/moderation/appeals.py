"""
Constitutional Appeals System
Manages constitutional appeals process with community oversight and due process protections
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4

logger = logging.getLogger(__name__)

class AppealStatus(Enum):
    """Appeal processing status"""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    COMMUNITY_REVIEW = "community_review"
    CONSTITUTIONAL_REVIEW = "constitutional_review"
    APPROVED = "approved"
    DENIED = "denied"
    PARTIALLY_GRANTED = "partially_granted"
    CLOSED = "closed"

class AppealType(Enum):
    """Types of constitutional appeals"""
    FIRST_AMENDMENT = "first_amendment"
    DUE_PROCESS = "due_process"
    VIEWPOINT_DISCRIMINATION = "viewpoint_discrimination"
    PRIOR_RESTRAINT = "prior_restraint"
    EQUAL_PROTECTION = "equal_protection"
    PROCEDURAL_ERROR = "procedural_error"
    NEW_EVIDENCE = "new_evidence"

class ReviewerType(Enum):
    """Types of appeal reviewers"""
    AUTOMATED = "automated"
    CONSTITUTIONAL_EXPERT = "constitutional_expert"
    COMMUNITY_PANEL = "community_panel"
    OVERSIGHT_BOARD = "oversight_board"

@dataclass
class AppealSubmission:
    """Appeal submission from user"""
    appeal_id: str
    content_id: str
    original_decision: str
    user_tier: str
    appeal_type: AppealType
    constitutional_grounds: str
    evidence_provided: Dict[str, Any]
    user_statement: str
    submitted_at: datetime
    expedited_request: bool = False

@dataclass
class AppealCase:
    """Complete appeal case"""
    appeal_id: str
    submission: AppealSubmission
    original_moderation: Dict[str, Any]
    status: AppealStatus
    assigned_reviewers: List[str] = field(default_factory=list)
    
    # Review process
    automated_review_result: Optional[Dict[str, Any]] = None
    constitutional_review_result: Optional[Dict[str, Any]] = None
    community_review_result: Optional[Dict[str, Any]] = None
    
    # Decision
    final_decision: Optional[str] = None
    decision_rationale: str = ""
    constitutional_analysis: Dict[str, Any] = field(default_factory=dict)
    remedial_actions: List[str] = field(default_factory=list)
    
    # Metadata
    review_deadline: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    community_oversight_involved: bool = False

@dataclass
class CommunityReviewer:
    """Community oversight reviewer"""
    reviewer_id: str
    tier_level: str
    constitutional_knowledge_score: float
    review_history: int
    bias_score: float
    specialties: List[str]
    active_cases: int
    max_capacity: int

class AppealsManager:
    """
    Constitutional appeals management system
    Handles due process appeals with community oversight and constitutional review
    """
    
    def __init__(self):
        self.active_appeals: Dict[str, AppealCase] = {}
        self.community_reviewers: Dict[str, CommunityReviewer] = {}
        self.constitutional_experts: Dict[str, Dict[str, Any]] = {}
        
        # Appeal processing SLAs
        self.appeal_slas = {
            "Bronze": {
                "standard": timedelta(days=5),
                "expedited": timedelta(days=2)
            },
            "Silver": {
                "standard": timedelta(days=3),
                "expedited": timedelta(days=1)
            },
            "Gold": {
                "standard": timedelta(days=2),
                "expedited": timedelta(hours=12)
            }
        }
        
        # Review requirements by appeal type
        self.review_requirements = {
            AppealType.FIRST_AMENDMENT: {
                "requires_constitutional_expert": True,
                "requires_community_review": True,
                "heightened_scrutiny": True,
                "burden_of_proof": "clear_and_convincing"
            },
            AppealType.VIEWPOINT_DISCRIMINATION: {
                "requires_constitutional_expert": True,
                "requires_community_review": True,
                "bias_detection_required": True,
                "viewpoint_neutrality_analysis": True
            },
            AppealType.DUE_PROCESS: {
                "requires_constitutional_expert": True,
                "requires_community_review": False,
                "procedural_analysis_required": True
            },
            AppealType.PRIOR_RESTRAINT: {
                "requires_constitutional_expert": True,
                "requires_community_review": True,
                "compelling_interest_analysis": True,
                "narrow_tailoring_required": True
            },
            AppealType.PROCEDURAL_ERROR: {
                "requires_constitutional_expert": False,
                "requires_community_review": False,
                "automated_review_sufficient": True
            }
        }
        
        self._initialize_review_panels()
        logger.info("Constitutional Appeals Manager initialized")

    def _initialize_review_panels(self):
        """Initialize community review panels and constitutional experts"""
        
        # Initialize constitutional experts (would be from database in production)
        sample_experts = [
            {
                "expert_id": "constitutional_scholar_1",
                "specialties": ["first_amendment", "due_process"],
                "credentials": "Constitutional Law Professor",
                "bias_score": 0.05,
                "expertise_level": 0.95
            },
            {
                "expert_id": "civil_rights_attorney",
                "specialties": ["equal_protection", "viewpoint_discrimination"],
                "credentials": "Civil Rights Attorney",
                "bias_score": 0.08,
                "expertise_level": 0.92
            },
            {
                "expert_id": "first_amendment_specialist",
                "specialties": ["first_amendment", "prior_restraint"],
                "credentials": "First Amendment Institute",
                "bias_score": 0.03,
                "expertise_level": 0.98
            }
        ]
        
        for expert_data in sample_experts:
            self.constitutional_experts[expert_data["expert_id"]] = expert_data
        
        # Initialize community reviewers (sample)
        sample_community_reviewers = [
            {
                "reviewer_id": "community_gold_1",
                "tier_level": "Gold",
                "constitutional_knowledge_score": 0.85,
                "review_history": 150,
                "bias_score": 0.12,
                "specialties": ["political_content", "free_speech"]
            },
            {
                "reviewer_id": "community_silver_1",
                "tier_level": "Silver", 
                "constitutional_knowledge_score": 0.75,
                "review_history": 75,
                "bias_score": 0.18,
                "specialties": ["community_standards"]
            }
        ]
        
        for reviewer_data in sample_community_reviewers:
            reviewer = CommunityReviewer(
                reviewer_id=reviewer_data["reviewer_id"],
                tier_level=reviewer_data["tier_level"],
                constitutional_knowledge_score=reviewer_data["constitutional_knowledge_score"],
                review_history=reviewer_data["review_history"],
                bias_score=reviewer_data["bias_score"],
                specialties=reviewer_data["specialties"],
                active_cases=0,
                max_capacity=5 if reviewer_data["tier_level"] == "Gold" else 3
            )
            self.community_reviewers[reviewer_data["reviewer_id"]] = reviewer

    async def process_appeal(
        self,
        content_id: str,
        appeal_reason: str,
        user_tier: str,
        additional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process constitutional appeal for moderation decision
        
        Args:
            content_id: Content that was moderated
            appeal_reason: User's reason for appeal
            user_tier: User's tier level
            additional_context: Additional appeal context
            
        Returns:
            Appeal processing result
        """
        try:
            appeal_id = str(uuid4())
            
            logger.info(f"Processing appeal {appeal_id} for content {content_id} from {user_tier} user")
            
            # Determine appeal type from reason
            appeal_type = await self._determine_appeal_type(appeal_reason, additional_context or {})
            
            # Create appeal submission
            submission = AppealSubmission(
                appeal_id=appeal_id,
                content_id=content_id,
                original_decision="",  # Would be populated from original moderation
                user_tier=user_tier,
                appeal_type=appeal_type,
                constitutional_grounds=appeal_reason,
                evidence_provided=additional_context or {},
                user_statement=appeal_reason,
                submitted_at=datetime.utcnow(),
                expedited_request=additional_context.get("expedited", False) if additional_context else False
            )
            
            # Retrieve original moderation decision
            original_moderation = await self._retrieve_original_moderation(content_id)
            
            # Determine review deadline
            sla_type = "expedited" if submission.expedited_request else "standard"
            review_deadline = datetime.utcnow() + self.appeal_slas[user_tier][sla_type]
            
            # Create appeal case
            appeal_case = AppealCase(
                appeal_id=appeal_id,
                submission=submission,
                original_moderation=original_moderation,
                status=AppealStatus.SUBMITTED,
                review_deadline=review_deadline
            )
            
            # Add to active appeals
            self.active_appeals[appeal_id] = appeal_case
            
            # Begin review process
            await self._initiate_review_process(appeal_case)
            
            result = {
                "appeal_id": appeal_id,
                "status": appeal_case.status.value,
                "appeal_type": appeal_type.value,
                "review_deadline": review_deadline.isoformat(),
                "review_process_initiated": True,
                "constitutional_review_required": self._requires_constitutional_review(appeal_type),
                "community_review_required": self._requires_community_review(appeal_type, user_tier)
            }
            
            logger.info(f"Appeal {appeal_id} submitted with type {appeal_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process appeal for content {content_id}: {str(e)}")
            return {
                "appeal_submitted": False,
                "error": str(e),
                "fallback_process_available": True
            }

    async def _determine_appeal_type(self, appeal_reason: str, context: Dict[str, Any]) -> AppealType:
        """Determine the type of constitutional appeal based on user's reason"""
        
        reason_lower = appeal_reason.lower()
        
        # First Amendment appeals
        if any(keyword in reason_lower for keyword in [
            "free speech", "first amendment", "protected speech", "political", "religious", "artistic"
        ]):
            return AppealType.FIRST_AMENDMENT
        
        # Viewpoint discrimination appeals
        if any(keyword in reason_lower for keyword in [
            "bias", "discrimination", "viewpoint", "political bias", "unfair"
        ]):
            return AppealType.VIEWPOINT_DISCRIMINATION
        
        # Due process appeals
        if any(keyword in reason_lower for keyword in [
            "due process", "unfair process", "no notice", "no explanation"
        ]):
            return AppealType.DUE_PROCESS
        
        # Prior restraint appeals
        if any(keyword in reason_lower for keyword in [
            "prior restraint", "censorship", "blocked unfairly", "suppressed"
        ]):
            return AppealType.PRIOR_RESTRAINT
        
        # Procedural error appeals
        if any(keyword in reason_lower for keyword in [
            "error", "mistake", "wrong", "misclassified"
        ]):
            return AppealType.PROCEDURAL_ERROR
        
        # New evidence appeals
        if any(keyword in reason_lower for keyword in [
            "new evidence", "additional context", "misunderstood"
        ]):
            return AppealType.NEW_EVIDENCE
        
        # Default to due process if unclear
        return AppealType.DUE_PROCESS

    async def _retrieve_original_moderation(self, content_id: str) -> Dict[str, Any]:
        """Retrieve original moderation decision for appeal review"""
        # This would query the moderation database in production
        return {
            "content_id": content_id,
            "decision": "restrict",
            "harm_level": "H2",
            "confidence": 0.75,
            "rationale": "Moderate harm potential detected",
            "timestamp": "2023-01-01T12:00:00Z"
        }

    async def _initiate_review_process(self, appeal_case: AppealCase):
        """Initiate the appropriate review process based on appeal type"""
        
        appeal_type = appeal_case.submission.appeal_type
        requirements = self.review_requirements[appeal_type]
        
        appeal_case.status = AppealStatus.UNDER_REVIEW
        
        # Start with automated review for eligible cases
        if requirements.get("automated_review_sufficient") and appeal_type == AppealType.PROCEDURAL_ERROR:
            await self._conduct_automated_review(appeal_case)
        
        # Initiate constitutional expert review if required
        elif requirements.get("requires_constitutional_expert"):
            await self._assign_constitutional_expert(appeal_case)
        
        # Initiate community review if required
        if requirements.get("requires_community_review"):
            await self._assign_community_reviewers(appeal_case)
            
        logger.info(f"Review process initiated for appeal {appeal_case.appeal_id}")

    async def _conduct_automated_review(self, appeal_case: AppealCase):
        """Conduct automated review for procedural appeals"""
        
        # Analyze original decision for procedural errors
        original_moderation = appeal_case.original_moderation
        
        automated_analysis = {
            "confidence_threshold_met": original_moderation.get("confidence", 0) >= 0.7,
            "rationale_provided": bool(original_moderation.get("rationale")),
            "appeal_eligible_decision": original_moderation.get("decision") in ["restrict", "quarantine", "block"],
            "proper_tier_handling": True,  # Would check against tier policies
            "constitutional_safeguards_applied": True  # Would verify safeguards
        }
        
        # Determine if procedural error occurred
        procedural_errors = []
        
        if not automated_analysis["confidence_threshold_met"]:
            procedural_errors.append("decision_made_with_insufficient_confidence")
            
        if not automated_analysis["rationale_provided"]:
            procedural_errors.append("inadequate_rationale_provided")
        
        # Create automated review result
        appeal_case.automated_review_result = {
            "review_completed_at": datetime.utcnow().isoformat(),
            "procedural_errors_found": procedural_errors,
            "analysis": automated_analysis,
            "recommendation": "grant_appeal" if procedural_errors else "deny_appeal",
            "confidence": 0.85 if not procedural_errors else 0.95
        }
        
        # If no constitutional expert required, finalize
        if not self.review_requirements[appeal_case.submission.appeal_type].get("requires_constitutional_expert"):
            await self._finalize_appeal_decision(appeal_case)

    async def _assign_constitutional_expert(self, appeal_case: AppealCase):
        """Assign constitutional expert for constitutional appeals"""
        
        appeal_type = appeal_case.submission.appeal_type
        required_specialties = []
        
        if appeal_type == AppealType.FIRST_AMENDMENT:
            required_specialties = ["first_amendment"]
        elif appeal_type == AppealType.VIEWPOINT_DISCRIMINATION:
            required_specialties = ["viewpoint_discrimination", "equal_protection"]
        elif appeal_type == AppealType.DUE_PROCESS:
            required_specialties = ["due_process"]
        elif appeal_type == AppealType.PRIOR_RESTRAINT:
            required_specialties = ["first_amendment", "prior_restraint"]
        
        # Find best matching expert
        best_expert = None
        best_score = 0
        
        for expert_id, expert_data in self.constitutional_experts.items():
            if expert_id in appeal_case.assigned_reviewers:
                continue  # Already assigned
            
            specialty_match = len(set(expert_data["specialties"]) & set(required_specialties))
            if specialty_match == 0:
                continue
            
            score = (specialty_match / len(required_specialties)) * expert_data["expertise_level"] * (1 - expert_data["bias_score"])
            
            if score > best_score:
                best_score = score
                best_expert = expert_id
        
        if best_expert:
            appeal_case.assigned_reviewers.append(best_expert)
            appeal_case.status = AppealStatus.CONSTITUTIONAL_REVIEW
            
            # Initiate constitutional review (would be async task in production)
            await self._conduct_constitutional_review(appeal_case, best_expert)
        else:
            logger.warning(f"No available constitutional expert for appeal {appeal_case.appeal_id}")

    async def _assign_community_reviewers(self, appeal_case: AppealCase):
        """Assign community reviewers for community oversight"""
        
        user_tier = appeal_case.submission.user_tier
        appeal_type = appeal_case.submission.appeal_type
        
        # Determine number of reviewers needed
        reviewers_needed = 3 if user_tier == "Gold" else 2
        
        # Find eligible community reviewers
        eligible_reviewers = []
        
        for reviewer_id, reviewer in self.community_reviewers.items():
            if reviewer.active_cases >= reviewer.max_capacity:
                continue
                
            # Prefer reviewers with relevant specialties
            relevance_score = 0
            if appeal_type == AppealType.FIRST_AMENDMENT and "free_speech" in reviewer.specialties:
                relevance_score += 0.3
            if appeal_type == AppealType.VIEWPOINT_DISCRIMINATION and "political_content" in reviewer.specialties:
                relevance_score += 0.3
            
            # Calculate overall score
            total_score = (
                reviewer.constitutional_knowledge_score * 0.4 +
                (1 - reviewer.bias_score) * 0.3 +
                (reviewer.review_history / 200) * 0.2 +  # Cap at 200 reviews
                relevance_score * 0.1
            )
            
            eligible_reviewers.append((reviewer_id, total_score))
        
        # Select top reviewers
        eligible_reviewers.sort(key=lambda x: x[1], reverse=True)
        selected_reviewers = eligible_reviewers[:reviewers_needed]
        
        for reviewer_id, _ in selected_reviewers:
            appeal_case.assigned_reviewers.append(reviewer_id)
            self.community_reviewers[reviewer_id].active_cases += 1
        
        if len(selected_reviewers) >= reviewers_needed:
            appeal_case.status = AppealStatus.COMMUNITY_REVIEW
            appeal_case.community_oversight_involved = True
            
            # Initiate community review
            await self._conduct_community_review(appeal_case, [r[0] for r in selected_reviewers])
        else:
            logger.warning(f"Insufficient community reviewers for appeal {appeal_case.appeal_id}")

    async def _conduct_constitutional_review(self, appeal_case: AppealCase, expert_id: str):
        """Conduct constitutional expert review"""
        
        expert_data = self.constitutional_experts[expert_id]
        appeal_type = appeal_case.submission.appeal_type
        
        # Simulate constitutional analysis
        constitutional_analysis = await self._perform_constitutional_analysis(
            appeal_case, expert_data["specialties"]
        )
        
        # Create constitutional review result
        appeal_case.constitutional_review_result = {
            "expert_id": expert_id,
            "expert_credentials": expert_data["credentials"],
            "review_completed_at": datetime.utcnow().isoformat(),
            "constitutional_analysis": constitutional_analysis,
            "recommendation": constitutional_analysis.get("recommendation", "deny_appeal"),
            "rationale": constitutional_analysis.get("expert_rationale", "Constitutional review completed"),
            "confidence": constitutional_analysis.get("confidence", 0.80)
        }
        
        logger.info(f"Constitutional review completed for appeal {appeal_case.appeal_id} by {expert_id}")

    async def _perform_constitutional_analysis(self, appeal_case: AppealCase, expert_specialties: List[str]) -> Dict[str, Any]:
        """Perform detailed constitutional analysis of the appeal"""
        
        appeal_type = appeal_case.submission.appeal_type
        original_decision = appeal_case.original_moderation.get("decision")
        user_grounds = appeal_case.submission.constitutional_grounds
        
        analysis = {
            "constitutional_principles_analyzed": [],
            "violations_found": [],
            "recommendation": "deny_appeal",
            "confidence": 0.75
        }
        
        # First Amendment analysis
        if appeal_type == AppealType.FIRST_AMENDMENT or "first_amendment" in expert_specialties:
            first_amendment_analysis = {
                "protected_speech_category": self._analyze_protected_speech(user_grounds),
                "government_interest": self._analyze_government_interest(appeal_case.original_moderation),
                "narrow_tailoring": self._analyze_narrow_tailoring(appeal_case.original_moderation),
                "less_restrictive_alternatives": self._identify_alternatives(original_decision)
            }
            
            analysis["constitutional_principles_analyzed"].append("first_amendment")
            
            if (first_amendment_analysis["protected_speech_category"] and 
                not first_amendment_analysis["narrow_tailoring"]):
                analysis["violations_found"].append("overbroad_restriction_of_protected_speech")
                analysis["recommendation"] = "grant_appeal"
                analysis["confidence"] = 0.85
        
        # Due process analysis
        if appeal_type == AppealType.DUE_PROCESS or "due_process" in expert_specialties:
            due_process_analysis = {
                "notice_adequate": bool(appeal_case.original_moderation.get("rationale")),
                "opportunity_to_respond": True,  # Appeal process itself
                "proportionality": self._analyze_proportionality(appeal_case.original_moderation),
                "procedural_safeguards": True  # Would check actual safeguards
            }
            
            analysis["constitutional_principles_analyzed"].append("due_process")
            
            if not due_process_analysis["notice_adequate"]:
                analysis["violations_found"].append("inadequate_notice")
                analysis["recommendation"] = "grant_appeal"
        
        # Viewpoint discrimination analysis
        if appeal_type == AppealType.VIEWPOINT_DISCRIMINATION:
            viewpoint_analysis = {
                "content_neutral": self._analyze_content_neutrality(appeal_case),
                "viewpoint_neutral": self._analyze_viewpoint_neutrality(appeal_case),
                "equal_treatment": True  # Would compare similar cases
            }
            
            analysis["constitutional_principles_analyzed"].append("viewpoint_neutrality")
            
            if not viewpoint_analysis["viewpoint_neutral"]:
                analysis["violations_found"].append("viewpoint_discrimination")
                analysis["recommendation"] = "grant_appeal"
                analysis["confidence"] = 0.90
        
        # Generate expert rationale
        analysis["expert_rationale"] = self._generate_expert_rationale(analysis, appeal_case)
        
        return analysis

    def _analyze_protected_speech(self, user_grounds: str) -> bool:
        """Analyze if content involves protected speech categories"""
        protected_indicators = [
            "political", "religious", "artistic", "scientific", "social commentary"
        ]
        return any(indicator in user_grounds.lower() for indicator in protected_indicators)

    def _analyze_government_interest(self, original_moderation: Dict[str, Any]) -> str:
        """Analyze the government interest in restriction"""
        harm_level = original_moderation.get("harm_level", "H1")
        if harm_level == "H3":
            return "compelling"
        elif harm_level == "H2":
            return "important"
        else:
            return "minimal"

    def _analyze_narrow_tailoring(self, original_moderation: Dict[str, Any]) -> bool:
        """Analyze if restriction is narrowly tailored"""
        confidence = original_moderation.get("confidence", 0)
        return confidence >= 0.8  # High confidence suggests narrow tailoring

    def _identify_alternatives(self, decision: str) -> List[str]:
        """Identify less restrictive alternatives"""
        alternatives = []
        if decision == "block":
            alternatives.extend(["warning", "restriction", "quarantine"])
        elif decision == "quarantine":
            alternatives.extend(["warning", "restriction"])
        elif decision == "restrict":
            alternatives.append("warning")
        return alternatives

    def _analyze_proportionality(self, original_moderation: Dict[str, Any]) -> bool:
        """Analyze proportionality of response"""
        harm_level = original_moderation.get("harm_level", "H1")
        decision = original_moderation.get("decision", "allow")
        
        proportional_responses = {
            "H0": ["allow"],
            "H1": ["allow", "allow_with_warning"],
            "H2": ["allow_with_warning", "restrict"],
            "H3": ["restrict", "quarantine", "block"]
        }
        
        return decision in proportional_responses.get(harm_level, [])

    def _analyze_content_neutrality(self, appeal_case: AppealCase) -> bool:
        """Analyze if restriction is content-neutral"""
        # This would involve complex analysis of the content and policy application
        return True  # Placeholder

    def _analyze_viewpoint_neutrality(self, appeal_case: AppealCase) -> bool:
        """Analyze if restriction is viewpoint-neutral"""
        # This would check for bias in policy application
        user_grounds = appeal_case.submission.constitutional_grounds.lower()
        bias_indicators = ["bias", "unfair", "discriminat", "target"]
        return not any(indicator in user_grounds for indicator in bias_indicators)

    def _generate_expert_rationale(self, analysis: Dict[str, Any], appeal_case: AppealCase) -> str:
        """Generate expert rationale for constitutional decision"""
        
        rationale_parts = []
        
        # Constitutional principles analyzed
        principles = analysis["constitutional_principles_analyzed"]
        rationale_parts.append(f"Constitutional analysis conducted on: {', '.join(principles)}")
        
        # Violations found
        violations = analysis["violations_found"]
        if violations:
            rationale_parts.append(f"Constitutional violations identified: {', '.join(violations)}")
        else:
            rationale_parts.append("No constitutional violations identified")
        
        # Recommendation rationale
        if analysis["recommendation"] == "grant_appeal":
            rationale_parts.append("Appeal granted based on constitutional analysis")
        else:
            rationale_parts.append("Original decision upheld as constitutionally sound")
        
        return " | ".join(rationale_parts)

    async def _conduct_community_review(self, appeal_case: AppealCase, reviewer_ids: List[str]):
        """Conduct community review process"""
        
        # Simulate community review votes
        review_votes = []
        
        for reviewer_id in reviewer_ids:
            reviewer = self.community_reviewers[reviewer_id]
            
            # Simulate review decision based on reviewer characteristics
            # Higher constitutional knowledge = more likely to identify violations
            grant_probability = reviewer.constitutional_knowledge_score * 0.6
            
            vote = {
                "reviewer_id": reviewer_id,
                "reviewer_tier": reviewer.tier_level,
                "constitutional_knowledge": reviewer.constitutional_knowledge_score,
                "vote": "grant" if grant_probability > 0.5 else "deny",
                "rationale": f"Community review by {reviewer.tier_level} tier reviewer",
                "timestamp": datetime.utcnow().isoformat()
            }
            review_votes.append(vote)
        
        # Tally votes
        grant_votes = len([v for v in review_votes if v["vote"] == "grant"])
        total_votes = len(review_votes)
        
        # Create community review result
        appeal_case.community_review_result = {
            "reviewers": reviewer_ids,
            "votes": review_votes,
            "grant_votes": grant_votes,
            "total_votes": total_votes,
            "consensus": grant_votes / total_votes,
            "recommendation": "grant_appeal" if grant_votes > total_votes / 2 else "deny_appeal",
            "review_completed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Community review completed for appeal {appeal_case.appeal_id}: "
                   f"{grant_votes}/{total_votes} votes to grant")

    async def _finalize_appeal_decision(self, appeal_case: AppealCase):
        """Finalize appeal decision based on all reviews"""
        
        # Collect all recommendations
        recommendations = []
        
        if appeal_case.automated_review_result:
            recommendations.append(appeal_case.automated_review_result["recommendation"])
        
        if appeal_case.constitutional_review_result:
            recommendations.append(appeal_case.constitutional_review_result["recommendation"])
        
        if appeal_case.community_review_result:
            recommendations.append(appeal_case.community_review_result["recommendation"])
        
        # Decision logic: Constitutional expert overrides, community provides input
        if appeal_case.constitutional_review_result:
            final_decision = appeal_case.constitutional_review_result["recommendation"]
            primary_rationale = appeal_case.constitutional_review_result["rationale"]
        elif appeal_case.community_review_result:
            final_decision = appeal_case.community_review_result["recommendation"]
            primary_rationale = f"Community consensus: {appeal_case.community_review_result['consensus']:.2f}"
        elif appeal_case.automated_review_result:
            final_decision = appeal_case.automated_review_result["recommendation"]
            primary_rationale = "Automated procedural review"
        else:
            final_decision = "deny_appeal"
            primary_rationale = "No review results available"
        
        # Set final decision
        if final_decision == "grant_appeal":
            appeal_case.status = AppealStatus.APPROVED
            appeal_case.final_decision = "granted"
        else:
            appeal_case.status = AppealStatus.DENIED
            appeal_case.final_decision = "denied"
        
        appeal_case.decision_rationale = primary_rationale
        appeal_case.resolved_at = datetime.utcnow()
        
        # Generate constitutional analysis summary
        appeal_case.constitutional_analysis = await self._generate_constitutional_summary(appeal_case)
        
        # Determine remedial actions if appeal granted
        if appeal_case.final_decision == "granted":
            appeal_case.remedial_actions = await self._determine_remedial_actions(appeal_case)
        
        # Clean up reviewer assignments
        for reviewer_id in appeal_case.assigned_reviewers:
            if reviewer_id in self.community_reviewers:
                self.community_reviewers[reviewer_id].active_cases = max(
                    0, self.community_reviewers[reviewer_id].active_cases - 1
                )
        
        logger.info(f"Appeal {appeal_case.appeal_id} finalized: {appeal_case.final_decision}")

    async def _generate_constitutional_summary(self, appeal_case: AppealCase) -> Dict[str, Any]:
        """Generate comprehensive constitutional analysis summary"""
        
        summary = {
            "appeal_type": appeal_case.submission.appeal_type.value,
            "constitutional_principles_considered": [],
            "violations_identified": [],
            "procedural_compliance": True,
            "community_input_received": bool(appeal_case.community_review_result),
            "expert_review_conducted": bool(appeal_case.constitutional_review_result)
        }
        
        # Add constitutional review findings
        if appeal_case.constitutional_review_result:
            constitutional_analysis = appeal_case.constitutional_review_result.get("constitutional_analysis", {})
            summary["constitutional_principles_considered"] = constitutional_analysis.get("constitutional_principles_analyzed", [])
            summary["violations_identified"] = constitutional_analysis.get("violations_found", [])
        
        # Add community perspective
        if appeal_case.community_review_result:
            summary["community_consensus"] = appeal_case.community_review_result["consensus"]
            summary["community_reviewers"] = len(appeal_case.community_review_result["reviewers"])
        
        return summary

    async def _determine_remedial_actions(self, appeal_case: AppealCase) -> List[str]:
        """Determine remedial actions for granted appeals"""
        
        actions = []
        
        # Reverse original decision
        original_decision = appeal_case.original_moderation.get("decision")
        if original_decision == "block":
            actions.append("unblock_content")
        elif original_decision == "quarantine":
            actions.append("release_from_quarantine")
        elif original_decision == "restrict":
            actions.append("remove_restrictions")
        
        # Constitutional protections
        if appeal_case.submission.appeal_type == AppealType.FIRST_AMENDMENT:
            actions.append("apply_first_amendment_protections")
            
        if appeal_case.submission.appeal_type == AppealType.VIEWPOINT_DISCRIMINATION:
            actions.append("implement_viewpoint_neutrality_safeguards")
        
        # Process improvements
        violations = appeal_case.constitutional_analysis.get("violations_identified", [])
        if "inadequate_notice" in violations:
            actions.append("improve_notification_process")
            
        if "overbroad_restriction_of_protected_speech" in violations:
            actions.append("refine_content_policies")
        
        # User tier specific actions
        if appeal_case.submission.user_tier == "Gold":
            actions.append("provide_gold_tier_protections")
        
        return actions

    def _requires_constitutional_review(self, appeal_type: AppealType) -> bool:
        """Check if appeal type requires constitutional expert review"""
        return self.review_requirements[appeal_type].get("requires_constitutional_expert", False)

    def _requires_community_review(self, appeal_type: AppealType, user_tier: str) -> bool:
        """Check if appeal requires community review"""
        base_requirement = self.review_requirements[appeal_type].get("requires_community_review", False)
        
        # Gold tier gets community review for all constitutional issues
        if user_tier == "Gold" and appeal_type in [AppealType.FIRST_AMENDMENT, AppealType.VIEWPOINT_DISCRIMINATION]:
            return True
        
        return base_requirement

    async def get_appeal_status(self, appeal_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of appeal"""
        
        if appeal_id not in self.active_appeals:
            return None
        
        appeal_case = self.active_appeals[appeal_id]
        
        return {
            "appeal_id": appeal_id,
            "content_id": appeal_case.submission.content_id,
            "status": appeal_case.status.value,
            "appeal_type": appeal_case.submission.appeal_type.value,
            "submitted_at": appeal_case.submission.submitted_at.isoformat(),
            "review_deadline": appeal_case.review_deadline.isoformat() if appeal_case.review_deadline else None,
            "assigned_reviewers": appeal_case.assigned_reviewers,
            "community_oversight_involved": appeal_case.community_oversight_involved,
            "final_decision": appeal_case.final_decision,
            "resolved_at": appeal_case.resolved_at.isoformat() if appeal_case.resolved_at else None
        }

    async def get_appeals_metrics(self) -> Dict[str, Any]:
        """Get appeals system performance metrics"""
        
        all_appeals = list(self.active_appeals.values())
        total_appeals = len(all_appeals)
        
        if total_appeals == 0:
            return {"no_appeals": True}
        
        # Status distribution
        status_distribution = {}
        for status in AppealStatus:
            status_distribution[status.value] = len([
                appeal for appeal in all_appeals if appeal.status == status
            ])
        
        # Appeal type distribution
        type_distribution = {}
        for appeal_type in AppealType:
            type_distribution[appeal_type.value] = len([
                appeal for appeal in all_appeals 
                if appeal.submission.appeal_type == appeal_type
            ])
        
        # Resolution metrics
        resolved_appeals = [appeal for appeal in all_appeals if appeal.resolved_at]
        if resolved_appeals:
            resolution_times = [
                (appeal.resolved_at - appeal.submission.submitted_at).total_seconds() / 3600
                for appeal in resolved_appeals
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            granted_appeals = len([appeal for appeal in resolved_appeals if appeal.final_decision == "granted"])
            grant_rate = granted_appeals / len(resolved_appeals)
        else:
            avg_resolution_time = 0
            grant_rate = 0
        
        # Constitutional compliance
        constitutional_reviews = len([
            appeal for appeal in all_appeals 
            if appeal.constitutional_review_result
        ])
        
        community_reviews = len([
            appeal for appeal in all_appeals 
            if appeal.community_review_result
        ])
        
        return {
            "total_appeals": total_appeals,
            "status_distribution": status_distribution,
            "type_distribution": type_distribution,
            "resolution_metrics": {
                "average_resolution_time_hours": avg_resolution_time,
                "grant_rate": grant_rate,
                "constitutional_reviews_conducted": constitutional_reviews,
                "community_reviews_conducted": community_reviews
            },
            "system_health": {
                "pending_appeals": status_distribution.get("submitted", 0) + status_distribution.get("under_review", 0),
                "overdue_appeals": len([
                    appeal for appeal in all_appeals
                    if appeal.review_deadline and datetime.utcnow() > appeal.review_deadline
                ])
            }
        }

    async def shutdown(self):
        """Gracefully shutdown appeals manager"""
        logger.info("Shutting down Constitutional Appeals Manager")
        
        # Log pending appeals
        pending_appeals = len([
            appeal for appeal in self.active_appeals.values()
            if appeal.status not in [AppealStatus.APPROVED, AppealStatus.DENIED, AppealStatus.CLOSED]
        ])
        
        if pending_appeals > 0:
            logger.warning(f"Shutting down with {pending_appeals} pending appeals")
        
        # Reset community reviewer assignments
        for reviewer in self.community_reviewers.values():
            reviewer.active_cases = 0