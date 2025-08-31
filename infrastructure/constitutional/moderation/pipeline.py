"""
Constitutional Machine-Only Moderation Pipeline
Real-time content analysis with constitutional harm classification and automated policy enforcement
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4

from infrastructure.ml.constitutional.harm_classifier import ConstitutionalHarmClassifier
from infrastructure.constitutional.moderation.policy_enforcement import PolicyEnforcement
from infrastructure.constitutional.moderation.response_actions import ResponseActions
from infrastructure.constitutional.moderation.escalation import EscalationManager
from infrastructure.constitutional.moderation.appeals import AppealsManager
from infrastructure.fog.security.tee_integration import TEESecurityManager
from infrastructure.fog.workload.router import WorkloadRouter
from infrastructure.constitutional.governance.pricing import ConstitutionalPricing
from infrastructure.constitutional.transparency.audit_logger import TransparencyLogger

logger = logging.getLogger(__name__)

class ModerationDecision(Enum):
    """Constitutional moderation decisions"""
    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    RESTRICT = "restrict"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    ESCALATE = "escalate"

class ProcessingStatus(Enum):
    """Content processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    APPEALED = "appealed"

@dataclass
class ContentAnalysis:
    """Content analysis results"""
    content_id: str
    harm_level: str  # H0, H1, H2, H3
    harm_categories: List[str]
    confidence_score: float
    constitutional_concerns: Dict[str, Any]
    viewpoint_bias_score: float
    timestamp: datetime
    processing_time_ms: int

@dataclass
class ModerationResult:
    """Complete moderation result"""
    content_id: str
    decision: ModerationDecision
    harm_analysis: ContentAnalysis
    policy_rationale: str
    response_actions: List[str]
    tier_level: str  # Bronze, Silver, Gold
    requires_escalation: bool
    appeal_eligible: bool
    audit_trail: Dict[str, Any]
    transparency_score: float

class ConstitutionalModerationPipeline:
    """
    Machine-only moderation pipeline with constitutional harm classification
    Processes content in real-time with minimal human intervention
    """
    
    def __init__(self):
        self.harm_classifier = ConstitutionalHarmClassifier()
        self.policy_enforcement = PolicyEnforcement()
        self.response_actions = ResponseActions()
        self.escalation_manager = EscalationManager()
        self.appeals_manager = AppealsManager()
        self.tee_security = TEESecurityManager()
        self.workload_router = WorkloadRouter()
        self.pricing = ConstitutionalPricing()
        self.audit_logger = TransparencyLogger()
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'average_processing_time': 0,
            'decisions_by_type': {},
            'harm_level_distribution': {},
            'escalation_rate': 0
        }
        
        # Constitutional principles cache
        self.constitutional_cache = {}
        
        logger.info("Constitutional Moderation Pipeline initialized")

    async def process_content(
        self, 
        content: str, 
        content_type: str,
        user_tier: str = "Bronze",
        context: Dict[str, Any] = None
    ) -> ModerationResult:
        """
        Process content through constitutional moderation pipeline
        
        Args:
            content: Content to moderate
            content_type: Type of content (text, code, etc.)
            user_tier: User tier (Bronze, Silver, Gold)
            context: Additional context information
            
        Returns:
            ModerationResult with decision and rationale
        """
        content_id = str(uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Processing content {content_id} for {user_tier} tier")
            
            # Step 1: Constitutional harm analysis
            harm_analysis = await self._analyze_constitutional_harm(
                content_id, content, content_type, context or {}
            )
            
            # Step 2: Policy enforcement decision
            enforcement_result = await self.policy_enforcement.evaluate_content(
                harm_analysis, user_tier, context
            )
            
            # Step 3: Determine response actions
            response_actions = await self.response_actions.determine_actions(
                enforcement_result, harm_analysis, user_tier
            )
            
            # Step 4: Check escalation requirements
            requires_escalation = await self._requires_escalation(
                harm_analysis, enforcement_result, user_tier
            )
            
            # Step 5: Build moderation result
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ModerationResult(
                content_id=content_id,
                decision=enforcement_result.decision,
                harm_analysis=harm_analysis,
                policy_rationale=enforcement_result.rationale,
                response_actions=response_actions.actions,
                tier_level=user_tier,
                requires_escalation=requires_escalation,
                appeal_eligible=self._is_appeal_eligible(enforcement_result, user_tier),
                audit_trail=self._build_audit_trail(
                    content_id, harm_analysis, enforcement_result, processing_time_ms
                ),
                transparency_score=self._calculate_transparency_score(
                    harm_analysis, enforcement_result
                )
            )
            
            # Step 6: Execute response actions
            await self._execute_response_actions(result)
            
            # Step 7: Handle escalation if required
            if requires_escalation:
                await self.escalation_manager.escalate_content(result)
            
            # Step 8: Update statistics and audit log
            await self._update_statistics(result)
            await self.audit_logger.log_moderation_decision(result)
            
            logger.info(f"Content {content_id} processed: {result.decision.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            # Return safe default with escalation
            return await self._create_error_result(content_id, str(e), user_tier)

    async def _analyze_constitutional_harm(
        self, 
        content_id: str, 
        content: str, 
        content_type: str,
        context: Dict[str, Any]
    ) -> ContentAnalysis:
        """Analyze content for constitutional harm using ML classifier"""
        start_time = time.time()
        
        try:
            # Use constitutional harm classifier
            classification_result = await self.harm_classifier.classify_content(
                content, content_type, context
            )
            
            # Extract constitutional concerns
            constitutional_concerns = await self._extract_constitutional_concerns(
                content, classification_result
            )
            
            # Calculate viewpoint bias score
            viewpoint_bias_score = await self._calculate_viewpoint_bias(
                content, classification_result
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return ContentAnalysis(
                content_id=content_id,
                harm_level=classification_result.harm_level,
                harm_categories=classification_result.categories,
                confidence_score=classification_result.confidence,
                constitutional_concerns=constitutional_concerns,
                viewpoint_bias_score=viewpoint_bias_score,
                timestamp=datetime.utcnow(),
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Harm analysis failed for {content_id}: {str(e)}")
            # Return conservative analysis
            return ContentAnalysis(
                content_id=content_id,
                harm_level="H2",  # Conservative default
                harm_categories=["analysis_error"],
                confidence_score=0.5,
                constitutional_concerns={"error": str(e)},
                viewpoint_bias_score=0.0,
                timestamp=datetime.utcnow(),
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    async def _extract_constitutional_concerns(
        self, 
        content: str, 
        classification_result: Any
    ) -> Dict[str, Any]:
        """Extract specific constitutional concerns from analysis"""
        concerns = {}
        
        # First Amendment concerns
        if any(cat in ["political_speech", "religious_expression", "artistic_expression"] 
               for cat in classification_result.categories):
            concerns["first_amendment"] = {
                "type": "protected_speech",
                "rationale": "Content may involve protected speech categories",
                "requires_heightened_scrutiny": True
            }
        
        # Viewpoint discrimination
        if classification_result.metadata.get("political_lean"):
            concerns["viewpoint_neutrality"] = {
                "detected_lean": classification_result.metadata["political_lean"],
                "confidence": classification_result.metadata.get("lean_confidence", 0.5),
                "requires_bias_check": True
            }
        
        # Due process concerns
        if classification_result.confidence < 0.7:
            concerns["due_process"] = {
                "low_confidence": True,
                "confidence_score": classification_result.confidence,
                "requires_human_review": True
            }
        
        return concerns

    async def _calculate_viewpoint_bias(
        self, 
        content: str, 
        classification_result: Any
    ) -> float:
        """Calculate potential viewpoint bias in moderation decision"""
        bias_factors = []
        
        # Check for political content bias
        if "political" in str(classification_result.categories).lower():
            political_lean = classification_result.metadata.get("political_lean", "neutral")
            if political_lean != "neutral":
                bias_factors.append(0.3)
        
        # Check for ideological bias
        ideological_markers = ["conservative", "liberal", "progressive", "traditional"]
        content_lower = content.lower()
        for marker in ideological_markers:
            if marker in content_lower:
                bias_factors.append(0.2)
        
        # Check for cultural bias
        if any(cat in ["cultural_expression", "religious_content"] 
               for cat in classification_result.categories):
            bias_factors.append(0.1)
        
        # Return average bias score (0.0 = no bias, 1.0 = high bias)
        return min(sum(bias_factors), 1.0) if bias_factors else 0.0

    async def _requires_escalation(
        self, 
        harm_analysis: ContentAnalysis,
        enforcement_result: Any,
        user_tier: str
    ) -> bool:
        """Determine if content requires human escalation"""
        
        # Gold tier: Escalate high-confidence severe cases
        if user_tier == "Gold":
            if (harm_analysis.harm_level == "H3" and 
                harm_analysis.confidence_score > 0.8):
                return True
            
            # Escalate constitutional concerns
            if harm_analysis.constitutional_concerns.get("first_amendment", {}).get("requires_heightened_scrutiny"):
                return True
        
        # Silver tier: Limited escalation for unclear cases
        elif user_tier == "Silver":
            if (harm_analysis.harm_level in ["H2", "H3"] and 
                harm_analysis.confidence_score < 0.6):
                return True
        
        # Bronze tier: No escalation (machine-only)
        
        # Always escalate system errors
        if "error" in harm_analysis.harm_categories:
            return True
        
        return False

    def _is_appeal_eligible(self, enforcement_result: Any, user_tier: str) -> bool:
        """Determine if decision is eligible for constitutional appeal"""
        
        # All restrictive decisions are appealable
        if enforcement_result.decision in [
            ModerationDecision.RESTRICT,
            ModerationDecision.QUARANTINE,
            ModerationDecision.BLOCK
        ]:
            return True
        
        # Gold tier users can appeal warnings
        if (user_tier == "Gold" and 
            enforcement_result.decision == ModerationDecision.ALLOW_WITH_WARNING):
            return True
        
        return False

    def _build_audit_trail(
        self, 
        content_id: str,
        harm_analysis: ContentAnalysis,
        enforcement_result: Any,
        processing_time_ms: int
    ) -> Dict[str, Any]:
        """Build comprehensive audit trail for transparency"""
        return {
            "content_id": content_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": processing_time_ms,
            "harm_classification": {
                "level": harm_analysis.harm_level,
                "categories": harm_analysis.harm_categories,
                "confidence": harm_analysis.confidence_score,
                "constitutional_concerns": harm_analysis.constitutional_concerns
            },
            "policy_decision": {
                "decision": enforcement_result.decision.value,
                "rationale": enforcement_result.rationale,
                "policy_version": "v1.0.0"
            },
            "constitutional_analysis": {
                "viewpoint_bias_score": harm_analysis.viewpoint_bias_score,
                "first_amendment_considerations": harm_analysis.constitutional_concerns.get("first_amendment"),
                "due_process_flags": harm_analysis.constitutional_concerns.get("due_process")
            },
            "system_metadata": {
                "classifier_version": self.harm_classifier.version,
                "pipeline_version": "1.0.0",
                "tee_attestation": self.tee_security.get_attestation()
            }
        }

    def _calculate_transparency_score(
        self, 
        harm_analysis: ContentAnalysis,
        enforcement_result: Any
    ) -> float:
        """Calculate transparency score for decision explainability"""
        score_factors = []
        
        # High confidence = high transparency
        score_factors.append(harm_analysis.confidence_score)
        
        # Clear rationale = high transparency
        if len(enforcement_result.rationale) > 50:
            score_factors.append(0.8)
        else:
            score_factors.append(0.4)
        
        # Constitutional analysis = high transparency
        if harm_analysis.constitutional_concerns:
            score_factors.append(0.9)
        else:
            score_factors.append(0.6)
        
        # Low viewpoint bias = high transparency
        score_factors.append(1.0 - harm_analysis.viewpoint_bias_score)
        
        return sum(score_factors) / len(score_factors)

    async def _execute_response_actions(self, result: ModerationResult):
        """Execute the determined response actions"""
        try:
            await self.response_actions.execute_actions(
                result.content_id,
                result.response_actions,
                result.harm_analysis,
                result.tier_level
            )
        except Exception as e:
            logger.error(f"Failed to execute response actions for {result.content_id}: {str(e)}")

    async def _update_statistics(self, result: ModerationResult):
        """Update pipeline performance statistics"""
        self.processing_stats['total_processed'] += 1
        
        # Track decision distribution
        decision = result.decision.value
        self.processing_stats['decisions_by_type'][decision] = \
            self.processing_stats['decisions_by_type'].get(decision, 0) + 1
        
        # Track harm level distribution
        harm_level = result.harm_analysis.harm_level
        self.processing_stats['harm_level_distribution'][harm_level] = \
            self.processing_stats['harm_level_distribution'].get(harm_level, 0) + 1
        
        # Update escalation rate
        if result.requires_escalation:
            escalations = sum(1 for d in self.processing_stats['decisions_by_type'].values() 
                            if d == 'escalate')
            self.processing_stats['escalation_rate'] = escalations / self.processing_stats['total_processed']
        
        # Update average processing time
        current_avg = self.processing_stats['average_processing_time']
        new_time = result.harm_analysis.processing_time_ms
        total = self.processing_stats['total_processed']
        self.processing_stats['average_processing_time'] = \
            ((current_avg * (total - 1)) + new_time) / total

    async def _create_error_result(
        self, 
        content_id: str, 
        error_message: str,
        user_tier: str
    ) -> ModerationResult:
        """Create safe error result with escalation"""
        
        error_analysis = ContentAnalysis(
            content_id=content_id,
            harm_level="H2",  # Conservative error handling
            harm_categories=["system_error"],
            confidence_score=0.0,
            constitutional_concerns={"system_error": error_message},
            viewpoint_bias_score=0.0,
            timestamp=datetime.utcnow(),
            processing_time_ms=0
        )
        
        return ModerationResult(
            content_id=content_id,
            decision=ModerationDecision.ESCALATE,
            harm_analysis=error_analysis,
            policy_rationale=f"System error requires human review: {error_message}",
            response_actions=["escalate_system_error"],
            tier_level=user_tier,
            requires_escalation=True,
            appeal_eligible=True,
            audit_trail={"error": error_message, "timestamp": datetime.utcnow().isoformat()},
            transparency_score=0.5
        )

    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get current pipeline performance metrics"""
        return {
            "processing_statistics": self.processing_stats,
            "constitutional_metrics": {
                "average_viewpoint_bias": await self._calculate_average_viewpoint_bias(),
                "first_amendment_flags": await self._count_constitutional_flags(),
                "transparency_average": await self._calculate_average_transparency()
            },
            "system_health": {
                "classifier_status": await self.harm_classifier.health_check(),
                "tee_status": await self.tee_security.health_check(),
                "escalation_queue_size": await self.escalation_manager.get_queue_size()
            }
        }

    async def _calculate_average_viewpoint_bias(self) -> float:
        """Calculate average viewpoint bias across recent decisions"""
        # Implementation would query recent decisions from audit log
        return 0.15  # Placeholder

    async def _count_constitutional_flags(self) -> int:
        """Count recent First Amendment flags"""
        # Implementation would query audit log
        return 5  # Placeholder

    async def _calculate_average_transparency(self) -> float:
        """Calculate average transparency score"""
        # Implementation would query recent decisions
        return 0.85  # Placeholder

    async def process_appeal(
        self, 
        content_id: str, 
        appeal_reason: str,
        user_tier: str
    ) -> Dict[str, Any]:
        """Process constitutional appeal for moderation decision"""
        return await self.appeals_manager.process_appeal(
            content_id, appeal_reason, user_tier
        )

    async def shutdown(self):
        """Gracefully shutdown the moderation pipeline"""
        logger.info("Shutting down Constitutional Moderation Pipeline")
        
        # Save statistics
        await self.audit_logger.log_system_event({
            "event": "pipeline_shutdown",
            "final_stats": self.processing_stats,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Cleanup resources
        await self.harm_classifier.shutdown()
        await self.tee_security.shutdown()
        await self.escalation_manager.shutdown()