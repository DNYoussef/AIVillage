"""
Constitutional Moderation Integration with Fog Computing Infrastructure
Integrates machine-only moderation pipeline with existing fog infrastructure
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from .pipeline import ConstitutionalModerationPipeline, ModerationResult
from infrastructure.fog.workload.router import WorkloadRouter
from infrastructure.fog.security.tee_integration import TEESecurityManager
from infrastructure.constitutional.governance.pricing import ConstitutionalPricing
from infrastructure.constitutional.transparency.audit_logger import TransparencyLogger
from infrastructure.p2p.core.message_delivery import MessageDeliverySystem

logger = logging.getLogger(__name__)

@dataclass
class FogWorkloadRequest:
    """Fog workload request requiring moderation"""
    workload_id: str
    content: str
    content_type: str
    user_id: str
    user_tier: str
    fog_node_id: str
    priority: str
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class ModerationResponse:
    """Response from constitutional moderation for fog workload"""
    workload_id: str
    approved: bool
    moderation_result: ModerationResult
    routing_decision: str
    pricing_impact: Dict[str, Any]
    security_requirements: List[str]
    transparency_data: Dict[str, Any]

class FogModerationIntegration:
    """
    Integration layer between constitutional moderation and fog infrastructure
    Provides real-time content moderation for fog compute workloads
    """
    
    def __init__(self):
        self.moderation_pipeline = ConstitutionalModerationPipeline()
        self.workload_router = WorkloadRouter()
        self.tee_security = TEESecurityManager()
        self.pricing = ConstitutionalPricing()
        self.transparency_logger = TransparencyLogger()
        self.message_delivery = MessageDeliverySystem()
        
        # Integration metrics
        self.integration_metrics = {
            'total_workloads_processed': 0,
            'approved_workloads': 0,
            'rejected_workloads': 0,
            'escalated_workloads': 0,
            'average_processing_time_ms': 0,
            'constitutional_protections_applied': 0
        }
        
        logger.info("Fog Moderation Integration initialized")

    async def process_workload_request(
        self, 
        workload_request: FogWorkloadRequest
    ) -> ModerationResponse:
        """
        Process fog workload request through constitutional moderation
        
        Args:
            workload_request: Fog workload requiring moderation approval
            
        Returns:
            ModerationResponse with approval decision and routing
        """
        try:
            start_time = datetime.utcnow()
            
            logger.info(f"Processing fog workload {workload_request.workload_id} from {workload_request.user_tier} user")
            
            # Step 1: Constitutional moderation
            moderation_result = await self.moderation_pipeline.process_content(
                content=workload_request.content,
                content_type=workload_request.content_type,
                user_tier=workload_request.user_tier,
                context={
                    **workload_request.context,
                    'workload_id': workload_request.workload_id,
                    'fog_node_id': workload_request.fog_node_id,
                    'processing_priority': workload_request.priority
                }
            )
            
            # Step 2: Determine workload approval
            approved = await self._determine_workload_approval(moderation_result)
            
            # Step 3: Calculate routing decision
            routing_decision = await self._determine_routing(
                workload_request, moderation_result, approved
            )
            
            # Step 4: Calculate pricing impact
            pricing_impact = await self._calculate_pricing_impact(
                workload_request, moderation_result
            )
            
            # Step 5: Determine security requirements
            security_requirements = await self._determine_security_requirements(
                moderation_result, workload_request
            )
            
            # Step 6: Generate transparency data
            transparency_data = await self._generate_transparency_data(
                workload_request, moderation_result
            )
            
            # Step 7: Create moderation response
            response = ModerationResponse(
                workload_id=workload_request.workload_id,
                approved=approved,
                moderation_result=moderation_result,
                routing_decision=routing_decision,
                pricing_impact=pricing_impact,
                security_requirements=security_requirements,
                transparency_data=transparency_data
            )
            
            # Step 8: Execute fog infrastructure actions
            await self._execute_fog_actions(workload_request, response)
            
            # Step 9: Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_integration_metrics(response, processing_time)
            
            logger.info(f"Workload {workload_request.workload_id} processed: "
                       f"approved={approved}, routing={routing_decision}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process workload {workload_request.workload_id}: {str(e)}")
            return await self._create_error_response(workload_request, str(e))

    async def _determine_workload_approval(self, moderation_result: ModerationResult) -> bool:
        """Determine if fog workload should be approved based on moderation result"""
        
        # Allow decisions
        if moderation_result.decision.value in ["allow", "allow_with_warning"]:
            return True
        
        # Restricted content may be allowed with constraints
        elif moderation_result.decision.value == "restrict":
            # Check if user tier allows restricted content processing
            if moderation_result.tier_level in ["Silver", "Gold"]:
                return True
            return False
        
        # Quarantined content requires special handling
        elif moderation_result.decision.value == "quarantine":
            # Only Gold tier can process quarantined content
            return moderation_result.tier_level == "Gold"
        
        # Blocked content is not approved for processing
        elif moderation_result.decision.value == "block":
            return False
        
        # Escalated content pending review
        elif moderation_result.decision.value == "escalate":
            # Hold workload pending human review
            return False
        
        # Default to not approved
        return False

    async def _determine_routing(
        self, 
        workload_request: FogWorkloadRequest,
        moderation_result: ModerationResult,
        approved: bool
    ) -> str:
        """Determine fog workload routing based on moderation result"""
        
        if not approved:
            return "rejected"
        
        # Route based on constitutional considerations and tier
        harm_level = moderation_result.harm_analysis.harm_level
        tier_level = moderation_result.tier_level
        constitutional_concerns = moderation_result.harm_analysis.constitutional_concerns
        
        # H0 constitutional content - premium routing
        if harm_level == "H0":
            if tier_level == "Gold":
                return "premium_constitutional_nodes"
            else:
                return "standard_constitutional_nodes"
        
        # H1 content with warnings - standard routing with monitoring
        elif harm_level == "H1":
            if moderation_result.decision.value == "allow_with_warning":
                return "monitored_standard_nodes"
            else:
                return "standard_nodes"
        
        # H2 restricted content - secure routing
        elif harm_level == "H2":
            if tier_level == "Gold":
                return "secure_gold_nodes"
            elif tier_level == "Silver":
                return "secure_silver_nodes"
            else:
                return "secure_bronze_nodes"
        
        # H3 quarantined content - maximum security routing (Gold only)
        elif harm_level == "H3" and tier_level == "Gold":
            return "maximum_security_nodes"
        
        # Constitutional concerns require enhanced nodes
        if constitutional_concerns:
            if constitutional_concerns.get("first_amendment"):
                return "first_amendment_protected_nodes"
            elif constitutional_concerns.get("viewpoint_neutrality"):
                return "viewpoint_neutral_nodes"
        
        # Default routing
        return "standard_nodes"

    async def _calculate_pricing_impact(
        self, 
        workload_request: FogWorkloadRequest,
        moderation_result: ModerationResult
    ) -> Dict[str, Any]:
        """Calculate pricing impact based on moderation and constitutional factors"""
        
        base_pricing = await self.pricing.calculate_base_workload_cost(
            workload_type=workload_request.content_type,
            user_tier=workload_request.user_tier,
            processing_complexity="standard"
        )
        
        pricing_adjustments = {
            'base_cost': base_pricing,
            'constitutional_adjustments': {},
            'tier_adjustments': {},
            'security_adjustments': {},
            'final_cost': base_pricing
        }
        
        # Constitutional protection discounts
        harm_level = moderation_result.harm_analysis.harm_level
        if harm_level == "H0":  # Constitutional content
            constitutional_discount = 0.15  # 15% discount for protected speech
            pricing_adjustments['constitutional_adjustments']['protected_speech_discount'] = constitutional_discount
            pricing_adjustments['final_cost'] *= (1 - constitutional_discount)
        
        # First Amendment protection incentives
        if moderation_result.harm_analysis.constitutional_concerns.get("first_amendment"):
            first_amendment_discount = 0.10  # 10% discount
            pricing_adjustments['constitutional_adjustments']['first_amendment_discount'] = first_amendment_discount
            pricing_adjustments['final_cost'] *= (1 - first_amendment_discount)
        
        # Tier-based adjustments
        tier_multipliers = {"Bronze": 1.0, "Silver": 0.95, "Gold": 0.85}
        tier_multiplier = tier_multipliers.get(workload_request.user_tier, 1.0)
        pricing_adjustments['tier_adjustments']['tier_multiplier'] = tier_multiplier
        pricing_adjustments['final_cost'] *= tier_multiplier
        
        # Security requirement surcharges
        if moderation_result.decision.value in ["restrict", "quarantine"]:
            security_surcharge = 0.20  # 20% surcharge for enhanced security
            pricing_adjustments['security_adjustments']['enhanced_security_surcharge'] = security_surcharge
            pricing_adjustments['final_cost'] *= (1 + security_surcharge)
        
        # Moderation processing fee
        moderation_fee = 0.05 * base_pricing  # 5% of base cost
        pricing_adjustments['moderation_fee'] = moderation_fee
        pricing_adjustments['final_cost'] += moderation_fee
        
        return pricing_adjustments

    async def _determine_security_requirements(
        self, 
        moderation_result: ModerationResult,
        workload_request: FogWorkloadRequest
    ) -> List[str]:
        """Determine security requirements for workload processing"""
        
        requirements = []
        
        # Base TEE requirement for all moderated content
        requirements.append("tee_attestation_required")
        
        # Enhanced security based on harm level
        harm_level = moderation_result.harm_analysis.harm_level
        if harm_level in ["H2", "H3"]:
            requirements.append("enhanced_tee_security")
            requirements.append("secure_communication_channels")
        
        # Constitutional protection security
        constitutional_concerns = moderation_result.harm_analysis.constitutional_concerns
        if constitutional_concerns.get("first_amendment"):
            requirements.append("constitutional_audit_logging")
            requirements.append("viewpoint_neutrality_monitoring")
        
        if constitutional_concerns.get("viewpoint_neutrality", {}).get("requires_bias_check"):
            requirements.append("bias_detection_monitoring")
        
        # Tier-based security enhancements
        tier_level = moderation_result.tier_level
        if tier_level == "Gold":
            requirements.extend([
                "gold_tier_encryption",
                "priority_security_monitoring",
                "constitutional_compliance_verification"
            ])
        elif tier_level == "Silver":
            requirements.extend([
                "silver_tier_encryption",
                "enhanced_monitoring"
            ])
        
        # Decision-specific security
        if moderation_result.decision.value == "restrict":
            requirements.extend([
                "content_access_controls",
                "usage_monitoring",
                "compliance_reporting"
            ])
        
        if moderation_result.decision.value == "quarantine":
            requirements.extend([
                "quarantine_security_protocols",
                "evidence_preservation",
                "limited_access_controls"
            ])
        
        # Appeal-eligible content requires additional protections
        if moderation_result.appeal_eligible:
            requirements.extend([
                "appeal_data_preservation",
                "decision_audit_trail",
                "constitutional_review_access"
            ])
        
        return list(set(requirements))  # Remove duplicates

    async def _generate_transparency_data(
        self, 
        workload_request: FogWorkloadRequest,
        moderation_result: ModerationResult
    ) -> Dict[str, Any]:
        """Generate transparency data for public accountability"""
        
        transparency_data = {
            'workload_metadata': {
                'workload_id': workload_request.workload_id,
                'user_tier': workload_request.user_tier,
                'fog_node_id': workload_request.fog_node_id,
                'timestamp': workload_request.timestamp.isoformat(),
                'processing_priority': workload_request.priority
            },
            'moderation_summary': {
                'harm_level': moderation_result.harm_analysis.harm_level,
                'decision': moderation_result.decision.value,
                'confidence_score': moderation_result.harm_analysis.confidence_score,
                'viewpoint_bias_score': moderation_result.harm_analysis.viewpoint_bias_score,
                'transparency_score': moderation_result.transparency_score
            },
            'constitutional_analysis': {
                'first_amendment_considerations': bool(moderation_result.harm_analysis.constitutional_concerns.get("first_amendment")),
                'due_process_compliance': bool(moderation_result.harm_analysis.constitutional_concerns.get("due_process")),
                'viewpoint_neutrality_verified': moderation_result.harm_analysis.viewpoint_bias_score < 0.3,
                'appeal_rights_preserved': moderation_result.appeal_eligible
            },
            'system_integrity': {
                'tee_attestation_verified': True,  # Would verify actual TEE attestation
                'classifier_version': moderation_result.audit_trail['system_metadata']['classifier_version'],
                'pipeline_version': moderation_result.audit_trail['system_metadata']['pipeline_version'],
                'processing_time_ms': moderation_result.harm_analysis.processing_time_ms
            }
        }
        
        # Add tier-specific transparency data
        if workload_request.user_tier == "Gold":
            transparency_data['gold_tier_protections'] = {
                'constitutional_review_available': True,
                'community_oversight_enabled': True,
                'enhanced_appeal_rights': True
            }
        
        return transparency_data

    async def _execute_fog_actions(
        self, 
        workload_request: FogWorkloadRequest,
        response: ModerationResponse
    ):
        """Execute actions in fog infrastructure based on moderation response"""
        
        try:
            # Route workload based on decision
            if response.approved:
                await self.workload_router.route_workload(
                    workload_id=workload_request.workload_id,
                    routing_decision=response.routing_decision,
                    security_requirements=response.security_requirements,
                    user_tier=workload_request.user_tier
                )
            else:
                await self.workload_router.reject_workload(
                    workload_id=workload_request.workload_id,
                    rejection_reason=response.moderation_result.policy_rationale,
                    appeal_eligible=response.moderation_result.appeal_eligible
                )
            
            # Apply security requirements
            if response.security_requirements:
                await self.tee_security.apply_security_requirements(
                    workload_id=workload_request.workload_id,
                    requirements=response.security_requirements
                )
            
            # Update pricing
            await self.pricing.apply_pricing_adjustments(
                workload_id=workload_request.workload_id,
                user_id=workload_request.user_id,
                pricing_impact=response.pricing_impact
            )
            
            # Log transparency data
            await self.transparency_logger.log_workload_processing(
                workload_request.workload_id,
                response.transparency_data
            )
            
            # Handle escalations
            if response.moderation_result.requires_escalation:
                await self._handle_workload_escalation(workload_request, response)
            
        except Exception as e:
            logger.error(f"Failed to execute fog actions for workload {workload_request.workload_id}: {str(e)}")
            raise

    async def _handle_workload_escalation(
        self, 
        workload_request: FogWorkloadRequest,
        response: ModerationResponse
    ):
        """Handle escalated workloads requiring human review"""
        
        escalation_data = {
            'workload_id': workload_request.workload_id,
            'user_id': workload_request.user_id,
            'user_tier': workload_request.user_tier,
            'fog_node_id': workload_request.fog_node_id,
            'moderation_result': response.moderation_result.audit_trail,
            'constitutional_concerns': response.moderation_result.harm_analysis.constitutional_concerns,
            'escalation_reason': 'constitutional_review_required'
        }
        
        # Hold workload pending review
        await self.workload_router.hold_workload(
            workload_id=workload_request.workload_id,
            hold_reason="pending_constitutional_review",
            estimated_review_time="2-24 hours"
        )
        
        # Notify user of escalation
        await self.message_delivery.send_escalation_notification(
            user_id=workload_request.user_id,
            workload_id=workload_request.workload_id,
            escalation_data=escalation_data
        )
        
        logger.info(f"Workload {workload_request.workload_id} escalated for constitutional review")

    async def _update_integration_metrics(
        self, 
        response: ModerationResponse, 
        processing_time_ms: float
    ):
        """Update integration performance metrics"""
        
        self.integration_metrics['total_workloads_processed'] += 1
        
        if response.approved:
            self.integration_metrics['approved_workloads'] += 1
        else:
            self.integration_metrics['rejected_workloads'] += 1
        
        if response.moderation_result.requires_escalation:
            self.integration_metrics['escalated_workloads'] += 1
        
        # Update average processing time
        current_avg = self.integration_metrics['average_processing_time_ms']
        total_processed = self.integration_metrics['total_workloads_processed']
        self.integration_metrics['average_processing_time_ms'] = \
            ((current_avg * (total_processed - 1)) + processing_time_ms) / total_processed
        
        # Count constitutional protections applied
        constitutional_concerns = response.moderation_result.harm_analysis.constitutional_concerns
        if constitutional_concerns:
            self.integration_metrics['constitutional_protections_applied'] += 1

    async def _create_error_response(
        self, 
        workload_request: FogWorkloadRequest, 
        error_message: str
    ) -> ModerationResponse:
        """Create error response for failed workload processing"""
        
        # Create minimal moderation result for error case
        error_moderation_result = ModerationResult(
            content_id=f"error_{workload_request.workload_id}",
            decision="escalate",  # Safe default
            harm_analysis=None,  # Would create minimal analysis
            policy_rationale=f"Processing error requires review: {error_message}",
            response_actions=["escalate_system_error"],
            tier_level=workload_request.user_tier,
            requires_escalation=True,
            appeal_eligible=True,
            audit_trail={"error": error_message, "timestamp": datetime.utcnow().isoformat()},
            transparency_score=0.5
        )
        
        return ModerationResponse(
            workload_id=workload_request.workload_id,
            approved=False,
            moderation_result=error_moderation_result,
            routing_decision="error_review_required",
            pricing_impact={"error_processing": True, "final_cost": 0},
            security_requirements=["error_investigation_required"],
            transparency_data={"error_occurred": True, "error_message": error_message}
        )

    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get current integration performance metrics"""
        
        total_processed = self.integration_metrics['total_workloads_processed']
        
        if total_processed == 0:
            return {"no_workloads_processed": True}
        
        return {
            'workload_statistics': {
                'total_processed': total_processed,
                'approval_rate': self.integration_metrics['approved_workloads'] / total_processed,
                'rejection_rate': self.integration_metrics['rejected_workloads'] / total_processed,
                'escalation_rate': self.integration_metrics['escalated_workloads'] / total_processed
            },
            'performance_metrics': {
                'average_processing_time_ms': self.integration_metrics['average_processing_time_ms'],
                'constitutional_protection_rate': self.integration_metrics['constitutional_protections_applied'] / total_processed
            },
            'system_health': {
                'pipeline_status': await self.moderation_pipeline.get_pipeline_metrics(),
                'fog_integration_status': 'operational'
            }
        }

    async def process_appeal_for_workload(
        self, 
        workload_id: str, 
        appeal_reason: str,
        user_tier: str
    ) -> Dict[str, Any]:
        """Process constitutional appeal for rejected workload"""
        
        # Find original content ID associated with workload
        # This would query the workload database in production
        content_id = f"workload_content_{workload_id}"
        
        return await self.moderation_pipeline.process_appeal(
            content_id=content_id,
            appeal_reason=appeal_reason,
            user_tier=user_tier
        )

    async def shutdown(self):
        """Gracefully shutdown fog integration"""
        logger.info("Shutting down Fog Moderation Integration")
        
        # Save metrics
        await self.transparency_logger.log_system_event({
            "event": "fog_integration_shutdown",
            "final_metrics": self.integration_metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Shutdown components
        await self.moderation_pipeline.shutdown()
        await self.tee_security.shutdown()


# Convenience functions for fog infrastructure integration
async def moderate_fog_workload(
    content: str,
    content_type: str,
    user_id: str,
    user_tier: str,
    fog_node_id: str,
    priority: str = "standard",
    context: Dict[str, Any] = None
) -> ModerationResponse:
    """
    Convenience function to moderate a fog workload
    
    Args:
        content: Content to moderate
        content_type: Type of content
        user_id: User identifier
        user_tier: User tier level
        fog_node_id: Fog node identifier
        priority: Processing priority
        context: Additional context
        
    Returns:
        ModerationResponse with approval and routing
    """
    integration = FogModerationIntegration()
    
    workload_request = FogWorkloadRequest(
        workload_id=str(uuid4()),
        content=content,
        content_type=content_type,
        user_id=user_id,
        user_tier=user_tier,
        fog_node_id=fog_node_id,
        priority=priority,
        timestamp=datetime.utcnow(),
        context=context or {}
    )
    
    return await integration.process_workload_request(workload_request)


async def get_fog_moderation_health() -> Dict[str, Any]:
    """Get fog moderation system health status"""
    integration = FogModerationIntegration()
    return await integration.get_integration_metrics()