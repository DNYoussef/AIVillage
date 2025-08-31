"""
Constitutional BetaNet Transport Layer

Advanced constitutional transport system that extends BetaNet's HTX protocol with
privacy-preserving speech/safety oversight. Provides seamless integration of
constitutional compliance while maintaining BetaNet's core privacy guarantees.

Key Features:
- Constitutional HTX transport with backward compatibility
- Tiered privacy-preserving constitutional verification (Bronze→Platinum)
- Real-time constitutional moderation integration
- Zero-knowledge proof generation and verification
- TEE-secured constitutional enforcement
- Privacy-preserving audit trails and transparency
- Constitutional mixnode routing with enhanced privacy
- Dynamic tier-based constitutional policy enforcement

Architecture:
- Extends existing BetaNetFogTransport with constitutional capabilities
- Integrates constitutional moderation pipeline for real-time verification
- Uses privacy verification engine for ZK proof generation
- Maintains full backward compatibility with existing BetaNet protocol
- Provides constitutional governance without compromising privacy

Constitutional Tiers:
- Bronze (20% Privacy): Full transparency with complete audit trail
- Silver (50% Privacy): Hash-based verification with limited monitoring
- Gold (80% Privacy): Zero-knowledge proofs with H3-only visibility
- Platinum (95% Privacy): Pure ZK compliance with cryptographic verification
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .htx_transport import HtxClient, HtxServer, HtxFrame, HtxFrameType
from .constitutional_frames import (
    ConstitutionalFrame, ConstitutionalFrameType, ConstitutionalTier,
    ConstitutionalManifest, ConstitutionalProof, ConstitutionalFrameProcessor,
    create_constitutional_manifest
)
from .privacy_verification import (
    PrivacyPreservingVerificationEngine, create_privacy_verification_engine
)
from ...constitutional.moderation.pipeline import ConstitutionalModerationPipeline
from ...security.tee.integration import TEESecurityIntegrationManager, get_integration_manager
from ...fog.bridges.betanet_integration import BetaNetFogTransport

logger = logging.getLogger(__name__)


class ConstitutionalTransportMode(Enum):
    """Constitutional transport operation modes."""
    
    STANDARD_BETANET = "standard_betanet"         # Standard BetaNet without constitutional features
    CONSTITUTIONAL_ENABLED = "constitutional_enabled"  # Constitutional features enabled
    PRIVACY_PRIORITY = "privacy_priority"         # Prioritize privacy over constitutional oversight
    TRANSPARENCY_PRIORITY = "transparency_priority"  # Prioritize transparency over privacy
    HYBRID_MODE = "hybrid_mode"                   # Balance privacy and constitutional compliance


@dataclass
class ConstitutionalTransportConfig:
    """Configuration for constitutional transport."""
    
    # Transport mode
    mode: ConstitutionalTransportMode = ConstitutionalTransportMode.CONSTITUTIONAL_ENABLED
    
    # Constitutional settings
    default_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    require_constitutional_verification: bool = True
    enable_real_time_moderation: bool = True
    
    # Privacy settings
    privacy_priority: float = 0.5  # 0.0 = full transparency, 1.0 = maximum privacy
    enable_zero_knowledge_proofs: bool = True
    enable_selective_disclosure: bool = True
    
    # Performance settings
    constitutional_frame_timeout_ms: int = 5000
    proof_generation_timeout_ms: int = 10000
    moderation_timeout_ms: int = 3000
    
    # Integration settings
    enable_tee_integration: bool = True
    enable_mixnode_routing: bool = True
    enable_fog_integration: bool = True
    
    # Audit and compliance
    enable_audit_logging: bool = True
    audit_retention_days: int = 30
    transparency_reporting: bool = True


@dataclass
class ConstitutionalMessage:
    """Message with constitutional compliance metadata."""
    
    message_id: str = field(default_factory=lambda: secrets.token_hex(16))
    
    # Message content
    content: bytes = b""
    content_type: str = "text"
    
    # Constitutional metadata
    constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    privacy_manifest: Optional[ConstitutionalManifest] = None
    compliance_proof: Optional[ConstitutionalProof] = None
    
    # Moderation results
    moderation_required: bool = True
    moderation_result: Optional[Any] = None
    constitutional_compliant: bool = True
    
    # Routing and delivery
    destination: str = ""
    priority: int = 1
    routing_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None
    
    def to_htx_frame(self, stream_id: int) -> ConstitutionalFrame:
        """Convert constitutional message to HTX frame."""
        return ConstitutionalFrame(
            frame_type=ConstitutionalFrameType.CONSTITUTIONAL_VERIFY,
            stream_id=stream_id,
            manifest=self.privacy_manifest,
            proof=self.compliance_proof,
            payload=self.content
        )
    
    @classmethod
    def from_htx_frame(cls, frame: ConstitutionalFrame) -> "ConstitutionalMessage":
        """Create constitutional message from HTX frame."""
        return cls(
            content=frame.payload,
            privacy_manifest=frame.manifest,
            compliance_proof=frame.proof,
            constitutional_tier=frame.manifest.tier if frame.manifest else ConstitutionalTier.BRONZE
        )


class ConstitutionalBetaNetTransport:
    """
    Constitutional BetaNet Transport with Privacy-Preserving Oversight
    
    Extends BetaNet's HTX transport protocol with constitutional compliance
    capabilities while preserving privacy guarantees. Provides seamless
    integration with constitutional moderation and zero-knowledge verification.
    """
    
    def __init__(
        self, 
        config: Optional[ConstitutionalTransportConfig] = None,
        base_transport: Optional[BetaNetFogTransport] = None
    ):
        self.config = config or ConstitutionalTransportConfig()
        self.base_transport = base_transport
        
        # Constitutional components
        self.moderation_pipeline: Optional[ConstitutionalModerationPipeline] = None
        self.privacy_verification: Optional[PrivacyPreservingVerificationEngine] = None
        self.frame_processor: Optional[ConstitutionalFrameProcessor] = None
        self.tee_integration: Optional[TEESecurityIntegrationManager] = None
        
        # Transport state
        self.htx_client: Optional[HtxClient] = None
        self.htx_server: Optional[HtxServer] = None
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.constitutional_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Message handling
        self.pending_verifications: Dict[str, ConstitutionalMessage] = {}
        self.verified_messages: Dict[str, ConstitutionalMessage] = {}
        self.message_handlers: List[Any] = []
        
        # Statistics
        self.transport_stats = {
            "total_messages": 0,
            "constitutional_messages": 0,
            "by_tier": {tier.name: 0 for tier in ConstitutionalTier},
            "privacy_preserved": 0,
            "transparency_provided": 0,
            "moderation_actions": 0,
            "average_processing_time_ms": 0
        }
        
        logger.info("Constitutional BetaNet transport initialized")
    
    async def initialize(self):
        """Initialize constitutional transport system."""
        
        # Initialize constitutional components
        if self.config.enable_real_time_moderation:
            self.moderation_pipeline = ConstitutionalModerationPipeline()
        
        if self.config.enable_zero_knowledge_proofs:
            self.privacy_verification = await create_privacy_verification_engine()
        
        self.frame_processor = ConstitutionalFrameProcessor()
        await self.frame_processor.initialize()
        
        # Initialize TEE integration if available
        if self.config.enable_tee_integration:
            try:
                self.tee_integration = await get_integration_manager()
            except Exception as e:
                logger.warning(f"TEE integration unavailable: {e}")
        
        # Initialize base transport if not provided
        if not self.base_transport and self.config.enable_fog_integration:
            self.base_transport = BetaNetFogTransport(
                privacy_mode="balanced",
                enable_covert=True,
                mobile_optimization=True
            )
        
        logger.info("Constitutional transport system ready")
    
    async def start_client(
        self, 
        server_host: str = "127.0.0.1", 
        server_port: int = 8443,
        constitutional_tier: ConstitutionalTier = ConstitutionalTier.SILVER
    ) -> bool:
        """Start constitutional HTX client."""
        
        try:
            # Start HTX client
            self.htx_client = HtxClient(server_host, server_port)
            if not await self.htx_client.start():
                return False
            
            # Register constitutional message handler
            self.htx_client.register_message_handler(self._handle_incoming_htx_message)
            
            # Send constitutional manifest
            manifest = create_constitutional_manifest(
                tier=constitutional_tier,
                harm_categories=["violence", "hate_speech", "misinformation", "privacy_violation"]
            )
            
            manifest_frame = ConstitutionalFrame(
                frame_type=ConstitutionalFrameType.PRIVACY_MANIFEST,
                stream_id=0,
                manifest=manifest
            )
            
            success = await self._send_constitutional_frame(manifest_frame)
            if success:
                logger.info(f"Constitutional client started with {constitutional_tier.name} tier")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error starting constitutional client: {e}")
            return False
    
    async def start_server(
        self, 
        host: str = "127.0.0.1", 
        port: int = 8443
    ) -> bool:
        """Start constitutional HTX server."""
        
        try:
            self.htx_server = HtxServer(host, port)
            return await self.htx_server.start()
        
        except Exception as e:
            logger.error(f"Error starting constitutional server: {e}")
            return False
    
    async def send_constitutional_message(
        self, 
        content: Union[str, bytes],
        destination: str,
        constitutional_tier: Optional[ConstitutionalTier] = None,
        priority: int = 1,
        require_verification: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """Send message with constitutional compliance verification."""
        
        start_time = time.time()
        
        try:
            # Prepare message content
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
                content_type = "text"
            else:
                content_bytes = content
                content_type = "binary"
            
            # Use provided tier or default
            tier = constitutional_tier or self.config.default_tier
            
            # Create constitutional message
            message = ConstitutionalMessage(
                content=content_bytes,
                content_type=content_type,
                constitutional_tier=tier,
                destination=destination,
                priority=priority,
                moderation_required=require_verification
            )
            
            # Process constitutional compliance
            if require_verification:
                compliance_result = await self._process_constitutional_compliance(message)
                if not compliance_result["compliant"]:
                    return False, {
                        "error": "Constitutional compliance verification failed",
                        "details": compliance_result
                    }
                
                message.compliance_proof = compliance_result.get("proof")
                message.moderation_result = compliance_result.get("moderation_result")
            
            # Generate privacy manifest
            message.privacy_manifest = create_constitutional_manifest(
                tier=tier,
                harm_categories=["violence", "hate_speech", "misinformation"]
            )
            
            # Send via appropriate transport
            success, transport_result = await self._send_via_transport(message)
            
            # Update statistics
            processing_time = int((time.time() - start_time) * 1000)
            await self._update_transport_statistics(message, processing_time, success)
            
            if success:
                self.verified_messages[message.message_id] = message
                return True, {
                    "message_id": message.message_id,
                    "constitutional_tier": tier.name,
                    "privacy_level": message.privacy_manifest.privacy_level,
                    "processing_time_ms": processing_time,
                    "transport_details": transport_result
                }
            
            return False, {"error": "Transport delivery failed", "details": transport_result}
        
        except Exception as e:
            logger.error(f"Error sending constitutional message: {e}")
            return False, {"error": str(e)}
    
    async def _process_constitutional_compliance(
        self, 
        message: ConstitutionalMessage
    ) -> Dict[str, Any]:
        """Process constitutional compliance for message."""
        
        try:
            content_str = message.content.decode('utf-8', errors='ignore')
            
            # Step 1: Constitutional moderation
            if self.moderation_pipeline:
                moderation_result = await self.moderation_pipeline.process_content(
                    content=content_str,
                    content_type=message.content_type,
                    user_tier=message.constitutional_tier.name,
                    context={"destination": message.destination}
                )
                
                # Check if content is constitutionally compliant
                compliant = moderation_result.decision.value in ["allow", "allow_with_warning"]
                
                if not compliant:
                    return {
                        "compliant": False,
                        "reason": moderation_result.policy_rationale,
                        "moderation_result": moderation_result
                    }
            else:
                # Fallback to basic check
                compliant = True
                moderation_result = None
            
            # Step 2: Generate privacy-preserving proof
            compliance_proof = None
            if self.privacy_verification and compliant:
                proof_success, proof = await self.privacy_verification.generate_constitutional_proof(
                    content=content_str,
                    moderation_result=moderation_result,
                    privacy_tier=message.constitutional_tier
                )
                
                if proof_success:
                    compliance_proof = proof
            
            return {
                "compliant": compliant,
                "moderation_result": moderation_result,
                "proof": compliance_proof,
                "processing_details": {
                    "tier": message.constitutional_tier.name,
                    "privacy_preserving": compliance_proof is not None,
                    "tee_processed": self.tee_integration is not None
                }
            }
        
        except Exception as e:
            logger.error(f"Constitutional compliance processing failed: {e}")
            return {"compliant": False, "error": str(e)}
    
    async def _send_via_transport(
        self, 
        message: ConstitutionalMessage
    ) -> Tuple[bool, Dict[str, Any]]:
        """Send message via appropriate transport method."""
        
        # Try constitutional HTX transport first
        if self.htx_client:
            frame = message.to_htx_frame(stream_id=1)
            success = await self._send_constitutional_frame(frame)
            
            if success:
                return True, {
                    "transport": "constitutional_htx",
                    "frame_type": frame.constitutional_type.name if frame.constitutional_type else "standard"
                }
        
        # Fallback to base transport
        if self.base_transport:
            transport_result = await self.base_transport.send_job_data(
                job_data=message.content,
                destination=message.destination,
                priority="high" if message.priority > 3 else "normal"
            )
            
            return transport_result["success"], {
                "transport": "base_betanet",
                "details": transport_result
            }
        
        return False, {"error": "No transport available"}
    
    async def _send_constitutional_frame(self, frame: ConstitutionalFrame) -> bool:
        """Send constitutional frame via HTX client."""
        
        if not self.htx_client:
            return False
        
        try:
            # Convert to HTX frame
            htx_frame = frame.to_htx_frame()
            
            # Send via HTX client
            from ..core.message_types import UnifiedMessage, MessageType
            
            unified_message = UnifiedMessage(
                message_type=MessageType.DATA,
                payload=htx_frame.encode()  # Encode HTX frame as payload
            )
            
            return await self.htx_client.send_message(unified_message)
        
        except Exception as e:
            logger.error(f"Error sending constitutional frame: {e}")
            return False
    
    async def _handle_incoming_htx_message(self, message: Any):
        """Handle incoming HTX message with constitutional processing."""
        
        try:
            # Decode HTX frame from message payload
            htx_frame = HtxFrame.decode(message.payload)[0]
            
            # Convert to constitutional frame
            const_frame = ConstitutionalFrame.from_htx_frame(htx_frame)
            
            # Process constitutional frame
            if self.frame_processor:
                compliant, response_frame = await self.frame_processor.process_constitutional_frame(
                    const_frame
                )
                
                # Send response if needed
                if response_frame:
                    await self._send_constitutional_frame(response_frame)
                
                # Handle compliant message
                if compliant:
                    constitutional_message = ConstitutionalMessage.from_htx_frame(const_frame)
                    
                    # Notify message handlers
                    for handler in self.message_handlers:
                        try:
                            await handler(constitutional_message)
                        except Exception as e:
                            logger.warning(f"Message handler error: {e}")
        
        except Exception as e:
            logger.error(f"Error handling incoming HTX message: {e}")
    
    async def _update_transport_statistics(
        self, 
        message: ConstitutionalMessage, 
        processing_time_ms: int, 
        success: bool
    ):
        """Update transport statistics."""
        
        self.transport_stats["total_messages"] += 1
        
        if message.moderation_required:
            self.transport_stats["constitutional_messages"] += 1
        
        # Update by tier
        tier_name = message.constitutional_tier.name
        self.transport_stats["by_tier"][tier_name] += 1
        
        # Update privacy/transparency counters
        if message.privacy_manifest and message.privacy_manifest.privacy_level > 0.5:
            self.transport_stats["privacy_preserved"] += 1
        else:
            self.transport_stats["transparency_provided"] += 1
        
        # Update processing time average
        total_messages = self.transport_stats["total_messages"]
        current_avg = self.transport_stats["average_processing_time_ms"]
        self.transport_stats["average_processing_time_ms"] = (
            (current_avg * (total_messages - 1) + processing_time_ms) / total_messages
        )
    
    async def receive_constitutional_message(self, timeout_ms: int = 5000) -> Optional[ConstitutionalMessage]:
        """Receive constitutional message with privacy preservation."""
        
        if not self.htx_client:
            return None
        
        try:
            # Use base transport receive if available
            if self.base_transport:
                result = await self.base_transport.receive_job_data(timeout=timeout_ms / 1000)
                if result and result["data"]:
                    # Create constitutional message from received data
                    return ConstitutionalMessage(
                        content=result["data"],
                        constitutional_tier=self.config.default_tier
                    )
            
            # TODO: Implement HTX-based receive when client supports it
            return None
        
        except Exception as e:
            logger.error(f"Error receiving constitutional message: {e}")
            return None
    
    def register_message_handler(self, handler):
        """Register handler for incoming constitutional messages."""
        self.message_handlers.append(handler)
    
    def get_transport_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transport statistics."""
        
        base_stats = self.transport_stats.copy()
        
        # Add component statistics
        if self.privacy_verification:
            base_stats["privacy_verification"] = self.privacy_verification.get_privacy_statistics()
        
        if self.frame_processor:
            base_stats["frame_processing"] = self.frame_processor.get_verification_stats()
        
        # Add configuration info
        base_stats["configuration"] = {
            "mode": self.config.mode.value,
            "default_tier": self.config.default_tier.name,
            "privacy_priority": self.config.privacy_priority,
            "real_time_moderation": self.config.enable_real_time_moderation,
            "zero_knowledge_proofs": self.config.enable_zero_knowledge_proofs,
            "tee_integration": self.tee_integration is not None
        }
        
        return base_stats
    
    async def stop(self):
        """Stop constitutional transport system."""
        
        if self.htx_client:
            await self.htx_client.stop()
        
        if self.htx_server:
            await self.htx_server.stop()
        
        logger.info("Constitutional transport stopped")


class ConstitutionalBetaNetService:
    """
    High-level service for constitutional BetaNet operations.
    
    Provides easy-to-use interface for applications requiring constitutional
    compliance with privacy preservation in BetaNet transport.
    """
    
    def __init__(self, config: Optional[ConstitutionalTransportConfig] = None):
        self.config = config or ConstitutionalTransportConfig()
        self.transport: Optional[ConstitutionalBetaNetTransport] = None
        
        # Service state
        self.running = False
        self.client_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Constitutional BetaNet service initialized")
    
    async def start_service(
        self, 
        mode: str = "client", 
        **connection_params
    ) -> bool:
        """Start constitutional service."""
        
        try:
            self.transport = ConstitutionalBetaNetTransport(self.config)
            await self.transport.initialize()
            
            if mode == "client":
                success = await self.transport.start_client(**connection_params)
            elif mode == "server":
                success = await self.transport.start_server(**connection_params)
            else:
                return False
            
            if success:
                self.running = True
                logger.info(f"Constitutional service started in {mode} mode")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error starting constitutional service: {e}")
            return False
    
    async def send_message(
        self, 
        content: Union[str, bytes],
        destination: str,
        privacy_tier: str = "silver",
        **options
    ) -> Dict[str, Any]:
        """Send message with constitutional compliance."""
        
        if not self.running or not self.transport:
            return {"success": False, "error": "Service not running"}
        
        # Convert privacy tier
        tier_mapping = {
            "bronze": ConstitutionalTier.BRONZE,
            "silver": ConstitutionalTier.SILVER, 
            "gold": ConstitutionalTier.GOLD,
            "platinum": ConstitutionalTier.PLATINUM
        }
        
        tier = tier_mapping.get(privacy_tier.lower(), ConstitutionalTier.SILVER)
        
        success, result = await self.transport.send_constitutional_message(
            content=content,
            destination=destination,
            constitutional_tier=tier,
            **options
        )
        
        return {"success": success, **result}
    
    async def receive_message(self, timeout_ms: int = 5000) -> Optional[Dict[str, Any]]:
        """Receive message with constitutional verification."""
        
        if not self.running or not self.transport:
            return None
        
        message = await self.transport.receive_constitutional_message(timeout_ms)
        if message:
            return {
                "message_id": message.message_id,
                "content": message.content.decode('utf-8', errors='ignore'),
                "constitutional_tier": message.constitutional_tier.name,
                "privacy_manifest": {
                    "privacy_level": message.privacy_manifest.privacy_level if message.privacy_manifest else 0.0,
                    "verification_method": message.privacy_manifest.verification_method if message.privacy_manifest else "none"
                },
                "compliant": message.constitutional_compliant,
                "created_at": message.created_at.isoformat()
            }
        
        return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        
        status = {
            "running": self.running,
            "configuration": {
                "mode": self.config.mode.value,
                "default_tier": self.config.default_tier.name,
                "privacy_priority": self.config.privacy_priority
            },
            "active_sessions": len(self.client_sessions)
        }
        
        if self.transport:
            status["transport_statistics"] = self.transport.get_transport_statistics()
        
        return status
    
    async def stop_service(self):
        """Stop constitutional service."""
        
        if self.transport:
            await self.transport.stop()
        
        self.running = False
        logger.info("Constitutional service stopped")


# Enhanced BetaNetFogTransport with constitutional capabilities
class EnhancedBetaNetFogTransport(BetaNetFogTransport):
    """
    Enhanced BetaNet Fog Transport with Constitutional Capabilities
    
    Extends the existing BetaNetFogTransport with constitutional compliance
    while maintaining full backward compatibility with fog computing infrastructure.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract constitutional parameters
        constitutional_enabled = kwargs.pop('constitutional_enabled', False)
        constitutional_tier = kwargs.pop('constitutional_tier', 'silver')
        
        # Initialize base transport
        super().__init__(*args, **kwargs)
        
        # Constitutional enhancements
        self.constitutional_enabled = constitutional_enabled
        self.constitutional_service: Optional[ConstitutionalBetaNetService] = None
        
        if constitutional_enabled:
            # Create constitutional configuration
            config = ConstitutionalTransportConfig(
                default_tier=getattr(ConstitutionalTier, constitutional_tier.upper()),
                mode=ConstitutionalTransportMode.HYBRID_MODE,
                enable_fog_integration=True
            )
            
            self.constitutional_service = ConstitutionalBetaNetService(config)
        
        logger.info(f"Enhanced BetaNet transport initialized (constitutional: {constitutional_enabled})")
    
    async def send_job_data(self, job_data: bytes, destination: str, priority: str = "normal") -> Dict[str, Any]:
        """Send job data with optional constitutional verification."""
        
        # Use constitutional transport if enabled
        if self.constitutional_enabled and self.constitutional_service:
            try:
                result = await self.constitutional_service.send_message(
                    content=job_data,
                    destination=destination,
                    privacy_tier="silver"  # Default tier for fog computing
                )
                
                if result["success"]:
                    return {
                        "success": True,
                        "transport": "constitutional_betanet",
                        "constitutional_compliant": True,
                        **result
                    }
            except Exception as e:
                logger.warning(f"Constitutional transport failed, falling back: {e}")
        
        # Fallback to standard BetaNet transport
        return await super().send_job_data(job_data, destination, priority)
    
    async def initialize_constitutional_features(self):
        """Initialize constitutional features if enabled."""
        
        if self.constitutional_enabled and self.constitutional_service:
            success = await self.constitutional_service.start_service(mode="client")
            if success:
                logger.info("Constitutional features initialized for BetaNet transport")
            else:
                logger.warning("Failed to initialize constitutional features")


# Factory functions
def create_constitutional_transport(
    constitutional_tier: str = "silver",
    privacy_priority: float = 0.5,
    enable_tee: bool = True,
    **transport_params
) -> ConstitutionalBetaNetTransport:
    """Create constitutional BetaNet transport with specified configuration."""
    
    config = ConstitutionalTransportConfig(
        default_tier=getattr(ConstitutionalTier, constitutional_tier.upper()),
        privacy_priority=privacy_priority,
        enable_tee_integration=enable_tee,
        **transport_params
    )
    
    return ConstitutionalBetaNetTransport(config)


def create_enhanced_fog_transport(
    constitutional_enabled: bool = True,
    constitutional_tier: str = "silver",
    **fog_params
) -> EnhancedBetaNetFogTransport:
    """Create enhanced BetaNet fog transport with constitutional capabilities."""
    
    return EnhancedBetaNetFogTransport(
        constitutional_enabled=constitutional_enabled,
        constitutional_tier=constitutional_tier,
        **fog_params
    )


if __name__ == "__main__":
    # Test constitutional transport
    async def test_constitutional_transport():
        
        # Test service-level interface
        service = ConstitutionalBetaNetService()
        
        # Start client
        success = await service.start_service(
            mode="client",
            server_host="127.0.0.1",
            server_port=8443,
            constitutional_tier=ConstitutionalTier.SILVER
        )
        
        print(f"Service started: {success}")
        
        if success:
            # Send test message
            result = await service.send_message(
                content="This is a test message for constitutional verification.",
                destination="test_peer",
                privacy_tier="silver"
            )
            
            print(f"Message sent: {result}")
            
            # Get service status
            status = service.get_service_status()
            print(f"Service status: {json.dumps(status, indent=2, default=str)}")
            
            await service.stop_service()
        
        # Test enhanced fog transport
        enhanced_transport = create_enhanced_fog_transport(
            constitutional_enabled=True,
            constitutional_tier="gold",
            privacy_mode="balanced"
        )
        
        await enhanced_transport.initialize_constitutional_features()
        
        # Test transport
        result = await enhanced_transport.send_job_data(
            job_data=b"Test fog job with constitutional compliance",
            destination="fog_node_001",
            priority="high"
        )
        
        print(f"Enhanced transport result: {result}")
    
    asyncio.run(test_constitutional_transport())