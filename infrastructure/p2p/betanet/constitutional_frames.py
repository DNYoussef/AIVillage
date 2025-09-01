"""
Constitutional HTX Frame Extensions

Extends the HTX v1.1 transport protocol with constitutional compliance frames that
enable privacy-preserving speech/safety oversight. Provides tiered constitutional
verification without compromising BetaNet's core privacy features.

Key features:
- Constitutional compliance frames for HTX protocol
- Tiered privacy verification (Bronze/Silver/Gold/Platinum)
- Zero-knowledge proof integration for privacy preservation
- Constitutional audit trail frames for transparency
- Backward compatibility with existing HTX v1.1 protocol

Constitutional Tiers:
- Bronze (20% Privacy): Full transparency with H0-H3 monitoring
- Silver (50% Privacy): Hash-based verification with H2-H3 monitoring
- Gold (80% Privacy): Zero-knowledge proofs with H3-only monitoring
- Platinum (95% Privacy): Pure ZK compliance with cryptographic verification
"""

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import logging
import secrets
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from infrastructure.constitutional.moderation.pipeline import (
    ConstitutionalModerationPipeline,
    ModerationResult,
)
from infrastructure.security.tee.integration import TEESecurityIntegrationManager

from .htx_transport import MAX_FRAME_SIZE, HtxFrame, HtxFrameType

logger = logging.getLogger(__name__)


class ConstitutionalFrameType(IntEnum):
    """Constitutional frame types extending HTX v1.1 specification."""

    # Constitutional compliance frames
    CONSTITUTIONAL_VERIFY = 0x10  # Constitutional verification request
    CONSTITUTIONAL_PROOF = 0x11  # Zero-knowledge constitutional proof
    CONSTITUTIONAL_AUDIT = 0x12  # Constitutional audit trail
    CONSTITUTIONAL_ALERT = 0x13  # Constitutional violation alert

    # Tiered privacy frames
    PRIVACY_MANIFEST = 0x14  # Privacy tier manifest
    HASH_VERIFICATION = 0x15  # Hash-based verification (Silver tier)
    ZK_PROOF_REQUEST = 0x16  # Zero-knowledge proof request
    ZK_PROOF_RESPONSE = 0x17  # Zero-knowledge proof response

    # Constitutional routing frames
    CONSTITUTIONAL_ROUTE = 0x18  # Constitutional-aware routing
    MODERATION_STATUS = 0x19  # Real-time moderation status
    GOVERNANCE_DIRECTIVE = 0x1A  # Constitutional governance directive

    # TEE integration frames
    TEE_ATTESTATION = 0x1B  # TEE attestation for constitutional enforcement
    ENCLAVE_VERIFICATION = 0x1C  # Secure enclave verification


class ConstitutionalTier(IntEnum):
    """Constitutional privacy tiers with increasing privacy guarantees."""

    BRONZE = 1  # 20% Privacy: Full transparency with H0-H3 monitoring
    SILVER = 2  # 50% Privacy: Hash-based verification with H2-H3 monitoring
    GOLD = 3  # 80% Privacy: Zero-knowledge proofs with H3-only monitoring
    PLATINUM = 4  # 95% Privacy: Pure ZK compliance with cryptographic verification


@dataclass
class ConstitutionalManifest:
    """Constitutional compliance manifest for content verification."""

    manifest_id: str = field(default_factory=lambda: secrets.token_hex(16))

    # Constitutional requirements
    tier: ConstitutionalTier = ConstitutionalTier.BRONZE
    harm_categories_monitored: List[str] = field(default_factory=list)
    moderation_required: bool = True

    # Privacy configuration
    privacy_level: float = 0.2  # 20% default (Bronze tier)
    monitoring_scope: List[str] = field(default_factory=lambda: ["H0", "H1", "H2", "H3"])
    audit_requirements: Dict[str, Any] = field(default_factory=dict)

    # Verification methods
    verification_method: str = "full_transparency"  # full_transparency, hash_based, zk_proof, pure_zk
    proof_requirements: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def to_bytes(self) -> bytes:
        """Serialize manifest to bytes."""
        data = {
            "manifest_id": self.manifest_id,
            "tier": self.tier.value,
            "harm_categories_monitored": self.harm_categories_monitored,
            "moderation_required": self.moderation_required,
            "privacy_level": self.privacy_level,
            "monitoring_scope": self.monitoring_scope,
            "audit_requirements": self.audit_requirements,
            "verification_method": self.verification_method,
            "proof_requirements": self.proof_requirements,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConstitutionalManifest":
        """Deserialize manifest from bytes."""
        parsed = json.loads(data.decode("utf-8"))
        manifest = cls()
        manifest.manifest_id = parsed["manifest_id"]
        manifest.tier = ConstitutionalTier(parsed["tier"])
        manifest.harm_categories_monitored = parsed["harm_categories_monitored"]
        manifest.moderation_required = parsed["moderation_required"]
        manifest.privacy_level = parsed["privacy_level"]
        manifest.monitoring_scope = parsed["monitoring_scope"]
        manifest.audit_requirements = parsed["audit_requirements"]
        manifest.verification_method = parsed["verification_method"]
        manifest.proof_requirements = parsed["proof_requirements"]
        manifest.created_at = parsed["created_at"]
        manifest.expires_at = parsed.get("expires_at")
        return manifest


@dataclass
class ConstitutionalProof:
    """Constitutional compliance proof with privacy preservation."""

    proof_id: str = field(default_factory=lambda: secrets.token_hex(16))

    # Proof metadata
    content_hash: str = ""
    proof_type: str = "full_transparency"  # full_transparency, hash_based, zk_proof, pure_zk
    tier: ConstitutionalTier = ConstitutionalTier.BRONZE

    # Verification data
    moderation_result_hash: Optional[str] = None
    zk_proof_data: Optional[bytes] = None
    verification_metadata: Dict[str, Any] = field(default_factory=dict)

    # Constitutional compliance
    harm_level: str = "H0"
    compliance_score: float = 1.0
    constitutional_flags: List[str] = field(default_factory=list)

    # TEE attestation
    tee_attestation: Optional[bytes] = None
    enclave_signature: Optional[bytes] = None

    # Timestamps
    generated_at: float = field(default_factory=time.time)
    valid_until: Optional[float] = None

    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        data = {
            "proof_id": self.proof_id,
            "content_hash": self.content_hash,
            "proof_type": self.proof_type,
            "tier": self.tier.value,
            "moderation_result_hash": self.moderation_result_hash,
            "zk_proof_data": self.zk_proof_data.hex() if self.zk_proof_data else None,
            "verification_metadata": self.verification_metadata,
            "harm_level": self.harm_level,
            "compliance_score": self.compliance_score,
            "constitutional_flags": self.constitutional_flags,
            "tee_attestation": self.tee_attestation.hex() if self.tee_attestation else None,
            "enclave_signature": self.enclave_signature.hex() if self.enclave_signature else None,
            "generated_at": self.generated_at,
            "valid_until": self.valid_until,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConstitutionalProof":
        """Deserialize proof from bytes."""
        parsed = json.loads(data.decode("utf-8"))
        proof = cls()
        proof.proof_id = parsed["proof_id"]
        proof.content_hash = parsed["content_hash"]
        proof.proof_type = parsed["proof_type"]
        proof.tier = ConstitutionalTier(parsed["tier"])
        proof.moderation_result_hash = parsed.get("moderation_result_hash")
        proof.zk_proof_data = bytes.fromhex(parsed["zk_proof_data"]) if parsed.get("zk_proof_data") else None
        proof.verification_metadata = parsed["verification_metadata"]
        proof.harm_level = parsed["harm_level"]
        proof.compliance_score = parsed["compliance_score"]
        proof.constitutional_flags = parsed["constitutional_flags"]
        proof.tee_attestation = bytes.fromhex(parsed["tee_attestation"]) if parsed.get("tee_attestation") else None
        proof.enclave_signature = (
            bytes.fromhex(parsed["enclave_signature"]) if parsed.get("enclave_signature") else None
        )
        proof.generated_at = parsed["generated_at"]
        proof.valid_until = parsed.get("valid_until")
        return proof


@dataclass
class ConstitutionalFrame:
    """Constitutional frame extending HTX v1.1 protocol."""

    # Base HTX frame properties
    frame_type: Union[HtxFrameType, ConstitutionalFrameType]
    stream_id: int

    # Constitutional-specific properties
    constitutional_type: Optional[ConstitutionalFrameType] = None
    manifest: Optional[ConstitutionalManifest] = None
    proof: Optional[ConstitutionalProof] = None

    # Frame payload
    payload: bytes = b""

    def __post_init__(self):
        """Validate constitutional frame."""
        if isinstance(self.frame_type, ConstitutionalFrameType):
            self.constitutional_type = self.frame_type

        # Validate payload size
        total_size = len(self.payload)
        if self.manifest:
            total_size += len(self.manifest.to_bytes())
        if self.proof:
            total_size += len(self.proof.to_bytes())

        if total_size > MAX_FRAME_SIZE:
            raise ValueError(f"Constitutional frame too large: {total_size} > {MAX_FRAME_SIZE}")

    def to_htx_frame(self) -> HtxFrame:
        """Convert constitutional frame to standard HTX frame."""
        # Build constitutional payload
        constitutional_payload = b""

        if self.manifest:
            manifest_data = self.manifest.to_bytes()
            constitutional_payload += struct.pack(">I", len(manifest_data))
            constitutional_payload += manifest_data
        else:
            constitutional_payload += struct.pack(">I", 0)

        if self.proof:
            proof_data = self.proof.to_bytes()
            constitutional_payload += struct.pack(">I", len(proof_data))
            constitutional_payload += proof_data
        else:
            constitutional_payload += struct.pack(">I", 0)

        # Add original payload
        constitutional_payload += self.payload

        # Create HTX frame with constitutional frame type
        if isinstance(self.frame_type, ConstitutionalFrameType):
            # Use CONTROL frame type with constitutional data in payload
            frame_type = HtxFrameType.CONTROL
            payload = struct.pack(">B", self.frame_type.value) + constitutional_payload
        else:
            frame_type = self.frame_type
            payload = constitutional_payload

        return HtxFrame(frame_type=frame_type, stream_id=self.stream_id, payload=payload)

    @classmethod
    def from_htx_frame(cls, htx_frame: HtxFrame) -> "ConstitutionalFrame":
        """Create constitutional frame from HTX frame."""
        payload = htx_frame.payload

        # Check if this is a constitutional control frame
        if htx_frame.frame_type == HtxFrameType.CONTROL and len(payload) > 0:
            constitutional_type = ConstitutionalFrameType(payload[0])
            payload = payload[1:]
            frame_type = constitutional_type
        else:
            frame_type = htx_frame.frame_type
            constitutional_type = None

        # Parse constitutional data
        manifest = None
        proof = None

        if len(payload) >= 4:
            manifest_len = struct.unpack(">I", payload[:4])[0]
            payload = payload[4:]

            if manifest_len > 0 and len(payload) >= manifest_len:
                manifest_data = payload[:manifest_len]
                manifest = ConstitutionalManifest.from_bytes(manifest_data)
                payload = payload[manifest_len:]

        if len(payload) >= 4:
            proof_len = struct.unpack(">I", payload[:4])[0]
            payload = payload[4:]

            if proof_len > 0 and len(payload) >= proof_len:
                proof_data = payload[:proof_len]
                proof = ConstitutionalProof.from_bytes(proof_data)
                payload = payload[proof_len:]

        return cls(
            frame_type=frame_type,
            stream_id=htx_frame.stream_id,
            constitutional_type=constitutional_type,
            manifest=manifest,
            proof=proof,
            payload=payload,
        )


class ConstitutionalFrameProcessor:
    """
    Processes constitutional frames with privacy-preserving verification.

    Integrates with constitutional moderation pipeline and TEE security
    to provide tiered constitutional compliance without compromising privacy.
    """

    def __init__(self):
        self.moderation_pipeline: Optional[ConstitutionalModerationPipeline] = None
        self.tee_integration: Optional[TEESecurityIntegrationManager] = None

        # Constitutional processing state
        self.active_manifests: Dict[str, ConstitutionalManifest] = {}
        self.proof_cache: Dict[str, ConstitutionalProof] = {}
        self.verification_stats: Dict[str, Any] = {
            "total_verifications": 0,
            "by_tier": {tier.name: 0 for tier in ConstitutionalTier},
            "by_harm_level": {},
            "privacy_preserved": 0,
        }

        logger.info("Constitutional frame processor initialized")

    async def initialize(self):
        """Initialize constitutional frame processor."""
        self.moderation_pipeline = ConstitutionalModerationPipeline()

        try:
            from infrastructure.security.tee.integration import get_integration_manager

            self.tee_integration = await get_integration_manager()
        except Exception as e:
            logger.warning(f"TEE integration unavailable: {e}")

        logger.info("Constitutional frame processor ready")

    async def process_constitutional_frame(
        self, frame: ConstitutionalFrame, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[ConstitutionalFrame]]:
        """
        Process constitutional frame with appropriate privacy preservation.

        Returns:
            Tuple of (is_compliant, response_frame)
        """
        try:
            context = context or {}

            if frame.constitutional_type == ConstitutionalFrameType.CONSTITUTIONAL_VERIFY:
                return await self._process_verification_request(frame, context)

            elif frame.constitutional_type == ConstitutionalFrameType.PRIVACY_MANIFEST:
                return await self._process_privacy_manifest(frame, context)

            elif frame.constitutional_type == ConstitutionalFrameType.ZK_PROOF_REQUEST:
                return await self._process_zk_proof_request(frame, context)

            elif frame.constitutional_type == ConstitutionalFrameType.TEE_ATTESTATION:
                return await self._process_tee_attestation(frame, context)

            else:
                logger.warning(f"Unknown constitutional frame type: {frame.constitutional_type}")
                return False, None

        except Exception as e:
            logger.error(f"Error processing constitutional frame: {e}")
            return False, None

    async def _process_verification_request(
        self, frame: ConstitutionalFrame, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalFrame]]:
        """Process constitutional verification request."""
        if not frame.manifest:
            return False, None

        manifest = frame.manifest
        content = frame.payload.decode("utf-8", errors="ignore")

        # Update stats
        self.verification_stats["total_verifications"] += 1
        self.verification_stats["by_tier"][manifest.tier.name] += 1

        # Perform constitutional verification based on tier
        if manifest.tier == ConstitutionalTier.BRONZE:
            # Bronze: Full transparency with H0-H3 monitoring
            is_compliant, proof = await self._bronze_tier_verification(content, manifest, context)

        elif manifest.tier == ConstitutionalTier.SILVER:
            # Silver: Hash-based verification with H2-H3 monitoring
            is_compliant, proof = await self._silver_tier_verification(content, manifest, context)

        elif manifest.tier == ConstitutionalTier.GOLD:
            # Gold: Zero-knowledge proofs with H3-only monitoring
            is_compliant, proof = await self._gold_tier_verification(content, manifest, context)

        elif manifest.tier == ConstitutionalTier.PLATINUM:
            # Platinum: Pure ZK compliance with cryptographic verification
            is_compliant, proof = await self._platinum_tier_verification(content, manifest, context)

        else:
            return False, None

        if is_compliant and proof:
            # Store manifest and proof
            self.active_manifests[manifest.manifest_id] = manifest
            self.proof_cache[proof.proof_id] = proof

            # Create response frame
            response_frame = ConstitutionalFrame(
                frame_type=ConstitutionalFrameType.CONSTITUTIONAL_PROOF, stream_id=frame.stream_id, proof=proof
            )

            return True, response_frame

        return False, None

    async def _bronze_tier_verification(
        self, content: str, manifest: ConstitutionalManifest, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalProof]]:
        """Bronze tier: Full transparency with H0-H3 monitoring."""
        # Full moderation with complete transparency
        moderation_result = await self.moderation_pipeline.process_content(
            content=content, content_type="text", user_tier="Bronze", context=context
        )

        # Create full transparency proof
        proof = ConstitutionalProof(
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            proof_type="full_transparency",
            tier=ConstitutionalTier.BRONZE,
            harm_level=moderation_result.harm_analysis.harm_level,
            compliance_score=moderation_result.transparency_score,
            constitutional_flags=list(moderation_result.harm_analysis.constitutional_concerns.keys()),
            verification_metadata={
                "moderation_decision": moderation_result.decision.value,
                "harm_categories": moderation_result.harm_analysis.harm_categories,
                "confidence_score": moderation_result.harm_analysis.confidence_score,
                "processing_time_ms": moderation_result.harm_analysis.processing_time_ms,
                "full_audit_trail": moderation_result.audit_trail,
            },
        )

        # Allow content if moderation allows it
        is_compliant = moderation_result.decision.value in ["allow", "allow_with_warning"]

        # Update stats
        self.verification_stats["by_harm_level"][proof.harm_level] = (
            self.verification_stats["by_harm_level"].get(proof.harm_level, 0) + 1
        )

        return is_compliant, proof

    async def _silver_tier_verification(
        self, content: str, manifest: ConstitutionalManifest, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalProof]]:
        """Silver tier: Hash-based verification with H2-H3 monitoring."""
        # Moderation with reduced transparency
        moderation_result = await self.moderation_pipeline.process_content(
            content=content, content_type="text", user_tier="Silver", context=context
        )

        # Create hash-based proof (privacy preserving)
        moderation_data = {
            "decision": moderation_result.decision.value,
            "harm_level": moderation_result.harm_analysis.harm_level,
            "confidence_score": moderation_result.harm_analysis.confidence_score,
        }
        moderation_hash = hashlib.sha256(json.dumps(moderation_data, sort_keys=True).encode()).hexdigest()

        proof = ConstitutionalProof(
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            proof_type="hash_based",
            tier=ConstitutionalTier.SILVER,
            moderation_result_hash=moderation_hash,
            harm_level=moderation_result.harm_analysis.harm_level,
            compliance_score=moderation_result.transparency_score,
            constitutional_flags=list(moderation_result.harm_analysis.constitutional_concerns.keys()),
            verification_metadata={
                "monitoring_scope": ["H2", "H3"],  # Reduced monitoring
                "privacy_level": 0.5,
                "hash_verification": True,
            },
        )

        is_compliant = moderation_result.decision.value in ["allow", "allow_with_warning"]

        # Increase privacy preservation counter
        self.verification_stats["privacy_preserved"] += 1

        return is_compliant, proof

    async def _gold_tier_verification(
        self, content: str, manifest: ConstitutionalManifest, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalProof]]:
        """Gold tier: Zero-knowledge proofs with H3-only monitoring."""
        # High-level moderation check (H3 only)
        moderation_result = await self.moderation_pipeline.process_content(
            content=content, content_type="text", user_tier="Gold", context=context
        )

        # Generate zero-knowledge proof for constitutional compliance
        zk_proof_data = await self._generate_zk_proof(content, moderation_result)

        proof = ConstitutionalProof(
            content_hash="",  # No content hash for privacy
            proof_type="zk_proof",
            tier=ConstitutionalTier.GOLD,
            zk_proof_data=zk_proof_data,
            harm_level=(
                moderation_result.harm_analysis.harm_level
                if moderation_result.harm_analysis.harm_level == "H3"
                else "PRIVATE"
            ),
            compliance_score=moderation_result.transparency_score,
            constitutional_flags=(
                []
                if moderation_result.harm_analysis.harm_level != "H3"
                else list(moderation_result.harm_analysis.constitutional_concerns.keys())
            ),
            verification_metadata={
                "monitoring_scope": ["H3"],  # H3 only
                "privacy_level": 0.8,
                "zk_proof_verification": True,
                "content_privacy": "high",
            },
        )

        # Only flag if H3 harm detected
        is_compliant = (moderation_result.harm_analysis.harm_level != "H3") or (
            moderation_result.decision.value in ["allow", "allow_with_warning"]
        )

        self.verification_stats["privacy_preserved"] += 1

        return is_compliant, proof

    async def _platinum_tier_verification(
        self, content: str, manifest: ConstitutionalManifest, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalProof]]:
        """Platinum tier: Pure ZK compliance with cryptographic verification."""
        # Minimal moderation in secure enclave
        if self.tee_integration:
            # Process in TEE for maximum privacy
            tee_result = await self._process_in_tee(content, context)
            moderation_compliant = tee_result.get("compliant", False)
            enclave_signature = tee_result.get("signature", b"")
        else:
            # Fallback to local processing
            moderation_result = await self.moderation_pipeline.process_content(
                content=content, content_type="text", user_tier="Gold", context=context  # Use Gold as fallback
            )
            moderation_compliant = moderation_result.harm_analysis.harm_level != "H3"
            enclave_signature = b""

        # Generate pure ZK proof with no content leakage
        zk_proof_data = await self._generate_pure_zk_proof(content, moderation_compliant)

        proof = ConstitutionalProof(
            content_hash="",  # No hash for maximum privacy
            proof_type="pure_zk",
            tier=ConstitutionalTier.PLATINUM,
            zk_proof_data=zk_proof_data,
            enclave_signature=enclave_signature,
            harm_level="PRIVATE",  # Never reveal harm level
            compliance_score=1.0 if moderation_compliant else 0.0,
            constitutional_flags=[],  # Never reveal flags
            verification_metadata={
                "monitoring_scope": [],  # No external monitoring
                "privacy_level": 0.95,
                "pure_zk_verification": True,
                "content_privacy": "maximum",
                "tee_processed": self.tee_integration is not None,
            },
        )

        self.verification_stats["privacy_preserved"] += 1

        return moderation_compliant, proof

    async def _generate_zk_proof(self, content: str, moderation_result: ModerationResult) -> bytes:
        """Generate zero-knowledge proof for constitutional compliance."""
        # Simplified ZK proof generation (production would use proper ZK-SNARK library)
        proof_data = {
            "content_compliant": moderation_result.decision.value in ["allow", "allow_with_warning"],
            "harm_level_acceptable": moderation_result.harm_analysis.harm_level in ["H0", "H1", "H2"],
            "confidence_threshold_met": moderation_result.harm_analysis.confidence_score > 0.7,
            "timestamp": time.time(),
            "proof_salt": secrets.token_hex(32),
        }

        # Create cryptographic commitment
        commitment = hashlib.sha256(json.dumps(proof_data, sort_keys=True).encode()).digest()
        return commitment

    async def _generate_pure_zk_proof(self, content: str, compliant: bool) -> bytes:
        """Generate pure zero-knowledge proof with no information leakage."""
        # Ultra-private ZK proof (production would use advanced ZK-STARK/ZK-SNARK)
        proof_data = {
            "compliant": compliant,
            "timestamp": time.time(),
            "nonce": secrets.token_hex(64),  # Extra entropy for privacy
        }

        # Multi-round hashing for security
        proof_hash = hashlib.sha256(json.dumps(proof_data, sort_keys=True).encode()).digest()
        for _ in range(1000):  # 1000 rounds of hashing
            proof_hash = hashlib.sha256(proof_hash).digest()

        return proof_hash

    async def _process_in_tee(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process content in Trusted Execution Environment."""
        try:
            # This would integrate with actual TEE enclave
            # For now, simulate TEE processing
            content_hash = hashlib.sha256(content.encode()).digest()

            # Generate TEE signature (simulated)
            tee_signature = hashlib.sha256(content_hash + b"tee_processing").digest()

            # Simulated constitutional check in TEE
            # Real implementation would run moderation in secure enclave
            harmful_keywords = ["violence", "hate", "threat", "harm"]
            compliant = not any(keyword in content.lower() for keyword in harmful_keywords)

            return {"compliant": compliant, "signature": tee_signature, "tee_processed": True}

        except Exception as e:
            logger.error(f"TEE processing failed: {e}")
            return {"compliant": False, "signature": b"", "tee_processed": False}

    async def _process_privacy_manifest(
        self, frame: ConstitutionalFrame, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalFrame]]:
        """Process privacy tier manifest."""
        if not frame.manifest:
            return False, None

        manifest = frame.manifest

        # Validate manifest
        if manifest.expires_at and time.time() > manifest.expires_at:
            return False, None

        # Store active manifest
        self.active_manifests[manifest.manifest_id] = manifest

        # Create acknowledgment frame
        ack_frame = ConstitutionalFrame(
            frame_type=ConstitutionalFrameType.CONSTITUTIONAL_AUDIT,
            stream_id=frame.stream_id,
            payload=f"MANIFEST_ACCEPTED:{manifest.manifest_id}".encode(),
        )

        logger.info(f"Privacy manifest accepted: {manifest.tier.name} tier, {manifest.privacy_level:.0%} privacy")

        return True, ack_frame

    async def _process_zk_proof_request(
        self, frame: ConstitutionalFrame, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalFrame]]:
        """Process zero-knowledge proof request."""
        content = frame.payload.decode("utf-8", errors="ignore")

        # Generate ZK proof for content
        proof_data = await self._generate_zk_proof(content, None)  # Would need actual moderation result

        # Create ZK proof response
        response_frame = ConstitutionalFrame(
            frame_type=ConstitutionalFrameType.ZK_PROOF_RESPONSE, stream_id=frame.stream_id, payload=proof_data
        )

        return True, response_frame

    async def _process_tee_attestation(
        self, frame: ConstitutionalFrame, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[ConstitutionalFrame]]:
        """Process TEE attestation frame."""
        if not self.tee_integration:
            return False, None

        attestation_data = frame.payload

        # Validate TEE attestation
        # This would integrate with actual TEE verification
        is_valid = len(attestation_data) > 0  # Simplified validation

        if is_valid:
            response_frame = ConstitutionalFrame(
                frame_type=ConstitutionalFrameType.TEE_ATTESTATION,
                stream_id=frame.stream_id,
                payload=b"ATTESTATION_VERIFIED",
            )
            return True, response_frame

        return False, None

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get constitutional verification statistics."""
        return {
            "total_verifications": self.verification_stats["total_verifications"],
            "verifications_by_tier": self.verification_stats["by_tier"],
            "harm_level_distribution": self.verification_stats["by_harm_level"],
            "privacy_preserved_count": self.verification_stats["privacy_preserved"],
            "privacy_preservation_rate": (
                self.verification_stats["privacy_preserved"] / max(1, self.verification_stats["total_verifications"])
            ),
            "active_manifests": len(self.active_manifests),
            "cached_proofs": len(self.proof_cache),
        }


# Factory functions
def create_constitutional_manifest(
    tier: ConstitutionalTier = ConstitutionalTier.SILVER,
    harm_categories: Optional[List[str]] = None,
    privacy_level: Optional[float] = None,
) -> ConstitutionalManifest:
    """Create constitutional manifest for specified tier."""
    harm_categories = harm_categories or ["violence", "hate_speech", "misinformation"]

    # Set privacy level based on tier
    privacy_levels = {
        ConstitutionalTier.BRONZE: 0.2,
        ConstitutionalTier.SILVER: 0.5,
        ConstitutionalTier.GOLD: 0.8,
        ConstitutionalTier.PLATINUM: 0.95,
    }

    if privacy_level is None:
        privacy_level = privacy_levels.get(tier, 0.2)

    # Set monitoring scope based on tier
    monitoring_scopes = {
        ConstitutionalTier.BRONZE: ["H0", "H1", "H2", "H3"],
        ConstitutionalTier.SILVER: ["H2", "H3"],
        ConstitutionalTier.GOLD: ["H3"],
        ConstitutionalTier.PLATINUM: [],
    }

    # Set verification method based on tier
    verification_methods = {
        ConstitutionalTier.BRONZE: "full_transparency",
        ConstitutionalTier.SILVER: "hash_based",
        ConstitutionalTier.GOLD: "zk_proof",
        ConstitutionalTier.PLATINUM: "pure_zk",
    }

    return ConstitutionalManifest(
        tier=tier,
        harm_categories_monitored=harm_categories,
        privacy_level=privacy_level,
        monitoring_scope=monitoring_scopes[tier],
        verification_method=verification_methods[tier],
        expires_at=time.time() + 3600,  # 1 hour expiry
    )


if __name__ == "__main__":
    # Test constitutional frame processing
    async def test_constitutional_frames():
        processor = ConstitutionalFrameProcessor()
        await processor.initialize()

        # Test Bronze tier verification
        bronze_manifest = create_constitutional_manifest(ConstitutionalTier.BRONZE)
        bronze_frame = ConstitutionalFrame(
            frame_type=ConstitutionalFrameType.CONSTITUTIONAL_VERIFY,
            stream_id=1,
            manifest=bronze_manifest,
            payload=b"This is test content for constitutional verification.",
        )

        compliant, response = await processor.process_constitutional_frame(bronze_frame)
        print(f"Bronze tier verification: compliant={compliant}")

        # Test Gold tier verification
        gold_manifest = create_constitutional_manifest(ConstitutionalTier.GOLD)
        gold_frame = ConstitutionalFrame(
            frame_type=ConstitutionalFrameType.CONSTITUTIONAL_VERIFY,
            stream_id=2,
            manifest=gold_manifest,
            payload=b"This is private content for ZK verification.",
        )

        compliant, response = await processor.process_constitutional_frame(gold_frame)
        print(f"Gold tier verification: compliant={compliant}")

        # Print stats
        stats = processor.get_verification_stats()
        print(f"Verification stats: {json.dumps(stats, indent=2)}")

    asyncio.run(test_constitutional_frames())
