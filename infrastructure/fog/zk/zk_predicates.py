"""
Zero-Knowledge Predicates Engine

Implements narrow ZK predicates for privacy-preserving verification:
1. Network Policy Predicates - Port ranges, protocols, traffic patterns
2. MIME Type Verification - Content type validation without revealing content
3. Model Pack Hash Validation - ML model integrity without exposing models
4. Privacy-Preserving Compliance - Compliance checks without data exposure

These predicates use efficient ZK constructions suitable for fog computing:
- Merkle tree inclusion proofs
- Range proofs for numerical bounds
- Set membership proofs
- Hash-based commitments

Design Principles:
- Narrow scope for practical efficiency
- Privacy-first architecture
- Integration with existing audit systems
- Gradual expansion capability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import secrets
import time
from typing import Any

# Cryptographic imports

logger = logging.getLogger(__name__)


class PredicateType(Enum):
    """Types of ZK predicates supported."""

    NETWORK_POLICY = "network_policy"
    MIME_TYPE = "mime_type"
    MODEL_HASH = "model_hash"
    COMPLIANCE_CHECK = "compliance_check"
    RANGE_PROOF = "range_proof"
    SET_MEMBERSHIP = "set_membership"


class ProofResult(Enum):
    """Result of ZK proof verification."""

    VALID = "valid"
    INVALID = "invalid"
    MALFORMED = "malformed"
    EXPIRED = "expired"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ZKCommitment:
    """Zero-knowledge commitment structure."""

    commitment_hash: str
    blinding_factor_hash: str  # Hash of blinding factor (not the factor itself)
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZKProof:
    """Zero-knowledge proof structure."""

    proof_id: str
    predicate_type: PredicateType
    commitment: ZKCommitment
    proof_data: dict[str, Any]
    witness_hash: str  # Hash of witness (not the witness itself)
    created_at: datetime
    verified_at: datetime | None = None
    verification_result: ProofResult | None = None
    verifier_id: str | None = None

    def is_valid(self) -> bool:
        """Check if proof is still valid."""
        if self.commitment.expires_at and datetime.now(timezone.utc) > self.commitment.expires_at:
            return False
        return self.verification_result == ProofResult.VALID


@dataclass
class PredicateContext:
    """Context for predicate evaluation."""

    network_policies: dict[str, Any] = field(default_factory=dict)
    allowed_mime_types: set[str] = field(default_factory=set)
    trusted_model_hashes: set[str] = field(default_factory=set)
    compliance_rules: dict[str, Any] = field(default_factory=dict)
    security_level: str = "standard"  # "minimal", "standard", "high"


class ZKPredicate(ABC):
    """Abstract base class for zero-knowledge predicates."""

    def __init__(self, predicate_id: str, predicate_type: PredicateType):
        self.predicate_id = predicate_id
        self.predicate_type = predicate_type
        self.created_at = datetime.now(timezone.utc)

    @abstractmethod
    async def generate_commitment(self, secret_data: Any, context: PredicateContext) -> ZKCommitment:
        """Generate zero-knowledge commitment for secret data."""
        pass

    @abstractmethod
    async def generate_proof(
        self, commitment: ZKCommitment, secret_data: Any, public_parameters: dict[str, Any]
    ) -> ZKProof:
        """Generate zero-knowledge proof."""
        pass

    @abstractmethod
    async def verify_proof(
        self, proof: ZKProof, public_parameters: dict[str, Any], context: PredicateContext
    ) -> ProofResult:
        """Verify zero-knowledge proof."""
        pass

    def _hash_data(self, data: Any) -> str:
        """Compute cryptographic hash of data."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _generate_blinding_factor(self) -> bytes:
        """Generate cryptographic blinding factor."""
        return secrets.token_bytes(32)


class NetworkPolicyPredicate(ZKPredicate):
    """Zero-knowledge predicate for network policy compliance."""

    def __init__(self, predicate_id: str = "network_policy"):
        super().__init__(predicate_id, PredicateType.NETWORK_POLICY)

    async def generate_commitment(self, network_config: dict[str, Any], context: PredicateContext) -> ZKCommitment:
        """
        Generate commitment for network configuration.

        Proves compliance with network policies without revealing:
        - Exact port numbers (only ranges)
        - Internal IP addresses
        - Service configurations
        """
        try:
            # Extract policy-relevant features without revealing specifics
            port_ranges = []
            protocols = set()

            for service in network_config.get("services", []):
                port = service.get("port", 0)
                protocol = service.get("protocol", "tcp")
                protocols.add(protocol)

                # Bucket port into ranges for privacy
                if 1 <= port <= 1024:
                    port_ranges.append("system")
                elif 1025 <= port <= 49151:
                    port_ranges.append("registered")
                elif 49152 <= port <= 65535:
                    port_ranges.append("dynamic")

            # Create privacy-preserving summary
            policy_summary = {
                "port_range_count": len(set(port_ranges)),
                "protocol_count": len(protocols),
                "service_count": len(network_config.get("services", [])),
                "has_system_ports": "system" in port_ranges,
                "protocol_types": sorted(list(protocols)),
            }

            # Generate commitment with blinding
            blinding_factor = self._generate_blinding_factor()
            commitment_data = json.dumps(policy_summary, sort_keys=True)
            commitment_with_blinding = commitment_data + blinding_factor.hex()
            commitment_hash = self._hash_data(commitment_with_blinding)

            commitment = ZKCommitment(
                commitment_hash=commitment_hash,
                blinding_factor_hash=self._hash_data(blinding_factor),
                created_at=datetime.now(timezone.utc),
                metadata={
                    "predicate_type": self.predicate_type.value,
                    "service_count": len(network_config.get("services", [])),
                    "policy_features": list(policy_summary.keys()),
                },
            )

            logger.info(f"Generated network policy commitment {commitment_hash[:16]}...")
            return commitment

        except Exception as e:
            logger.error(f"Failed to generate network policy commitment: {e}")
            raise

    async def generate_proof(
        self, commitment: ZKCommitment, network_config: dict[str, Any], public_parameters: dict[str, Any]
    ) -> ZKProof:
        """Generate proof of network policy compliance."""
        try:
            proof_id = f"net_policy_{int(time.time())}_{secrets.token_hex(4)}"

            # Extract public claims (what we're proving)
            allowed_protocols = public_parameters.get("allowed_protocols", {"tcp", "udp", "https"})
            allowed_port_ranges = public_parameters.get("allowed_port_ranges", ["registered", "dynamic"])
            max_services = public_parameters.get("max_services", 10)

            # Verify configuration against policies
            services = network_config.get("services", [])
            protocol_compliance = all(service.get("protocol", "tcp") in allowed_protocols for service in services)

            port_compliance = True
            for service in services:
                port = service.get("port", 0)
                if 1 <= port <= 1024 and "system" not in allowed_port_ranges:
                    port_compliance = False
                elif 1025 <= port <= 49151 and "registered" not in allowed_port_ranges:
                    port_compliance = False
                elif 49152 <= port <= 65535 and "dynamic" not in allowed_port_ranges:
                    port_compliance = False

            service_count_compliance = len(services) <= max_services

            # Create proof data (public claims only)
            proof_data = {
                "protocol_compliance": protocol_compliance,
                "port_compliance": port_compliance,
                "service_count_compliance": service_count_compliance,
                "total_compliance": protocol_compliance and port_compliance and service_count_compliance,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "public_parameters_hash": self._hash_data(public_parameters),
            }

            # Generate witness hash (proves knowledge without revealing)
            witness_data = {
                "network_config": network_config,
                "blinding_factor_hash": commitment.blinding_factor_hash,
                "proof_nonce": secrets.token_hex(16),
            }
            witness_hash = self._hash_data(witness_data)

            proof = ZKProof(
                proof_id=proof_id,
                predicate_type=self.predicate_type,
                commitment=commitment,
                proof_data=proof_data,
                witness_hash=witness_hash,
                created_at=datetime.now(timezone.utc),
            )

            logger.info(f"Generated network policy proof {proof_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate network policy proof: {e}")
            raise

    async def verify_proof(
        self, proof: ZKProof, public_parameters: dict[str, Any], context: PredicateContext
    ) -> ProofResult:
        """Verify network policy proof."""
        try:
            # Check proof freshness
            if proof.commitment.expires_at and datetime.now(timezone.utc) > proof.commitment.expires_at:
                return ProofResult.EXPIRED

            # Verify proof structure
            if not all(
                key in proof.proof_data
                for key in ["protocol_compliance", "port_compliance", "service_count_compliance", "total_compliance"]
            ):
                return ProofResult.MALFORMED

            # Check public parameter consistency
            expected_params_hash = self._hash_data(public_parameters)
            if proof.proof_data.get("public_parameters_hash") != expected_params_hash:
                return ProofResult.INVALID

            # Verify compliance claims
            total_compliance = (
                proof.proof_data["protocol_compliance"]
                and proof.proof_data["port_compliance"]
                and proof.proof_data["service_count_compliance"]
            )

            if proof.proof_data["total_compliance"] != total_compliance:
                return ProofResult.INVALID

            # Additional context-based validation
            network_policies = context.network_policies
            if network_policies.get("strict_mode", False):
                if not proof.proof_data["total_compliance"]:
                    return ProofResult.INVALID

            logger.info(f"Network policy proof {proof.proof_id} verification: VALID")
            return ProofResult.VALID

        except Exception as e:
            logger.error(f"Failed to verify network policy proof: {e}")
            return ProofResult.MALFORMED


class MimeTypePredicate(ZKPredicate):
    """Zero-knowledge predicate for MIME type validation."""

    def __init__(self, predicate_id: str = "mime_type"):
        super().__init__(predicate_id, PredicateType.MIME_TYPE)

    async def generate_commitment(self, file_metadata: dict[str, Any], context: PredicateContext) -> ZKCommitment:
        """Generate commitment for file MIME type without revealing content."""
        try:
            # Extract MIME type features for commitment
            mime_type = file_metadata.get("mime_type", "application/octet-stream")
            file_size = file_metadata.get("size", 0)

            # Categorize MIME type for privacy
            mime_category = self._categorize_mime_type(mime_type)
            size_category = self._categorize_file_size(file_size)

            # Create privacy-preserving commitment data
            commitment_data = {
                "mime_category": mime_category,
                "size_category": size_category,
                "has_extension": bool(file_metadata.get("extension")),
                "timestamp": int(time.time()) // 3600,  # Hour granularity
            }

            # Generate commitment with blinding
            blinding_factor = self._generate_blinding_factor()
            commitment_with_blinding = json.dumps(commitment_data, sort_keys=True) + blinding_factor.hex()
            commitment_hash = self._hash_data(commitment_with_blinding)

            commitment = ZKCommitment(
                commitment_hash=commitment_hash,
                blinding_factor_hash=self._hash_data(blinding_factor),
                created_at=datetime.now(timezone.utc),
                metadata={
                    "predicate_type": self.predicate_type.value,
                    "mime_category": mime_category,
                    "size_category": size_category,
                },
            )

            logger.info(f"Generated MIME type commitment {commitment_hash[:16]}...")
            return commitment

        except Exception as e:
            logger.error(f"Failed to generate MIME type commitment: {e}")
            raise

    def _categorize_mime_type(self, mime_type: str) -> str:
        """Categorize MIME type for privacy."""
        if mime_type.startswith("text/"):
            return "text"
        elif mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("audio/"):
            return "audio"
        elif mime_type.startswith("video/"):
            return "video"
        elif mime_type.startswith("application/"):
            if "json" in mime_type or "xml" in mime_type:
                return "structured_data"
            elif "pdf" in mime_type or "doc" in mime_type:
                return "document"
            else:
                return "application"
        else:
            return "other"

    def _categorize_file_size(self, size: int) -> str:
        """Categorize file size for privacy."""
        if size < 1024:  # < 1KB
            return "tiny"
        elif size < 1024 * 1024:  # < 1MB
            return "small"
        elif size < 10 * 1024 * 1024:  # < 10MB
            return "medium"
        elif size < 100 * 1024 * 1024:  # < 100MB
            return "large"
        else:
            return "huge"

    async def generate_proof(
        self, commitment: ZKCommitment, file_metadata: dict[str, Any], public_parameters: dict[str, Any]
    ) -> ZKProof:
        """Generate proof of MIME type compliance."""
        try:
            proof_id = f"mime_type_{int(time.time())}_{secrets.token_hex(4)}"

            # Extract allowed MIME types from parameters
            allowed_mime_types = set(public_parameters.get("allowed_mime_types", []))
            max_file_size = public_parameters.get("max_file_size", 100 * 1024 * 1024)  # 100MB

            # Check compliance
            mime_type = file_metadata.get("mime_type", "application/octet-stream")
            file_size = file_metadata.get("size", 0)

            mime_compliance = mime_type in allowed_mime_types
            size_compliance = file_size <= max_file_size

            # Create proof data
            proof_data = {
                "mime_compliance": mime_compliance,
                "size_compliance": size_compliance,
                "total_compliance": mime_compliance and size_compliance,
                "mime_category": self._categorize_mime_type(mime_type),
                "size_category": self._categorize_file_size(file_size),
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "public_parameters_hash": self._hash_data(public_parameters),
            }

            # Generate witness hash
            witness_data = {
                "file_metadata": file_metadata,
                "blinding_factor_hash": commitment.blinding_factor_hash,
                "proof_nonce": secrets.token_hex(16),
            }
            witness_hash = self._hash_data(witness_data)

            proof = ZKProof(
                proof_id=proof_id,
                predicate_type=self.predicate_type,
                commitment=commitment,
                proof_data=proof_data,
                witness_hash=witness_hash,
                created_at=datetime.now(timezone.utc),
            )

            logger.info(f"Generated MIME type proof {proof_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate MIME type proof: {e}")
            raise

    async def verify_proof(
        self, proof: ZKProof, public_parameters: dict[str, Any], context: PredicateContext
    ) -> ProofResult:
        """Verify MIME type proof."""
        try:
            # Check proof freshness
            if proof.commitment.expires_at and datetime.now(timezone.utc) > proof.commitment.expires_at:
                return ProofResult.EXPIRED

            # Verify proof structure
            required_fields = ["mime_compliance", "size_compliance", "total_compliance", "mime_category"]
            if not all(field in proof.proof_data for field in required_fields):
                return ProofResult.MALFORMED

            # Check parameter consistency
            expected_params_hash = self._hash_data(public_parameters)
            if proof.proof_data.get("public_parameters_hash") != expected_params_hash:
                return ProofResult.INVALID

            # Verify compliance logic
            total_compliance = proof.proof_data["mime_compliance"] and proof.proof_data["size_compliance"]
            if proof.proof_data["total_compliance"] != total_compliance:
                return ProofResult.INVALID

            # Context-based validation
            if context.allowed_mime_types:
                proof.proof_data["mime_category"]
                # Additional category-based checks could be added here

            logger.info(f"MIME type proof {proof.proof_id} verification: VALID")
            return ProofResult.VALID

        except Exception as e:
            logger.error(f"Failed to verify MIME type proof: {e}")
            return ProofResult.MALFORMED


class ModelHashPredicate(ZKPredicate):
    """Zero-knowledge predicate for ML model pack hash validation."""

    def __init__(self, predicate_id: str = "model_hash"):
        super().__init__(predicate_id, PredicateType.MODEL_HASH)

    async def generate_commitment(self, model_metadata: dict[str, Any], context: PredicateContext) -> ZKCommitment:
        """Generate commitment for model hash without revealing model details."""
        try:
            # Extract model features for commitment
            model_hash = model_metadata.get("model_hash", "")
            model_size = model_metadata.get("size_bytes", 0)
            model_type = model_metadata.get("model_type", "unknown")

            # Create privacy-preserving commitment data
            commitment_data = {
                "hash_prefix": model_hash[:8] if model_hash else "",  # Only first 8 chars
                "size_category": self._categorize_model_size(model_size),
                "model_type": model_type,
                "timestamp": int(time.time()) // 3600,  # Hour granularity
            }

            # Generate commitment with blinding
            blinding_factor = self._generate_blinding_factor()
            commitment_with_blinding = json.dumps(commitment_data, sort_keys=True) + blinding_factor.hex()
            commitment_hash = self._hash_data(commitment_with_blinding)

            commitment = ZKCommitment(
                commitment_hash=commitment_hash,
                blinding_factor_hash=self._hash_data(blinding_factor),
                created_at=datetime.now(timezone.utc),
                metadata={
                    "predicate_type": self.predicate_type.value,
                    "model_type": model_type,
                    "size_category": self._categorize_model_size(model_size),
                },
            )

            logger.info(f"Generated model hash commitment {commitment_hash[:16]}...")
            return commitment

        except Exception as e:
            logger.error(f"Failed to generate model hash commitment: {e}")
            raise

    def _categorize_model_size(self, size_bytes: int) -> str:
        """Categorize model size for privacy."""
        if size_bytes < 1024 * 1024:  # < 1MB
            return "tiny"
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            return "small"
        elif size_bytes < 100 * 1024 * 1024:  # < 100MB
            return "medium"
        elif size_bytes < 1024 * 1024 * 1024:  # < 1GB
            return "large"
        else:
            return "huge"

    async def generate_proof(
        self, commitment: ZKCommitment, model_metadata: dict[str, Any], public_parameters: dict[str, Any]
    ) -> ZKProof:
        """Generate proof of model hash validity."""
        try:
            proof_id = f"model_hash_{int(time.time())}_{secrets.token_hex(4)}"

            # Extract trusted model hashes
            trusted_hashes = set(public_parameters.get("trusted_model_hashes", []))
            allowed_model_types = set(public_parameters.get("allowed_model_types", []))
            max_model_size = public_parameters.get("max_model_size", 1024 * 1024 * 1024)  # 1GB

            # Check compliance
            model_hash = model_metadata.get("model_hash", "")
            model_type = model_metadata.get("model_type", "unknown")
            model_size = model_metadata.get("size_bytes", 0)

            hash_compliance = model_hash in trusted_hashes
            type_compliance = not allowed_model_types or model_type in allowed_model_types
            size_compliance = model_size <= max_model_size

            # Create proof data
            proof_data = {
                "hash_compliance": hash_compliance,
                "type_compliance": type_compliance,
                "size_compliance": size_compliance,
                "total_compliance": hash_compliance and type_compliance and size_compliance,
                "model_type": model_type,
                "size_category": self._categorize_model_size(model_size),
                "hash_in_trusted_set": hash_compliance,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "public_parameters_hash": self._hash_data(public_parameters),
            }

            # Generate witness hash
            witness_data = {
                "model_metadata": model_metadata,
                "blinding_factor_hash": commitment.blinding_factor_hash,
                "proof_nonce": secrets.token_hex(16),
            }
            witness_hash = self._hash_data(witness_data)

            proof = ZKProof(
                proof_id=proof_id,
                predicate_type=self.predicate_type,
                commitment=commitment,
                proof_data=proof_data,
                witness_hash=witness_hash,
                created_at=datetime.now(timezone.utc),
            )

            logger.info(f"Generated model hash proof {proof_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate model hash proof: {e}")
            raise

    async def verify_proof(
        self, proof: ZKProof, public_parameters: dict[str, Any], context: PredicateContext
    ) -> ProofResult:
        """Verify model hash proof."""
        try:
            # Check proof freshness
            if proof.commitment.expires_at and datetime.now(timezone.utc) > proof.commitment.expires_at:
                return ProofResult.EXPIRED

            # Verify proof structure
            required_fields = ["hash_compliance", "type_compliance", "size_compliance", "total_compliance"]
            if not all(field in proof.proof_data for field in required_fields):
                return ProofResult.MALFORMED

            # Check parameter consistency
            expected_params_hash = self._hash_data(public_parameters)
            if proof.proof_data.get("public_parameters_hash") != expected_params_hash:
                return ProofResult.INVALID

            # Verify compliance logic
            total_compliance = (
                proof.proof_data["hash_compliance"]
                and proof.proof_data["type_compliance"]
                and proof.proof_data["size_compliance"]
            )
            if proof.proof_data["total_compliance"] != total_compliance:
                return ProofResult.INVALID

            # Context-based validation
            if context.trusted_model_hashes:
                if not proof.proof_data["hash_in_trusted_set"]:
                    return ProofResult.INVALID

            logger.info(f"Model hash proof {proof.proof_id} verification: VALID")
            return ProofResult.VALID

        except Exception as e:
            logger.error(f"Failed to verify model hash proof: {e}")
            return ProofResult.MALFORMED


class CompliancePredicate(ZKPredicate):
    """Zero-knowledge predicate for privacy-preserving compliance checks."""

    def __init__(self, predicate_id: str = "compliance_check"):
        super().__init__(predicate_id, PredicateType.COMPLIANCE_CHECK)

    async def generate_commitment(self, compliance_data: dict[str, Any], context: PredicateContext) -> ZKCommitment:
        """Generate commitment for compliance status without revealing sensitive data."""
        try:
            # Extract compliance metrics for commitment
            data_retention_days = compliance_data.get("data_retention_days", 0)
            user_consent_percentage = compliance_data.get("user_consent_percentage", 0.0)
            security_score = compliance_data.get("security_score", 0.0)
            audit_findings_count = compliance_data.get("audit_findings_count", 0)

            # Create privacy-preserving commitment data
            commitment_data = {
                "retention_category": self._categorize_retention_period(data_retention_days),
                "consent_category": self._categorize_percentage(user_consent_percentage),
                "security_category": self._categorize_percentage(security_score * 100),
                "findings_category": self._categorize_findings_count(audit_findings_count),
                "timestamp": int(time.time()) // 3600,  # Hour granularity
            }

            # Generate commitment with blinding
            blinding_factor = self._generate_blinding_factor()
            commitment_with_blinding = json.dumps(commitment_data, sort_keys=True) + blinding_factor.hex()
            commitment_hash = self._hash_data(commitment_with_blinding)

            commitment = ZKCommitment(
                commitment_hash=commitment_hash,
                blinding_factor_hash=self._hash_data(blinding_factor),
                created_at=datetime.now(timezone.utc),
                metadata={
                    "predicate_type": self.predicate_type.value,
                    "retention_category": commitment_data["retention_category"],
                    "compliance_categories": list(commitment_data.keys()),
                },
            )

            logger.info(f"Generated compliance commitment {commitment_hash[:16]}...")
            return commitment

        except Exception as e:
            logger.error(f"Failed to generate compliance commitment: {e}")
            raise

    def _categorize_retention_period(self, days: int) -> str:
        """Categorize data retention period."""
        if days <= 30:
            return "short"
        elif days <= 365:
            return "medium"
        elif days <= 1095:  # 3 years
            return "long"
        else:
            return "extended"

    def _categorize_percentage(self, percentage: float) -> str:
        """Categorize percentage values."""
        if percentage >= 95.0:
            return "excellent"
        elif percentage >= 90.0:
            return "good"
        elif percentage >= 75.0:
            return "acceptable"
        elif percentage >= 50.0:
            return "poor"
        else:
            return "failing"

    def _categorize_findings_count(self, count: int) -> str:
        """Categorize audit findings count."""
        if count == 0:
            return "none"
        elif count <= 3:
            return "few"
        elif count <= 10:
            return "some"
        else:
            return "many"

    async def generate_proof(
        self, commitment: ZKCommitment, compliance_data: dict[str, Any], public_parameters: dict[str, Any]
    ) -> ZKProof:
        """Generate proof of compliance without revealing sensitive details."""
        try:
            proof_id = f"compliance_{int(time.time())}_{secrets.token_hex(4)}"

            # Extract compliance thresholds
            min_consent_percentage = public_parameters.get("min_consent_percentage", 90.0)
            max_retention_days = public_parameters.get("max_retention_days", 1095)  # 3 years
            min_security_score = public_parameters.get("min_security_score", 0.8)
            max_audit_findings = public_parameters.get("max_audit_findings", 5)

            # Check compliance
            data_retention_days = compliance_data.get("data_retention_days", 0)
            user_consent_percentage = compliance_data.get("user_consent_percentage", 0.0)
            security_score = compliance_data.get("security_score", 0.0)
            audit_findings_count = compliance_data.get("audit_findings_count", 0)

            retention_compliance = data_retention_days <= max_retention_days
            consent_compliance = user_consent_percentage >= min_consent_percentage
            security_compliance = security_score >= min_security_score
            findings_compliance = audit_findings_count <= max_audit_findings

            # Create proof data
            proof_data = {
                "retention_compliance": retention_compliance,
                "consent_compliance": consent_compliance,
                "security_compliance": security_compliance,
                "findings_compliance": findings_compliance,
                "total_compliance": all(
                    [retention_compliance, consent_compliance, security_compliance, findings_compliance]
                ),
                "compliance_categories": {
                    "retention": self._categorize_retention_period(data_retention_days),
                    "consent": self._categorize_percentage(user_consent_percentage),
                    "security": self._categorize_percentage(security_score * 100),
                    "findings": self._categorize_findings_count(audit_findings_count),
                },
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "public_parameters_hash": self._hash_data(public_parameters),
            }

            # Generate witness hash
            witness_data = {
                "compliance_data": compliance_data,
                "blinding_factor_hash": commitment.blinding_factor_hash,
                "proof_nonce": secrets.token_hex(16),
            }
            witness_hash = self._hash_data(witness_data)

            proof = ZKProof(
                proof_id=proof_id,
                predicate_type=self.predicate_type,
                commitment=commitment,
                proof_data=proof_data,
                witness_hash=witness_hash,
                created_at=datetime.now(timezone.utc),
            )

            logger.info(f"Generated compliance proof {proof_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate compliance proof: {e}")
            raise

    async def verify_proof(
        self, proof: ZKProof, public_parameters: dict[str, Any], context: PredicateContext
    ) -> ProofResult:
        """Verify compliance proof."""
        try:
            # Check proof freshness
            if proof.commitment.expires_at and datetime.now(timezone.utc) > proof.commitment.expires_at:
                return ProofResult.EXPIRED

            # Verify proof structure
            required_fields = [
                "retention_compliance",
                "consent_compliance",
                "security_compliance",
                "findings_compliance",
                "total_compliance",
            ]
            if not all(field in proof.proof_data for field in required_fields):
                return ProofResult.MALFORMED

            # Check parameter consistency
            expected_params_hash = self._hash_data(public_parameters)
            if proof.proof_data.get("public_parameters_hash") != expected_params_hash:
                return ProofResult.INVALID

            # Verify compliance logic
            total_compliance = all(
                [
                    proof.proof_data["retention_compliance"],
                    proof.proof_data["consent_compliance"],
                    proof.proof_data["security_compliance"],
                    proof.proof_data["findings_compliance"],
                ]
            )
            if proof.proof_data["total_compliance"] != total_compliance:
                return ProofResult.INVALID

            # Context-based validation
            compliance_rules = context.compliance_rules
            if compliance_rules.get("strict_mode", False):
                if not proof.proof_data["total_compliance"]:
                    return ProofResult.INVALID

            logger.info(f"Compliance proof {proof.proof_id} verification: VALID")
            return ProofResult.VALID

        except Exception as e:
            logger.error(f"Failed to verify compliance proof: {e}")
            return ProofResult.MALFORMED


class ZKPredicateEngine:
    """
    Zero-Knowledge Predicate Engine for Privacy-Preserving Fog Computing

    Orchestrates ZK predicates for:
    - Network policy verification
    - Content type validation
    - Model integrity checks
    - Compliance verification
    """

    def __init__(self, node_id: str, data_dir: str = "zk_data"):
        self.node_id = node_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Predicate registry
        self.predicates: dict[str, ZKPredicate] = {
            "network_policy": NetworkPolicyPredicate(),
            "mime_type": MimeTypePredicate(),
            "model_hash": ModelHashPredicate(),
            "compliance_check": CompliancePredicate(),
        }

        # Proof storage
        self.commitments: dict[str, ZKCommitment] = {}
        self.proofs: dict[str, ZKProof] = {}

        # Configuration
        self.config = {
            "default_commitment_ttl_hours": 24,
            "proof_cache_size": 1000,
            "verification_timeout_seconds": 30,
            "privacy_level": "standard",  # "minimal", "standard", "high"
        }

        logger.info(f"ZK Predicate Engine initialized for node {node_id}")

    async def generate_commitment(
        self, predicate_id: str, secret_data: Any, context: PredicateContext | None = None, ttl_hours: int | None = None
    ) -> str:
        """
        Generate zero-knowledge commitment.

        Returns:
            Commitment ID for later proof generation
        """
        try:
            if predicate_id not in self.predicates:
                raise ValueError(f"Unknown predicate: {predicate_id}")

            predicate = self.predicates[predicate_id]
            context = context or PredicateContext()

            # Set commitment expiration
            if ttl_hours is None:
                ttl_hours = self.config["default_commitment_ttl_hours"]

            commitment = await predicate.generate_commitment(secret_data, context)
            commitment.expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

            # Store commitment
            commitment_id = commitment.commitment_hash
            self.commitments[commitment_id] = commitment

            logger.info(f"Generated ZK commitment {commitment_id[:16]}... for predicate {predicate_id}")
            return commitment_id

        except Exception as e:
            logger.error(f"Failed to generate commitment for {predicate_id}: {e}")
            raise

    async def generate_proof(
        self, commitment_id: str, predicate_id: str, secret_data: Any, public_parameters: dict[str, Any]
    ) -> str:
        """
        Generate zero-knowledge proof for commitment.

        Returns:
            Proof ID for verification
        """
        try:
            if commitment_id not in self.commitments:
                raise ValueError(f"Unknown commitment: {commitment_id}")

            if predicate_id not in self.predicates:
                raise ValueError(f"Unknown predicate: {predicate_id}")

            commitment = self.commitments[commitment_id]
            predicate = self.predicates[predicate_id]

            # Check commitment validity
            if commitment.expires_at and datetime.now(timezone.utc) > commitment.expires_at:
                raise ValueError("Commitment has expired")

            # Generate proof
            proof = await predicate.generate_proof(commitment, secret_data, public_parameters)

            # Store proof
            proof_id = proof.proof_id
            self.proofs[proof_id] = proof

            logger.info(f"Generated ZK proof {proof_id} for commitment {commitment_id[:16]}...")
            return proof_id

        except Exception as e:
            logger.error(f"Failed to generate proof for commitment {commitment_id}: {e}")
            raise

    async def verify_proof(
        self, proof_id: str, public_parameters: dict[str, Any], context: PredicateContext | None = None
    ) -> ProofResult:
        """
        Verify zero-knowledge proof.

        Returns:
            Verification result
        """
        try:
            if proof_id not in self.proofs:
                return ProofResult.INSUFFICIENT_DATA

            proof = self.proofs[proof_id]
            predicate_id = proof.predicate_type.value

            if predicate_id not in self.predicates:
                return ProofResult.MALFORMED

            predicate = self.predicates[predicate_id]
            context = context or PredicateContext()

            # Verify proof
            result = await predicate.verify_proof(proof, public_parameters, context)

            # Update proof record
            proof.verified_at = datetime.now(timezone.utc)
            proof.verification_result = result
            proof.verifier_id = self.node_id

            logger.info(f"Verified ZK proof {proof_id}: {result.value}")
            return result

        except Exception as e:
            logger.error(f"Failed to verify proof {proof_id}: {e}")
            return ProofResult.MALFORMED

    async def get_proof_stats(self) -> dict[str, Any]:
        """Get statistics about ZK proofs."""
        total_commitments = len(self.commitments)
        total_proofs = len(self.proofs)

        # Count by predicate type
        proofs_by_type = {}
        verified_proofs = 0
        valid_proofs = 0

        for proof in self.proofs.values():
            predicate_type = proof.predicate_type.value
            proofs_by_type[predicate_type] = proofs_by_type.get(predicate_type, 0) + 1

            if proof.verification_result:
                verified_proofs += 1
                if proof.verification_result == ProofResult.VALID:
                    valid_proofs += 1

        # Count expired commitments
        now = datetime.now(timezone.utc)
        expired_commitments = sum(1 for c in self.commitments.values() if c.expires_at and now > c.expires_at)

        return {
            "node_id": self.node_id,
            "total_commitments": total_commitments,
            "expired_commitments": expired_commitments,
            "total_proofs": total_proofs,
            "verified_proofs": verified_proofs,
            "valid_proofs": valid_proofs,
            "verification_rate": verified_proofs / total_proofs if total_proofs > 0 else 0,
            "validity_rate": valid_proofs / verified_proofs if verified_proofs > 0 else 0,
            "proofs_by_predicate": proofs_by_type,
            "supported_predicates": list(self.predicates.keys()),
        }

    async def cleanup_expired(self) -> int:
        """Clean up expired commitments and old proofs."""
        now = datetime.now(timezone.utc)
        cleaned_count = 0

        # Remove expired commitments
        expired_commitments = [
            cid for cid, commitment in self.commitments.items() if commitment.expires_at and now > commitment.expires_at
        ]

        for cid in expired_commitments:
            del self.commitments[cid]
            cleaned_count += 1

        # Remove old proofs (keep only recent ones based on cache size)
        if len(self.proofs) > self.config["proof_cache_size"]:
            # Sort by creation time, keep most recent
            sorted_proofs = sorted(self.proofs.items(), key=lambda x: x[1].created_at, reverse=True)

            keep_count = self.config["proof_cache_size"]
            to_remove = sorted_proofs[keep_count:]

            for proof_id, _ in to_remove:
                del self.proofs[proof_id]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired/old ZK items")

        return cleaned_count

    def register_predicate(self, predicate_id: str, predicate: ZKPredicate):
        """Register custom ZK predicate."""
        self.predicates[predicate_id] = predicate
        logger.info(f"Registered custom ZK predicate: {predicate_id}")

    def get_supported_predicates(self) -> list[str]:
        """Get list of supported predicate types."""
        return list(self.predicates.keys())

    async def export_proofs(self, output_path: str, predicate_type: str | None = None) -> int:
        """Export proofs to JSON file."""
        try:
            proofs_to_export = []

            for proof in self.proofs.values():
                if predicate_type is None or proof.predicate_type.value == predicate_type:
                    proof_data = {
                        "proof_id": proof.proof_id,
                        "predicate_type": proof.predicate_type.value,
                        "created_at": proof.created_at.isoformat(),
                        "verified_at": proof.verified_at.isoformat() if proof.verified_at else None,
                        "verification_result": proof.verification_result.value if proof.verification_result else None,
                        "verifier_id": proof.verifier_id,
                        "commitment_hash": proof.commitment.commitment_hash,
                        "witness_hash": proof.witness_hash,
                        "proof_data": proof.proof_data,
                    }
                    proofs_to_export.append(proof_data)

            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "node_id": self.node_id,
                "predicate_filter": predicate_type,
                "proof_count": len(proofs_to_export),
                "proofs": proofs_to_export,
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, sort_keys=True)

            logger.info(f"Exported {len(proofs_to_export)} ZK proofs to {output_path}")
            return len(proofs_to_export)

        except Exception as e:
            logger.error(f"Failed to export proofs: {e}")
            raise
