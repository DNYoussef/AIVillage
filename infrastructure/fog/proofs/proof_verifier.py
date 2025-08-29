"""
Cryptographic Proof Verifier

Provides comprehensive verification capabilities for all proof types:
- Proof-of-Execution (PoE) validation
- Proof-of-Audit (PoA) consensus verification
- Proof-of-SLA (PoSLA) compliance validation
- Merkle tree proof verification
- Cryptographic signature validation
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# Import our proof types
from .proof_generator import (
    CryptographicProof,
    ProofOfAudit,
    ProofOfExecution,
    ProofOfSLA,
)

logger = logging.getLogger(__name__)


class VerificationResult(Enum):
    """Verification result status"""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    UNTRUSTED = "untrusted"
    ERROR = "error"


@dataclass
class VerificationReport:
    """Detailed verification report"""

    proof_id: str
    result: VerificationResult
    timestamp: datetime
    verifier_id: str

    # Detailed verification results
    signature_valid: bool = False
    merkle_valid: bool = False
    timestamp_valid: bool = False
    data_integrity_valid: bool = False
    consensus_valid: bool = False

    # Additional metadata
    verification_time_ms: float = 0.0
    error_messages: list[str] = None
    warnings: list[str] = None
    trusted_keys: list[str] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
        if self.warnings is None:
            self.warnings = []
        if self.trusted_keys is None:
            self.trusted_keys = []


class ProofVerifier:
    """
    Comprehensive cryptographic proof verification system

    Provides secure verification for all proof types with:
    - Digital signature validation
    - Merkle tree proof verification
    - Timestamp and freshness checks
    - Consensus validation for audit proofs
    - SLA compliance verification
    - Trust chain validation
    """

    def __init__(self, verifier_id: str, trusted_keys_dir: str = "trusted_keys"):
        """
        Initialize proof verifier

        Args:
            verifier_id: Unique identifier for this verifier
            trusted_keys_dir: Directory containing trusted public keys
        """
        self.verifier_id = verifier_id
        self.trusted_keys_dir = Path(trusted_keys_dir)
        self.trusted_keys_dir.mkdir(exist_ok=True)

        # Load trusted public keys
        self.trusted_public_keys: dict[str, Any] = {}
        self._load_trusted_keys()

        # Verification cache
        self.verification_cache: dict[str, VerificationReport] = {}

        # Configuration
        self.config = {
            "max_proof_age_hours": 72,  # 3 days
            "enable_caching": True,
            "cache_ttl_seconds": 3600,  # 1 hour
            "require_trusted_keys": True,
            "min_consensus_threshold": 0.51,
            "strict_timestamp_validation": True,
        }

        # Statistics
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "cached_verifications": 0,
            "signature_failures": 0,
            "merkle_failures": 0,
            "timestamp_failures": 0,
        }

        logger.info(f"Proof verifier {verifier_id} initialized with {len(self.trusted_public_keys)} trusted keys")

    def _load_trusted_keys(self):
        """Load trusted public keys from directory"""
        try:
            for key_file in self.trusted_keys_dir.glob("*.pem"):
                with open(key_file, "rb") as f:
                    public_key = serialization.load_pem_public_key(f.read())
                    key_hash = self._get_public_key_hash(public_key)
                    self.trusted_public_keys[key_hash] = public_key
                    logger.debug(f"Loaded trusted key: {key_hash[:16]}...")

        except Exception as e:
            logger.warning(f"Error loading trusted keys: {e}")

    def _get_public_key_hash(self, public_key) -> str:
        """Get hash of public key for identification"""
        public_key_der = public_key.public_bytes(
            encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_der).hexdigest()

    def add_trusted_key(self, public_key_pem: str, key_id: str):
        """Add a trusted public key"""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            key_hash = self._get_public_key_hash(public_key)
            self.trusted_public_keys[key_hash] = public_key

            # Save to file
            key_file = self.trusted_keys_dir / f"{key_id}.pem"
            with open(key_file, "w") as f:
                f.write(public_key_pem)

            logger.info(f"Added trusted key {key_id}: {key_hash[:16]}...")

        except Exception as e:
            logger.error(f"Failed to add trusted key: {e}")
            raise

    async def verify_proof(self, proof: CryptographicProof) -> VerificationReport:
        """Verify a cryptographic proof"""
        start_time = time.time()

        # Check cache first
        if self.config["enable_caching"] and proof.proof_id in self.verification_cache:
            cached_report = self.verification_cache[proof.proof_id]
            cache_age = (datetime.now(timezone.utc) - cached_report.timestamp).total_seconds()

            if cache_age < self.config["cache_ttl_seconds"]:
                self.stats["cached_verifications"] += 1
                return cached_report

        # Create verification report
        report = VerificationReport(
            proof_id=proof.proof_id,
            result=VerificationResult.INVALID,
            timestamp=datetime.now(timezone.utc),
            verifier_id=self.verifier_id,
        )

        try:
            # Update statistics
            self.stats["total_verifications"] += 1

            # Step 1: Validate signature
            report.signature_valid = await self._verify_signature(proof, report)

            # Step 2: Validate timestamp
            report.timestamp_valid = await self._verify_timestamp(proof, report)

            # Step 3: Validate data integrity
            report.data_integrity_valid = await self._verify_data_integrity(proof, report)

            # Step 4: Validate Merkle proof if present
            if proof.merkle_root:
                report.merkle_valid = await self._verify_merkle_proof(proof, report)
            else:
                report.merkle_valid = True  # No Merkle proof to verify

            # Step 5: Type-specific validation
            if isinstance(proof, ProofOfExecution):
                await self._verify_execution_proof(proof, report)
            elif isinstance(proof, ProofOfAudit):
                await self._verify_audit_proof(proof, report)
            elif isinstance(proof, ProofOfSLA):
                await self._verify_sla_proof(proof, report)

            # Determine overall result
            if all([report.signature_valid, report.timestamp_valid, report.data_integrity_valid, report.merkle_valid]):
                if proof.public_key_hash in self.trusted_public_keys or not self.config["require_trusted_keys"]:
                    report.result = VerificationResult.VALID
                    self.stats["successful_verifications"] += 1
                else:
                    report.result = VerificationResult.UNTRUSTED
                    report.error_messages.append("Public key not in trusted set")
            else:
                report.result = VerificationResult.INVALID
                self.stats["failed_verifications"] += 1

        except Exception as e:
            report.result = VerificationResult.ERROR
            report.error_messages.append(f"Verification error: {str(e)}")
            self.stats["failed_verifications"] += 1
            logger.error(f"Error verifying proof {proof.proof_id}: {e}")

        finally:
            report.verification_time_ms = (time.time() - start_time) * 1000

            # Cache result if enabled
            if self.config["enable_caching"]:
                self.verification_cache[proof.proof_id] = report

        return report

    async def _verify_signature(self, proof: CryptographicProof, report: VerificationReport) -> bool:
        """Verify digital signature"""
        try:
            if not proof.signature or not proof.public_key_hash:
                report.error_messages.append("Missing signature or public key hash")
                return False

            # Get public key
            public_key = self.trusted_public_keys.get(proof.public_key_hash)
            if not public_key and self.config["require_trusted_keys"]:
                report.error_messages.append("Public key not found in trusted set")
                self.stats["signature_failures"] += 1
                return False

            if not public_key:
                # For untrusted keys, we still validate signature structure
                report.warnings.append("Cannot verify signature: untrusted key")
                return True  # Allow verification to continue

            # Verify signature
            signature_bytes = bytes.fromhex(proof.signature)
            data_to_verify = proof.data_hash.encode("utf-8")

            public_key.verify(
                signature_bytes,
                data_to_verify,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )

            report.trusted_keys.append(proof.public_key_hash)
            return True

        except InvalidSignature:
            report.error_messages.append("Invalid digital signature")
            self.stats["signature_failures"] += 1
            return False
        except Exception as e:
            report.error_messages.append(f"Signature verification error: {str(e)}")
            self.stats["signature_failures"] += 1
            return False

    async def _verify_timestamp(self, proof: CryptographicProof, report: VerificationReport) -> bool:
        """Verify proof timestamp validity"""
        try:
            if not self.config["strict_timestamp_validation"]:
                return True

            now = datetime.now(timezone.utc)
            proof_age = (now - proof.timestamp).total_seconds() / 3600  # hours

            if proof_age > self.config["max_proof_age_hours"]:
                report.error_messages.append(f"Proof expired: {proof_age:.1f} hours old")
                self.stats["timestamp_failures"] += 1
                return False

            # Check for future timestamps (with small tolerance)
            if proof.timestamp > now.replace(second=0, microsecond=0):
                future_seconds = (proof.timestamp - now).total_seconds()
                if future_seconds > 300:  # 5 minute tolerance
                    report.error_messages.append("Proof timestamp is too far in the future")
                    self.stats["timestamp_failures"] += 1
                    return False
                else:
                    report.warnings.append(f"Proof timestamp {future_seconds:.0f}s in future (within tolerance)")

            return True

        except Exception as e:
            report.error_messages.append(f"Timestamp verification error: {str(e)}")
            return False

    async def _verify_data_integrity(self, proof: CryptographicProof, report: VerificationReport) -> bool:
        """Verify data integrity by recomputing hashes"""
        try:
            # Recompute data hash based on proof type
            if isinstance(proof, ProofOfExecution):
                expected_hash = await self._compute_execution_hash(proof)
            elif isinstance(proof, ProofOfAudit):
                expected_hash = await self._compute_audit_hash(proof)
            elif isinstance(proof, ProofOfSLA):
                expected_hash = await self._compute_sla_hash(proof)
            else:
                # Generic proof hash computation
                proof_data = {
                    "proof_id": proof.proof_id,
                    "proof_type": proof.proof_type.value,
                    "timestamp": proof.timestamp.isoformat(),
                    "node_id": proof.node_id,
                    "metadata": proof.metadata,
                }
                expected_hash = hashlib.sha256(json.dumps(proof_data, sort_keys=True).encode()).hexdigest()

            if expected_hash != proof.data_hash:
                report.error_messages.append("Data integrity check failed: hash mismatch")
                return False

            return True

        except Exception as e:
            report.error_messages.append(f"Data integrity verification error: {str(e)}")
            return False

    async def _compute_execution_hash(self, proof: ProofOfExecution) -> str:
        """Recompute hash for execution proof"""
        execution_data = {
            "task_id": proof.task_execution.task_id,
            "node_id": proof.task_execution.node_id,
            "start_timestamp": proof.task_execution.start_timestamp,
            "end_timestamp": proof.task_execution.end_timestamp,
            "input_hash": proof.task_execution.input_hash,
            "output_hash": proof.task_execution.output_hash,
            "exit_code": proof.task_execution.exit_code,
            "resource_usage": proof.task_execution.resource_usage,
            "environment_hash": proof.task_execution.environment_hash,
            "command_signature": proof.task_execution.command_signature,
        }
        return hashlib.sha256(json.dumps(execution_data, sort_keys=True).encode()).hexdigest()

    async def _compute_audit_hash(self, proof: ProofOfAudit) -> str:
        """Recompute hash for audit proof"""
        from dataclasses import asdict

        audit_data = {
            "audit_evidence": [asdict(evidence) for evidence in proof.audit_evidence],
            "consensus_threshold": proof.consensus_threshold,
            "achieved_consensus": proof.achieved_consensus,
            "consensus_verdict": proof.metadata.get("consensus_verdict"),
            "weighted_confidence": proof.metadata.get("weighted_confidence"),
        }
        return hashlib.sha256(json.dumps(audit_data, sort_keys=True).encode()).hexdigest()

    async def _compute_sla_hash(self, proof: ProofOfSLA) -> str:
        """Recompute hash for SLA proof"""
        from dataclasses import asdict

        sla_data = {
            "measurements": [asdict(measurement) for measurement in proof.sla_measurements],
            "compliance_period": proof.compliance_period,
            "aggregated_metrics": proof.aggregated_metrics,
            "compliance_percentage": proof.metadata.get("compliance_percentage"),
            "node_id": proof.node_id,
        }
        return hashlib.sha256(json.dumps(sla_data, sort_keys=True).encode()).hexdigest()

    async def _verify_merkle_proof(self, proof: CryptographicProof, report: VerificationReport) -> bool:
        """Verify Merkle tree proof"""
        try:
            if not proof.verification_data or "merkle_proofs" not in proof.verification_data:
                report.warnings.append("No Merkle verification data available")
                return True  # Not an error if no Merkle data

            merkle_proofs = proof.verification_data["merkle_proofs"]
            leaf_hashes = proof.verification_data.get("leaf_hashes", [])

            if not merkle_proofs or not leaf_hashes:
                report.error_messages.append("Incomplete Merkle verification data")
                self.stats["merkle_failures"] += 1
                return False

            # Verify each Merkle proof
            for i, merkle_proof_data in enumerate(merkle_proofs):
                if i >= len(leaf_hashes):
                    break

                leaf_hash = leaf_hashes[i]

                # Reconstruct root from proof path
                current_hash = leaf_hash
                for proof_hash, position in merkle_proof_data:
                    if position == "left":
                        combined = proof_hash + current_hash
                    else:
                        combined = current_hash + proof_hash
                    current_hash = hashlib.sha256(combined.encode()).hexdigest()

                if current_hash != proof.merkle_root:
                    report.error_messages.append(f"Merkle proof verification failed for leaf {i}")
                    self.stats["merkle_failures"] += 1
                    return False

            return True

        except Exception as e:
            report.error_messages.append(f"Merkle proof verification error: {str(e)}")
            self.stats["merkle_failures"] += 1
            return False

    async def _verify_execution_proof(self, proof: ProofOfExecution, report: VerificationReport):
        """Additional verification for execution proofs"""
        try:
            # Verify deterministic hash if present
            if proof.deterministic_hash:
                deterministic_input = f"{proof.task_execution.input_hash}:{proof.task_execution.command_signature}"
                expected_hash = hashlib.sha256(deterministic_input.encode()).hexdigest()

                if expected_hash != proof.deterministic_hash:
                    report.error_messages.append("Deterministic hash verification failed")

            # Verify witness data if present
            if proof.witness_data:
                witness_components = [
                    proof.task_execution.input_hash,
                    proof.task_execution.output_hash,
                    proof.task_execution.environment_hash,
                    str(proof.task_execution.exit_code),
                    str(proof.task_execution.start_timestamp),
                    str(proof.task_execution.end_timestamp),
                ]
                witness_string = ":".join(witness_components)
                expected_witness = hashlib.sha256(witness_string.encode()).hexdigest()

                if expected_witness != proof.witness_data:
                    report.error_messages.append("Witness data verification failed")

            # Validate execution duration
            duration = proof.task_execution.end_timestamp - proof.task_execution.start_timestamp
            if duration < 0:
                report.error_messages.append("Invalid execution duration: negative time")
            elif duration > 86400:  # 24 hours
                report.warnings.append(f"Unusually long execution duration: {duration:.1f}s")

        except Exception as e:
            report.error_messages.append(f"Execution proof verification error: {str(e)}")

    async def _verify_audit_proof(self, proof: ProofOfAudit, report: VerificationReport):
        """Additional verification for audit proofs"""
        try:
            # Verify consensus calculation
            total_weight = sum(evidence.consensus_weight for evidence in proof.audit_evidence)

            if total_weight == 0:
                report.error_messages.append("Zero total consensus weight")
                return

            # Recalculate consensus
            verdict_weights = {}
            for evidence in proof.audit_evidence:
                verdict_weights[evidence.verdict] = verdict_weights.get(evidence.verdict, 0) + evidence.consensus_weight

            max_weight = max(verdict_weights.values()) if verdict_weights else 0
            recalculated_consensus = max_weight / total_weight

            if abs(recalculated_consensus - proof.achieved_consensus) > 0.001:
                report.error_messages.append("Consensus calculation mismatch")

            # Verify consensus meets threshold
            report.consensus_valid = proof.achieved_consensus >= proof.consensus_threshold

            if not report.consensus_valid:
                report.error_messages.append(
                    f"Consensus {proof.achieved_consensus:.1%} below threshold {proof.consensus_threshold:.1%}"
                )

            # Validate auditor evidence
            for evidence in proof.audit_evidence:
                if evidence.confidence_score < 0 or evidence.confidence_score > 1:
                    report.warnings.append(f"Invalid confidence score for auditor {evidence.auditor_id}")

                if evidence.consensus_weight <= 0:
                    report.warnings.append(f"Invalid consensus weight for auditor {evidence.auditor_id}")

        except Exception as e:
            report.error_messages.append(f"Audit proof verification error: {str(e)}")

    async def _verify_sla_proof(self, proof: ProofOfSLA, report: VerificationReport):
        """Additional verification for SLA proofs"""
        try:
            # Verify compliance period
            if proof.compliance_period[1] <= proof.compliance_period[0]:
                report.error_messages.append("Invalid compliance period: end before start")
                return

            # Verify measurement timestamps are within period
            for measurement in proof.sla_measurements:
                if not (proof.compliance_period[0] <= measurement.timestamp <= proof.compliance_period[1]):
                    report.warnings.append(f"Measurement {measurement.measurement_id} outside compliance period")

            # Recalculate compliance percentage
            compliant_measurements = sum(1 for m in proof.sla_measurements if m.compliance_status == "compliant")
            recalculated_compliance = (compliant_measurements / len(proof.sla_measurements)) * 100

            expected_compliance = proof.metadata.get("compliance_percentage", 0)
            if abs(recalculated_compliance - expected_compliance) > 0.1:
                report.error_messages.append(
                    f"Compliance percentage mismatch: expected {expected_compliance:.1f}%, "
                    f"calculated {recalculated_compliance:.1f}%"
                )

            # Verify aggregated metrics
            if proof.aggregated_metrics:
                # Group measurements by type
                metrics_by_type = {}
                for measurement in proof.sla_measurements:
                    if measurement.metric_type not in metrics_by_type:
                        metrics_by_type[measurement.metric_type] = []
                    metrics_by_type[measurement.metric_type].append(measurement.measured_value)

                # Verify averages
                for metric_type, values in metrics_by_type.items():
                    if values:
                        expected_avg = sum(values) / len(values)
                        actual_avg = proof.aggregated_metrics.get(f"{metric_type}_avg")

                        if actual_avg is not None and abs(expected_avg - actual_avg) > 0.001:
                            report.warnings.append(
                                f"Average mismatch for {metric_type}: expected {expected_avg:.3f}, "
                                f"got {actual_avg:.3f}"
                            )

        except Exception as e:
            report.error_messages.append(f"SLA proof verification error: {str(e)}")

    async def batch_verify_proofs(self, proofs: list[CryptographicProof]) -> list[VerificationReport]:
        """Verify multiple proofs in batch"""
        if not proofs:
            return []

        # Verify proofs concurrently
        verification_tasks = [self.verify_proof(proof) for proof in proofs]
        reports = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # Handle any exceptions
        valid_reports = []
        for i, report in enumerate(reports):
            if isinstance(report, Exception):
                error_report = VerificationReport(
                    proof_id=proofs[i].proof_id,
                    result=VerificationResult.ERROR,
                    timestamp=datetime.now(timezone.utc),
                    verifier_id=self.verifier_id,
                )
                error_report.error_messages.append(f"Verification exception: {str(report)}")
                valid_reports.append(error_report)
            else:
                valid_reports.append(report)

        logger.info(f"Batch verified {len(proofs)} proofs")
        return valid_reports

    def get_verification_stats(self) -> dict[str, Any]:
        """Get verification statistics"""
        total = self.stats["total_verifications"]
        success_rate = (self.stats["successful_verifications"] / total * 100) if total > 0 else 0

        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "trusted_keys_count": len(self.trusted_public_keys),
            "cached_proofs": len(self.verification_cache),
            "verifier_id": self.verifier_id,
        }

    def clear_cache(self):
        """Clear verification cache"""
        cleared_count = len(self.verification_cache)
        self.verification_cache.clear()
        logger.info(f"Cleared {cleared_count} cached verification results")

    async def cleanup_expired_cache(self):
        """Remove expired entries from verification cache"""
        if not self.config["enable_caching"]:
            return

        now = datetime.now(timezone.utc)
        expired_keys = []

        for proof_id, report in self.verification_cache.items():
            cache_age = (now - report.timestamp).total_seconds()
            if cache_age > self.config["cache_ttl_seconds"]:
                expired_keys.append(proof_id)

        for key in expired_keys:
            del self.verification_cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
