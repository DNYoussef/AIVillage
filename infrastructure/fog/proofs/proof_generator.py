"""
Unified Proof Generation System

Generates cryptographic proofs for fog computing operations including:
- Proof-of-Execution (PoE): Task completion and output verification
- Proof-of-Audit (PoA): AI auditor consensus proofs
- Proof-of-SLA (PoSLA): Performance and compliance proofs
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding

logger = logging.getLogger(__name__)


class ProofType(Enum):
    """Types of cryptographic proofs supported by the system"""

    PROOF_OF_EXECUTION = "poe"
    PROOF_OF_AUDIT = "poa"
    PROOF_OF_SLA = "posla"
    MERKLE_BATCH = "merkle_batch"


@dataclass
class TaskExecution:
    """Record of task execution for proof generation"""

    task_id: str
    node_id: str
    start_timestamp: float
    end_timestamp: float
    input_hash: str
    output_hash: str
    exit_code: int
    resource_usage: dict[str, Any]
    environment_hash: str
    command_signature: str


@dataclass
class AuditEvidence:
    """Evidence from AI auditor for proof generation"""

    audit_id: str
    auditor_id: str
    task_id: str
    timestamp: float
    verdict: str  # "pass", "fail", "warning"
    confidence_score: float
    evidence_hashes: list[str]
    consensus_weight: float


@dataclass
class SLAMeasurement:
    """SLA performance measurements for proof generation"""

    measurement_id: str
    node_id: str
    timestamp: float
    metric_type: str  # "latency", "throughput", "availability", "error_rate"
    measured_value: float
    target_value: float
    compliance_status: str  # "compliant", "breach", "warning"
    measurement_hash: str


@dataclass
class CryptographicProof:
    """Base cryptographic proof structure"""

    proof_id: str
    proof_type: ProofType
    timestamp: datetime
    node_id: str
    data_hash: str
    merkle_root: str | None = None
    signature: str | None = None
    public_key_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    verification_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProofOfExecution(CryptographicProof):
    """Proof-of-Execution specific fields"""

    task_execution: TaskExecution
    computation_trace: list[str] = field(default_factory=list)
    deterministic_hash: str | None = None
    witness_data: str | None = None


@dataclass
class ProofOfAudit(CryptographicProof):
    """Proof-of-Audit specific fields"""

    audit_evidence: list[AuditEvidence]
    consensus_threshold: float
    achieved_consensus: float
    auditor_signatures: dict[str, str] = field(default_factory=dict)
    consensus_proof: str | None = None


@dataclass
class ProofOfSLA(CryptographicProof):
    """Proof-of-SLA specific fields"""

    sla_measurements: list[SLAMeasurement]
    compliance_period: tuple[float, float]
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    attestation_signature: str | None = None


class ProofGenerator:
    """
    Unified cryptographic proof generation system for fog computing

    Generates tamper-proof cryptographic evidence for:
    - Task execution and outputs (PoE)
    - AI auditor consensus (PoA)
    - SLA compliance (PoSLA)
    """

    def __init__(self, node_id: str, private_key_path: str | None = None):
        """
        Initialize proof generator

        Args:
            node_id: Unique identifier for this fog node
            private_key_path: Path to private key file for signing
        """
        self.node_id = node_id
        self.private_key = None
        self.public_key = None

        # Load or generate cryptographic keys
        if private_key_path and Path(private_key_path).exists():
            self._load_keys(private_key_path)
        else:
            self._generate_keys()

        # Proof storage
        self.generated_proofs: dict[str, CryptographicProof] = {}

        # Configuration
        self.config = {
            "hash_algorithm": "sha256",
            "signature_algorithm": "rsa-pss",
            "proof_retention_days": 90,
            "merkle_batch_size": 100,
            "enable_zero_knowledge": True,
        }

        logger.info(f"Proof generator initialized for node {node_id}")

    def _generate_keys(self):
        """Generate RSA key pair for proof signing"""
        try:
            self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            self.public_key = self.private_key.public_key()

            logger.info("Generated new RSA key pair for proof signing")

        except Exception as e:
            logger.error(f"Failed to generate cryptographic keys: {e}")
            raise

    def _load_keys(self, private_key_path: str):
        """Load RSA key pair from file"""
        try:
            with open(private_key_path, "rb") as key_file:
                self.private_key = serialization.load_pem_private_key(key_file.read(), password=None)
            self.public_key = self.private_key.public_key()

            logger.info(f"Loaded RSA key pair from {private_key_path}")

        except Exception as e:
            logger.error(f"Failed to load keys from {private_key_path}: {e}")
            raise

    def _compute_hash(self, data: str | bytes | dict[str, Any]) -> str:
        """Compute cryptographic hash of data"""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        if isinstance(data, str):
            data = data.encode("utf-8")

        return hashlib.sha256(data).hexdigest()

    def _sign_data(self, data: str) -> str:
        """Generate digital signature for data"""
        try:
            signature = self.private_key.sign(
                data.encode("utf-8"),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )

            return signature.hex()

        except Exception as e:
            logger.error(f"Failed to sign data: {e}")
            raise

    def _get_public_key_hash(self) -> str:
        """Get hash of public key for verification"""
        public_key_der = self.public_key.public_bytes(
            encoding=Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_der).hexdigest()

    async def generate_proof_of_execution(
        self, task_execution: TaskExecution, computation_trace: list[str] | None = None, include_witness: bool = True
    ) -> ProofOfExecution:
        """
        Generate Proof-of-Execution for completed task

        Args:
            task_execution: Task execution record
            computation_trace: Optional detailed computation steps
            include_witness: Include witness data for verification

        Returns:
            Cryptographic proof of task execution
        """
        try:
            proof_id = f"poe_{task_execution.task_id}_{int(time.time())}"

            # Create execution data hash
            execution_data = {
                "task_id": task_execution.task_id,
                "node_id": task_execution.node_id,
                "start_timestamp": task_execution.start_timestamp,
                "end_timestamp": task_execution.end_timestamp,
                "input_hash": task_execution.input_hash,
                "output_hash": task_execution.output_hash,
                "exit_code": task_execution.exit_code,
                "resource_usage": task_execution.resource_usage,
                "environment_hash": task_execution.environment_hash,
                "command_signature": task_execution.command_signature,
            }

            data_hash = self._compute_hash(execution_data)

            # Generate deterministic hash for reproducibility
            deterministic_input = f"{task_execution.input_hash}:{task_execution.command_signature}"
            deterministic_hash = self._compute_hash(deterministic_input)

            # Generate witness data if requested
            witness_data = None
            if include_witness:
                witness_data = self._generate_witness_data(task_execution)

            # Create proof structure
            proof = ProofOfExecution(
                proof_id=proof_id,
                proof_type=ProofType.PROOF_OF_EXECUTION,
                timestamp=datetime.now(timezone.utc),
                node_id=self.node_id,
                data_hash=data_hash,
                public_key_hash=self._get_public_key_hash(),
                task_execution=task_execution,
                computation_trace=computation_trace or [],
                deterministic_hash=deterministic_hash,
                witness_data=witness_data,
                metadata={
                    "execution_duration": task_execution.end_timestamp - task_execution.start_timestamp,
                    "resource_efficiency": self._calculate_resource_efficiency(task_execution.resource_usage),
                    "verification_level": "cryptographic",
                },
            )

            # Sign the proof
            proof.signature = self._sign_data(data_hash)

            # Store proof
            self.generated_proofs[proof_id] = proof

            logger.info(f"Generated PoE proof {proof_id} for task {task_execution.task_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate PoE proof: {e}")
            raise

    async def generate_proof_of_audit(
        self, audit_evidence: list[AuditEvidence], consensus_threshold: float = 0.67
    ) -> ProofOfAudit:
        """
        Generate Proof-of-Audit from AI auditor consensus

        Args:
            audit_evidence: List of audit evidence from multiple auditors
            consensus_threshold: Required consensus threshold (0.0-1.0)

        Returns:
            Cryptographic proof of audit consensus
        """
        try:
            if not audit_evidence:
                raise ValueError("No audit evidence provided")

            proof_id = f"poa_{audit_evidence[0].task_id}_{int(time.time())}"

            # Calculate consensus metrics
            total_weight = sum(evidence.consensus_weight for evidence in audit_evidence)
            weighted_confidence = (
                sum(evidence.confidence_score * evidence.consensus_weight for evidence in audit_evidence) / total_weight
                if total_weight > 0
                else 0.0
            )

            # Determine consensus verdict
            verdict_weights = {}
            for evidence in audit_evidence:
                verdict_weights[evidence.verdict] = verdict_weights.get(evidence.verdict, 0) + evidence.consensus_weight

            achieved_consensus = max(verdict_weights.values()) / total_weight if total_weight > 0 else 0.0
            consensus_verdict = max(verdict_weights.keys(), key=lambda k: verdict_weights[k])

            # Create audit data hash
            audit_data = {
                "audit_evidence": [asdict(evidence) for evidence in audit_evidence],
                "consensus_threshold": consensus_threshold,
                "achieved_consensus": achieved_consensus,
                "consensus_verdict": consensus_verdict,
                "weighted_confidence": weighted_confidence,
            }

            data_hash = self._compute_hash(audit_data)

            # Collect auditor signatures
            auditor_signatures = {}
            for evidence in audit_evidence:
                # In a real implementation, this would verify actual auditor signatures
                auditor_signatures[evidence.auditor_id] = self._compute_hash(
                    f"{evidence.auditor_id}:{evidence.audit_id}:{evidence.verdict}"
                )

            # Generate consensus proof
            consensus_proof = self._generate_consensus_proof(audit_evidence, achieved_consensus)

            # Create proof structure
            proof = ProofOfAudit(
                proof_id=proof_id,
                proof_type=ProofType.PROOF_OF_AUDIT,
                timestamp=datetime.now(timezone.utc),
                node_id=self.node_id,
                data_hash=data_hash,
                public_key_hash=self._get_public_key_hash(),
                audit_evidence=audit_evidence,
                consensus_threshold=consensus_threshold,
                achieved_consensus=achieved_consensus,
                auditor_signatures=auditor_signatures,
                consensus_proof=consensus_proof,
                metadata={
                    "consensus_verdict": consensus_verdict,
                    "weighted_confidence": weighted_confidence,
                    "auditor_count": len(audit_evidence),
                    "consensus_achieved": achieved_consensus >= consensus_threshold,
                },
            )

            # Sign the proof
            proof.signature = self._sign_data(data_hash)

            # Store proof
            self.generated_proofs[proof_id] = proof

            logger.info(f"Generated PoA proof {proof_id} with {achieved_consensus:.2%} consensus")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate PoA proof: {e}")
            raise

    async def generate_proof_of_sla(
        self, sla_measurements: list[SLAMeasurement], compliance_period: tuple[float, float]
    ) -> ProofOfSLA:
        """
        Generate Proof-of-SLA for performance compliance

        Args:
            sla_measurements: List of SLA measurements during period
            compliance_period: (start_timestamp, end_timestamp) of measurement period

        Returns:
            Cryptographic proof of SLA compliance
        """
        try:
            if not sla_measurements:
                raise ValueError("No SLA measurements provided")

            proof_id = f"posla_{self.node_id}_{int(compliance_period[0])}_{int(compliance_period[1])}"

            # Aggregate metrics by type
            aggregated_metrics = self._aggregate_sla_metrics(sla_measurements)

            # Calculate overall compliance
            compliant_measurements = sum(1 for m in sla_measurements if m.compliance_status == "compliant")
            compliance_percentage = compliant_measurements / len(sla_measurements) * 100

            # Create SLA data hash
            sla_data = {
                "measurements": [asdict(measurement) for measurement in sla_measurements],
                "compliance_period": compliance_period,
                "aggregated_metrics": aggregated_metrics,
                "compliance_percentage": compliance_percentage,
                "node_id": self.node_id,
            }

            data_hash = self._compute_hash(sla_data)

            # Generate attestation signature (self-attestation for now)
            attestation_data = f"{proof_id}:{compliance_percentage}:{data_hash}"
            attestation_signature = self._sign_data(attestation_data)

            # Create proof structure
            proof = ProofOfSLA(
                proof_id=proof_id,
                proof_type=ProofType.PROOF_OF_SLA,
                timestamp=datetime.now(timezone.utc),
                node_id=self.node_id,
                data_hash=data_hash,
                public_key_hash=self._get_public_key_hash(),
                sla_measurements=sla_measurements,
                compliance_period=compliance_period,
                aggregated_metrics=aggregated_metrics,
                attestation_signature=attestation_signature,
                metadata={
                    "compliance_percentage": compliance_percentage,
                    "measurement_count": len(sla_measurements),
                    "period_duration": compliance_period[1] - compliance_period[0],
                    "overall_status": "compliant" if compliance_percentage >= 95.0 else "breach",
                },
            )

            # Sign the proof
            proof.signature = self._sign_data(data_hash)

            # Store proof
            self.generated_proofs[proof_id] = proof

            logger.info(f"Generated PoSLA proof {proof_id} with {compliance_percentage:.1f}% compliance")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate PoSLA proof: {e}")
            raise

    def _generate_witness_data(self, task_execution: TaskExecution) -> str:
        """Generate witness data for proof verification"""
        witness_components = [
            task_execution.input_hash,
            task_execution.output_hash,
            task_execution.environment_hash,
            str(task_execution.exit_code),
            str(task_execution.start_timestamp),
            str(task_execution.end_timestamp),
        ]

        witness_string = ":".join(witness_components)
        return self._compute_hash(witness_string)

    def _calculate_resource_efficiency(self, resource_usage: dict[str, Any]) -> float:
        """Calculate resource efficiency metric"""
        # Simple efficiency calculation based on resource utilization
        cpu_efficiency = min(resource_usage.get("cpu_percent", 0) / 100.0, 1.0)
        memory_efficiency = min(resource_usage.get("memory_percent", 0) / 100.0, 1.0)

        return (cpu_efficiency + memory_efficiency) / 2.0

    def _generate_consensus_proof(self, audit_evidence: list[AuditEvidence], achieved_consensus: float) -> str:
        """Generate cryptographic proof of auditor consensus"""
        # Simplified consensus proof - in production would use more sophisticated crypto
        evidence_hashes = []
        for evidence in audit_evidence:
            evidence_hash = self._compute_hash(
                {
                    "auditor_id": evidence.auditor_id,
                    "verdict": evidence.verdict,
                    "confidence_score": evidence.confidence_score,
                    "consensus_weight": evidence.consensus_weight,
                }
            )
            evidence_hashes.append(evidence_hash)

        consensus_data = {
            "evidence_hashes": sorted(evidence_hashes),
            "achieved_consensus": achieved_consensus,
            "timestamp": time.time(),
        }

        return self._compute_hash(consensus_data)

    def _aggregate_sla_metrics(self, measurements: list[SLAMeasurement]) -> dict[str, float]:
        """Aggregate SLA measurements by metric type"""
        metrics_by_type = {}

        for measurement in measurements:
            if measurement.metric_type not in metrics_by_type:
                metrics_by_type[measurement.metric_type] = []
            metrics_by_type[measurement.metric_type].append(measurement.measured_value)

        aggregated = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                aggregated[f"{metric_type}_avg"] = sum(values) / len(values)
                aggregated[f"{metric_type}_min"] = min(values)
                aggregated[f"{metric_type}_max"] = max(values)

        return aggregated

    async def create_merkle_batch_proof(self, proofs: list[CryptographicProof]) -> CryptographicProof:
        """
        Create Merkle tree batch proof for multiple individual proofs

        Args:
            proofs: List of proofs to batch together

        Returns:
            Merkle batch proof containing all input proofs
        """
        try:
            if not proofs:
                raise ValueError("No proofs provided for batching")

            # Import here to avoid circular dependency
            from .merkle_tree import MerkleTree

            proof_id = f"batch_{int(time.time())}_{len(proofs)}"

            # Create Merkle tree from proof hashes
            proof_hashes = [proof.data_hash for proof in proofs]
            merkle_tree = MerkleTree(proof_hashes)
            merkle_root = merkle_tree.get_root()

            # Create batch data
            batch_data = {
                "proof_ids": [proof.proof_id for proof in proofs],
                "proof_hashes": proof_hashes,
                "merkle_root": merkle_root,
                "batch_size": len(proofs),
                "node_id": self.node_id,
            }

            data_hash = self._compute_hash(batch_data)

            # Create batch proof
            batch_proof = CryptographicProof(
                proof_id=proof_id,
                proof_type=ProofType.MERKLE_BATCH,
                timestamp=datetime.now(timezone.utc),
                node_id=self.node_id,
                data_hash=data_hash,
                merkle_root=merkle_root,
                public_key_hash=self._get_public_key_hash(),
                metadata={
                    "batch_size": len(proofs),
                    "proof_types": list(set(proof.proof_type.value for proof in proofs)),
                    "time_range": {
                        "start": min(proof.timestamp for proof in proofs).isoformat(),
                        "end": max(proof.timestamp for proof in proofs).isoformat(),
                    },
                },
                verification_data={
                    "merkle_tree_depth": merkle_tree.depth,
                    "leaf_hashes": proof_hashes,
                    "merkle_proofs": merkle_tree.get_all_proofs(),
                },
            )

            # Sign the batch proof
            batch_proof.signature = self._sign_data(data_hash)

            # Store batch proof
            self.generated_proofs[proof_id] = batch_proof

            logger.info(f"Created Merkle batch proof {proof_id} containing {len(proofs)} proofs")
            return batch_proof

        except Exception as e:
            logger.error(f"Failed to create Merkle batch proof: {e}")
            raise

    def get_proof(self, proof_id: str) -> CryptographicProof | None:
        """Retrieve a generated proof by ID"""
        return self.generated_proofs.get(proof_id)

    def list_proofs(
        self, proof_type: ProofType | None = None, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[CryptographicProof]:
        """List generated proofs with optional filtering"""
        proofs = list(self.generated_proofs.values())

        if proof_type:
            proofs = [p for p in proofs if p.proof_type == proof_type]

        if start_time:
            proofs = [p for p in proofs if p.timestamp >= start_time]

        if end_time:
            proofs = [p for p in proofs if p.timestamp <= end_time]

        return sorted(proofs, key=lambda p: p.timestamp, reverse=True)

    def get_statistics(self) -> dict[str, Any]:
        """Get proof generation statistics"""
        proofs = list(self.generated_proofs.values())

        stats = {
            "total_proofs": len(proofs),
            "proofs_by_type": {},
            "recent_activity": 0,
            "node_id": self.node_id,
            "public_key_hash": self._get_public_key_hash(),
        }

        # Count by type
        for proof in proofs:
            proof_type = proof.proof_type.value
            stats["proofs_by_type"][proof_type] = stats["proofs_by_type"].get(proof_type, 0) + 1

        # Count recent activity (last 24 hours)
        recent_threshold = datetime.now(timezone.utc).timestamp() - 86400
        stats["recent_activity"] = sum(1 for proof in proofs if proof.timestamp.timestamp() >= recent_threshold)

        return stats

    async def export_proofs(self, output_path: str, proof_ids: list[str] | None = None) -> int:
        """
        Export proofs to JSON file

        Args:
            output_path: Path to output file
            proof_ids: Optional list of specific proof IDs to export

        Returns:
            Number of proofs exported
        """
        try:
            proofs_to_export = []

            if proof_ids:
                for proof_id in proof_ids:
                    if proof_id in self.generated_proofs:
                        proofs_to_export.append(self.generated_proofs[proof_id])
            else:
                proofs_to_export = list(self.generated_proofs.values())

            # Convert to exportable format
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "node_id": self.node_id,
                "proof_count": len(proofs_to_export),
                "proofs": [],
            }

            for proof in proofs_to_export:
                proof_dict = asdict(proof)
                proof_dict["timestamp"] = proof.timestamp.isoformat()
                proof_dict["proof_type"] = proof.proof_type.value
                export_data["proofs"].append(proof_dict)

            # Write to file
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, sort_keys=True)

            logger.info(f"Exported {len(proofs_to_export)} proofs to {output_path}")
            return len(proofs_to_export)

        except Exception as e:
            logger.error(f"Failed to export proofs: {e}")
            raise
