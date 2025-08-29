"""
Cryptographic Proof Generator

Creates Proof-of-Execution (PoE), Proof-of-Audit (PoA), and Proof-of-SLA (PoSLA)
with Merkle tree aggregation for verifiable fog computing tasks.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ProofType(Enum):
    """Types of cryptographic proofs supported"""

    EXECUTION = "proof_of_execution"  # PoE - task execution proof
    AUDIT = "proof_of_audit"  # PoA - audit trail proof
    SLA = "proof_of_sla"  # PoSLA - SLA compliance proof
    BATCH = "batch_proof"  # Aggregated proof batch


@dataclass
class ExecutionProof:
    """Proof-of-Execution data structure"""

    task_id: str
    node_id: str
    execution_hash: str
    input_hash: str
    output_hash: str
    timestamp: float
    duration: float
    resource_usage: dict[str, float]
    merkle_path: list[str] = field(default_factory=list)
    signature: str | None = None


@dataclass
class AuditProof:
    """Proof-of-Audit data structure"""

    audit_id: str
    target_task_id: str
    auditor_node_id: str
    audit_result: str  # "passed", "failed", "warning"
    compliance_score: float
    audit_hash: str
    timestamp: float
    evidence_hashes: list[str] = field(default_factory=list)
    merkle_path: list[str] = field(default_factory=list)
    signature: str | None = None


@dataclass
class SLAProof:
    """Proof-of-SLA data structure"""

    sla_id: str
    service_id: str
    node_id: str
    sla_metrics: dict[str, float]  # response_time, availability, throughput
    compliance_status: str  # "met", "violated", "warning"
    measurement_period: tuple[float, float]  # start, end timestamps
    sla_hash: str
    timestamp: float
    merkle_path: list[str] = field(default_factory=list)
    signature: str | None = None


@dataclass
class BatchProof:
    """Aggregated batch proof containing multiple proof types"""

    batch_id: str
    batch_root: str  # Merkle root of all proofs in batch
    proof_count: int
    proof_types: dict[str, int]  # count by proof type
    timestamp: float
    node_contributors: list[str]
    batch_hash: str
    signature: str | None = None


class ProofGenerator:
    """
    Generates cryptographic proofs for fog computing tasks

    Features:
    - Proof-of-Execution for task completion verification
    - Proof-of-Audit for compliance and quality validation
    - Proof-of-SLA for service level agreement compliance
    - Merkle tree aggregation for batch verification
    - Integration with Betanet blockchain anchoring
    """

    def __init__(self, node_id: str, private_key: str | None = None):
        self.node_id = node_id
        self.private_key = private_key
        self.proof_cache: dict[str, Any] = {}
        self.batch_proofs: list[Any] = []
        self.merkle_trees: dict[str, Any] = {}

        # Statistics
        self.stats = {
            "proofs_generated": 0,
            "execution_proofs": 0,
            "audit_proofs": 0,
            "sla_proofs": 0,
            "batch_proofs": 0,
            "merkle_aggregations": 0,
        }

        logger.info(f"Proof generator initialized for node {node_id}")

    async def generate_execution_proof(
        self,
        task_id: str,
        input_data: bytes,
        output_data: bytes,
        execution_time: float,
        resource_usage: dict[str, float],
    ) -> ExecutionProof:
        """
        Generate Proof-of-Execution for completed task

        Args:
            task_id: Unique task identifier
            input_data: Task input data
            output_data: Task output data
            execution_time: Task execution duration in seconds
            resource_usage: Resource consumption metrics

        Returns:
            ExecutionProof with cryptographic validation
        """
        try:
            # Calculate input/output hashes
            input_hash = hashlib.sha256(input_data).hexdigest()
            output_hash = hashlib.sha256(output_data).hexdigest()

            # Create execution context for proof
            execution_context = {
                "task_id": task_id,
                "node_id": self.node_id,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "execution_time": execution_time,
                "resource_usage": resource_usage,
                "timestamp": time.time(),
            }

            # Calculate execution hash
            context_json = json.dumps(execution_context, sort_keys=True)
            execution_hash = hashlib.sha256(context_json.encode()).hexdigest()

            # Create proof object
            proof = ExecutionProof(
                task_id=task_id,
                node_id=self.node_id,
                execution_hash=execution_hash,
                input_hash=input_hash,
                output_hash=output_hash,
                timestamp=execution_context["timestamp"],
                duration=execution_time,
                resource_usage=resource_usage,
            )

            # Sign proof if private key available
            if self.private_key:
                proof.signature = await self._sign_proof(execution_hash)

            # Cache proof for batch aggregation
            self.proof_cache[f"execution_{task_id}"] = proof
            self.batch_proofs.append(proof)

            # Update statistics
            self.stats["proofs_generated"] += 1
            self.stats["execution_proofs"] += 1

            logger.info(f"Generated execution proof for task {task_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate execution proof: {e}")
            raise

    async def generate_audit_proof(
        self, audit_id: str, target_task_id: str, audit_result: str, compliance_score: float, evidence_data: list[bytes]
    ) -> AuditProof:
        """
        Generate Proof-of-Audit for task validation

        Args:
            audit_id: Unique audit identifier
            target_task_id: Task being audited
            audit_result: Audit outcome ("passed", "failed", "warning")
            compliance_score: Numerical compliance score (0-100)
            evidence_data: List of evidence bytes for audit

        Returns:
            AuditProof with cryptographic validation
        """
        try:
            # Calculate evidence hashes
            evidence_hashes = [hashlib.sha256(evidence).hexdigest() for evidence in evidence_data]

            # Create audit context
            audit_context = {
                "audit_id": audit_id,
                "target_task_id": target_task_id,
                "auditor_node_id": self.node_id,
                "audit_result": audit_result,
                "compliance_score": compliance_score,
                "evidence_hashes": evidence_hashes,
                "timestamp": time.time(),
            }

            # Calculate audit hash
            context_json = json.dumps(audit_context, sort_keys=True)
            audit_hash = hashlib.sha256(context_json.encode()).hexdigest()

            # Create proof object
            proof = AuditProof(
                audit_id=audit_id,
                target_task_id=target_task_id,
                auditor_node_id=self.node_id,
                audit_result=audit_result,
                compliance_score=compliance_score,
                audit_hash=audit_hash,
                timestamp=audit_context["timestamp"],
                evidence_hashes=evidence_hashes,
            )

            # Sign proof if private key available
            if self.private_key:
                proof.signature = await self._sign_proof(audit_hash)

            # Cache proof for batch aggregation
            self.proof_cache[f"audit_{audit_id}"] = proof
            self.batch_proofs.append(proof)

            # Update statistics
            self.stats["proofs_generated"] += 1
            self.stats["audit_proofs"] += 1

            logger.info(f"Generated audit proof {audit_id} for task {target_task_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate audit proof: {e}")
            raise

    async def generate_sla_proof(
        self,
        sla_id: str,
        service_id: str,
        sla_metrics: dict[str, float],
        measurement_period: tuple[float, float],
        compliance_status: str,
    ) -> SLAProof:
        """
        Generate Proof-of-SLA for service level compliance

        Args:
            sla_id: Unique SLA identifier
            service_id: Service being measured
            sla_metrics: Performance metrics (response_time, availability, etc.)
            measurement_period: (start_time, end_time) for measurement
            compliance_status: "met", "violated", or "warning"

        Returns:
            SLAProof with cryptographic validation
        """
        try:
            # Create SLA context
            sla_context = {
                "sla_id": sla_id,
                "service_id": service_id,
                "node_id": self.node_id,
                "sla_metrics": sla_metrics,
                "compliance_status": compliance_status,
                "measurement_period": measurement_period,
                "timestamp": time.time(),
            }

            # Calculate SLA hash
            context_json = json.dumps(sla_context, sort_keys=True)
            sla_hash = hashlib.sha256(context_json.encode()).hexdigest()

            # Create proof object
            proof = SLAProof(
                sla_id=sla_id,
                service_id=service_id,
                node_id=self.node_id,
                sla_metrics=sla_metrics,
                compliance_status=compliance_status,
                measurement_period=measurement_period,
                sla_hash=sla_hash,
                timestamp=sla_context["timestamp"],
            )

            # Sign proof if private key available
            if self.private_key:
                proof.signature = await self._sign_proof(sla_hash)

            # Cache proof for batch aggregation
            self.proof_cache[f"sla_{sla_id}"] = proof
            self.batch_proofs.append(proof)

            # Update statistics
            self.stats["proofs_generated"] += 1
            self.stats["sla_proofs"] += 1

            logger.info(f"Generated SLA proof {sla_id} for service {service_id}")
            return proof

        except Exception as e:
            logger.error(f"Failed to generate SLA proof: {e}")
            raise

    async def generate_batch_proof(self, batch_size: int = 100) -> BatchProof | None:
        """
        Generate aggregated batch proof from accumulated individual proofs

        Args:
            batch_size: Maximum number of proofs to include in batch

        Returns:
            BatchProof with Merkle root aggregation
        """
        if len(self.batch_proofs) < 2:
            logger.debug("Insufficient proofs for batch generation")
            return None

        try:
            # Select proofs for batch
            proofs_to_batch = self.batch_proofs[:batch_size]
            self.batch_proofs = self.batch_proofs[batch_size:]

            # Calculate proof type distribution
            proof_types = {"execution": 0, "audit": 0, "sla": 0}

            proof_hashes = []
            node_contributors = set()

            for proof in proofs_to_batch:
                if isinstance(proof, ExecutionProof):
                    proof_types["execution"] += 1
                    proof_hashes.append(proof.execution_hash)
                    node_contributors.add(proof.node_id)
                elif isinstance(proof, AuditProof):
                    proof_types["audit"] += 1
                    proof_hashes.append(proof.audit_hash)
                    node_contributors.add(proof.auditor_node_id)
                elif isinstance(proof, SLAProof):
                    proof_types["sla"] += 1
                    proof_hashes.append(proof.sla_hash)
                    node_contributors.add(proof.node_id)

            # Build Merkle tree from proof hashes
            merkle_root = await self._build_merkle_tree(proof_hashes)

            # Create batch context
            batch_id = f"batch_{self.node_id}_{int(time.time())}"
            batch_context = {
                "batch_id": batch_id,
                "batch_root": merkle_root,
                "proof_count": len(proofs_to_batch),
                "proof_types": proof_types,
                "node_contributors": list(node_contributors),
                "timestamp": time.time(),
            }

            # Calculate batch hash
            context_json = json.dumps(batch_context, sort_keys=True)
            batch_hash = hashlib.sha256(context_json.encode()).hexdigest()

            # Create batch proof
            batch_proof = BatchProof(
                batch_id=batch_id,
                batch_root=merkle_root,
                proof_count=len(proofs_to_batch),
                proof_types=proof_types,
                timestamp=batch_context["timestamp"],
                node_contributors=list(node_contributors),
                batch_hash=batch_hash,
            )

            # Sign batch proof
            if self.private_key:
                batch_proof.signature = await self._sign_proof(batch_hash)

            # Update statistics
            self.stats["batch_proofs"] += 1
            self.stats["merkle_aggregations"] += 1

            logger.info(
                f"Generated batch proof {batch_id} with {len(proofs_to_batch)} proofs " f"(root: {merkle_root[:16]}...)"
            )

            return batch_proof

        except Exception as e:
            logger.error(f"Failed to generate batch proof: {e}")
            raise

    async def _build_merkle_tree(self, hashes: list[str]) -> str:
        """
        Build Merkle tree from list of hashes and return root

        Args:
            hashes: List of hash strings

        Returns:
            Merkle root hash
        """
        if not hashes:
            return ""

        if len(hashes) == 1:
            return hashes[0]

        # Build tree level by level
        current_level = hashes[:]

        while len(current_level) > 1:
            next_level = []

            # Process pairs of hashes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # Handle odd number of nodes by duplicating last hash
                right = current_level[i + 1] if i + 1 < len(current_level) else left

                # Combine hashes
                combined = left + right
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)

            current_level = next_level

        return current_level[0]

    async def _sign_proof(self, proof_hash: str) -> str:
        """
        Sign proof hash with node's private key

        Args:
            proof_hash: Hash to sign

        Returns:
            Signature string
        """
        # Simplified signing - in production would use proper cryptographic signing
        if not self.private_key:
            return ""

        signature_data = f"{self.node_id}:{proof_hash}:{self.private_key}"
        return hashlib.sha256(signature_data.encode()).hexdigest()

    def get_proof_cache(self) -> dict[str, Any]:
        """Get cached proofs for inspection"""
        return self.proof_cache.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get proof generation statistics"""
        return {
            **self.stats,
            "cached_proofs": len(self.proof_cache),
            "pending_batch_proofs": len(self.batch_proofs),
            "node_id": self.node_id,
        }

    async def clear_cache(self):
        """Clear proof cache and reset counters"""
        self.proof_cache.clear()
        self.batch_proofs.clear()
        self.merkle_trees.clear()
        logger.info("Proof generator cache cleared")

    def set_private_key(self, private_key: str):
        """Set private key for proof signing"""
        self.private_key = private_key
        logger.info("Private key updated for proof signing")
