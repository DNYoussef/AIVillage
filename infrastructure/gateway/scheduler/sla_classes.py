"""
SLA Classes for Fog Computing

Implements three-tier SLA system with replication and attestation:
- S (replicated+attested): Mission-critical jobs with cryptographic attestation
- A (replicated): High-availability jobs with multi-node replication
- B (best-effort): Standard jobs with single-node placement

Features:
- Multi-node job replication for S and A classes
- Cryptographic attestation for S-class jobs
- SLA compliance monitoring and violation tracking
- Resource allocation policies per SLA tier
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..monitoring.metrics import FogMetricsCollector, SLAClass

logger = logging.getLogger(__name__)


class ReplicationStrategy(Enum):
    """Replication strategies for different SLA classes"""

    NONE = "none"  # B-class: single node
    ACTIVE_PASSIVE = "active_passive"  # A-class: active + standby
    ACTIVE_ACTIVE = "active_active"  # S-class: multiple active replicas


class AttestationMethod(Enum):
    """Cryptographic attestation methods for S-class jobs"""

    MERKLE_PROOF = "merkle_proof"
    DIGITAL_SIGNATURE = "digital_signature"
    ZERO_KNOWLEDGE = "zero_knowledge"


@dataclass
class SLARequirements:
    """SLA requirements and guarantees for each service class"""

    sla_class: SLAClass
    max_placement_latency_ms: int
    min_success_rate: float
    replication_strategy: ReplicationStrategy
    min_replicas: int
    requires_attestation: bool
    max_resource_ratio: float  # Maximum % of cluster resources
    priority_weight: float  # Higher = more priority

    def __post_init__(self):
        """Validate SLA requirements consistency"""
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0.0 and 1.0")
        if not 0.0 <= self.max_resource_ratio <= 1.0:
            raise ValueError("max_resource_ratio must be between 0.0 and 1.0")


@dataclass
class JobReplica:
    """Information about a single job replica"""

    replica_id: str
    node_id: str
    status: str  # "pending", "running", "completed", "failed"
    started_at: float | None = None
    completed_at: float | None = None
    result_hash: str | None = None
    attestation_proof: dict[str, Any] | None = None
    resource_usage: dict[str, float] = field(default_factory=dict)


@dataclass
class ReplicatedJob:
    """Multi-replica job execution tracking"""

    job_id: str
    sla_class: SLAClass
    namespace: str
    requirements: SLARequirements
    replicas: list[JobReplica] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    consensus_result: dict[str, Any] | None = None
    attestation_complete: bool = False
    sla_violations: list[str] = field(default_factory=list)

    def get_active_replicas(self) -> list[JobReplica]:
        """Get replicas that are currently running"""
        return [r for r in self.replicas if r.status == "running"]

    def get_completed_replicas(self) -> list[JobReplica]:
        """Get replicas that have completed successfully"""
        return [r for r in self.replicas if r.status == "completed"]

    def is_consensus_reached(self) -> bool:
        """Check if enough replicas have completed for consensus"""
        completed = len(self.get_completed_replicas())
        required = max(1, (self.requirements.min_replicas + 1) // 2)  # Majority
        return completed >= required

    def calculate_placement_latency(self) -> float | None:
        """Calculate job placement latency in milliseconds"""
        if not self.replicas:
            return None

        first_start = min(r.started_at for r in self.replicas if r.started_at)
        if first_start:
            return (first_start - self.created_at) * 1000
        return None


class SLAManager:
    """
    SLA Management System

    Manages three-tier SLA system with replication, attestation, and compliance
    monitoring for fog computing infrastructure.
    """

    def __init__(self, metrics_collector: FogMetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_jobs: dict[str, ReplicatedJob] = {}
        self.sla_definitions = self._initialize_sla_definitions()

        # Compliance tracking
        self.violation_counts: dict[SLAClass, int] = {SLAClass.S: 0, SLAClass.A: 0, SLAClass.B: 0}

        logger.info("SLA Manager initialized with three-tier system")

    def _initialize_sla_definitions(self) -> dict[SLAClass, SLARequirements]:
        """Initialize SLA class definitions and requirements"""
        return {
            SLAClass.S: SLARequirements(
                sla_class=SLAClass.S,
                max_placement_latency_ms=250,
                min_success_rate=0.999,
                replication_strategy=ReplicationStrategy.ACTIVE_ACTIVE,
                min_replicas=3,
                requires_attestation=True,
                max_resource_ratio=0.3,  # Max 30% of cluster
                priority_weight=3.0,
            ),
            SLAClass.A: SLARequirements(
                sla_class=SLAClass.A,
                max_placement_latency_ms=500,
                min_success_rate=0.99,
                replication_strategy=ReplicationStrategy.ACTIVE_PASSIVE,
                min_replicas=2,
                requires_attestation=False,
                max_resource_ratio=0.5,  # Max 50% of cluster
                priority_weight=2.0,
            ),
            SLAClass.B: SLARequirements(
                sla_class=SLAClass.B,
                max_placement_latency_ms=1000,
                min_success_rate=0.95,
                replication_strategy=ReplicationStrategy.NONE,
                min_replicas=1,
                requires_attestation=False,
                max_resource_ratio=1.0,  # Can use all remaining resources
                priority_weight=1.0,
            ),
        }

    def get_sla_requirements(self, sla_class: SLAClass) -> SLARequirements:
        """Get SLA requirements for a given class"""
        return self.sla_definitions[sla_class]

    def create_replicated_job(self, job_id: str, sla_class: SLAClass, namespace: str) -> ReplicatedJob:
        """Create a new replicated job with appropriate SLA requirements"""
        requirements = self.get_sla_requirements(sla_class)

        job = ReplicatedJob(job_id=job_id, sla_class=sla_class, namespace=namespace, requirements=requirements)

        self.active_jobs[job_id] = job

        logger.info(f"Created {sla_class.value} job {job_id} requiring {requirements.min_replicas} replicas")
        return job

    def add_job_replica(self, job_id: str, node_id: str) -> str:
        """Add a replica for a job on the specified node"""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.active_jobs[job_id]
        replica_id = f"{job_id}-replica-{len(job.replicas)}"

        replica = JobReplica(replica_id=replica_id, node_id=node_id, status="pending")

        job.replicas.append(replica)

        logger.debug(f"Added replica {replica_id} for job {job_id} on node {node_id}")
        return replica_id

    def start_replica(self, job_id: str, replica_id: str):
        """Mark a replica as started"""
        job = self.active_jobs.get(job_id)
        if not job:
            return

        for replica in job.replicas:
            if replica.replica_id == replica_id:
                replica.status = "running"
                replica.started_at = time.time()

                # Check placement latency SLA
                latency = job.calculate_placement_latency()
                if latency and latency > job.requirements.max_placement_latency_ms:
                    violation = f"placement_latency_{latency:.1f}ms"
                    job.sla_violations.append(violation)
                    self.metrics_collector.record_sla_violation(job.sla_class, "placement_latency")
                    self.violation_counts[job.sla_class] += 1

                logger.debug(f"Started replica {replica_id} for job {job_id}")
                break

    def complete_replica(self, job_id: str, replica_id: str, result_data: dict[str, Any]):
        """Mark a replica as completed with result data"""
        job = self.active_jobs.get(job_id)
        if not job:
            return

        for replica in job.replicas:
            if replica.replica_id == replica_id:
                replica.status = "completed"
                replica.completed_at = time.time()
                replica.result_hash = self._compute_result_hash(result_data)

                # Generate attestation for S-class jobs
                if job.requirements.requires_attestation:
                    replica.attestation_proof = self._generate_attestation(replica, result_data)

                logger.debug(f"Completed replica {replica_id} for job {job_id}")

                # Check if we can reach consensus
                if job.is_consensus_reached():
                    self._finalize_job_consensus(job)

                break

    def fail_replica(self, job_id: str, replica_id: str, reason: str):
        """Mark a replica as failed"""
        job = self.active_jobs.get(job_id)
        if not job:
            return

        for replica in job.replicas:
            if replica.replica_id == replica_id:
                replica.status = "failed"
                replica.completed_at = time.time()

                logger.warning(f"Failed replica {replica_id} for job {job_id}: {reason}")

                # Check if we need to spawn additional replicas
                self._handle_replica_failure(job, replica)
                break

    def _compute_result_hash(self, result_data: dict[str, Any]) -> str:
        """Compute deterministic hash of job result"""
        result_json = json.dumps(result_data, sort_keys=True)
        return hashlib.sha256(result_json.encode()).hexdigest()

    def _generate_attestation(self, replica: JobReplica, result_data: dict[str, Any]) -> dict[str, Any]:
        """Generate cryptographic attestation for S-class job result"""
        attestation = {
            "method": AttestationMethod.DIGITAL_SIGNATURE.value,
            "timestamp": time.time(),
            "node_id": replica.node_id,
            "result_hash": replica.result_hash,
            "execution_signature": self._sign_execution(replica, result_data),
            "merkle_root": self._compute_merkle_root(result_data),
        }

        logger.debug(f"Generated attestation for replica {replica.replica_id}")
        return attestation

    def _sign_execution(self, replica: JobReplica, result_data: dict[str, Any]) -> str:
        """Create digital signature for job execution (placeholder)"""
        # In production, this would use actual cryptographic signing
        execution_data = {
            "replica_id": replica.replica_id,
            "node_id": replica.node_id,
            "result_hash": replica.result_hash,
            "timestamp": replica.completed_at,
        }

        signature_payload = json.dumps(execution_data, sort_keys=True)
        return hashlib.sha256(signature_payload.encode()).hexdigest()

    def _compute_merkle_root(self, result_data: dict[str, Any]) -> str:
        """Compute Merkle root for result verification (placeholder)"""
        # In production, this would build a proper Merkle tree
        data_str = json.dumps(result_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _finalize_job_consensus(self, job: ReplicatedJob):
        """Finalize job when consensus is reached"""
        completed_replicas = job.get_completed_replicas()

        # Simple majority consensus (in production, would be more sophisticated)
        result_hashes = [r.result_hash for r in completed_replicas]
        consensus_hash = max(set(result_hashes), key=result_hashes.count)

        consensus_replicas = [r for r in completed_replicas if r.result_hash == consensus_hash]

        job.consensus_result = {
            "consensus_hash": consensus_hash,
            "participating_replicas": len(consensus_replicas),
            "total_replicas": len(job.replicas),
            "finalized_at": time.time(),
        }

        # Mark attestation complete for S-class jobs
        if job.requirements.requires_attestation:
            job.attestation_complete = all(r.attestation_proof is not None for r in consensus_replicas)

        logger.info(f"Job {job.job_id} consensus reached with {len(consensus_replicas)} replicas")

    def _handle_replica_failure(self, job: ReplicatedJob, failed_replica: JobReplica):
        """Handle replica failure and determine if additional replicas needed"""
        active_replicas = len(job.get_active_replicas())
        completed_replicas = len(job.get_completed_replicas())

        # Check if we need more replicas to meet minimum requirements
        total_good_replicas = active_replicas + completed_replicas

        if total_good_replicas < job.requirements.min_replicas:
            shortage = job.requirements.min_replicas - total_good_replicas
            logger.warning(f"Job {job.job_id} needs {shortage} additional replicas due to failures")

            # This would trigger the scheduler to create additional replicas
            # For now, just log the requirement

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get detailed status for a job"""
        job = self.active_jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "sla_class": job.sla_class.value,
            "namespace": job.namespace,
            "replicas": len(job.replicas),
            "active_replicas": len(job.get_active_replicas()),
            "completed_replicas": len(job.get_completed_replicas()),
            "consensus_reached": job.is_consensus_reached(),
            "attestation_complete": job.attestation_complete,
            "sla_violations": job.sla_violations,
            "created_at": job.created_at,
            "placement_latency_ms": job.calculate_placement_latency(),
        }

    def get_sla_compliance_report(self) -> dict[str, Any]:
        """Generate SLA compliance report"""
        total_jobs = len(self.active_jobs)

        compliance_by_class = {}
        for sla_class in SLAClass:
            class_jobs = [j for j in self.active_jobs.values() if j.sla_class == sla_class]
            violations = self.violation_counts[sla_class]

            compliance_by_class[sla_class.value] = {
                "total_jobs": len(class_jobs),
                "violations": violations,
                "compliance_rate": 1.0 - (violations / max(1, len(class_jobs))),
                "avg_replicas": sum(len(j.replicas) for j in class_jobs) / max(1, len(class_jobs)),
                "consensus_rate": sum(1 for j in class_jobs if j.is_consensus_reached()) / max(1, len(class_jobs)),
            }

        return {
            "total_jobs": total_jobs,
            "total_violations": sum(self.violation_counts.values()),
            "overall_compliance": 1.0 - (sum(self.violation_counts.values()) / max(1, total_jobs)),
            "by_class": compliance_by_class,
            "report_timestamp": time.time(),
        }

    def cleanup_completed_jobs(self, retention_hours: int = 24):
        """Clean up completed jobs older than retention period"""
        cutoff_time = time.time() - (retention_hours * 3600)

        to_remove = []
        for job_id, job in self.active_jobs.items():
            if job.consensus_result and job.consensus_result.get("finalized_at", 0) < cutoff_time:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.active_jobs[job_id]
            logger.debug(f"Cleaned up completed job {job_id}")

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed jobs")


# Convenience functions for integration with scheduler
def create_job_with_sla(job_id: str, sla_class: SLAClass, namespace: str, sla_manager: SLAManager) -> ReplicatedJob:
    """Create a job with appropriate SLA requirements"""
    return sla_manager.create_replicated_job(job_id, sla_class, namespace)


def validate_sla_class(sla_class_str: str) -> SLAClass:
    """Validate and convert string to SLAClass enum"""
    try:
        return SLAClass(sla_class_str.lower())
    except ValueError:
        logger.warning(f"Invalid SLA class '{sla_class_str}', defaulting to B")
        return SLAClass.B


def get_replication_requirements(sla_class: SLAClass) -> dict[str, Any]:
    """Get replication requirements for a given SLA class"""
    requirements_map = {
        SLAClass.S: {
            "min_replicas": 3,
            "strategy": "active_active",
            "requires_attestation": True,
            "max_latency_ms": 250,
        },
        SLAClass.A: {
            "min_replicas": 2,
            "strategy": "active_passive",
            "requires_attestation": False,
            "max_latency_ms": 500,
        },
        SLAClass.B: {"min_replicas": 1, "strategy": "none", "requires_attestation": False, "max_latency_ms": 1000},
    }

    return requirements_map[sla_class]


if __name__ == "__main__":
    # Demo SLA system
    from ..monitoring.metrics import FogMetricsCollector

    metrics = FogMetricsCollector()
    sla_manager = SLAManager(metrics)

    # Create S-class job with replication and attestation
    job = sla_manager.create_replicated_job("critical-job-1", SLAClass.S, "production")

    # Add replicas
    replica1 = sla_manager.add_job_replica("critical-job-1", "node-01")
    replica2 = sla_manager.add_job_replica("critical-job-1", "node-02")
    replica3 = sla_manager.add_job_replica("critical-job-1", "node-03")

    # Simulate execution
    sla_manager.start_replica("critical-job-1", replica1)
    sla_manager.start_replica("critical-job-1", replica2)
    sla_manager.start_replica("critical-job-1", replica3)

    # Complete replicas
    result = {"output": "success", "computation_time": 42.5}
    sla_manager.complete_replica("critical-job-1", replica1, result)
    sla_manager.complete_replica("critical-job-1", replica2, result)

    # Generate compliance report
    report = sla_manager.get_sla_compliance_report()
    print("SLA Compliance Report:")
    print(json.dumps(report, indent=2))
