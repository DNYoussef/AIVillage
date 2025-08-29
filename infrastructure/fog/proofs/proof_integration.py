"""
Proof System Integration

Integrates cryptographic proof generation with fog computing infrastructure:
- Task execution proof hooks
- SLA monitoring integration
- Tokenomics reward integration
- Audit system integration
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any

from .proof_generator import AuditEvidence, CryptographicProof, ProofGenerator, SLAMeasurement, TaskExecution
from .proof_verifier import ProofVerifier, VerificationResult

logger = logging.getLogger(__name__)


class ProofIntegrationError(Exception):
    """Proof system integration error"""

    pass


@dataclass
class TaskProofRequest:
    """Request to generate proof for task execution"""

    task_id: str
    node_id: str
    input_data: str
    output_data: str
    command: str
    environment: dict[str, Any]
    resource_usage: dict[str, Any]
    start_time: float
    end_time: float
    exit_code: int
    include_witness: bool = True
    computation_trace: list[str] = None


@dataclass
class AuditProofRequest:
    """Request to generate audit consensus proof"""

    task_id: str
    audit_results: list[dict[str, Any]]
    consensus_threshold: float = 0.67


@dataclass
class SLAProofRequest:
    """Request to generate SLA compliance proof"""

    node_id: str
    start_timestamp: float
    end_timestamp: float
    measurements: list[dict[str, Any]]


class ProofSystemIntegration:
    """
    Comprehensive proof system integration for fog computing

    Provides unified interface for:
    - Automatic proof generation from task execution
    - Integration with monitoring and SLA systems
    - Tokenomics reward calculation and distribution
    - Audit consensus proof generation
    - Batch proof processing and verification
    """

    def __init__(
        self,
        node_id: str,
        proof_storage_dir: str = "proof_storage",
        enable_auto_proofs: bool = True,
        batch_size: int = 50,
    ):
        self.node_id = node_id
        self.proof_storage_dir = Path(proof_storage_dir)
        self.proof_storage_dir.mkdir(exist_ok=True)
        self.enable_auto_proofs = enable_auto_proofs
        self.batch_size = batch_size

        # Initialize proof components
        self.proof_generator = ProofGenerator(
            node_id=node_id, private_key_path=str(self.proof_storage_dir / "node_private_key.pem")
        )
        self.proof_verifier = ProofVerifier(
            verifier_id=f"{node_id}_verifier", trusted_keys_dir=str(self.proof_storage_dir / "trusted_keys")
        )

        # Proof queues and batching
        self.pending_execution_proofs: list[TaskProofRequest] = []
        self.pending_audit_proofs: list[AuditProofRequest] = []
        self.pending_sla_proofs: list[SLAProofRequest] = []

        # Integration hooks
        self.task_hooks: list[Callable] = []
        self.audit_hooks: list[Callable] = []
        self.sla_hooks: list[Callable] = []
        self.reward_hooks: list[Callable] = []

        # Background processing
        self._background_tasks: set = set()
        self._running = False

        # Statistics
        self.stats = {
            "execution_proofs_generated": 0,
            "audit_proofs_generated": 0,
            "sla_proofs_generated": 0,
            "batch_proofs_generated": 0,
            "proofs_verified": 0,
            "rewards_distributed": 0,
            "integration_errors": 0,
        }

        logger.info(f"Proof system integration initialized for node {node_id}")

    async def start(self):
        """Start the proof system integration"""
        if self._running:
            return

        self._running = True
        logger.info("Starting proof system integration")

        # Start background processing tasks
        if self.enable_auto_proofs:
            tasks = [self._proof_batch_processor(), self._proof_cleanup_task(), self._statistics_reporter()]

            for task_coro in tasks:
                task = asyncio.create_task(task_coro)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    async def stop(self):
        """Stop the proof system integration"""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping proof system integration")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def generate_execution_proof(self, request: TaskProofRequest) -> CryptographicProof | None:
        """Generate proof of execution for a completed task"""
        try:
            # Create task execution record
            task_execution = TaskExecution(
                task_id=request.task_id,
                node_id=request.node_id,
                start_timestamp=request.start_time,
                end_timestamp=request.end_time,
                input_hash=self._compute_hash(request.input_data),
                output_hash=self._compute_hash(request.output_data),
                exit_code=request.exit_code,
                resource_usage=request.resource_usage,
                environment_hash=self._compute_hash(request.environment),
                command_signature=self._compute_hash(request.command),
            )

            # Generate proof
            proof = await self.proof_generator.generate_proof_of_execution(
                task_execution=task_execution,
                computation_trace=request.computation_trace,
                include_witness=request.include_witness,
            )

            # Store proof
            await self._store_proof(proof)

            # Execute hooks
            await self._execute_hooks(self.task_hooks, proof)

            # Update statistics
            self.stats["execution_proofs_generated"] += 1

            logger.info(f"Generated execution proof for task {request.task_id}")
            return proof

        except Exception as e:
            self.stats["integration_errors"] += 1
            logger.error(f"Failed to generate execution proof: {e}")
            raise ProofIntegrationError(f"Execution proof generation failed: {e}")

    async def generate_audit_proof(self, request: AuditProofRequest) -> CryptographicProof | None:
        """Generate proof of audit consensus"""
        try:
            # Convert audit results to evidence objects
            audit_evidence = []
            for i, result in enumerate(request.audit_results):
                evidence = AuditEvidence(
                    audit_id=f"audit_{request.task_id}_{i}",
                    auditor_id=result.get("auditor_id", f"auditor_{i}"),
                    task_id=request.task_id,
                    timestamp=result.get("timestamp", time.time()),
                    verdict=result.get("verdict", "pass"),
                    confidence_score=result.get("confidence", 0.95),
                    evidence_hashes=result.get("evidence_hashes", []),
                    consensus_weight=result.get("weight", 1.0),
                )
                audit_evidence.append(evidence)

            # Generate proof
            proof = await self.proof_generator.generate_proof_of_audit(
                audit_evidence=audit_evidence, consensus_threshold=request.consensus_threshold
            )

            # Store proof
            await self._store_proof(proof)

            # Execute hooks
            await self._execute_hooks(self.audit_hooks, proof)

            # Update statistics
            self.stats["audit_proofs_generated"] += 1

            logger.info(f"Generated audit proof for task {request.task_id}")
            return proof

        except Exception as e:
            self.stats["integration_errors"] += 1
            logger.error(f"Failed to generate audit proof: {e}")
            raise ProofIntegrationError(f"Audit proof generation failed: {e}")

    async def generate_sla_proof(self, request: SLAProofRequest) -> CryptographicProof | None:
        """Generate proof of SLA compliance"""
        try:
            # Convert measurements to SLA measurement objects
            sla_measurements = []
            for i, measurement in enumerate(request.measurements):
                sla_measurement = SLAMeasurement(
                    measurement_id=f"sla_{request.node_id}_{i}",
                    node_id=request.node_id,
                    timestamp=measurement.get("timestamp", time.time()),
                    metric_type=measurement.get("metric_type", "latency"),
                    measured_value=measurement.get("value", 0.0),
                    target_value=measurement.get("target", 0.0),
                    compliance_status=measurement.get("status", "compliant"),
                    measurement_hash=self._compute_hash(measurement),
                )
                sla_measurements.append(sla_measurement)

            # Generate proof
            proof = await self.proof_generator.generate_proof_of_sla(
                sla_measurements=sla_measurements, compliance_period=(request.start_timestamp, request.end_timestamp)
            )

            # Store proof
            await self._store_proof(proof)

            # Execute hooks
            await self._execute_hooks(self.sla_hooks, proof)

            # Update statistics
            self.stats["sla_proofs_generated"] += 1

            logger.info(f"Generated SLA proof for node {request.node_id}")
            return proof

        except Exception as e:
            self.stats["integration_errors"] += 1
            logger.error(f"Failed to generate SLA proof: {e}")
            raise ProofIntegrationError(f"SLA proof generation failed: {e}")

    async def verify_proof(self, proof: CryptographicProof) -> bool:
        """Verify a cryptographic proof"""
        try:
            report = await self.proof_verifier.verify_proof(proof)

            self.stats["proofs_verified"] += 1

            is_valid = report.result == VerificationResult.VALID

            if is_valid:
                logger.info(f"Proof {proof.proof_id} verified successfully")
            else:
                logger.warning(f"Proof {proof.proof_id} verification failed: {report.error_messages}")

            return is_valid

        except Exception as e:
            self.stats["integration_errors"] += 1
            logger.error(f"Failed to verify proof: {e}")
            return False

    async def create_batch_proof(self, proof_ids: list[str]) -> CryptographicProof | None:
        """Create Merkle batch proof from individual proofs"""
        try:
            # Get individual proofs
            proofs = []
            for proof_id in proof_ids:
                proof = self.proof_generator.get_proof(proof_id)
                if proof:
                    proofs.append(proof)

            if not proofs:
                return None

            # Create batch proof
            batch_proof = await self.proof_generator.create_merkle_batch_proof(proofs)

            # Store batch proof
            await self._store_proof(batch_proof)

            self.stats["batch_proofs_generated"] += 1

            logger.info(f"Created batch proof from {len(proofs)} individual proofs")
            return batch_proof

        except Exception as e:
            self.stats["integration_errors"] += 1
            logger.error(f"Failed to create batch proof: {e}")
            raise ProofIntegrationError(f"Batch proof creation failed: {e}")

    def queue_execution_proof(self, request: TaskProofRequest):
        """Queue execution proof for batch processing"""
        if self.enable_auto_proofs:
            self.pending_execution_proofs.append(request)
            logger.debug(f"Queued execution proof for task {request.task_id}")

    def queue_audit_proof(self, request: AuditProofRequest):
        """Queue audit proof for batch processing"""
        if self.enable_auto_proofs:
            self.pending_audit_proofs.append(request)
            logger.debug(f"Queued audit proof for task {request.task_id}")

    def queue_sla_proof(self, request: SLAProofRequest):
        """Queue SLA proof for batch processing"""
        if self.enable_auto_proofs:
            self.pending_sla_proofs.append(request)
            logger.debug(f"Queued SLA proof for node {request.node_id}")

    def add_task_hook(self, hook_func: Callable):
        """Add hook function for task execution proofs"""
        self.task_hooks.append(hook_func)
        logger.debug(f"Added task hook: {hook_func.__name__}")

    def add_audit_hook(self, hook_func: Callable):
        """Add hook function for audit proofs"""
        self.audit_hooks.append(hook_func)
        logger.debug(f"Added audit hook: {hook_func.__name__}")

    def add_sla_hook(self, hook_func: Callable):
        """Add hook function for SLA proofs"""
        self.sla_hooks.append(hook_func)
        logger.debug(f"Added SLA hook: {hook_func.__name__}")

    def add_reward_hook(self, hook_func: Callable):
        """Add hook function for reward distribution"""
        self.reward_hooks.append(hook_func)
        logger.debug(f"Added reward hook: {hook_func.__name__}")

    async def _proof_batch_processor(self):
        """Background task to process queued proofs in batches"""
        while self._running:
            try:
                # Process execution proofs
                if self.pending_execution_proofs:
                    batch = self.pending_execution_proofs[: self.batch_size]
                    self.pending_execution_proofs = self.pending_execution_proofs[self.batch_size :]

                    for request in batch:
                        try:
                            await self.generate_execution_proof(request)
                        except Exception as e:
                            logger.error(f"Batch processing error for execution proof: {e}")

                # Process audit proofs
                if self.pending_audit_proofs:
                    batch = self.pending_audit_proofs[: self.batch_size]
                    self.pending_audit_proofs = self.pending_audit_proofs[self.batch_size :]

                    for request in batch:
                        try:
                            await self.generate_audit_proof(request)
                        except Exception as e:
                            logger.error(f"Batch processing error for audit proof: {e}")

                # Process SLA proofs
                if self.pending_sla_proofs:
                    batch = self.pending_sla_proofs[: self.batch_size]
                    self.pending_sla_proofs = self.pending_sla_proofs[self.batch_size :]

                    for request in batch:
                        try:
                            await self.generate_sla_proof(request)
                        except Exception as e:
                            logger.error(f"Batch processing error for SLA proof: {e}")

                # Sleep between batch processing cycles
                await asyncio.sleep(30)  # Process batches every 30 seconds

            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(60)

    async def _proof_cleanup_task(self):
        """Background task to clean up old proofs and cache"""
        while self._running:
            try:
                # Clean up verification cache
                await self.proof_verifier.cleanup_expired_cache()

                # Clean up old proof files (implementation would go here)
                # This could include archiving or deleting old proofs based on retention policy

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Proof cleanup error: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes

    async def _statistics_reporter(self):
        """Background task to report statistics"""
        while self._running:
            try:
                logger.info(f"Proof system stats: {self.get_statistics()}")
                await asyncio.sleep(900)  # Report every 15 minutes

            except Exception as e:
                logger.error(f"Statistics reporter error: {e}")
                await asyncio.sleep(300)

    async def _store_proof(self, proof: CryptographicProof):
        """Store proof to persistent storage"""
        try:
            proof_file = self.proof_storage_dir / f"{proof.proof_id}.json"

            # Convert proof to JSON
            proof_dict = asdict(proof)
            proof_dict["timestamp"] = proof.timestamp.isoformat()
            proof_dict["proof_type"] = proof.proof_type.value

            # Write to file
            import json

            with open(proof_file, "w") as f:
                json.dump(proof_dict, f, indent=2)

            logger.debug(f"Stored proof {proof.proof_id} to {proof_file}")

        except Exception as e:
            logger.error(f"Failed to store proof: {e}")

    async def _execute_hooks(self, hooks: list[Callable], proof: CryptographicProof):
        """Execute registered hook functions"""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(proof)
                else:
                    hook(proof)
            except Exception as e:
                logger.error(f"Hook execution error: {e}")

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data"""
        import hashlib
        import json

        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        elif not isinstance(data, str):
            data = str(data)

        return hashlib.sha256(data.encode()).hexdigest()

    def get_statistics(self) -> dict[str, Any]:
        """Get proof system integration statistics"""
        return {
            **self.stats,
            "queued_execution_proofs": len(self.pending_execution_proofs),
            "queued_audit_proofs": len(self.pending_audit_proofs),
            "queued_sla_proofs": len(self.pending_sla_proofs),
            "active_hooks": {
                "task_hooks": len(self.task_hooks),
                "audit_hooks": len(self.audit_hooks),
                "sla_hooks": len(self.sla_hooks),
                "reward_hooks": len(self.reward_hooks),
            },
            "proof_generator_stats": self.proof_generator.get_statistics(),
            "proof_verifier_stats": self.proof_verifier.get_verification_stats(),
        }

    async def get_proof_by_id(self, proof_id: str) -> CryptographicProof | None:
        """Get proof by ID from storage"""
        # First check generator cache
        proof = self.proof_generator.get_proof(proof_id)
        if proof:
            return proof

        # Load from file storage
        try:
            proof_file = self.proof_storage_dir / f"{proof_id}.json"
            if proof_file.exists():
                import json

                with open(proof_file) as f:
                    proof_dict = json.load(f)

                # Convert back to proof object (simplified - full implementation would
                # properly reconstruct the specific proof type)
                logger.debug(f"Loaded proof {proof_id} from storage")
                return proof_dict  # Return as dict for now

        except Exception as e:
            logger.error(f"Failed to load proof from storage: {e}")

        return None

    async def list_proofs(self, proof_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List stored proofs with optional filtering"""
        proofs = []

        try:
            # List files in proof storage directory
            proof_files = list(self.proof_storage_dir.glob("*.json"))
            proof_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # Most recent first

            for proof_file in proof_files[:limit]:
                try:
                    import json

                    with open(proof_file) as f:
                        proof_dict = json.load(f)

                    # Filter by type if specified
                    if proof_type and proof_dict.get("proof_type") != proof_type:
                        continue

                    proofs.append(
                        {
                            "proof_id": proof_dict.get("proof_id"),
                            "proof_type": proof_dict.get("proof_type"),
                            "timestamp": proof_dict.get("timestamp"),
                            "node_id": proof_dict.get("node_id"),
                            "data_hash": proof_dict.get("data_hash"),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error reading proof file {proof_file}: {e}")

        except Exception as e:
            logger.error(f"Error listing proofs: {e}")

        return proofs
