"""
Proof System API Management

Provides REST API endpoints for cryptographic proof system:
- Proof generation endpoints
- Proof verification endpoints
- Batch processing endpoints
- Tokenomics integration endpoints
- Monitoring and statistics endpoints
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .proof_generator import CryptographicProof, ProofType
from .proof_integration import AuditProofRequest, ProofSystemIntegration, SLAProofRequest, TaskProofRequest
from .tokenomics_integration import ProofTokenomicsIntegration

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class TaskExecutionRequest(BaseModel):
    """Request to generate execution proof"""

    task_id: str
    node_id: str
    input_data: str
    output_data: str
    command: str
    environment: dict[str, Any] = Field(default_factory=dict)
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    start_time: float
    end_time: float
    exit_code: int = 0
    include_witness: bool = True
    computation_trace: list[str] | None = None


class AuditConsensusRequest(BaseModel):
    """Request to generate audit consensus proof"""

    task_id: str
    audit_results: list[dict[str, Any]]
    consensus_threshold: float = 0.67


class SLAComplianceRequest(BaseModel):
    """Request to generate SLA compliance proof"""

    node_id: str
    start_timestamp: float
    end_timestamp: float
    measurements: list[dict[str, Any]]


class ProofVerificationRequest(BaseModel):
    """Request to verify a proof"""

    proof_data: dict[str, Any]


class BatchProofRequest(BaseModel):
    """Request to create batch proof"""

    proof_ids: list[str]


class ProofResponse(BaseModel):
    """Response containing proof data"""

    proof_id: str
    proof_type: str
    timestamp: str
    node_id: str
    data_hash: str
    signature: str | None = None
    merkle_root: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    message: str | None = None


class VerificationResponse(BaseModel):
    """Response containing verification results"""

    proof_id: str
    result: str
    timestamp: str
    verifier_id: str
    signature_valid: bool = False
    merkle_valid: bool = False
    timestamp_valid: bool = False
    data_integrity_valid: bool = False
    consensus_valid: bool = False
    verification_time_ms: float = 0.0
    error_messages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RewardResponse(BaseModel):
    """Response containing reward information"""

    proof_id: str
    base_reward: float
    quality_bonus: float
    verification_bonus: float
    consensus_bonus: float
    total_reward: float
    penalty_amount: float
    net_reward: float
    distributed: bool = False
    recipient_account: str | None = None


class ProofStatsResponse(BaseModel):
    """Response containing proof system statistics"""

    total_proofs_generated: int
    proofs_by_type: dict[str, int]
    total_verifications: int
    verification_success_rate: float
    total_rewards_distributed: float
    active_integrations: dict[str, Any]


class ProofAPIManager:
    """API management for the proof system"""

    def __init__(self, node_id: str, fog_token_system, host: str = "0.0.0.0", port: int = 8080):
        self.node_id = node_id
        self.host = host
        self.port = port

        # Initialize proof system components
        self.proof_integration = ProofSystemIntegration(
            node_id=node_id, proof_storage_dir="proof_storage", enable_auto_proofs=True
        )

        self.tokenomics_integration = ProofTokenomicsIntegration(fog_token_system=fog_token_system)

        # API statistics
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "proof_generation_requests": 0,
            "verification_requests": 0,
            "reward_distribution_requests": 0,
        }

        # Create FastAPI app with lifespan
        self.app = self._create_app()

        logger.info(f"Proof API manager initialized for node {node_id}")

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Lifespan context manager for FastAPI app"""
        # Startup
        await self.proof_integration.start()
        logger.info("Proof API started")

        yield

        # Shutdown
        await self.proof_integration.stop()
        logger.info("Proof API stopped")

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        app = FastAPI(
            title="Fog Computing Proof System API",
            description="Cryptographic proof generation and verification API",
            version="1.0.0",
            lifespan=self.lifespan,
        )

        # Add middleware for request counting
        @app.middleware("http")
        async def count_requests(request, call_next):
            self.api_stats["total_requests"] += 1
            try:
                response = await call_next(request)
                if response.status_code < 400:
                    self.api_stats["successful_requests"] += 1
                else:
                    self.api_stats["failed_requests"] += 1
                return response
            except Exception:
                self.api_stats["failed_requests"] += 1
                raise

        # Proof generation endpoints
        @app.post("/api/v1/proofs/execution", response_model=ProofResponse)
        async def generate_execution_proof(request: TaskExecutionRequest, background_tasks: BackgroundTasks):
            """Generate proof of execution for a task"""
            try:
                self.api_stats["proof_generation_requests"] += 1

                # Create proof request
                proof_request = TaskProofRequest(
                    task_id=request.task_id,
                    node_id=request.node_id,
                    input_data=request.input_data,
                    output_data=request.output_data,
                    command=request.command,
                    environment=request.environment,
                    resource_usage=request.resource_usage,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    exit_code=request.exit_code,
                    include_witness=request.include_witness,
                    computation_trace=request.computation_trace,
                )

                # Generate proof
                proof = await self.proof_integration.generate_execution_proof(proof_request)

                if not proof:
                    raise HTTPException(status_code=500, detail="Failed to generate execution proof")

                # Schedule reward calculation in background
                background_tasks.add_task(self._calculate_and_distribute_reward, proof, request.node_id)

                return ProofResponse(
                    proof_id=proof.proof_id,
                    proof_type=proof.proof_type.value,
                    timestamp=proof.timestamp.isoformat(),
                    node_id=proof.node_id,
                    data_hash=proof.data_hash,
                    signature=proof.signature,
                    merkle_root=proof.merkle_root,
                    metadata=proof.metadata,
                    message="Execution proof generated successfully",
                )

            except Exception as e:
                logger.error(f"Error generating execution proof: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v1/proofs/audit", response_model=ProofResponse)
        async def generate_audit_proof(request: AuditConsensusRequest, background_tasks: BackgroundTasks):
            """Generate proof of audit consensus"""
            try:
                self.api_stats["proof_generation_requests"] += 1

                # Create audit proof request
                audit_request = AuditProofRequest(
                    task_id=request.task_id,
                    audit_results=request.audit_results,
                    consensus_threshold=request.consensus_threshold,
                )

                # Generate proof
                proof = await self.proof_integration.generate_audit_proof(audit_request)

                if not proof:
                    raise HTTPException(status_code=500, detail="Failed to generate audit proof")

                # Schedule reward calculation for audit participants
                auditor_ids = [result.get("auditor_id") for result in request.audit_results if result.get("auditor_id")]
                background_tasks.add_task(self._distribute_consensus_rewards, auditor_ids)

                return ProofResponse(
                    proof_id=proof.proof_id,
                    proof_type=proof.proof_type.value,
                    timestamp=proof.timestamp.isoformat(),
                    node_id=proof.node_id,
                    data_hash=proof.data_hash,
                    signature=proof.signature,
                    merkle_root=proof.merkle_root,
                    metadata=proof.metadata,
                    message="Audit proof generated successfully",
                )

            except Exception as e:
                logger.error(f"Error generating audit proof: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v1/proofs/sla", response_model=ProofResponse)
        async def generate_sla_proof(request: SLAComplianceRequest, background_tasks: BackgroundTasks):
            """Generate proof of SLA compliance"""
            try:
                self.api_stats["proof_generation_requests"] += 1

                # Create SLA proof request
                sla_request = SLAProofRequest(
                    node_id=request.node_id,
                    start_timestamp=request.start_timestamp,
                    end_timestamp=request.end_timestamp,
                    measurements=request.measurements,
                )

                # Generate proof
                proof = await self.proof_integration.generate_sla_proof(sla_request)

                if not proof:
                    raise HTTPException(status_code=500, detail="Failed to generate SLA proof")

                # Schedule reward calculation
                background_tasks.add_task(self._calculate_and_distribute_reward, proof, request.node_id)

                return ProofResponse(
                    proof_id=proof.proof_id,
                    proof_type=proof.proof_type.value,
                    timestamp=proof.timestamp.isoformat(),
                    node_id=proof.node_id,
                    data_hash=proof.data_hash,
                    signature=proof.signature,
                    merkle_root=proof.merkle_root,
                    metadata=proof.metadata,
                    message="SLA proof generated successfully",
                )

            except Exception as e:
                logger.error(f"Error generating SLA proof: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v1/proofs/batch", response_model=ProofResponse)
        async def create_batch_proof(request: BatchProofRequest):
            """Create Merkle batch proof from individual proofs"""
            try:
                self.api_stats["proof_generation_requests"] += 1

                # Create batch proof
                batch_proof = await self.proof_integration.create_batch_proof(request.proof_ids)

                if not batch_proof:
                    raise HTTPException(status_code=500, detail="Failed to create batch proof")

                return ProofResponse(
                    proof_id=batch_proof.proof_id,
                    proof_type=batch_proof.proof_type.value,
                    timestamp=batch_proof.timestamp.isoformat(),
                    node_id=batch_proof.node_id,
                    data_hash=batch_proof.data_hash,
                    signature=batch_proof.signature,
                    merkle_root=batch_proof.merkle_root,
                    metadata=batch_proof.metadata,
                    message="Batch proof created successfully",
                )

            except Exception as e:
                logger.error(f"Error creating batch proof: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Proof verification endpoints
        @app.post("/api/v1/proofs/verify", response_model=VerificationResponse)
        async def verify_proof(request: ProofVerificationRequest, background_tasks: BackgroundTasks):
            """Verify a cryptographic proof"""
            try:
                self.api_stats["verification_requests"] += 1

                # Parse proof data (simplified - would need full proof reconstruction)
                proof_dict = request.proof_data

                # For demonstration, create a mock proof object
                # In production, this would properly reconstruct the proof from the data
                mock_proof = type(
                    "MockProof",
                    (),
                    {
                        "proof_id": proof_dict.get("proof_id", "unknown"),
                        "proof_type": ProofType(proof_dict.get("proof_type", "poe")),
                        "data_hash": proof_dict.get("data_hash", ""),
                        "signature": proof_dict.get("signature"),
                        "public_key_hash": proof_dict.get("public_key_hash"),
                        "merkle_root": proof_dict.get("merkle_root"),
                        "timestamp": datetime.fromisoformat(proof_dict.get("timestamp", datetime.now().isoformat())),
                        "node_id": proof_dict.get("node_id", ""),
                        "metadata": proof_dict.get("metadata", {}),
                        "verification_data": proof_dict.get("verification_data", {}),
                    },
                )()

                # Verify proof
                is_valid = await self.proof_integration.verify_proof(mock_proof)

                # Schedule verification reward
                background_tasks.add_task(
                    self._distribute_verification_reward, self.node_id, mock_proof.proof_type, is_valid
                )

                return VerificationResponse(
                    proof_id=mock_proof.proof_id,
                    result="valid" if is_valid else "invalid",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    verifier_id=self.proof_integration.proof_verifier.verifier_id,
                    signature_valid=is_valid,
                    merkle_valid=is_valid,
                    timestamp_valid=is_valid,
                    data_integrity_valid=is_valid,
                    consensus_valid=is_valid,
                    verification_time_ms=50.0,  # Mock timing
                    error_messages=[] if is_valid else ["Verification failed"],
                    warnings=[],
                )

            except Exception as e:
                logger.error(f"Error verifying proof: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Query endpoints
        @app.get("/api/v1/proofs/{proof_id}", response_model=dict[str, Any])
        async def get_proof(proof_id: str):
            """Get proof by ID"""
            try:
                proof = await self.proof_integration.get_proof_by_id(proof_id)
                if not proof:
                    raise HTTPException(status_code=404, detail="Proof not found")

                return proof if isinstance(proof, dict) else asdict(proof)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting proof: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v1/proofs", response_model=list[dict[str, Any]])
        async def list_proofs(proof_type: str | None = None, limit: int = 100):
            """List proofs with optional filtering"""
            try:
                proofs = await self.proof_integration.list_proofs(proof_type=proof_type, limit=limit)
                return proofs

            except Exception as e:
                logger.error(f"Error listing proofs: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Reward endpoints
        @app.get("/api/v1/rewards/calculate/{proof_id}", response_model=RewardResponse)
        async def calculate_reward(proof_id: str):
            """Calculate reward for a specific proof"""
            try:
                self.api_stats["reward_distribution_requests"] += 1

                # Get proof
                proof = await self.proof_integration.get_proof_by_id(proof_id)
                if not proof:
                    raise HTTPException(status_code=404, detail="Proof not found")

                # Calculate reward (mock implementation)
                from decimal import Decimal

                reward = type(
                    "MockReward",
                    (),
                    {
                        "proof_id": proof_id,
                        "base_reward": Decimal("10.0"),
                        "quality_bonus": Decimal("2.0"),
                        "verification_bonus": Decimal("1.0"),
                        "consensus_bonus": Decimal("0.5"),
                        "total_reward": Decimal("13.5"),
                        "penalty_amount": Decimal("0.0"),
                        "net_reward": Decimal("13.5"),
                    },
                )()

                return RewardResponse(
                    proof_id=reward.proof_id,
                    base_reward=float(reward.base_reward),
                    quality_bonus=float(reward.quality_bonus),
                    verification_bonus=float(reward.verification_bonus),
                    consensus_bonus=float(reward.consensus_bonus),
                    total_reward=float(reward.total_reward),
                    penalty_amount=float(reward.penalty_amount),
                    net_reward=float(reward.net_reward),
                    distributed=False,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error calculating reward: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Statistics endpoints
        @app.get("/api/v1/stats", response_model=ProofStatsResponse)
        async def get_statistics():
            """Get proof system statistics"""
            try:
                integration_stats = self.proof_integration.get_statistics()
                tokenomics_stats = self.tokenomics_integration.get_reward_statistics()

                return ProofStatsResponse(
                    total_proofs_generated=integration_stats.get("execution_proofs_generated", 0)
                    + integration_stats.get("audit_proofs_generated", 0)
                    + integration_stats.get("sla_proofs_generated", 0),
                    proofs_by_type={
                        "execution": integration_stats.get("execution_proofs_generated", 0),
                        "audit": integration_stats.get("audit_proofs_generated", 0),
                        "sla": integration_stats.get("sla_proofs_generated", 0),
                        "batch": integration_stats.get("batch_proofs_generated", 0),
                    },
                    total_verifications=integration_stats.get("proofs_verified", 0),
                    verification_success_rate=95.0,  # Mock value
                    total_rewards_distributed=float(tokenomics_stats.get("total_rewards_distributed", 0)),
                    active_integrations={
                        "api_stats": self.api_stats,
                        "integration_stats": integration_stats,
                        "tokenomics_stats": tokenomics_stats,
                    },
                )

            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
            }

        return app

    async def _calculate_and_distribute_reward(self, proof: CryptographicProof, recipient_account: str):
        """Background task to calculate and distribute proof rewards"""
        try:
            # Calculate reward
            proof_reward = await self.tokenomics_integration.calculate_proof_reward(proof)

            # Distribute reward
            success = await self.tokenomics_integration.distribute_proof_reward(proof_reward, recipient_account)

            if success:
                logger.info(
                    f"Distributed {proof_reward.net_reward} FOG to {recipient_account} for proof {proof.proof_id}"
                )
            else:
                logger.error(f"Failed to distribute reward for proof {proof.proof_id}")

        except Exception as e:
            logger.error(f"Error in reward calculation/distribution: {e}")

    async def _distribute_verification_reward(
        self, verifier_account: str, proof_type: ProofType, verification_successful: bool
    ):
        """Background task to distribute verification rewards"""
        try:
            success = await self.tokenomics_integration.distribute_verification_reward(
                verifier_account, proof_type, verification_successful
            )

            if success:
                logger.debug(f"Distributed verification reward to {verifier_account}")

        except Exception as e:
            logger.error(f"Error distributing verification reward: {e}")

    async def _distribute_consensus_rewards(self, participant_accounts: list[str]):
        """Background task to distribute consensus participation rewards"""
        try:
            distributed = await self.tokenomics_integration.distribute_consensus_reward(
                participant_accounts, "audit_participation"
            )

            logger.info(f"Distributed consensus rewards to {distributed}/{len(participant_accounts)} participants")

        except Exception as e:
            logger.error(f"Error distributing consensus rewards: {e}")

    async def start_server(self):
        """Start the API server"""
        import uvicorn

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="info")

        server = uvicorn.Server(config)

        logger.info(f"Starting proof API server on {self.host}:{self.port}")
        await server.serve()

    def get_api_statistics(self) -> dict[str, Any]:
        """Get API usage statistics"""
        total_requests = self.api_stats["total_requests"]
        success_rate = (self.api_stats["successful_requests"] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self.api_stats,
            "success_rate_percent": success_rate,
            "node_id": self.node_id,
            "endpoints": {
                "/api/v1/proofs/execution": "Generate execution proof",
                "/api/v1/proofs/audit": "Generate audit proof",
                "/api/v1/proofs/sla": "Generate SLA proof",
                "/api/v1/proofs/batch": "Create batch proof",
                "/api/v1/proofs/verify": "Verify proof",
                "/api/v1/proofs/{id}": "Get proof by ID",
                "/api/v1/proofs": "List proofs",
                "/api/v1/rewards/calculate/{id}": "Calculate reward",
                "/api/v1/stats": "Get statistics",
                "/api/v1/health": "Health check",
            },
        }


# Factory function for creating proof API
def create_proof_api(node_id: str, fog_token_system, host: str = "0.0.0.0", port: int = 8080) -> ProofAPIManager:
    """Create and configure proof API manager"""
    return ProofAPIManager(node_id=node_id, fog_token_system=fog_token_system, host=host, port=port)
