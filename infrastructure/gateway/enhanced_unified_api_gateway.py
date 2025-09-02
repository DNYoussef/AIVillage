"""
Enhanced Unified API Gateway - Complete Fog Computing Integration

Integrates all 8 enhanced fog computing components with the existing backend:
1. TEE Runtime Management
2. Cryptographic Proof System 
3. Zero-Knowledge Predicates
4. Market-based Dynamic Pricing
5. Heterogeneous Byzantine Quorum
6. Onion Routing Integration
7. Bayesian Reputation System
8. VRF Neighbor Selection

Production-ready API gateway with comprehensive fog computing capabilities.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

# Import authentication
# Import enhanced fog computing components
from infrastructure.fog.integration.fog_system_manager import FogSystemManager
from infrastructure.fog.market.pricing_manager import (
    DynamicPricingManager,
    ResourceLane,
    get_pricing_manager,
)
from infrastructure.fog.privacy.onion_routing import OnionRouter
from infrastructure.fog.proofs.proof_generator import ProofGenerator
from infrastructure.fog.proofs.proof_verifier import ProofVerifier
from infrastructure.fog.quorum.quorum_manager import ByzantineQuorumManager
from infrastructure.fog.reputation.bayesian_reputation import BayesianReputationEngine
from infrastructure.fog.scheduler.placement import (
    FogScheduler,
    JobClass,
    JobRequest,
    PlacementStrategy,
    get_scheduler,
)
from infrastructure.fog.tee.tee_runtime_manager import TEERuntimeManager
from infrastructure.fog.vrf.vrf_neighbor_selection import VRFNeighborSelector
from infrastructure.gateway.auth import (
    JWTBearer,
    JWTHandler,
    TokenPayload,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class APIResponse(BaseModel):
    """Standardized API response format."""

    success: bool = True
    data: Any = None
    message: str = ""
    error_code: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str | None = None


class HealthCheckResponse(BaseModel):
    """Health check response format."""

    status: str
    timestamp: datetime
    services: dict[str, dict[str, Any]]
    version: str = "2.0.0"


class TEERuntimeRequest(BaseModel):
    """TEE runtime operation request."""

    operation: str = Field(..., description="TEE operation: create, deploy, execute, terminate")
    code: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class ProofRequest(BaseModel):
    """Proof generation request."""

    proof_type: str = Field(..., description="Type of proof to generate")
    statement: dict[str, Any] = Field(..., description="Statement to prove")
    witness: dict[str, Any] | None = None
    public_inputs: dict[str, Any] = Field(default_factory=dict)


class ZKPredicateRequest(BaseModel):
    """Zero-knowledge predicate request."""

    predicate_type: str = Field(..., description="Type of predicate")
    conditions: list[dict[str, Any]] = Field(..., description="Predicate conditions")
    private_inputs: dict[str, Any] = Field(default_factory=dict)


class PricingRequest(BaseModel):
    """Dynamic pricing request."""

    resource_lane: str = Field(..., description="Resource lane: cpu, memory, storage, etc.")
    quantity: float = Field(default=1.0, ge=0.0)
    duration_hours: float = Field(default=1.0, ge=0.0)
    node_id: str | None = None


class JobSchedulingRequest(BaseModel):
    """Job scheduling request."""

    job_id: str
    namespace: str = "default"
    job_class: str = "B_CLASS"
    cpu_cores: float = Field(default=1.0, ge=0.1)
    memory_gb: float = Field(default=1.0, ge=0.1)
    disk_gb: float = Field(default=1.0, ge=0.1)
    estimated_duration_hours: float = Field(default=1.0, ge=0.0)
    placement_strategy: str = "nsga_ii"


class QuorumRequest(BaseModel):
    """Byzantine quorum consensus request."""

    proposal: dict[str, Any] = Field(..., description="Proposal for consensus")
    quorum_type: str = Field(default="byzantine", description="Type of quorum consensus")
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class OnionRoutingRequest(BaseModel):
    """Onion routing request."""

    destination: str = Field(..., description="Destination endpoint")
    data: dict[str, Any] = Field(..., description="Data to route")
    circuit_length: int = Field(default=3, ge=2, le=6)


class ReputationUpdateRequest(BaseModel):
    """Reputation update request."""

    node_id: str = Field(..., description="Node ID to update")
    event_type: str = Field(..., description="Type of reputation event")
    success: bool = Field(..., description="Whether operation was successful")
    performance_score: float | None = Field(default=None, ge=0.0, le=1.0)


class VRFNeighborRequest(BaseModel):
    """VRF neighbor selection request."""

    node_id: str = Field(..., description="Node ID requesting neighbors")
    neighbor_count: int = Field(default=5, ge=1, le=20)
    selection_criteria: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# SERVICE MANAGER WITH FOG INTEGRATION
# ============================================================================


class EnhancedServiceManager:
    """Enhanced service manager with complete fog computing integration."""

    def __init__(self):
        # Core fog system manager
        self.fog_system_manager = FogSystemManager()

        # Individual fog components
        self.pricing_manager: DynamicPricingManager | None = None
        self.scheduler: FogScheduler | None = None
        self.tee_runtime = TEERuntimeManager()
        self.proof_generator = ProofGenerator()
        self.proof_verifier = ProofVerifier()
        self.quorum_manager = ByzantineQuorumManager()
        self.onion_router = OnionRouter()
        self.reputation_engine = BayesianReputationEngine()
        self.vrf_selector = VRFNeighborSelector()

        # Service status
        self.services = {
            "fog_system_manager": {"available": True, "status": "stopped"},
            "tee_runtime": {"available": True, "status": "stopped"},
            "proof_system": {"available": True, "status": "stopped"},
            "dynamic_pricing": {"available": True, "status": "stopped"},
            "job_scheduler": {"available": True, "status": "stopped"},
            "byzantine_quorum": {"available": True, "status": "stopped"},
            "onion_routing": {"available": True, "status": "stopped"},
            "reputation_system": {"available": True, "status": "stopped"},
            "vrf_neighbor": {"available": True, "status": "stopped"},
        }

    async def start_services(self):
        """Initialize all enhanced fog computing services."""
        logger.info("🚀 Starting enhanced fog computing services...")

        try:
            # Start main fog system manager
            await self.fog_system_manager.start()
            self.services["fog_system_manager"]["status"] = "running"
            logger.info("✅ Fog system manager started")

            # Start individual components
            await self._start_individual_components()

            logger.info("🎯 Enhanced fog computing services initialization complete")

        except Exception as e:
            logger.error(f"❌ Enhanced fog service startup failed: {e}")
            await self._handle_startup_failure(e)

    async def _start_individual_components(self):
        """Start individual fog computing components."""

        # TEE Runtime Manager
        try:
            await self.tee_runtime.start()
            self.services["tee_runtime"]["status"] = "running"
            logger.info("✅ TEE runtime manager started")
        except Exception as e:
            logger.error(f"❌ TEE runtime startup failed: {e}")
            self.services["tee_runtime"]["status"] = "error"

        # Proof System
        try:
            await self.proof_generator.initialize()
            await self.proof_verifier.initialize()
            self.services["proof_system"]["status"] = "running"
            logger.info("✅ Proof system started")
        except Exception as e:
            logger.error(f"❌ Proof system startup failed: {e}")
            self.services["proof_system"]["status"] = "error"

        # Dynamic Pricing Manager
        try:
            self.pricing_manager = await get_pricing_manager()
            self.services["dynamic_pricing"]["status"] = "running"
            logger.info("✅ Dynamic pricing manager started")
        except Exception as e:
            logger.error(f"❌ Pricing manager startup failed: {e}")
            self.services["dynamic_pricing"]["status"] = "error"

        # Job Scheduler
        try:
            self.scheduler = await get_scheduler()
            self.services["job_scheduler"]["status"] = "running"
            logger.info("✅ Job scheduler started")
        except Exception as e:
            logger.error(f"❌ Job scheduler startup failed: {e}")
            self.services["job_scheduler"]["status"] = "error"

        # Byzantine Quorum Manager
        try:
            await self.quorum_manager.start()
            self.services["byzantine_quorum"]["status"] = "running"
            logger.info("✅ Byzantine quorum manager started")
        except Exception as e:
            logger.error(f"❌ Quorum manager startup failed: {e}")
            self.services["byzantine_quorum"]["status"] = "error"

        # Onion Router
        try:
            await self.onion_router.start()
            self.services["onion_routing"]["status"] = "running"
            logger.info("✅ Onion router started")
        except Exception as e:
            logger.error(f"❌ Onion router startup failed: {e}")
            self.services["onion_routing"]["status"] = "error"

        # Reputation System
        try:
            await self.reputation_engine.initialize()
            self.services["reputation_system"]["status"] = "running"
            logger.info("✅ Reputation system started")
        except Exception as e:
            logger.error(f"❌ Reputation system startup failed: {e}")
            self.services["reputation_system"]["status"] = "error"

        # VRF Neighbor Selector
        try:
            await self.vrf_selector.initialize()
            self.services["vrf_neighbor"]["status"] = "running"
            logger.info("✅ VRF neighbor selector started")
        except Exception as e:
            logger.error(f"❌ VRF selector startup failed: {e}")
            self.services["vrf_neighbor"]["status"] = "error"

    async def _handle_startup_failure(self, error: Exception):
        """Handle service startup failures."""
        logger.error(f"Service startup failed: {error}")
        # Mark all services as error state
        for service in self.services.values():
            if service["status"] == "stopped":
                service["status"] = "error"

    async def stop_services(self):
        """Cleanup all enhanced fog services."""
        logger.info("🔄 Stopping enhanced fog computing services...")

        # Stop individual components
        if self.fog_system_manager:
            await self.fog_system_manager.stop()
        if self.tee_runtime:
            await self.tee_runtime.stop()
        if self.quorum_manager:
            await self.quorum_manager.stop()
        if self.onion_router:
            await self.onion_router.stop()
        if self.vrf_selector:
            await self.vrf_selector.cleanup()

        logger.info("✅ All enhanced fog services stopped")

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "services": self.services,
            "fog_system": self.fog_system_manager.get_status() if self.fog_system_manager else {},
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("🚀 Enhanced Unified API Gateway starting up...")
    await service_manager.start_services()

    yield

    # Shutdown
    logger.info("🔄 Enhanced Unified API Gateway shutting down...")
    await service_manager.stop_services()


# Initialize enhanced service manager
service_manager = EnhancedServiceManager()

# Initialize JWT handler
jwt_handler = JWTHandler(
    secret_key=os.getenv("JWT_SECRET_KEY"), require_mfa=os.getenv("REQUIRE_MFA", "false").lower() == "true"
)

# Create FastAPI app
app = FastAPI(
    title="AIVillage Enhanced Unified API Gateway",
    description="Production-ready unified API with complete fog computing integration",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# SECURITY: Import secure CORS configuration
try:
    from src.security.cors_config import SECURE_CORS_CONFIG
except ImportError:
    # Fallback secure configuration if import fails
    import os

    env = os.getenv("AIVILLAGE_ENV", "development")
    if env == "production":
        cors_origins = ["https://aivillage.app", "https://www.aivillage.app"]
    else:
        cors_origins = ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]

    SECURE_CORS_CONFIG = {
        "allow_origins": cors_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Accept", "Content-Type", "Authorization", "X-Requested-With"],
    }

# Add secure CORS middleware - NO WILDCARDS
app.add_middleware(CORSMiddleware, **SECURE_CORS_CONFIG)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security dependencies
security = HTTPBearer()
jwt_auth = JWTBearer(jwt_handler, required_scopes=["read"])
jwt_admin = JWTBearer(jwt_handler, required_scopes=["admin"])

# ============================================================================
# CORE API ENDPOINTS
# ============================================================================


@app.get("/", response_model=APIResponse)
async def root():
    """Enhanced API root endpoint."""
    return APIResponse(
        data={
            "service": "AIVillage Enhanced Unified API Gateway",
            "version": "2.0.0",
            "status": "operational",
            "enhanced_features": [
                "TEE Runtime Management",
                "Cryptographic Proof System",
                "Zero-Knowledge Predicates",
                "Market-based Dynamic Pricing",
                "Heterogeneous Byzantine Quorum",
                "Onion Routing Integration",
                "Bayesian Reputation System",
                "VRF Neighbor Selection",
            ],
            "fog_endpoints": {
                "tee": "POST /v1/fog/tee/runtime",
                "proofs": "POST /v1/fog/proofs/generate",
                "zk_predicates": "POST /v1/fog/zk/predicate",
                "pricing": "GET /v1/fog/pricing/quote",
                "scheduler": "POST /v1/fog/jobs/schedule",
                "quorum": "POST /v1/fog/quorum/consensus",
                "onion": "POST /v1/fog/onion/route",
                "reputation": "POST /v1/fog/reputation/update",
                "vrf": "GET /v1/fog/vrf/neighbors",
                "system": "GET /v1/fog/system/status",
            },
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Enhanced health check with all fog services."""
    services_status = {}
    overall_status = "healthy"

    system_status = service_manager.get_system_status()

    for service_name, service_info in system_status["services"].items():
        if service_info["status"] == "error":
            overall_status = "degraded"
        services_status[service_name] = {"status": service_info["status"], "available": service_info["available"]}

    return HealthCheckResponse(status=overall_status, timestamp=datetime.now(), services=services_status)


# ============================================================================
# TEE RUNTIME ENDPOINTS (/v1/fog/tee/)
# ============================================================================


@app.post("/v1/fog/tee/runtime", response_model=APIResponse)
async def tee_runtime_operation(request: TEERuntimeRequest, token: TokenPayload = Depends(jwt_auth)):
    """Execute TEE runtime operations."""

    if service_manager.services["tee_runtime"]["status"] != "running":
        raise HTTPException(status_code=503, detail="TEE runtime service unavailable")

    try:
        result = await service_manager.tee_runtime.execute_operation(
            operation=request.operation, code=request.code, inputs=request.inputs, config=request.config
        )

        return APIResponse(data=result, message=f"TEE {request.operation} operation completed")

    except Exception as e:
        logger.error(f"TEE runtime operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TEE operation failed: {str(e)}")


@app.get("/v1/fog/tee/status", response_model=APIResponse)
async def get_tee_status(token: TokenPayload = Depends(jwt_auth)):
    """Get TEE runtime status and enclaves."""

    try:
        status = await service_manager.tee_runtime.get_status()
        return APIResponse(data=status, message="TEE status retrieved")

    except Exception as e:
        logger.error(f"Failed to get TEE status: {e}")
        raise HTTPException(status_code=500, detail=f"TEE status retrieval failed: {str(e)}")


# ============================================================================
# CRYPTOGRAPHIC PROOF ENDPOINTS (/v1/fog/proofs/)
# ============================================================================


@app.post("/v1/fog/proofs/generate", response_model=APIResponse)
async def generate_proof(request: ProofRequest, token: TokenPayload = Depends(jwt_auth)):
    """Generate cryptographic proofs."""

    if service_manager.services["proof_system"]["status"] != "running":
        raise HTTPException(status_code=503, detail="Proof system service unavailable")

    try:
        proof = await service_manager.proof_generator.generate_proof(
            proof_type=request.proof_type,
            statement=request.statement,
            witness=request.witness,
            public_inputs=request.public_inputs,
        )

        return APIResponse(
            data={"proof": proof, "proof_type": request.proof_type},
            message="Cryptographic proof generated successfully",
        )

    except Exception as e:
        logger.error(f"Proof generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proof generation failed: {str(e)}")


@app.post("/v1/fog/proofs/verify", response_model=APIResponse)
async def verify_proof(proof_data: dict, token: TokenPayload = Depends(jwt_auth)):
    """Verify cryptographic proofs."""

    try:
        is_valid = await service_manager.proof_verifier.verify_proof(proof_data)

        return APIResponse(
            data={"valid": is_valid, "verification_time": datetime.now().isoformat()},
            message="Proof verification completed",
        )

    except Exception as e:
        logger.error(f"Proof verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Proof verification failed: {str(e)}")


# ============================================================================
# ZERO-KNOWLEDGE PREDICATE ENDPOINTS (/v1/fog/zk/)
# ============================================================================


@app.post("/v1/fog/zk/predicate", response_model=APIResponse)
async def evaluate_zk_predicate(request: ZKPredicateRequest, token: TokenPayload = Depends(jwt_auth)):
    """Evaluate zero-knowledge predicates."""

    try:
        result = await service_manager.proof_generator.evaluate_zk_predicate(
            predicate_type=request.predicate_type, conditions=request.conditions, private_inputs=request.private_inputs
        )

        return APIResponse(data=result, message="Zero-knowledge predicate evaluated")

    except Exception as e:
        logger.error(f"ZK predicate evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"ZK evaluation failed: {str(e)}")


# ============================================================================
# DYNAMIC PRICING ENDPOINTS (/v1/fog/pricing/)
# ============================================================================


@app.get("/v1/fog/pricing/quote", response_model=APIResponse)
async def get_pricing_quote(
    resource_lane: str,
    quantity: float = 1.0,
    duration_hours: float = 1.0,
    node_id: str | None = None,
    token: TokenPayload = Depends(jwt_auth),
):
    """Get dynamic pricing quote for resources."""

    if service_manager.services["dynamic_pricing"]["status"] != "running":
        raise HTTPException(status_code=503, detail="Dynamic pricing service unavailable")

    try:
        lane = ResourceLane(resource_lane)
        quote = await service_manager.pricing_manager.get_resource_price(
            lane=lane, quantity=Decimal(str(quantity)), duration_hours=Decimal(str(duration_hours)), node_id=node_id
        )

        return APIResponse(data=quote, message=f"Pricing quote for {resource_lane} generated")

    except Exception as e:
        logger.error(f"Pricing quote failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pricing failed: {str(e)}")


@app.post("/v1/fog/pricing/bulk", response_model=APIResponse)
async def get_bulk_pricing_quote(
    resource_requirements: dict[str, tuple[float, float]], token: TokenPayload = Depends(jwt_auth)
):
    """Get bulk pricing quote for multiple resources."""

    try:
        # Convert to appropriate format
        requirements = {
            ResourceLane(k): (Decimal(str(v[0])), Decimal(str(v[1]))) for k, v in resource_requirements.items()
        }

        quote = await service_manager.pricing_manager.get_bulk_pricing_quote(requirements)

        return APIResponse(data=quote, message="Bulk pricing quote generated")

    except Exception as e:
        logger.error(f"Bulk pricing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk pricing failed: {str(e)}")


@app.get("/v1/fog/pricing/analytics", response_model=APIResponse)
async def get_pricing_analytics(token: TokenPayload = Depends(jwt_admin)):
    """Get comprehensive pricing analytics."""

    try:
        analytics = await service_manager.pricing_manager.get_market_analytics()

        return APIResponse(data=analytics, message="Pricing analytics retrieved")

    except Exception as e:
        logger.error(f"Pricing analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


# ============================================================================
# JOB SCHEDULER ENDPOINTS (/v1/fog/jobs/)
# ============================================================================


@app.post("/v1/fog/jobs/schedule", response_model=APIResponse)
async def schedule_job(request: JobSchedulingRequest, token: TokenPayload = Depends(jwt_auth)):
    """Schedule fog computing job."""

    if service_manager.services["job_scheduler"]["status"] != "running":
        raise HTTPException(status_code=503, detail="Job scheduler service unavailable")

    try:
        job_request = JobRequest(
            job_id=request.job_id,
            namespace=request.namespace,
            job_class=JobClass(request.job_class.upper()),
            cpu_cores=request.cpu_cores,
            memory_gb=request.memory_gb,
            disk_gb=request.disk_gb,
            estimated_duration_hours=request.estimated_duration_hours,
        )

        strategy = PlacementStrategy(request.placement_strategy.upper())
        result = await service_manager.scheduler.schedule_job(job_request, strategy)

        return APIResponse(data=result, message=f"Job {request.job_id} scheduling completed")

    except Exception as e:
        logger.error(f"Job scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job scheduling failed: {str(e)}")


@app.get("/v1/fog/jobs/scheduler/stats", response_model=APIResponse)
async def get_scheduler_stats(token: TokenPayload = Depends(jwt_auth)):
    """Get job scheduler performance statistics."""

    try:
        stats = service_manager.scheduler.get_scheduler_stats()

        return APIResponse(data=stats, message="Scheduler statistics retrieved")

    except Exception as e:
        logger.error(f"Scheduler stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


# ============================================================================
# BYZANTINE QUORUM ENDPOINTS (/v1/fog/quorum/)
# ============================================================================


@app.post("/v1/fog/quorum/consensus", response_model=APIResponse)
async def initiate_consensus(request: QuorumRequest, token: TokenPayload = Depends(jwt_auth)):
    """Initiate Byzantine quorum consensus."""

    if service_manager.services["byzantine_quorum"]["status"] != "running":
        raise HTTPException(status_code=503, detail="Byzantine quorum service unavailable")

    try:
        result = await service_manager.quorum_manager.initiate_consensus(
            proposal=request.proposal, timeout_seconds=request.timeout_seconds
        )

        return APIResponse(data=result, message="Byzantine consensus completed")

    except Exception as e:
        logger.error(f"Consensus failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consensus failed: {str(e)}")


@app.get("/v1/fog/quorum/status", response_model=APIResponse)
async def get_quorum_status(token: TokenPayload = Depends(jwt_auth)):
    """Get quorum system status."""

    try:
        status = await service_manager.quorum_manager.get_status()

        return APIResponse(data=status, message="Quorum status retrieved")

    except Exception as e:
        logger.error(f"Quorum status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


# ============================================================================
# ONION ROUTING ENDPOINTS (/v1/fog/onion/)
# ============================================================================


@app.post("/v1/fog/onion/route", response_model=APIResponse)
async def create_onion_route(request: OnionRoutingRequest, token: TokenPayload = Depends(jwt_auth)):
    """Create onion routing circuit and route data."""

    if service_manager.services["onion_routing"]["status"] != "running":
        raise HTTPException(status_code=503, detail="Onion routing service unavailable")

    try:
        result = await service_manager.onion_router.route_data(
            destination=request.destination, data=request.data, circuit_length=request.circuit_length
        )

        return APIResponse(data=result, message="Data routed through onion circuit")

    except Exception as e:
        logger.error(f"Onion routing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Onion routing failed: {str(e)}")


@app.get("/v1/fog/onion/circuits", response_model=APIResponse)
async def get_onion_circuits(token: TokenPayload = Depends(jwt_auth)):
    """Get active onion circuits."""

    try:
        circuits = await service_manager.onion_router.get_active_circuits()

        return APIResponse(data={"circuits": circuits}, message="Active circuits retrieved")

    except Exception as e:
        logger.error(f"Circuit retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Circuit retrieval failed: {str(e)}")


# ============================================================================
# REPUTATION SYSTEM ENDPOINTS (/v1/fog/reputation/)
# ============================================================================


@app.post("/v1/fog/reputation/update", response_model=APIResponse)
async def update_reputation(request: ReputationUpdateRequest, token: TokenPayload = Depends(jwt_auth)):
    """Update node reputation score."""

    if service_manager.services["reputation_system"]["status"] != "running":
        raise HTTPException(status_code=503, detail="Reputation system service unavailable")

    try:
        result = await service_manager.reputation_engine.update_reputation(
            node_id=request.node_id,
            event_type=request.event_type,
            success=request.success,
            performance_score=request.performance_score,
        )

        return APIResponse(data=result, message=f"Reputation updated for node {request.node_id}")

    except Exception as e:
        logger.error(f"Reputation update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reputation update failed: {str(e)}")


@app.get("/v1/fog/reputation/score/{node_id}", response_model=APIResponse)
async def get_reputation_score(node_id: str, token: TokenPayload = Depends(jwt_auth)):
    """Get reputation score for a node."""

    try:
        score = service_manager.reputation_engine.get_reputation_score(node_id)
        trust_score = service_manager.reputation_engine.get_trust_score(node_id)

        return APIResponse(
            data={
                "node_id": node_id,
                "reputation_score": score,
                "trust_score": trust_score,
                "timestamp": datetime.now().isoformat(),
            },
            message="Reputation score retrieved",
        )

    except Exception as e:
        logger.error(f"Reputation retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reputation retrieval failed: {str(e)}")


# ============================================================================
# VRF NEIGHBOR SELECTION ENDPOINTS (/v1/fog/vrf/)
# ============================================================================


@app.get("/v1/fog/vrf/neighbors", response_model=APIResponse)
async def get_vrf_neighbors(node_id: str, neighbor_count: int = 5, token: TokenPayload = Depends(jwt_auth)):
    """Get VRF-selected neighbors for a node."""

    if service_manager.services["vrf_neighbor"]["status"] != "running":
        raise HTTPException(status_code=503, detail="VRF neighbor service unavailable")

    try:
        neighbors = await service_manager.vrf_selector.select_neighbors(node_id=node_id, neighbor_count=neighbor_count)

        return APIResponse(
            data={"neighbors": neighbors, "selection_method": "vrf"}, message=f"VRF neighbors selected for {node_id}"
        )

    except Exception as e:
        logger.error(f"VRF neighbor selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"VRF selection failed: {str(e)}")


# ============================================================================
# SYSTEM STATUS ENDPOINTS (/v1/fog/system/)
# ============================================================================


@app.get("/v1/fog/system/status", response_model=APIResponse)
async def get_fog_system_status(token: TokenPayload = Depends(jwt_auth)):
    """Get comprehensive fog system status."""

    try:
        status = service_manager.get_system_status()

        # Add component-specific status
        if service_manager.pricing_manager:
            status["pricing_analytics"] = await service_manager.pricing_manager.get_market_analytics()

        if service_manager.scheduler:
            status["scheduler_stats"] = service_manager.scheduler.get_scheduler_stats()

        return APIResponse(data=status, message="Fog system status retrieved")

    except Exception as e:
        logger.error(f"System status failed: {e}")
        raise HTTPException(status_code=500, detail=f"System status failed: {str(e)}")


@app.get("/v1/fog/system/metrics", response_model=APIResponse)
async def get_fog_system_metrics(token: TokenPayload = Depends(jwt_admin)):
    """Get detailed fog system performance metrics."""

    try:
        metrics = await service_manager.fog_system_manager.get_comprehensive_metrics()

        return APIResponse(data=metrics, message="Fog system metrics retrieved")

    except Exception as e:
        logger.error(f"System metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"System metrics failed: {str(e)}")


# ============================================================================
# WEBSOCKET ENDPOINT FOR REAL-TIME UPDATES
# ============================================================================


@app.websocket("/ws/fog")
async def fog_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fog system updates."""

    await websocket.accept()

    try:
        # Send welcome message with fog capabilities
        await websocket.send_json(
            {
                "type": "fog_connection_established",
                "message": "Connected to Enhanced Fog Computing Gateway",
                "capabilities": [
                    "tee_runtime",
                    "proof_system",
                    "zk_predicates",
                    "dynamic_pricing",
                    "job_scheduler",
                    "byzantine_quorum",
                    "onion_routing",
                    "reputation_system",
                    "vrf_neighbor",
                ],
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()

                # SECURITY: Message size limit check
                if len(data) > 1024 * 1024:  # 1MB limit
                    await websocket.send_json(
                        {"type": "error", "message": "Message too large", "code": "MESSAGE_TOO_LARGE"}
                    )
                    continue

                # SECURITY FIX: Use safe JSON parsing instead of eval()
                message = json.loads(data)

                # SECURITY: Input validation - only allow specific message types
                allowed_types = {"ping", "get_fog_status", "subscribe_pricing", "unsubscribe_pricing"}
                msg_type = message.get("type")

                if not msg_type or msg_type not in allowed_types:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid or unsupported message type",
                            "allowed_types": list(allowed_types),
                            "code": "INVALID_MESSAGE_TYPE",
                        }
                    )
                    continue

                # SECURITY: Additional message validation
                if not isinstance(message, dict):
                    await websocket.send_json(
                        {"type": "error", "message": "Message must be a JSON object", "code": "INVALID_MESSAGE_FORMAT"}
                    )
                    continue

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

                elif msg_type == "get_fog_status":
                    status = service_manager.get_system_status()
                    await websocket.send_json(
                        {
                            "type": "fog_status_update",
                            "data": status,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                elif msg_type == "subscribe_pricing":
                    # Subscribe to pricing updates
                    if service_manager.pricing_manager:
                        analytics = await service_manager.pricing_manager.get_market_analytics()
                        await websocket.send_json(
                            {
                                "type": "pricing_update",
                                "data": analytics,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received from fog WebSocket: {e}")
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON format", "code": "JSON_DECODE_ERROR"}
                )
            except Exception as e:
                logger.error(f"Fog WebSocket message processing error: {e}")
                await websocket.send_json(
                    {"type": "error", "message": "Message processing failed", "code": "PROCESSING_ERROR"}
                )

    except WebSocketDisconnect:
        logger.info("Fog WebSocket client disconnected")


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced AIVillage Unified API Gateway...")

    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    # Run server
    uvicorn.run("enhanced_unified_api_gateway:app", host=host, port=port, reload=debug, log_level="info")
