"""
RESTful Marketplace API for Federated AI Resource Allocation

Provides comprehensive API endpoints for:
- Federated inference and training resource requests
- Size-tier pricing and quotes
- Real-time auction management
- Dynamic resource discovery and allocation
- QoS guarantees and SLA management

Key Features:
- OpenAPI/Swagger compatible endpoints
- Async FastAPI implementation
- Size-tier based pricing
- Multi-criteria resource matching
- Real-time bidding and allocation
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .auction_engine import (
    AuctionEngine,
    create_federated_inference_auction,
    create_federated_training_auction,
    get_auction_engine,
)
from .pricing_manager import (
    DynamicPricingManager,
    ResourceLane,
    UserSizeTier,
    get_pricing_manager,
)
from .market_orchestrator import (
    MarketOrchestrator,
    AllocationStrategy,
    TaskPriority,
    get_market_orchestrator,
)

logger = logging.getLogger(__name__)

# Pydantic models for API


class WorkloadType(str, Enum):
    """Supported workload types"""

    INFERENCE = "inference"
    TRAINING = "training"
    GENERAL = "general"


class ModelSize(str, Enum):
    """Supported model sizes"""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


class PrivacyLevel(str, Enum):
    """Privacy requirement levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReliabilityLevel(str, Enum):
    """Reliability requirement levels"""

    BEST_EFFORT = "best_effort"
    STANDARD = "standard"
    HIGH = "high"
    GUARANTEED = "guaranteed"


class FederatedInferenceRequest(BaseModel):
    """Request for federated inference resources"""

    requester_id: str = Field(..., description="Unique identifier for the requester")
    user_tier: UserSizeTier = Field(..., description="User size tier for pricing")
    model_size: ModelSize = Field(..., description="AI model size category")
    requests_count: int = Field(1, ge=1, le=10000, description="Number of inference requests")
    participants_needed: int = Field(1, ge=1, le=1000, description="Required participant nodes")
    privacy_level: PrivacyLevel = Field(PrivacyLevel.MEDIUM, description="Privacy requirement level")
    max_latency_ms: float = Field(100.0, ge=1.0, le=5000.0, description="Maximum acceptable latency")
    max_budget: float = Field(100.0, ge=0.01, description="Maximum budget for the task")
    preferred_regions: List[str] = Field(default_factory=list, description="Preferred geographic regions")
    deadline_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Task deadline in minutes")

    class Config:
        schema_extra = {
            "example": {
                "requester_id": "user_123",
                "user_tier": "medium",
                "model_size": "large",
                "requests_count": 100,
                "participants_needed": 10,
                "privacy_level": "high",
                "max_latency_ms": 200.0,
                "max_budget": 500.0,
                "preferred_regions": ["us-east", "eu-west"],
                "deadline_minutes": 60,
            }
        }


class FederatedTrainingRequest(BaseModel):
    """Request for federated training resources"""

    requester_id: str = Field(..., description="Unique identifier for the requester")
    user_tier: UserSizeTier = Field(..., description="User size tier for pricing")
    model_size: ModelSize = Field(..., description="AI model size category")
    duration_hours: float = Field(1.0, ge=0.1, le=168.0, description="Training duration in hours")
    participants_needed: int = Field(5, ge=2, le=1000, description="Required participant nodes")
    privacy_level: PrivacyLevel = Field(PrivacyLevel.HIGH, description="Privacy requirement level")
    reliability_requirement: ReliabilityLevel = Field(ReliabilityLevel.HIGH, description="Reliability requirement")
    min_trust_score: float = Field(0.7, ge=0.0, le=1.0, description="Minimum trust score for participants")
    max_budget: float = Field(1000.0, ge=1.0, description="Maximum budget for training")
    preferred_regions: List[str] = Field(default_factory=list, description="Preferred geographic regions")
    deadline_hours: Optional[float] = Field(None, ge=1.0, le=720.0, description="Task deadline in hours")

    class Config:
        schema_extra = {
            "example": {
                "requester_id": "researcher_456",
                "user_tier": "large",
                "model_size": "xlarge",
                "duration_hours": 8.0,
                "participants_needed": 25,
                "privacy_level": "critical",
                "reliability_requirement": "guaranteed",
                "min_trust_score": 0.8,
                "max_budget": 5000.0,
                "preferred_regions": ["us-west", "asia-pacific"],
                "deadline_hours": 24.0,
            }
        }


class ResourceQuoteRequest(BaseModel):
    """Request for resource pricing quote"""

    user_tier: UserSizeTier = Field(..., description="User size tier")
    workload_type: WorkloadType = Field(..., description="Type of workload")
    model_size: ModelSize = Field(..., description="Model size")
    requests_count: Optional[int] = Field(1, ge=1, description="Number of inference requests")
    duration_hours: Optional[float] = Field(1.0, ge=0.1, description="Training duration hours")
    participants_needed: int = Field(1, ge=1, description="Required participants")
    privacy_level: PrivacyLevel = Field(PrivacyLevel.MEDIUM, description="Privacy level")
    reliability_requirement: Optional[ReliabilityLevel] = Field(
        ReliabilityLevel.STANDARD, description="Reliability level"
    )


class AllocationStatusResponse(BaseModel):
    """Resource allocation status response"""

    request_id: str
    status: str
    progress_percentage: float
    estimated_completion_time: Optional[datetime]
    allocated_nodes: List[Dict[str, Any]]
    total_cost: Optional[float]
    quality_metrics: Dict[str, float]
    error_message: Optional[str]


class MarketplaceAPI:
    """Main marketplace API class"""

    def __init__(self):
        self.app = FastAPI(
            title="AI Village Federated Marketplace API",
            description="RESTful API for federated AI resource allocation with size-tier pricing",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Market components
        self.auction_engine: Optional[AuctionEngine] = None
        self.pricing_manager: Optional[DynamicPricingManager] = None
        self.market_orchestrator: Optional[MarketOrchestrator] = None

        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}

        # Setup routes
        self._setup_routes()

        logger.info("Marketplace API initialized")

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize market components on startup"""
            await self.initialize_market_components()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            logger.info("Marketplace API shutting down")

        # Health check endpoint
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(UTC).isoformat(),
                "components": {
                    "auction_engine": self.auction_engine is not None,
                    "pricing_manager": self.pricing_manager is not None,
                    "market_orchestrator": self.market_orchestrator is not None,
                },
            }

        # Federated inference endpoints
        @self.app.post("/federated/inference/request", tags=["Federated Inference"])
        async def request_federated_inference(
            request: FederatedInferenceRequest, background_tasks: BackgroundTasks
        ) -> Dict[str, Any]:
            """Request federated inference resources"""

            if not self.market_orchestrator:
                raise HTTPException(status_code=503, detail="Market orchestrator not available")

            try:
                # Generate request ID
                request_id = f"inf_{uuid.uuid4().hex[:8]}"

                # Create allocation request
                allocation_id = await self.market_orchestrator.request_resources(
                    requester_id=request.requester_id,
                    task_spec={
                        "workload_type": "federated_inference",
                        "model_size": request.model_size.value,
                        "requests_count": request.requests_count,
                        "participants_needed": request.participants_needed,
                        "privacy_level": request.privacy_level.value,
                    },
                    cpu_cores=self._calculate_inference_cpu_requirements(request.model_size.value),
                    memory_gb=self._calculate_inference_memory_requirements(request.model_size.value),
                    duration_hours=1.0,  # Inference is typically short-lived
                    allocation_strategy=AllocationStrategy.BALANCED,
                    task_priority=TaskPriority.NORMAL,
                    max_budget=request.max_budget,
                    max_latency_ms=request.max_latency_ms,
                    required_regions=request.preferred_regions,
                )

                # Track request
                self.active_requests[request_id] = {
                    "allocation_id": allocation_id,
                    "request_type": "federated_inference",
                    "user_tier": request.user_tier.value,
                    "created_at": datetime.now(UTC),
                    "request_data": request.dict(),
                }

                return {
                    "request_id": request_id,
                    "allocation_id": allocation_id,
                    "status": "submitted",
                    "message": "Federated inference request submitted successfully",
                    "estimated_completion_time": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                }

            except Exception as e:
                logger.error(f"Error processing federated inference request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/federated/training/request", tags=["Federated Training"])
        async def request_federated_training(
            request: FederatedTrainingRequest, background_tasks: BackgroundTasks
        ) -> Dict[str, Any]:
            """Request federated training resources"""

            if not self.market_orchestrator:
                raise HTTPException(status_code=503, detail="Market orchestrator not available")

            try:
                # Generate request ID
                request_id = f"train_{uuid.uuid4().hex[:8]}"

                # Create allocation request
                allocation_id = await self.market_orchestrator.request_resources(
                    requester_id=request.requester_id,
                    task_spec={
                        "workload_type": "federated_training",
                        "model_size": request.model_size.value,
                        "duration_hours": request.duration_hours,
                        "participants_needed": request.participants_needed,
                        "privacy_level": request.privacy_level.value,
                        "reliability_requirement": request.reliability_requirement.value,
                    },
                    cpu_cores=self._calculate_training_cpu_requirements(request.model_size.value),
                    memory_gb=self._calculate_training_memory_requirements(request.model_size.value),
                    duration_hours=request.duration_hours,
                    allocation_strategy=AllocationStrategy.BEST_QUALITY,
                    task_priority=TaskPriority.HIGH,
                    max_budget=request.max_budget,
                    min_trust_score=request.min_trust_score,
                    required_regions=request.preferred_regions,
                )

                # Track request
                self.active_requests[request_id] = {
                    "allocation_id": allocation_id,
                    "request_type": "federated_training",
                    "user_tier": request.user_tier.value,
                    "created_at": datetime.now(UTC),
                    "request_data": request.dict(),
                }

                return {
                    "request_id": request_id,
                    "allocation_id": allocation_id,
                    "status": "submitted",
                    "message": "Federated training request submitted successfully",
                    "estimated_completion_time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                }

            except Exception as e:
                logger.error(f"Error processing federated training request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Pricing endpoints
        @self.app.post("/pricing/quote", tags=["Pricing"])
        async def get_pricing_quote(request: ResourceQuoteRequest) -> Dict[str, Any]:
            """Get pricing quote for resources"""

            if not self.pricing_manager:
                raise HTTPException(status_code=503, detail="Pricing manager not available")

            try:
                if request.workload_type == WorkloadType.INFERENCE:
                    quote = await self.pricing_manager.get_federated_inference_price(
                        user_tier=request.user_tier,
                        model_size=request.model_size.value,
                        requests_count=request.requests_count or 1,
                        participants_needed=request.participants_needed,
                        privacy_level=request.privacy_level.value,
                    )
                elif request.workload_type == WorkloadType.TRAINING:
                    quote = await self.pricing_manager.get_federated_training_price(
                        user_tier=request.user_tier,
                        model_size=request.model_size.value,
                        duration_hours=request.duration_hours or 1.0,
                        participants_needed=request.participants_needed,
                        privacy_level=request.privacy_level.value,
                        reliability_requirement=request.reliability_requirement.value,
                    )
                else:
                    # General resource pricing
                    quote = await self.pricing_manager.get_resource_price(
                        lane=ResourceLane.CPU,
                        quantity=Decimal("4.0"),
                        duration_hours=Decimal(str(request.duration_hours or 1.0)),
                    )

                return {
                    "quote_id": f"quote_{uuid.uuid4().hex[:8]}",
                    "pricing_details": quote,
                    "valid_until": (datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
                    "terms": {
                        "currency": "USD",
                        "payment_due": "upon_completion",
                        "cancellation_policy": "free_cancellation_before_allocation",
                    },
                }

            except Exception as e:
                logger.error(f"Error getting pricing quote: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/pricing/tiers", tags=["Pricing"])
        async def get_pricing_tiers() -> Dict[str, Any]:
            """Get information about available pricing tiers"""

            return {
                "pricing_tiers": {
                    "small": {
                        "description": "Mobile-first users with budget constraints",
                        "inference_price_range": "$0.01 - $0.10 per request",
                        "training_price_range": "$1 - $10 per hour",
                        "max_concurrent_jobs": 5,
                        "uptime_guarantee": "95%",
                        "support": "community",
                    },
                    "medium": {
                        "description": "Hybrid cloud-edge users",
                        "inference_price_range": "$0.10 - $1.00 per request",
                        "training_price_range": "$10 - $100 per hour",
                        "max_concurrent_jobs": 20,
                        "uptime_guarantee": "98%",
                        "support": "standard",
                    },
                    "large": {
                        "description": "Cloud-heavy users with high performance needs",
                        "inference_price_range": "$1.00 - $10.00 per request",
                        "training_price_range": "$100 - $1000 per hour",
                        "max_concurrent_jobs": 50,
                        "uptime_guarantee": "99%",
                        "support": "priority",
                    },
                    "enterprise": {
                        "description": "Dedicated enterprise with custom SLAs",
                        "inference_price_range": "$10.00+ per request (custom)",
                        "training_price_range": "$1000+ per hour (custom)",
                        "max_concurrent_jobs": 200,
                        "uptime_guarantee": "99.9%",
                        "support": "dedicated",
                    },
                },
                "tier_selection_guide": {
                    "small": "Ideal for mobile apps, prototype development, low-volume inference",
                    "medium": "Good for production apps, research projects, moderate workloads",
                    "large": "Suitable for enterprise applications, heavy ML workloads",
                    "enterprise": "Custom solutions for large organizations with specific requirements",
                },
            }

        # Status and monitoring endpoints
        @self.app.get("/requests/{request_id}/status", tags=["Resource Allocation"])
        async def get_request_status(request_id: str) -> AllocationStatusResponse:
            """Get status of resource allocation request"""

            if request_id not in self.active_requests:
                raise HTTPException(status_code=404, detail="Request not found")

            try:
                request_info = self.active_requests[request_id]
                allocation_id = request_info["allocation_id"]

                # Get status from market orchestrator
                if self.market_orchestrator:
                    status_info = await self.market_orchestrator.get_allocation_status(allocation_id)

                    if status_info:
                        return AllocationStatusResponse(
                            request_id=request_id,
                            status=status_info["status"],
                            progress_percentage=self._calculate_progress(status_info["status"]),
                            estimated_completion_time=self._estimate_completion_time(status_info),
                            allocated_nodes=status_info["allocation_details"].get("allocated_nodes", []),
                            total_cost=status_info["allocation_details"].get("total_cost"),
                            quality_metrics=self._calculate_quality_metrics(status_info),
                            error_message=None,
                        )

                # Fallback status
                return AllocationStatusResponse(
                    request_id=request_id,
                    status="unknown",
                    progress_percentage=0.0,
                    estimated_completion_time=None,
                    allocated_nodes=[],
                    total_cost=None,
                    quality_metrics={},
                    error_message="Unable to retrieve status",
                )

            except Exception as e:
                logger.error(f"Error getting request status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/market/analytics", tags=["Market Analytics"])
        async def get_market_analytics() -> Dict[str, Any]:
            """Get comprehensive market analytics"""

            try:
                analytics = {}

                if self.market_orchestrator:
                    orchestrator_stats = await self.market_orchestrator.get_market_statistics()
                    analytics["orchestrator"] = orchestrator_stats

                if self.pricing_manager:
                    pricing_analytics = await self.pricing_manager.get_market_analytics()
                    analytics["pricing"] = pricing_analytics

                if self.auction_engine:
                    auction_stats = await self.auction_engine.get_market_statistics()
                    analytics["auctions"] = auction_stats

                # Add API-specific metrics
                analytics["api_metrics"] = {
                    "active_requests": len(self.active_requests),
                    "total_requests": len(self.active_requests),  # Would track historical in production
                    "request_types": self._get_request_type_breakdown(),
                    "tier_usage": self._get_tier_usage_breakdown(),
                }

                return analytics

            except Exception as e:
                logger.error(f"Error getting market analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Auction management endpoints
        @self.app.post("/auctions/federated-inference", tags=["Auctions"])
        async def create_inference_auction(
            requester_id: str,
            model_size: ModelSize,
            participants_needed: int,
            duration_hours: float = 1.0,
            privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM,
            max_latency_ms: float = 100.0,
            reserve_price: Optional[float] = None,
        ) -> Dict[str, Any]:
            """Create auction for federated inference workload"""

            try:
                auction_id = await create_federated_inference_auction(
                    requester_id=requester_id,
                    model_size=model_size.value,
                    participants_needed=participants_needed,
                    duration_hours=duration_hours,
                    privacy_level=privacy_level.value,
                    max_latency_ms=max_latency_ms,
                    reserve_price=reserve_price,
                )

                return {
                    "auction_id": auction_id,
                    "auction_type": "federated_inference",
                    "status": "open",
                    "message": "Federated inference auction created successfully",
                }

            except Exception as e:
                logger.error(f"Error creating inference auction: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/auctions/federated-training", tags=["Auctions"])
        async def create_training_auction(
            requester_id: str,
            model_size: ModelSize,
            participants_needed: int,
            duration_hours: float,
            privacy_level: PrivacyLevel = PrivacyLevel.HIGH,
            reliability_requirement: ReliabilityLevel = ReliabilityLevel.HIGH,
            reserve_price: Optional[float] = None,
        ) -> Dict[str, Any]:
            """Create auction for federated training workload"""

            try:
                auction_id = await create_federated_training_auction(
                    requester_id=requester_id,
                    model_size=model_size.value,
                    participants_needed=participants_needed,
                    duration_hours=duration_hours,
                    privacy_level=privacy_level.value,
                    reliability_requirement=reliability_requirement.value,
                    reserve_price=reserve_price,
                )

                return {
                    "auction_id": auction_id,
                    "auction_type": "federated_training",
                    "status": "open",
                    "message": "Federated training auction created successfully",
                }

            except Exception as e:
                logger.error(f"Error creating training auction: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/auctions/{auction_id}/status", tags=["Auctions"])
        async def get_auction_status(auction_id: str) -> Dict[str, Any]:
            """Get auction status and details"""

            if not self.auction_engine:
                raise HTTPException(status_code=503, detail="Auction engine not available")

            try:
                status = await self.auction_engine.get_auction_status(auction_id)

                if not status:
                    raise HTTPException(status_code=404, detail="Auction not found")

                return status

            except Exception as e:
                logger.error(f"Error getting auction status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def initialize_market_components(self):
        """Initialize market components"""
        try:
            self.auction_engine = await get_auction_engine()
            self.pricing_manager = await get_pricing_manager()
            self.market_orchestrator = await get_market_orchestrator()

            logger.info("Market components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing market components: {e}")
            raise

    def _calculate_inference_cpu_requirements(self, model_size: str) -> float:
        """Calculate CPU requirements for inference based on model size"""
        multipliers = {"small": 2.0, "medium": 4.0, "large": 8.0, "xlarge": 16.0}
        return multipliers.get(model_size, 4.0)

    def _calculate_inference_memory_requirements(self, model_size: str) -> float:
        """Calculate memory requirements for inference based on model size"""
        multipliers = {"small": 4.0, "medium": 8.0, "large": 16.0, "xlarge": 32.0}
        return multipliers.get(model_size, 8.0)

    def _calculate_training_cpu_requirements(self, model_size: str) -> float:
        """Calculate CPU requirements for training based on model size"""
        multipliers = {"small": 4.0, "medium": 8.0, "large": 16.0, "xlarge": 32.0}
        return multipliers.get(model_size, 8.0)

    def _calculate_training_memory_requirements(self, model_size: str) -> float:
        """Calculate memory requirements for training based on model size"""
        multipliers = {"small": 8.0, "medium": 16.0, "large": 32.0, "xlarge": 64.0}
        return multipliers.get(model_size, 16.0)

    def _calculate_progress(self, status: str) -> float:
        """Calculate progress percentage from status"""
        progress_map = {
            "requested": 10.0,
            "pricing": 20.0,
            "auctioning": 40.0,
            "matching": 60.0,
            "allocated": 80.0,
            "deploying": 90.0,
            "active": 100.0,
            "completed": 100.0,
            "failed": 0.0,
            "cancelled": 0.0,
        }
        return progress_map.get(status, 0.0)

    def _estimate_completion_time(self, status_info: Dict[str, Any]) -> Optional[datetime]:
        """Estimate completion time based on status"""
        status = status_info.get("status", "")

        if status in ["completed", "failed", "cancelled"]:
            return None

        # Simple estimation based on status
        minutes_remaining = {
            "requested": 15,
            "pricing": 10,
            "auctioning": 8,
            "matching": 5,
            "allocated": 3,
            "deploying": 2,
            "active": 1,
        }.get(status, 15)

        return datetime.now(UTC) + timedelta(minutes=minutes_remaining)

    def _calculate_quality_metrics(self, status_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics from status info"""
        return {
            "trust_score": 0.8,  # Would calculate from actual allocations
            "latency_ms": 150.0,
            "reliability_score": 0.95,
            "privacy_score": 0.9,
        }

    def _get_request_type_breakdown(self) -> Dict[str, int]:
        """Get breakdown of request types"""
        breakdown = {}
        for req_info in self.active_requests.values():
            req_type = req_info.get("request_type", "unknown")
            breakdown[req_type] = breakdown.get(req_type, 0) + 1
        return breakdown

    def _get_tier_usage_breakdown(self) -> Dict[str, int]:
        """Get breakdown of tier usage"""
        breakdown = {}
        for req_info in self.active_requests.values():
            tier = req_info.get("user_tier", "unknown")
            breakdown[tier] = breakdown.get(tier, 0) + 1
        return breakdown


# Global API instance
_marketplace_api: Optional[MarketplaceAPI] = None


def get_marketplace_api() -> MarketplaceAPI:
    """Get global marketplace API instance"""
    global _marketplace_api

    if _marketplace_api is None:
        _marketplace_api = MarketplaceAPI()

    return _marketplace_api


def start_marketplace_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the marketplace API server"""
    api = get_marketplace_api()
    uvicorn.run(api.app, host=host, port=port)


if __name__ == "__main__":
    start_marketplace_api()
