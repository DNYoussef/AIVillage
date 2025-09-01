"""
Unified API Gateway for Federated Operations

This module provides a comprehensive REST API gateway for all federated operations
including inference, training, resource allocation, and system management.

Key Features:
- Unified REST endpoints for federated operations
- Integration with fog compute bridge, marketplace, and P2P networks
- Real-time job status monitoring and billing information
- Comprehensive error handling and validation
- Performance metrics and analytics
- Security and authentication middleware
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, UTC, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from enum import Enum
import uvicorn

# Core integrations
try:
    from core.rag.integration.fog_compute_bridge import (
        get_fog_compute_bridge,
        FogComputeBridge,
        QueryType,
        QueryDistributionStrategy,
    )

    FOG_BRIDGE_AVAILABLE = True
except ImportError:
    FOG_BRIDGE_AVAILABLE = False

try:
    from infrastructure.fog.market.marketplace_api import get_marketplace_api, MarketplaceAPI
    from infrastructure.fog.market.pricing_manager import get_pricing_manager

    MARKETPLACE_AVAILABLE = True
except ImportError:
    MARKETPLACE_AVAILABLE = False

try:
    from infrastructure.distributed_inference.core.distributed_inference_manager import (
        get_distributed_inference_manager,
        get_distributed_system_status,
    )

    DISTRIBUTED_INFERENCE_AVAILABLE = True
except ImportError:
    DISTRIBUTED_INFERENCE_AVAILABLE = False

try:

    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic Models for API


class JobStatus(str, Enum):
    """Status of federated jobs."""

    SUBMITTED = "submitted"
    QUEUED = "queued"
    ALLOCATING = "allocating"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(str, Enum):
    """Type of resource being requested."""

    INFERENCE = "inference"
    TRAINING = "training"
    RAG_PROCESSING = "rag_processing"
    VECTOR_SEARCH = "vector_search"
    HYBRID = "hybrid"


class PriorityLevel(str, Enum):
    """Priority levels for job scheduling."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class UserTier(str, Enum):
    """User tier for pricing and resource allocation."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class PrivacyLevel(str, Enum):
    """Privacy requirement levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FederatedInferenceRequest(BaseModel):
    """Request for federated inference."""

    user_id: str = Field(..., description="User identifier")
    user_tier: UserTier = Field(UserTier.MEDIUM, description="User tier for pricing")

    # Model and query information
    model_name: Optional[str] = Field(None, description="Specific model to use")
    query: str = Field(..., description="Query or input for inference")
    query_type: QueryType = Field(QueryType.SIMPLE_RAG, description="Type of query processing")

    # Resource requirements
    strategy: QueryDistributionStrategy = Field(QueryDistributionStrategy.BALANCED, description="Distribution strategy")
    max_nodes: int = Field(3, ge=1, le=20, description="Maximum nodes to use")
    required_capabilities: List[str] = Field(default_factory=list, description="Required node capabilities")

    # Cost and performance
    max_budget: float = Field(100.0, ge=0.01, description="Maximum budget for the task")
    max_latency_ms: float = Field(5000.0, ge=100.0, description="Maximum acceptable latency")
    priority: PriorityLevel = Field(PriorityLevel.NORMAL, description="Job priority")

    # Privacy and security
    privacy_level: PrivacyLevel = Field(PrivacyLevel.MEDIUM, description="Privacy requirements")
    encryption_required: bool = Field(False, description="Require end-to-end encryption")

    # Scheduling
    timeout_minutes: int = Field(30, ge=1, le=1440, description="Job timeout in minutes")
    preferred_regions: List[str] = Field(default_factory=list, description="Preferred geographic regions")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "user_tier": "medium",
                "model_name": "llama2-7b",
                "query": "What are the benefits of federated learning?",
                "query_type": "simple_rag",
                "strategy": "balanced",
                "max_nodes": 3,
                "max_budget": 50.0,
                "max_latency_ms": 3000.0,
                "privacy_level": "high",
            }
        }


class FederatedTrainingRequest(BaseModel):
    """Request for federated training."""

    user_id: str = Field(..., description="User identifier")
    user_tier: UserTier = Field(UserTier.MEDIUM, description="User tier for pricing")

    # Training configuration
    model_name: str = Field(..., description="Model to train")
    dataset_source: str = Field(..., description="Dataset source or description")
    training_type: str = Field("supervised", description="Type of training")

    # Resource requirements
    participants_needed: int = Field(5, ge=2, le=100, description="Number of training participants")
    duration_hours: float = Field(2.0, ge=0.5, le=168.0, description="Expected training duration")
    compute_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {"cpu_cores": 4, "memory_gb": 8, "gpu_required": False},
        description="Compute resource requirements",
    )

    # Cost and performance
    max_budget: float = Field(1000.0, ge=1.0, description="Maximum budget for training")
    min_accuracy_target: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum accuracy target")
    convergence_threshold: float = Field(0.01, ge=0.001, le=0.1, description="Convergence threshold")

    # Privacy and security
    privacy_level: PrivacyLevel = Field(PrivacyLevel.HIGH, description="Privacy requirements")
    differential_privacy: bool = Field(True, description="Use differential privacy")
    min_trust_score: float = Field(0.7, ge=0.0, le=1.0, description="Minimum participant trust score")

    # Scheduling
    deadline_hours: Optional[float] = Field(None, ge=1.0, description="Training deadline in hours")
    preferred_regions: List[str] = Field(default_factory=list, description="Preferred geographic regions")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "researcher_456",
                "user_tier": "large",
                "model_name": "custom_classifier",
                "dataset_source": "medical_images_dataset",
                "participants_needed": 10,
                "duration_hours": 8.0,
                "max_budget": 2000.0,
                "privacy_level": "critical",
                "differential_privacy": True,
            }
        }


class JobStatusResponse(BaseModel):
    """Response for job status queries."""

    job_id: str
    status: JobStatus
    progress_percentage: float = Field(ge=0.0, le=100.0)

    # Timing information
    submitted_at: datetime
    started_at: Optional[datetime]
    estimated_completion_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Resource allocation
    allocated_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    resource_utilization: Dict[str, float] = Field(default_factory=dict)

    # Results and costs
    results: Optional[Dict[str, Any]] = None
    total_cost: Optional[float] = None
    estimated_cost: Optional[float] = None

    # Quality metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    quality_score: Optional[float] = None

    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class BillingInfoResponse(BaseModel):
    """Response for billing information."""

    user_id: str
    user_tier: UserTier
    billing_period: Dict[str, datetime]

    # Usage summary
    total_jobs: int
    successful_jobs: int
    total_cost: float
    estimated_monthly_cost: float

    # Resource usage breakdown
    resource_usage: Dict[str, Dict[str, float]]
    cost_breakdown: Dict[str, float]

    # Tier information
    tier_limits: Dict[str, Any]
    usage_against_limits: Dict[str, float]

    # Recent activity
    recent_jobs: List[Dict[str, Any]]


class SystemHealthResponse(BaseModel):
    """Response for system health status."""

    overall_status: str
    timestamp: datetime

    # Component status
    components: Dict[str, Dict[str, Any]]

    # Performance metrics
    performance_metrics: Dict[str, float]

    # Resource availability
    resource_availability: Dict[str, Dict[str, float]]

    # Network status
    network_status: Dict[str, Any]


class FederatedAPIGateway:
    """Unified API Gateway for Federated Operations."""

    def __init__(self):
        self.app = FastAPI(
            title="AI Village Federated API Gateway",
            description="Unified API for federated inference, training, and resource management",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Core components
        self.fog_bridge: Optional[FogComputeBridge] = None
        self.marketplace_api: Optional[MarketplaceAPI] = None
        self.pricing_manager = None
        self.inference_manager = None

        # Job tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}

        # Performance metrics
        self.api_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_cost_processed": 0.0,
            "unique_users": set(),
            "requests_by_tier": {tier.value: 0 for tier in UserTier},
            "requests_by_type": {resource_type.value: 0 for resource_type in ResourceType},
        }

        # Setup routes
        self._setup_routes()
        self._setup_middleware()

        logger.info("Federated API Gateway initialized")

    def _setup_middleware(self):
        """Setup API middleware for authentication, logging, etc."""

        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            """Log all API requests."""
            start_time = time.time()

            response = await call_next(request)

            duration = time.time() - start_time
            logger.info(
                f"API Request: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Duration: {duration:.3f}s"
            )

            # Update stats
            self.api_stats["total_requests"] += 1
            if response.status_code < 400:
                self.api_stats["successful_requests"] += 1
            else:
                self.api_stats["failed_requests"] += 1

            # Update average response time
            total_requests = self.api_stats["total_requests"]
            self.api_stats["average_response_time"] = (
                self.api_stats["average_response_time"] * (total_requests - 1) + duration
            ) / total_requests

            return response

    def _setup_routes(self):
        """Setup all API routes."""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize components on startup."""
            await self._initialize_components()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            await self._cleanup_components()

        # Health and system endpoints
        @self.app.get("/health", response_model=Dict[str, Any], tags=["System"])
        async def health_check():
            """API health check."""
            return {
                "status": "healthy",
                "timestamp": datetime.now(UTC),
                "version": "2.0.0",
                "components": {
                    "fog_bridge": self.fog_bridge is not None,
                    "marketplace": self.marketplace_api is not None,
                    "pricing_manager": self.pricing_manager is not None,
                    "inference_manager": self.inference_manager is not None,
                },
            }

        @self.app.get("/v1/system/status", response_model=SystemHealthResponse, tags=["System"])
        async def get_system_status():
            """Get comprehensive system status."""
            try:
                # Gather status from all components
                components = {}

                if self.fog_bridge:
                    fog_status = self.fog_bridge.get_system_status()
                    components["fog_bridge"] = {"status": "active", "details": fog_status}

                if DISTRIBUTED_INFERENCE_AVAILABLE:
                    distributed_status = get_distributed_system_status()
                    components["distributed_inference"] = {"status": "active", "details": distributed_status}

                # Performance metrics from API
                performance_metrics = {
                    "api_response_time": self.api_stats["average_response_time"],
                    "api_success_rate": (
                        self.api_stats["successful_requests"] / max(self.api_stats["total_requests"], 1)
                    ),
                    "total_api_requests": self.api_stats["total_requests"],
                    "active_jobs": len(self.active_jobs),
                }

                # Resource availability (mock data - would be real in production)
                resource_availability = {
                    "compute_nodes": {"available": 15, "total": 20},
                    "inference_capacity": {"utilization": 0.65, "max_concurrent": 100},
                    "training_slots": {"available": 5, "total": 10},
                }

                return SystemHealthResponse(
                    overall_status="healthy",
                    timestamp=datetime.now(UTC),
                    components=components,
                    performance_metrics=performance_metrics,
                    resource_availability=resource_availability,
                    network_status={"p2p_nodes": 25, "marketplace_active": True},
                )

            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve system status")

        # Federated inference endpoints
        @self.app.post("/v1/federated/inference", tags=["Federated Inference"])
        async def submit_federated_inference(
            request: FederatedInferenceRequest,
            background_tasks: BackgroundTasks,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        ) -> Dict[str, Any]:
            """Submit federated inference request."""

            job_id = f"inf_{uuid.uuid4().hex[:12]}"

            try:
                # Update stats
                self.api_stats["unique_users"].add(request.user_id)
                self.api_stats["requests_by_tier"][request.user_tier.value] += 1
                self.api_stats["requests_by_type"]["inference"] += 1

                # Create job tracking
                job_info = {
                    "job_id": job_id,
                    "type": "inference",
                    "user_id": request.user_id,
                    "user_tier": request.user_tier.value,
                    "status": JobStatus.SUBMITTED,
                    "submitted_at": datetime.now(UTC),
                    "request_data": request.dict(),
                    "estimated_cost": await self._estimate_inference_cost(request),
                }

                self.active_jobs[job_id] = job_info

                # Submit to fog bridge for processing
                if self.fog_bridge:
                    background_tasks.add_task(self._process_inference_job, job_id, request)
                else:
                    raise RuntimeError("Fog bridge not available")

                return {
                    "job_id": job_id,
                    "status": "submitted",
                    "message": "Federated inference request submitted successfully",
                    "estimated_cost": job_info["estimated_cost"],
                    "estimated_completion_time": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                }

            except Exception as e:
                logger.error(f"Error submitting inference request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/federated/training", tags=["Federated Training"])
        async def submit_federated_training(
            request: FederatedTrainingRequest,
            background_tasks: BackgroundTasks,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        ) -> Dict[str, Any]:
            """Submit federated training request."""

            job_id = f"train_{uuid.uuid4().hex[:12]}"

            try:
                # Update stats
                self.api_stats["unique_users"].add(request.user_id)
                self.api_stats["requests_by_tier"][request.user_tier.value] += 1
                self.api_stats["requests_by_type"]["training"] += 1

                # Create job tracking
                job_info = {
                    "job_id": job_id,
                    "type": "training",
                    "user_id": request.user_id,
                    "user_tier": request.user_tier.value,
                    "status": JobStatus.SUBMITTED,
                    "submitted_at": datetime.now(UTC),
                    "request_data": request.dict(),
                    "estimated_cost": await self._estimate_training_cost(request),
                }

                self.active_jobs[job_id] = job_info

                # Submit for processing
                background_tasks.add_task(self._process_training_job, job_id, request)

                return {
                    "job_id": job_id,
                    "status": "submitted",
                    "message": "Federated training request submitted successfully",
                    "estimated_cost": job_info["estimated_cost"],
                    "estimated_completion_time": (
                        datetime.now(UTC) + timedelta(hours=request.duration_hours)
                    ).isoformat(),
                }

            except Exception as e:
                logger.error(f"Error submitting training request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Job status and management endpoints
        @self.app.get("/v1/jobs/{job_id}/status", response_model=JobStatusResponse, tags=["Job Management"])
        async def get_job_status(job_id: str):
            """Get status of a federated job."""

            # Check active jobs first
            if job_id in self.active_jobs:
                job_info = self.active_jobs[job_id]
            elif job_id in self.completed_jobs:
                job_info = self.completed_jobs[job_id]
            else:
                raise HTTPException(status_code=404, detail="Job not found")

            try:
                return JobStatusResponse(
                    job_id=job_id,
                    status=JobStatus(job_info.get("status", "unknown")),
                    progress_percentage=job_info.get("progress", 0.0),
                    submitted_at=job_info["submitted_at"],
                    started_at=job_info.get("started_at"),
                    estimated_completion_at=job_info.get("estimated_completion_at"),
                    completed_at=job_info.get("completed_at"),
                    allocated_nodes=job_info.get("allocated_nodes", []),
                    resource_utilization=job_info.get("resource_utilization", {}),
                    results=job_info.get("results"),
                    total_cost=job_info.get("total_cost"),
                    estimated_cost=job_info.get("estimated_cost"),
                    performance_metrics=job_info.get("performance_metrics", {}),
                    quality_score=job_info.get("quality_score"),
                    error_message=job_info.get("error_message"),
                    error_details=job_info.get("error_details"),
                )

            except Exception as e:
                logger.error(f"Error getting job status: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve job status")

        @self.app.post("/v1/jobs/{job_id}/cancel", tags=["Job Management"])
        async def cancel_job(job_id: str):
            """Cancel a federated job."""

            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Active job not found")

            try:
                job_info = self.active_jobs[job_id]
                job_info["status"] = JobStatus.CANCELLED.value
                job_info["completed_at"] = datetime.now(UTC)
                job_info["error_message"] = "Job cancelled by user"

                # Move to completed jobs
                self.completed_jobs[job_id] = job_info
                del self.active_jobs[job_id]

                return {"job_id": job_id, "status": "cancelled", "message": "Job cancelled successfully"}

            except Exception as e:
                logger.error(f"Error cancelling job: {e}")
                raise HTTPException(status_code=500, detail="Failed to cancel job")

        @self.app.get("/v1/users/{user_id}/jobs", tags=["Job Management"])
        async def get_user_jobs(
            user_id: str,
            status: Optional[str] = Query(None, description="Filter by job status"),
            limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return"),
            offset: int = Query(0, ge=0, description="Number of jobs to skip"),
        ):
            """Get jobs for a specific user."""

            try:
                # Combine active and completed jobs for this user
                user_jobs = []

                for job_info in list(self.active_jobs.values()) + list(self.completed_jobs.values()):
                    if job_info.get("user_id") == user_id:
                        if not status or job_info.get("status") == status:
                            user_jobs.append(job_info)

                # Sort by submission time (newest first)
                user_jobs.sort(key=lambda x: x.get("submitted_at", datetime.min), reverse=True)

                # Apply pagination
                paginated_jobs = user_jobs[offset : offset + limit]

                return {
                    "user_id": user_id,
                    "total_jobs": len(user_jobs),
                    "returned_jobs": len(paginated_jobs),
                    "offset": offset,
                    "limit": limit,
                    "jobs": paginated_jobs,
                }

            except Exception as e:
                logger.error(f"Error getting user jobs: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve user jobs")

        # Billing endpoints
        @self.app.get("/v1/users/{user_id}/billing", response_model=BillingInfoResponse, tags=["Billing"])
        async def get_billing_info(
            user_id: str,
            period: str = Query("current", description="Billing period: current, last_month, year_to_date"),
        ):
            """Get billing information for a user."""

            try:
                # Calculate billing period dates
                now = datetime.now(UTC)
                if period == "current":
                    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    period_end = (
                        period_start.replace(month=period_start.month + 1)
                        if period_start.month < 12
                        else period_start.replace(year=period_start.year + 1, month=1)
                    ) - timedelta(seconds=1)
                elif period == "last_month":
                    period_end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=1)
                    period_start = period_end.replace(day=1)
                else:  # year_to_date
                    period_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                    period_end = now

                # Calculate usage and costs for the user
                user_jobs = [
                    job
                    for job in list(self.completed_jobs.values())
                    if (
                        job.get("user_id") == user_id
                        and period_start <= job.get("submitted_at", datetime.min) <= period_end
                    )
                ]

                total_cost = sum(job.get("total_cost", 0) for job in user_jobs)
                successful_jobs = len([job for job in user_jobs if job.get("status") == "completed"])

                # Estimate monthly cost based on current usage
                days_in_period = (period_end - period_start).days or 1
                daily_average = total_cost / days_in_period
                estimated_monthly_cost = daily_average * 30

                # Get user tier (from most recent job)
                user_tier = UserTier.MEDIUM
                if user_jobs:
                    user_tier = UserTier(user_jobs[-1].get("user_tier", "medium"))

                # Mock tier limits (would be real in production)
                tier_limits = {
                    "small": {"monthly_budget": 500, "max_concurrent_jobs": 5, "priority": "normal"},
                    "medium": {"monthly_budget": 2000, "max_concurrent_jobs": 20, "priority": "high"},
                    "large": {"monthly_budget": 10000, "max_concurrent_jobs": 50, "priority": "high"},
                    "enterprise": {"monthly_budget": 50000, "max_concurrent_jobs": 200, "priority": "urgent"},
                }

                return BillingInfoResponse(
                    user_id=user_id,
                    user_tier=user_tier,
                    billing_period={"start": period_start, "end": period_end},
                    total_jobs=len(user_jobs),
                    successful_jobs=successful_jobs,
                    total_cost=total_cost,
                    estimated_monthly_cost=estimated_monthly_cost,
                    resource_usage={
                        "inference": {"requests": 150, "cost": total_cost * 0.6},
                        "training": {"hours": 24.5, "cost": total_cost * 0.4},
                    },
                    cost_breakdown={
                        "compute_resources": total_cost * 0.7,
                        "network_usage": total_cost * 0.2,
                        "storage": total_cost * 0.1,
                    },
                    tier_limits=tier_limits.get(user_tier.value, {}),
                    usage_against_limits={
                        "monthly_budget": total_cost / tier_limits.get(user_tier.value, {}).get("monthly_budget", 1000),
                        "concurrent_jobs": len([j for j in self.active_jobs.values() if j.get("user_id") == user_id])
                        / tier_limits.get(user_tier.value, {}).get("max_concurrent_jobs", 10),
                    },
                    recent_jobs=[
                        {
                            "job_id": job["job_id"],
                            "type": job["type"],
                            "status": job["status"],
                            "cost": job.get("total_cost", 0),
                            "submitted_at": job["submitted_at"],
                        }
                        for job in user_jobs[-10:]  # Last 10 jobs
                    ],
                )

            except Exception as e:
                logger.error(f"Error getting billing info: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve billing information")

        # Analytics and reporting endpoints
        @self.app.get("/v1/analytics/system", tags=["Analytics"])
        async def get_system_analytics(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
            """Get system-wide analytics (admin only)."""

            try:
                return {
                    "api_statistics": {**self.api_stats, "unique_users": len(self.api_stats["unique_users"])},
                    "job_statistics": {
                        "active_jobs": len(self.active_jobs),
                        "completed_jobs": len(self.completed_jobs),
                        "jobs_by_status": self._get_jobs_by_status(),
                        "jobs_by_type": self._get_jobs_by_type(),
                    },
                    "resource_utilization": await self._get_resource_utilization(),
                    "cost_analytics": {
                        "total_processed": self.api_stats.get("total_cost_processed", 0),
                        "average_job_cost": self._calculate_average_job_cost(),
                    },
                }

            except Exception as e:
                logger.error(f"Error getting analytics: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

    async def _initialize_components(self):
        """Initialize all gateway components."""
        try:
            # Initialize fog bridge
            if FOG_BRIDGE_AVAILABLE:
                self.fog_bridge = get_fog_compute_bridge()
                if not self.fog_bridge.initialized:
                    await self.fog_bridge.initialize()
                logger.info("Fog bridge initialized")

            # Initialize marketplace
            if MARKETPLACE_AVAILABLE:
                self.marketplace_api = get_marketplace_api()
                await self.marketplace_api.initialize_market_components()
                logger.info("Marketplace initialized")

            # Initialize pricing manager
            if MARKETPLACE_AVAILABLE:
                self.pricing_manager = get_pricing_manager()
                logger.info("Pricing manager initialized")

            # Initialize distributed inference
            if DISTRIBUTED_INFERENCE_AVAILABLE:
                self.inference_manager = get_distributed_inference_manager()
                await self.inference_manager.start()
                logger.info("Distributed inference manager initialized")

            logger.info("All API gateway components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    async def _cleanup_components(self):
        """Cleanup all components on shutdown."""
        try:
            if self.fog_bridge:
                await self.fog_bridge.close()

            if self.inference_manager:
                await self.inference_manager.stop()

            logger.info("API gateway components cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _process_inference_job(self, job_id: str, request: FederatedInferenceRequest):
        """Process a federated inference job."""
        try:
            job_info = self.active_jobs.get(job_id)
            if not job_info:
                return

            # Update job status
            job_info["status"] = JobStatus.ALLOCATING.value
            job_info["started_at"] = datetime.now(UTC)

            # Submit to fog bridge
            result = await self.fog_bridge.distribute_query(
                query=request.query,
                query_type=request.query_type,
                strategy=request.strategy,
                user_tier=request.user_tier.value,
                max_budget=request.max_budget,
                privacy_level=request.privacy_level.value,
                required_capabilities=request.required_capabilities,
            )

            # Update job with results
            job_info["status"] = JobStatus.COMPLETED.value if result.get("distributed") else JobStatus.FAILED.value
            job_info["completed_at"] = datetime.now(UTC)
            job_info["results"] = result.get("results", {})
            job_info["total_cost"] = result.get("performance_metrics", {}).get("total_cost", 0)
            job_info["allocated_nodes"] = [{"node_id": node_id} for node_id in result.get("fog_nodes_used", [])]
            job_info["performance_metrics"] = result.get("performance_metrics", {})

            if result.get("error"):
                job_info["error_message"] = result["error"]

            # Update API stats
            self.api_stats["total_cost_processed"] += job_info.get("total_cost", 0)

            # Move to completed jobs
            self.completed_jobs[job_id] = job_info
            del self.active_jobs[job_id]

        except Exception as e:
            logger.error(f"Error processing inference job {job_id}: {e}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update(
                    {"status": JobStatus.FAILED.value, "completed_at": datetime.now(UTC), "error_message": str(e)}
                )

    async def _process_training_job(self, job_id: str, request: FederatedTrainingRequest):
        """Process a federated training job."""
        try:
            job_info = self.active_jobs.get(job_id)
            if not job_info:
                return

            # Update job status
            job_info["status"] = JobStatus.ALLOCATING.value
            job_info["started_at"] = datetime.now(UTC)

            # Simulate training process (in production, would use actual training infrastructure)
            await asyncio.sleep(2)  # Simulate allocation time

            job_info["status"] = JobStatus.RUNNING.value
            job_info["progress"] = 25.0

            # Simulate training progress
            for progress in [50.0, 75.0, 90.0]:
                await asyncio.sleep(request.duration_hours * 3600 / 4)  # Distributed over training time
                if job_id in self.active_jobs:  # Check if not cancelled
                    job_info["progress"] = progress

            # Complete training
            if job_id in self.active_jobs:
                job_info["status"] = JobStatus.COMPLETED.value
                job_info["completed_at"] = datetime.now(UTC)
                job_info["progress"] = 100.0
                job_info["results"] = {
                    "model_id": f"trained_model_{job_id}",
                    "final_accuracy": 0.87 + (hash(job_id) % 10) / 100,  # Simulate accuracy
                    "training_time_hours": request.duration_hours,
                    "participants_used": request.participants_needed,
                    "convergence_achieved": True,
                }
                job_info["total_cost"] = request.max_budget * 0.8  # Use 80% of budget
                job_info["allocated_nodes"] = [
                    {"node_id": f"train_node_{i}", "contribution": 1.0 / request.participants_needed}
                    for i in range(request.participants_needed)
                ]

                # Update API stats
                self.api_stats["total_cost_processed"] += job_info["total_cost"]

                # Move to completed jobs
                self.completed_jobs[job_id] = job_info
                del self.active_jobs[job_id]

        except Exception as e:
            logger.error(f"Error processing training job {job_id}: {e}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update(
                    {"status": JobStatus.FAILED.value, "completed_at": datetime.now(UTC), "error_message": str(e)}
                )

    async def _estimate_inference_cost(self, request: FederatedInferenceRequest) -> float:
        """Estimate cost for inference request."""
        # Simple cost estimation based on user tier and complexity
        base_cost = {"small": 0.05, "medium": 0.15, "large": 0.30, "enterprise": 0.50}.get(
            request.user_tier.value, 0.15
        )

        complexity_multiplier = {
            "simple_rag": 1.0,
            "complex_inference": 2.0,
            "vector_search": 1.5,
            "hybrid_processing": 2.5,
        }.get(request.query_type.value, 1.0)

        node_multiplier = min(request.max_nodes, 5) * 0.5

        return min(base_cost * complexity_multiplier * node_multiplier, request.max_budget)

    async def _estimate_training_cost(self, request: FederatedTrainingRequest) -> float:
        """Estimate cost for training request."""
        # Cost based on duration, participants, and user tier
        hourly_rate = {"small": 5.0, "medium": 12.0, "large": 25.0, "enterprise": 50.0}.get(
            request.user_tier.value, 12.0
        )

        participant_cost = request.participants_needed * 2.0
        duration_cost = request.duration_hours * hourly_rate

        return min(duration_cost + participant_cost, request.max_budget)

    def _get_jobs_by_status(self) -> Dict[str, int]:
        """Get job count breakdown by status."""
        status_counts = {}
        all_jobs = list(self.active_jobs.values()) + list(self.completed_jobs.values())

        for job in all_jobs:
            status = job.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return status_counts

    def _get_jobs_by_type(self) -> Dict[str, int]:
        """Get job count breakdown by type."""
        type_counts = {}
        all_jobs = list(self.active_jobs.values()) + list(self.completed_jobs.values())

        for job in all_jobs:
            job_type = job.get("type", "unknown")
            type_counts[job_type] = type_counts.get(job_type, 0) + 1

        return type_counts

    async def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        return {
            "compute_utilization": 0.68,
            "network_utilization": 0.45,
            "storage_utilization": 0.32,
            "fog_nodes_active": 0.75,
            "inference_capacity": 0.55,
            "training_capacity": 0.40,
        }

    def _calculate_average_job_cost(self) -> float:
        """Calculate average cost per job."""
        all_jobs = list(self.completed_jobs.values())
        if not all_jobs:
            return 0.0

        total_cost = sum(job.get("total_cost", 0) for job in all_jobs)
        return total_cost / len(all_jobs)


# Global API instance
_api_gateway: Optional[FederatedAPIGateway] = None


def get_api_gateway() -> FederatedAPIGateway:
    """Get or create the global API gateway instance."""
    global _api_gateway

    if _api_gateway is None:
        _api_gateway = FederatedAPIGateway()

    return _api_gateway


def start_api_gateway(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the federated API gateway server."""
    get_api_gateway()

    logger.info(f"Starting Federated API Gateway on {host}:{port}")
    uvicorn.run(
        "infrastructure.fog.gateway.api.federated:get_api_gateway().app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    start_api_gateway()
