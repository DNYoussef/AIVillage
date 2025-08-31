"""
Service Interface Specifications for Fog Coordinator Refactoring

This file defines the abstract interfaces and data models for the 6 services
that will replace the monolithic fog_coordinator.py implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time
import torch

# =============================================================================
# Common Data Models
# =============================================================================

@dataclass
class ServiceEvent:
    """Base class for inter-service events."""
    event_type: str
    source_service: str
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceInstance:
    """Information about a registered service."""
    service_id: str
    service_type: str
    endpoint: str
    health_status: str = "unknown"
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HealthStatus(Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class DeviceCapability:
    """Device capability information."""
    device_id: str
    cpu_cores: int
    ram_mb: float
    storage_gb: float
    compute_gflops: float
    memory_gb: float
    bandwidth_mbps: float
    battery_percent: Optional[float] = None
    trust_score: float = 0.5
    latency_ms: float = 100.0
    evolution_capacity: float = 1.0

@dataclass
class ResourceRequirements:
    """Resource requirements specification."""
    min_cpu_cores: int = 1
    min_ram_mb: float = 1024
    min_compute_gflops: float = 1.0
    min_memory_gb: float = 2.0
    min_bandwidth_mbps: float = 10.0
    min_trust_score: float = 0.6
    min_reliability: float = 0.7
    max_nodes: int = 10
    duration_minutes: int = 60

@dataclass
class ComputeWorkload:
    """Compute workload specification."""
    workload_type: str
    total_workload: float
    deadline: float
    priority: str = "normal"
    data_size_mb: float = 0.0
    memory_requirements_gb: float = 1.0

# =============================================================================
# 1. Fog Orchestration Service Interface
# =============================================================================

class ServiceRegistry:
    """Service registry for managing service instances."""
    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.service_types: Dict[str, List[str]] = {}

class IFogOrchestrationService(ABC):
    """Interface for the Fog Orchestration Service."""

    @abstractmethod
    async def initialize_services(self) -> ServiceRegistry:
        """Initialize the service registry and discover available services."""
        pass

    @abstractmethod
    async def register_service(self, service_id: str, service_instance: Any) -> bool:
        """Register a service instance in the registry."""
        pass

    @abstractmethod
    async def discover_service(self, service_type: str) -> Optional[ServiceInstance]:
        """Discover a service instance by type."""
        pass

    @abstractmethod
    async def health_check_all_services(self) -> Dict[str, HealthStatus]:
        """Perform health checks on all registered services."""
        pass

    @abstractmethod
    async def shutdown_services(self, graceful: bool = True) -> bool:
        """Shutdown all services gracefully or forcefully."""
        pass

    @abstractmethod
    async def route_event(self, event: ServiceEvent) -> None:
        """Route an event to the appropriate service handler."""
        pass

    @abstractmethod
    async def broadcast_event(self, event: ServiceEvent) -> None:
        """Broadcast an event to all interested services."""
        pass

# =============================================================================
# 2. Fog Harvesting Service Interface
# =============================================================================

@dataclass
class SuitabilityScore:
    """Device suitability assessment result."""
    device_id: str
    overall_score: float
    component_scores: Dict[str, float]
    is_suitable: bool
    reasons: List[str] = field(default_factory=list)

@dataclass
class AllocationPlan:
    """Resource allocation plan."""
    allocation_id: str
    participants: List[str]
    workload_distribution: Dict[str, ComputeWorkload]
    estimated_completion_time: float
    total_cost: float

@dataclass
class BurstRequest:
    """Compute burst request specification."""
    requester_wallet: str
    compute_requirements: ResourceRequirements
    duration_minutes: int
    max_cost_tokens: float
    priority: str = "normal"

@dataclass
class DeviceHealth:
    """Device health status."""
    device_id: str
    is_online: bool
    cpu_usage_percent: float
    memory_usage_percent: float
    last_seen: float
    response_time_ms: float

class IFogHarvestingService(ABC):
    """Interface for the Fog Harvesting Service."""

    @abstractmethod
    async def discover_devices(self) -> List[DeviceCapability]:
        """Discover available devices and their capabilities."""
        pass

    @abstractmethod
    async def assess_device_suitability(self, device_id: str) -> SuitabilityScore:
        """Assess a device's suitability for compute tasks."""
        pass

    @abstractmethod
    async def select_participants(self, requirements: ResourceRequirements) -> List[str]:
        """Select suitable participants based on requirements."""
        pass

    @abstractmethod
    async def allocate_resources(self, participants: List[str], workload: ComputeWorkload) -> AllocationPlan:
        """Allocate resources to participants for a workload."""
        pass

    @abstractmethod
    async def coordinate_burst(self, burst_request: BurstRequest) -> str:
        """Coordinate a compute burst, returning the burst ID."""
        pass

    @abstractmethod
    async def monitor_device_health(self, device_id: str) -> DeviceHealth:
        """Monitor the health status of a specific device."""
        pass

# =============================================================================
# 3. Fog Marketplace Service Interface  
# =============================================================================

@dataclass
class ResourceSpec:
    """Resource specification for pricing."""
    compute_gflops: float
    memory_gb: float
    storage_gb: float
    network_mbps: float
    duration_hours: float
    quality_level: str = "standard"

@dataclass
class PricingQuote:
    """Pricing quote for resources."""
    resource_spec: ResourceSpec
    base_price: float
    quality_multiplier: float
    demand_multiplier: float
    total_price: float
    validity_seconds: int = 300
    quote_id: str = ""

@dataclass
class ResourceRequest:
    """Resource request for auction."""
    request_id: str
    resource_spec: ResourceSpec
    max_budget: float
    deadline: float
    requester_wallet: str

@dataclass
class ResourceBid:
    """Bid for resource auction."""
    bid_id: str
    bidder_id: str
    price_per_unit: float
    available_resources: ResourceSpec
    quality_score: float

@dataclass
class AuctionResult:
    """Result of resource auction."""
    auction_id: str
    winning_bids: List[ResourceBid]
    total_cost: float
    resource_allocation: Dict[str, ResourceSpec]

@dataclass
class CompletedJob:
    """Information about a completed job."""
    job_id: str
    resource_providers: List[str]
    total_cost: float
    completion_time: float
    quality_metrics: Dict[str, float]

@dataclass
class RevenueDistribution:
    """Revenue distribution plan."""
    job_id: str
    total_revenue: float
    provider_shares: Dict[str, float]
    platform_fee: float
    validator_rewards: Dict[str, float]

@dataclass
class MarketMetrics:
    """Marketplace metrics."""
    total_volume: float
    active_providers: int
    active_consumers: int
    average_price_per_gflop: float
    market_utilization: float

class IFogMarketplaceService(ABC):
    """Interface for the Fog Marketplace Service."""

    @abstractmethod
    async def price_resources(self, resource_spec: ResourceSpec) -> PricingQuote:
        """Calculate pricing for specified resources."""
        pass

    @abstractmethod
    async def create_auction(self, resource_request: ResourceRequest) -> str:
        """Create a new resource auction, returning auction ID."""
        pass

    @abstractmethod
    async def submit_bid(self, auction_id: str, bid: ResourceBid) -> bool:
        """Submit a bid for a resource auction."""
        pass

    @abstractmethod
    async def settle_auction(self, auction_id: str) -> AuctionResult:
        """Settle an auction and return the results."""
        pass

    @abstractmethod
    async def distribute_revenue(self, completed_job: CompletedJob) -> RevenueDistribution:
        """Distribute revenue from a completed job."""
        pass

    @abstractmethod
    async def get_market_metrics(self) -> MarketMetrics:
        """Get current marketplace metrics."""
        pass

# =============================================================================
# 4. Fog Privacy Service Interface
# =============================================================================

@dataclass
class HiddenServiceSpec:
    """Hidden service specification."""
    service_data: bytes
    privacy_level: str
    max_hosting_nodes: int
    access_control: Dict[str, Any]
    compute_budget: float

@dataclass
class RouteSpec:
    """Onion routing specification."""
    source_node: str
    destination_node: str
    min_hops: int
    max_latency_ms: float
    anonymity_level: str

@dataclass
class OnionRoute:
    """Established onion route."""
    route_id: str
    node_sequence: List[str]
    encryption_keys: List[bytes]
    estimated_latency_ms: float

@dataclass
class ParticipantData:
    """Participant data for Byzantine detection."""
    participant_id: str
    gradients: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    timestamp: float

class IFogPrivacyService(ABC):
    """Interface for the Fog Privacy Service."""

    @abstractmethod
    async def initialize_privacy_budgets(self, participants: List[str]) -> Dict[str, float]:
        """Initialize privacy budgets for participants."""
        pass

    @abstractmethod
    async def apply_differential_privacy(self, data: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Apply differential privacy noise to data."""
        pass

    @abstractmethod
    async def create_hidden_service(self, service_spec: HiddenServiceSpec) -> str:
        """Create a hidden service, returning service ID."""
        pass

    @abstractmethod
    async def setup_onion_routing(self, route_spec: RouteSpec) -> OnionRoute:
        """Set up an onion route for anonymous communication."""
        pass

    @abstractmethod
    async def secure_aggregate(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform secure aggregation of gradients."""
        pass

    @abstractmethod
    async def detect_byzantine_behavior(self, participant_data: List[ParticipantData]) -> List[str]:
        """Detect Byzantine participants and return their IDs."""
        pass

# =============================================================================
# 5. Fog Tokenomics Service Interface
# =============================================================================

@dataclass
class WalletInfo:
    """Wallet information."""
    wallet_id: str
    balance: float
    total_earned: float
    total_spent: float
    last_transaction: float
    status: str

@dataclass
class Transaction:
    """Transaction record."""
    transaction_id: str
    from_wallet: str
    to_wallet: str
    amount: float
    transaction_type: str
    timestamp: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransactionResult:
    """Transaction processing result."""
    transaction_id: str
    status: str
    confirmation_time: float
    gas_used: float
    error_message: Optional[str] = None

@dataclass
class WorkProof:
    """Proof of computational work."""
    worker_id: str
    work_type: str
    compute_units: float
    quality_score: float
    verification_data: Dict[str, Any]
    timestamp: float

class IFogTokenomicsService(ABC):
    """Interface for the Fog Tokenomics Service."""

    @abstractmethod
    async def initialize_wallet(self, wallet_id: str) -> WalletInfo:
        """Initialize a new wallet."""
        pass

    @abstractmethod
    async def process_payment(self, transaction: Transaction) -> TransactionResult:
        """Process a payment transaction."""
        pass

    @abstractmethod
    async def calculate_mining_rewards(self, work_proof: WorkProof) -> float:
        """Calculate mining rewards for completed work."""
        pass

    @abstractmethod
    async def distribute_rewards(self, rewards: Dict[str, float]) -> bool:
        """Distribute rewards to multiple wallets."""
        pass

    @abstractmethod
    async def verify_balance(self, wallet_id: str, amount: float) -> bool:
        """Verify if wallet has sufficient balance."""
        pass

    @abstractmethod
    async def get_transaction_history(self, wallet_id: str) -> List[Transaction]:
        """Get transaction history for a wallet."""
        pass

# =============================================================================
# 6. Fog System Stats Service Interface
# =============================================================================

@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: float
    active_services: int
    total_devices: int
    compute_utilization: float
    memory_utilization: float
    network_throughput_mbps: float
    error_rate: float
    response_time_ms: float

@dataclass
class MetricEvent:
    """Individual metric event."""
    metric_name: str
    metric_value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class TimeWindow:
    """Time window specification."""
    start_time: float
    end_time: float
    granularity: str = "minute"  # second, minute, hour, day

@dataclass
class PerformanceStats:
    """Performance statistics over a time window."""
    time_window: TimeWindow
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    throughput: float
    error_count: int
    availability_percent: float

@dataclass
class HealthReport:
    """System health report."""
    overall_health: HealthStatus
    service_health: Dict[str, HealthStatus]
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    generated_at: float = field(default_factory=time.time)

@dataclass
class AlertConfig:
    """Alert configuration."""
    metric_name: str
    threshold: float
    comparison: str  # "greater", "less", "equal"
    duration_seconds: int
    cooldown_seconds: int
    notification_channels: List[str]

class ExportFormat(Enum):
    """Export format options."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    PROMETHEUS = "prometheus"

class IFogSystemStatsService(ABC):
    """Interface for the Fog System Stats Service."""

    @abstractmethod
    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        pass

    @abstractmethod
    async def record_event(self, event: MetricEvent) -> None:
        """Record a metric event."""
        pass

    @abstractmethod
    async def get_performance_stats(self, time_window: TimeWindow) -> PerformanceStats:
        """Get performance statistics for a time window."""
        pass

    @abstractmethod
    async def generate_health_report(self) -> HealthReport:
        """Generate a comprehensive health report."""
        pass

    @abstractmethod
    async def setup_alerts(self, alert_config: AlertConfig) -> None:
        """Set up monitoring alerts."""
        pass

    @abstractmethod
    async def export_analytics(self, format: ExportFormat) -> bytes:
        """Export analytics data in specified format."""
        pass

# =============================================================================
# Service Factory Interface
# =============================================================================

class IServiceFactory(ABC):
    """Factory interface for creating service instances."""

    @abstractmethod
    async def create_orchestration_service(self) -> IFogOrchestrationService:
        """Create orchestration service instance."""
        pass

    @abstractmethod
    async def create_harvesting_service(self) -> IFogHarvestingService:
        """Create harvesting service instance."""
        pass

    @abstractmethod
    async def create_marketplace_service(self) -> IFogMarketplaceService:
        """Create marketplace service instance."""
        pass

    @abstractmethod
    async def create_privacy_service(self) -> IFogPrivacyService:
        """Create privacy service instance."""
        pass

    @abstractmethod
    async def create_tokenomics_service(self) -> IFogTokenomicsService:
        """Create tokenomics service instance."""
        pass

    @abstractmethod
    async def create_system_stats_service(self) -> IFogSystemStatsService:
        """Create system stats service instance."""
        pass

# =============================================================================
# Integration Events
# =============================================================================

class ServiceEventTypes:
    """Predefined service event types."""
    # Device events
    DEVICE_DISCOVERED = "device_discovered"
    DEVICE_OFFLINE = "device_offline"
    DEVICE_CAPABILITY_UPDATED = "device_capability_updated"
    
    # Market events  
    RESOURCE_PRICED = "resource_priced"
    AUCTION_CREATED = "auction_created"
    AUCTION_COMPLETED = "auction_completed"
    PAYMENT_PROCESSED = "payment_processed"
    
    # Privacy events
    PRIVACY_BUDGET_UPDATED = "privacy_budget_updated"
    BYZANTINE_DETECTED = "byzantine_detected"
    HIDDEN_SERVICE_DEPLOYED = "hidden_service_deployed"
    
    # System events
    SERVICE_HEALTH_CHANGED = "service_health_changed"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"
    SYSTEM_ERROR = "system_error"