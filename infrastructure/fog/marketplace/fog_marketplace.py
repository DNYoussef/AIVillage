"""
Fog Computing Marketplace - Decentralized AWS Alternative

Provides a marketplace for fog computing resources where users can:
- Buy compute/storage/bandwidth from fog nodes
- Sell idle resources for tokens
- Host websites and services with censorship resistance
- Get guaranteed SLAs at fraction of cloud costs

Key Features:
- Dynamic pricing based on supply/demand
- Multi-region fog zones for redundancy
- Token-based payment system
- SLA enforcement with penalties
- Hidden service hosting support
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import hashlib
import logging
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of services available in fog marketplace"""

    COMPUTE_INSTANCE = "compute_instance"  # Virtual machines
    SERVERLESS_FUNCTION = "serverless_function"  # Lambda-like functions
    CONTAINER_HOSTING = "container_hosting"  # Docker containers
    STATIC_WEBSITE = "static_website"  # Static site hosting
    DYNAMIC_WEBSITE = "dynamic_website"  # Dynamic web apps
    DATABASE = "database"  # Database hosting
    OBJECT_STORAGE = "object_storage"  # S3-like storage
    BLOCK_STORAGE = "block_storage"  # EBS-like volumes
    CDN = "cdn"  # Content delivery
    ML_INFERENCE = "ml_inference"  # ML model serving
    BATCH_PROCESSING = "batch_processing"  # Batch jobs
    HIDDEN_SERVICE = "hidden_service"  # .fog onion service


class ServiceTier(Enum):
    """Service tier levels with different guarantees"""

    BASIC = "basic"  # Best effort, no SLA
    STANDARD = "standard"  # 99% uptime SLA
    PREMIUM = "premium"  # 99.9% uptime SLA
    ENTERPRISE = "enterprise"  # 99.99% uptime SLA


class PricingModel(Enum):
    """Pricing models for services"""

    PAY_AS_YOU_GO = "pay_as_you_go"  # Per hour/GB
    RESERVED = "reserved"  # Pre-paid discount
    SPOT = "spot"  # Auction-based pricing
    FLAT_RATE = "flat_rate"  # Monthly subscription


@dataclass
class ServiceOffering:
    """A service offered in the marketplace"""

    offering_id: str
    provider_id: str  # Fog node or cluster ID
    service_type: ServiceType
    service_tier: ServiceTier
    pricing_model: PricingModel

    # Specifications
    cpu_cores: int | None = None
    memory_gb: float | None = None
    storage_gb: float | None = None
    bandwidth_mbps: float | None = None
    gpu_available: bool = False

    # Pricing (in tokens per hour/GB)
    base_price: Decimal = Decimal("1.0")
    current_price: Decimal = Decimal("1.0")  # Dynamic pricing

    # Availability
    regions: list[str] = field(default_factory=list)
    availability_zones: list[str] = field(default_factory=list)
    capacity_available: int = 100
    capacity_total: int = 100

    # SLA guarantees
    uptime_guarantee: float = 99.0  # Percentage
    latency_guarantee_ms: int | None = None
    throughput_guarantee_mbps: float | None = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Reviews and ratings
    rating: float = 0.0  # 0-5 stars
    review_count: int = 0
    reliability_score: float = 100.0  # Historical reliability


@dataclass
class ServiceRequest:
    """Request for fog computing services"""

    request_id: str
    customer_id: str
    service_type: ServiceType
    service_tier: ServiceTier

    # Requirements
    cpu_cores: int | None = None
    memory_gb: float | None = None
    storage_gb: float | None = None
    bandwidth_mbps: float | None = None
    gpu_required: bool = False

    # Preferences
    preferred_regions: list[str] = field(default_factory=list)
    max_price_per_hour: Decimal | None = None
    duration_hours: int | None = None

    # Special requirements
    hidden_service: bool = False  # Requires .fog address
    encrypted_storage: bool = True
    redundancy_factor: int = 1  # Number of replicas

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "pending"


@dataclass
class ServiceContract:
    """Active service contract between customer and provider"""

    contract_id: str
    request_id: str
    offering_id: str
    customer_id: str
    provider_id: str

    # Service details
    service_type: ServiceType
    service_tier: ServiceTier
    specifications: dict[str, Any]

    # Pricing
    agreed_price: Decimal
    pricing_model: PricingModel

    # Duration
    start_time: datetime
    end_time: datetime | None
    billing_period: str = "hourly"
    auto_renew: bool = False

    # SLA terms
    sla_terms: dict[str, Any] = field(default_factory=dict)
    penalty_rate: Decimal = Decimal("0.1")  # 10% penalty for SLA breach

    # Usage tracking
    usage_hours: float = 0.0
    usage_gb: float = 0.0
    tokens_spent: int = 0
    tokens_refunded: int = 0

    # Performance metrics
    uptime_percent: float = 100.0
    sla_breaches: int = 0

    # Hidden service specific
    onion_address: str | None = None
    introduction_points: list[str] = field(default_factory=list)

    status: str = "active"  # active, paused, terminated


@dataclass
class MarketplaceStats:
    """Marketplace statistics and metrics"""

    total_providers: int = 0
    total_customers: int = 0
    total_offerings: int = 0
    active_contracts: int = 0

    # Resource totals
    total_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    total_storage_gb: float = 0.0
    total_bandwidth_gbps: float = 0.0

    # Usage stats
    compute_hours_served: float = 0.0
    storage_gb_hours_served: float = 0.0
    bandwidth_gb_served: float = 0.0

    # Economics
    total_tokens_transacted: int = 0
    average_price_per_hour: Decimal = Decimal("0")

    # Service distribution
    service_type_distribution: dict[str, int] = field(default_factory=dict)
    region_distribution: dict[str, int] = field(default_factory=dict)


class FogMarketplace:
    """
    Decentralized marketplace for fog computing resources.
    Connects resource providers with consumers using token economics.
    """

    def __init__(
        self,
        marketplace_id: str,
        base_token_rate: int = 100,
        enable_hidden_services: bool = True,
        enable_spot_pricing: bool = True,
    ):
        self.marketplace_id = marketplace_id
        self.base_token_rate = base_token_rate
        self.enable_hidden_services = enable_hidden_services
        self.enable_spot_pricing = enable_spot_pricing

        # Marketplace data
        self.offerings: dict[str, ServiceOffering] = {}
        self.requests: dict[str, ServiceRequest] = {}
        self.contracts: dict[str, ServiceContract] = {}

        # Provider/customer tracking
        self.providers: dict[str, list[str]] = {}  # provider_id -> offering_ids
        self.customers: dict[str, list[str]] = {}  # customer_id -> contract_ids

        # Pricing engine
        self.price_history: dict[ServiceType, list[tuple[datetime, Decimal]]] = {}
        self.demand_metrics: dict[ServiceType, float] = {}

        # Hidden service registry
        self.hidden_services: dict[str, str] = {}  # onion_address -> contract_id

        # Statistics
        self.stats = MarketplaceStats()

        logger.info(f"FogMarketplace initialized: {marketplace_id}")

    async def register_offering(self, provider_id: str, offering: ServiceOffering) -> bool:
        """Register a new service offering from a provider"""

        try:
            # Validate offering
            if not self._validate_offering(offering):
                logger.warning(f"Invalid offering from {provider_id}")
                return False

            # Store offering
            self.offerings[offering.offering_id] = offering

            # Track provider
            if provider_id not in self.providers:
                self.providers[provider_id] = []
            self.providers[provider_id].append(offering.offering_id)

            # Update stats
            self.stats.total_offerings += 1
            if offering.cpu_cores:
                self.stats.total_cpu_cores += offering.cpu_cores * offering.capacity_total
            if offering.memory_gb:
                self.stats.total_memory_gb += offering.memory_gb * offering.capacity_total
            if offering.storage_gb:
                self.stats.total_storage_gb += offering.storage_gb * offering.capacity_total

            # Update pricing based on supply
            await self._update_dynamic_pricing(offering.service_type)

            logger.info(
                f"Registered offering {offering.offering_id}: " f"{offering.service_type.value} from {provider_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register offering: {e}")
            return False

    def _validate_offering(self, offering: ServiceOffering) -> bool:
        """Validate service offering specifications"""

        # Check required fields based on service type
        if offering.service_type in [ServiceType.COMPUTE_INSTANCE, ServiceType.CONTAINER_HOSTING]:
            if not offering.cpu_cores or not offering.memory_gb:
                return False

        if offering.service_type in [ServiceType.OBJECT_STORAGE, ServiceType.BLOCK_STORAGE]:
            if not offering.storage_gb:
                return False

        # Validate SLA guarantees match tier
        tier_uptime = {
            ServiceTier.BASIC: 0.0,
            ServiceTier.STANDARD: 99.0,
            ServiceTier.PREMIUM: 99.9,
            ServiceTier.ENTERPRISE: 99.99,
        }

        if offering.uptime_guarantee < tier_uptime.get(offering.service_tier, 0):
            return False

        return True

    async def submit_request(self, customer_id: str, request: ServiceRequest) -> str | None:
        """Submit a service request and find matching offerings"""

        try:
            # Store request
            self.requests[request.request_id] = request

            # Find matching offerings
            matches = await self._find_matching_offerings(request)

            if not matches:
                logger.warning(f"No matching offerings for request {request.request_id}")
                request.status = "no_matches"
                return None

            # Select best offering based on price and rating
            best_offering = await self._select_best_offering(matches, request)

            if not best_offering:
                request.status = "failed"
                return None

            # Create contract
            contract = await self._create_contract(request, best_offering, customer_id)

            if contract:
                request.status = "fulfilled"
                return contract.contract_id
            else:
                request.status = "failed"
                return None

        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            request.status = "error"
            return None

    async def _find_matching_offerings(self, request: ServiceRequest) -> list[ServiceOffering]:
        """Find offerings that match request requirements"""

        matches = []

        for offering in self.offerings.values():
            # Check service type
            if offering.service_type != request.service_type:
                continue

            # Check tier
            if offering.service_tier.value < request.service_tier.value:
                continue

            # Check specifications
            if request.cpu_cores and offering.cpu_cores:
                if offering.cpu_cores < request.cpu_cores:
                    continue

            if request.memory_gb and offering.memory_gb:
                if offering.memory_gb < request.memory_gb:
                    continue

            if request.storage_gb and offering.storage_gb:
                if offering.storage_gb < request.storage_gb:
                    continue

            if request.gpu_required and not offering.gpu_available:
                continue

            # Check availability
            if offering.capacity_available <= 0:
                continue

            # Check price
            if request.max_price_per_hour:
                if offering.current_price > request.max_price_per_hour:
                    continue

            # Check regions
            if request.preferred_regions:
                if not any(r in offering.regions for r in request.preferred_regions):
                    continue

            matches.append(offering)

        return matches

    async def _select_best_offering(
        self, offerings: list[ServiceOffering], request: ServiceRequest
    ) -> ServiceOffering | None:
        """Select best offering based on price, rating, and reliability"""

        if not offerings:
            return None

        # Score offerings
        scored_offerings = []
        for offering in offerings:
            # Calculate score (0-100)
            price_score = float(
                50
                * (1 - float(offering.current_price) / float(request.max_price_per_hour or offering.current_price * 2))
            )
            rating_score = offering.rating * 10  # 0-50
            reliability_score = offering.reliability_score / 2  # 0-50

            total_score = price_score + rating_score * 0.3 + reliability_score * 0.2

            scored_offerings.append((offering, total_score))

        # Sort by score
        scored_offerings.sort(key=lambda x: x[1], reverse=True)

        return scored_offerings[0][0] if scored_offerings else None

    async def _create_contract(
        self, request: ServiceRequest, offering: ServiceOffering, customer_id: str
    ) -> ServiceContract | None:
        """Create a service contract between customer and provider"""

        try:
            contract_id = str(uuid.uuid4())

            # Prepare specifications
            specifications = {
                "cpu_cores": offering.cpu_cores or request.cpu_cores,
                "memory_gb": offering.memory_gb or request.memory_gb,
                "storage_gb": offering.storage_gb or request.storage_gb,
                "bandwidth_mbps": offering.bandwidth_mbps or request.bandwidth_mbps,
                "gpu_available": offering.gpu_available,
            }

            # Calculate duration and pricing
            duration_hours = request.duration_hours or 1
            total_price = offering.current_price * duration_hours

            # Create contract
            contract = ServiceContract(
                contract_id=contract_id,
                request_id=request.request_id,
                offering_id=offering.offering_id,
                customer_id=customer_id,
                provider_id=offering.provider_id,
                service_type=offering.service_type,
                service_tier=offering.service_tier,
                specifications=specifications,
                agreed_price=offering.current_price,
                pricing_model=offering.pricing_model,
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC) + timedelta(hours=duration_hours) if duration_hours else None,
                sla_terms={
                    "uptime_guarantee": offering.uptime_guarantee,
                    "latency_guarantee_ms": offering.latency_guarantee_ms,
                    "throughput_guarantee_mbps": offering.throughput_guarantee_mbps,
                },
            )

            # Handle hidden service requests
            if request.hidden_service and self.enable_hidden_services:
                # Generate .fog address
                onion_address = self._generate_onion_address(contract_id)
                contract.onion_address = onion_address
                self.hidden_services[onion_address] = contract_id

            # Store contract
            self.contracts[contract_id] = contract

            # Update customer tracking
            if customer_id not in self.customers:
                self.customers[customer_id] = []
            self.customers[customer_id].append(contract_id)

            # Update offering capacity
            offering.capacity_available -= 1

            # Update stats
            self.stats.active_contracts += 1

            logger.info(
                f"Created contract {contract_id}: "
                f"{customer_id} -> {offering.provider_id}, "
                f"price: {total_price} tokens"
            )

            return contract

        except Exception as e:
            logger.error(f"Failed to create contract: {e}")
            return None

    def _generate_onion_address(self, contract_id: str) -> str:
        """Generate a .fog onion address for hidden service"""
        # Simplified generation - in production would use proper crypto
        hash_bytes = hashlib.sha256(contract_id.encode()).digest()[:10]
        address = base64.b32encode(hash_bytes).decode().lower().rstrip("=")
        return f"{address}.fog"

    async def _update_dynamic_pricing(self, service_type: ServiceType):
        """Update dynamic pricing based on supply and demand"""

        if not self.enable_spot_pricing:
            return

        # Count supply and demand
        supply = sum(1 for o in self.offerings.values() if o.service_type == service_type and o.capacity_available > 0)

        demand = sum(1 for r in self.requests.values() if r.service_type == service_type and r.status == "pending")

        # Calculate demand ratio
        demand_ratio = demand / max(supply, 1)
        self.demand_metrics[service_type] = demand_ratio

        # Adjust prices based on demand
        for offering in self.offerings.values():
            if offering.service_type != service_type:
                continue

            if offering.pricing_model != PricingModel.SPOT:
                continue

            # Price adjustment factor (0.5x to 2x)
            adjustment = min(2.0, max(0.5, demand_ratio))
            offering.current_price = offering.base_price * Decimal(str(adjustment))
            offering.last_updated = datetime.now(UTC)

        # Record price history
        if service_type not in self.price_history:
            self.price_history[service_type] = []

        avg_price = Decimal("0")
        count = 0
        for o in self.offerings.values():
            if o.service_type == service_type:
                avg_price += o.current_price
                count += 1

        if count > 0:
            avg_price /= count
            self.price_history[service_type].append((datetime.now(UTC), avg_price))

    async def process_usage(self, contract_id: str, usage_metrics: dict[str, Any]) -> bool:
        """Process usage metrics and update billing"""

        if contract_id not in self.contracts:
            logger.warning(f"Unknown contract: {contract_id}")
            return False

        contract = self.contracts[contract_id]

        # Update usage
        contract.usage_hours += usage_metrics.get("hours", 0)
        contract.usage_gb += usage_metrics.get("gb", 0)

        # Calculate tokens
        hours_cost = int(contract.agreed_price * Decimal(str(contract.usage_hours)))
        contract.tokens_spent = hours_cost

        # Check SLA compliance
        uptime = usage_metrics.get("uptime_percent", 100)
        if uptime < contract.sla_terms.get("uptime_guarantee", 99):
            contract.sla_breaches += 1
            # Apply penalty
            penalty = int(contract.tokens_spent * contract.penalty_rate)
            contract.tokens_refunded += penalty
            logger.info(f"SLA breach on contract {contract_id}: {penalty} tokens refunded")

        contract.uptime_percent = uptime

        return True

    async def terminate_contract(self, contract_id: str, reason: str = "completed") -> bool:
        """Terminate a service contract"""

        if contract_id not in self.contracts:
            return False

        contract = self.contracts[contract_id]
        contract.status = "terminated"
        contract.end_time = datetime.now(UTC)

        # Free up capacity
        if contract.offering_id in self.offerings:
            self.offerings[contract.offering_id].capacity_available += 1

        # Update stats
        self.stats.active_contracts -= 1
        self.stats.compute_hours_served += contract.usage_hours
        self.stats.total_tokens_transacted += contract.tokens_spent

        # Clean up hidden service
        if contract.onion_address:
            del self.hidden_services[contract.onion_address]

        logger.info(
            f"Terminated contract {contract_id}: "
            f"{contract.usage_hours:.2f} hours, "
            f"{contract.tokens_spent} tokens"
        )

        return True

    def get_market_stats(self) -> MarketplaceStats:
        """Get current marketplace statistics"""

        # Update provider count
        self.stats.total_providers = len(self.providers)
        self.stats.total_customers = len(self.customers)

        # Service type distribution
        for offering in self.offerings.values():
            service_type = offering.service_type.value
            if service_type not in self.stats.service_type_distribution:
                self.stats.service_type_distribution[service_type] = 0
            self.stats.service_type_distribution[service_type] += 1

        # Calculate average price
        total_price = Decimal("0")
        count = 0
        for offering in self.offerings.values():
            total_price += offering.current_price
            count += 1

        if count > 0:
            self.stats.average_price_per_hour = total_price / count

        return self.stats

    def search_offerings(
        self,
        service_type: ServiceType | None = None,
        max_price: Decimal | None = None,
        min_rating: float | None = None,
        region: str | None = None,
    ) -> list[ServiceOffering]:
        """Search for service offerings with filters"""

        results = []

        for offering in self.offerings.values():
            # Apply filters
            if service_type and offering.service_type != service_type:
                continue

            if max_price and offering.current_price > max_price:
                continue

            if min_rating and offering.rating < min_rating:
                continue

            if region and region not in offering.regions:
                continue

            results.append(offering)

        # Sort by rating and price
        results.sort(key=lambda x: (x.rating, -float(x.current_price)), reverse=True)

        return results
