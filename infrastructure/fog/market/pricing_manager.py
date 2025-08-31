"""
Dynamic Pricing Manager for Fog Computing Market

Implements sophisticated pricing strategies with anti-manipulation safeguards:
- Dynamic pricing bands based on supply/demand
- Resource lane pricing (CPU, Memory, Storage, Bandwidth)
- Market manipulation detection and prevention
- Price volatility management
- Multi-tier pricing strategies

Key Features:
- Real-time price discovery with circuit breakers
- Lane-specific pricing for different resource types
- Anti-manipulation algorithms with anomaly detection
- Dynamic reserve price calculation
- Historical pricing analytics and forecasting
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import logging
import statistics
from typing import Any
import uuid

# Import reputation system for pricing integration
from ..reputation import BayesianReputationEngine, integrate_with_pricing

# Import constitutional pricing components
from .constitutional_pricing import ConstitutionalPricingEngine
from .audit_pricing import AuditTrailManager

# Set high precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class ResourceLane(str, Enum):
    """Resource lanes for differentiated pricing"""

    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    GPU = "gpu"
    SPECIALIZED = "specialized"
    FEDERATED_INFERENCE = "federated_inference"
    FEDERATED_TRAINING = "federated_training"
    TEE_SECURE = "tee_secure"  # TEE-enhanced workloads
    CONSTITUTIONAL = "constitutional"  # Constitutional compliance pricing


class PricingStrategy(str, Enum):
    """Pricing strategy types"""

    DYNAMIC = "dynamic"  # Supply/demand based
    STABLE = "stable"  # Stability-focused
    AGGRESSIVE = "aggressive"  # Rapid price discovery
    CONSERVATIVE = "conservative"  # Slow price changes
    CIRCUIT_BREAKER = "circuit_breaker"  # Emergency mode


class MarketCondition(str, Enum):
    """Market condition indicators"""

    NORMAL = "normal"
    HIGH_DEMAND = "high_demand"
    LOW_DEMAND = "low_demand"
    VOLATILE = "volatile"
    MANIPULATION = "manipulation_detected"
    EMERGENCY = "emergency"


class UserSizeTier(str, Enum):
    """User size tiers for different pricing models"""
    
    BRONZE = "bronze"    # Mobile-first, basic users
    SILVER = "silver"    # Hybrid cloud-edge users  
    GOLD = "gold"        # Cloud-heavy, premium users
    PLATINUM = "platinum" # Dedicated enterprise users
    
    # Legacy compatibility
    SMALL = "bronze"     # Alias for bronze
    MEDIUM = "silver"    # Alias for silver
    LARGE = "gold"       # Alias for gold
    ENTERPRISE = "platinum"  # Alias for platinum


@dataclass
class PriceBand:
    """Price band for a resource lane"""

    lane: ResourceLane

    # Price boundaries
    floor_price: Decimal  # Absolute minimum price
    ceiling_price: Decimal  # Absolute maximum price
    current_price: Decimal  # Current market price
    target_price: Decimal  # Algorithm target price

    # Price movement constraints
    max_hourly_change: Decimal = Decimal("0.1")  # 10% max change per hour
    max_daily_change: Decimal = Decimal("0.5")  # 50% max change per day
    volatility_threshold: Decimal = Decimal("0.3")  # 30% volatility triggers alerts

    # Market metrics
    supply_units: Decimal = Decimal("0")
    demand_units: Decimal = Decimal("0")
    utilization_rate: Decimal = Decimal("0")

    # Historical tracking
    price_history: list[tuple[datetime, Decimal]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Ensure all prices are Decimal"""
        for attr in ["floor_price", "ceiling_price", "current_price", "target_price"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    def calculate_volatility(self, hours: int = 24) -> Decimal:
        """Calculate price volatility over specified hours"""

        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        recent_prices = [price for timestamp, price in self.price_history if timestamp > cutoff]

        if len(recent_prices) < 2:
            return Decimal("0")

        prices_float = [float(p) for p in recent_prices]
        mean_price = statistics.mean(prices_float)

        if mean_price == 0:
            return Decimal("0")

        variance = statistics.variance(prices_float)
        volatility = Decimal(str((variance**0.5) / mean_price))

        return volatility

    def update_price(self, new_price: Decimal, enforce_limits: bool = True) -> bool:
        """Update price with safety constraints"""

        if enforce_limits:
            # Check floor/ceiling constraints
            if new_price < self.floor_price:
                new_price = self.floor_price
            elif new_price > self.ceiling_price:
                new_price = self.ceiling_price

            # Check hourly change limit
            if self.price_history:
                last_hour_cutoff = datetime.now(UTC) - timedelta(hours=1)
                recent_prices = [price for timestamp, price in self.price_history if timestamp > last_hour_cutoff]

                if recent_prices:
                    max_allowed_price = recent_prices[-1] * (Decimal("1") + self.max_hourly_change)
                    min_allowed_price = recent_prices[-1] * (Decimal("1") - self.max_hourly_change)

                    new_price = max(min_allowed_price, min(max_allowed_price, new_price))

        # Store old price and update
        old_price = self.current_price
        self.current_price = new_price
        self.last_updated = datetime.now(UTC)

        # Add to history
        self.price_history.append((self.last_updated, new_price))

        # Keep only last 7 days of history
        cutoff = datetime.now(UTC) - timedelta(days=7)
        self.price_history = [(timestamp, price) for timestamp, price in self.price_history if timestamp > cutoff]

        price_change = abs((new_price - old_price) / old_price) if old_price > 0 else Decimal("0")

        logger.debug(
            f"{self.lane.value} price updated: "
            f"${float(old_price):.6f} -> ${float(new_price):.6f} "
            f"({float(price_change * 100):.2f}% change)"
        )

        return True

    def is_volatile(self) -> bool:
        """Check if price is volatile"""
        return self.calculate_volatility() > self.volatility_threshold

    def get_supply_demand_ratio(self) -> Decimal:
        """Get supply/demand ratio"""
        if self.demand_units == 0:
            return Decimal("999")  # Very high supply relative to demand
        return self.supply_units / self.demand_units


@dataclass
class PricingAnomalyAlert:
    """Alert for pricing anomalies and potential manipulation"""

    alert_id: str
    lane: ResourceLane
    anomaly_type: str  # "volatility", "manipulation", "circuit_breaker", "demand_spike"

    severity: str = "medium"  # "low", "medium", "high", "critical"
    description: str = ""

    # Data points
    current_price: Decimal = Decimal("0")
    expected_price: Decimal = Decimal("0")
    deviation_percentage: Decimal = Decimal("0")

    # Context
    market_condition: MarketCondition = MarketCondition.NORMAL
    suggested_action: str = ""

    # Timing
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None

    def calculate_severity(self):
        """Calculate alert severity based on deviation"""
        deviation = abs(self.deviation_percentage)

        if deviation > Decimal("0.5"):  # >50% deviation
            self.severity = "critical"
        elif deviation > Decimal("0.3"):  # >30% deviation
            self.severity = "high"
        elif deviation > Decimal("0.15"):  # >15% deviation
            self.severity = "medium"
        else:
            self.severity = "low"


@dataclass 
class SizeTierPricing:
    """Size-tier specific pricing configuration"""
    
    tier: UserSizeTier
    
    # Inference pricing (per request)
    inference_price_min: Decimal
    inference_price_max: Decimal
    inference_price_base: Decimal
    
    # Training pricing (per hour)
    training_price_min: Decimal
    training_price_max: Decimal  
    training_price_base: Decimal
    
    # Resource limits and guarantees
    max_concurrent_jobs: int = 10
    guaranteed_uptime_percentage: Decimal = Decimal("95")
    max_latency_sla_ms: Decimal = Decimal("500")
    
    # Volume discounts
    volume_discount_threshold: int = 100  # requests/month
    volume_discount_percentage: Decimal = Decimal("0.1")  # 10%
    
    # Priority and QoS
    priority_multiplier: Decimal = Decimal("1.0")
    dedicated_support: bool = False
    custom_pricing: bool = False


@dataclass
class H200HourPricing:
    """H200-hour equivalent pricing calculation"""
    
    device_computing_power: Decimal  # TOPS_d - device computing power
    utilization_rate: Decimal        # u - utilization rate (0-1)
    time_hours: Decimal              # t - time in hours
    h200_reference_tops: Decimal = Decimal("989")  # H200 reference TOPS
    
    def calculate_h200_hours(self) -> Decimal:
        """Calculate H200-hour equivalent: H200h(d) = (TOPS_d × u × t) / T_ref"""
        return (self.device_computing_power * self.utilization_rate * self.time_hours) / self.h200_reference_tops
    
    def calculate_constitutional_multiplier(self, transparency_level: str) -> Decimal:
        """Calculate constitutional compliance multiplier"""
        multipliers = {
            "basic": Decimal("1.0"),
            "enhanced": Decimal("1.1"),
            "full_audit": Decimal("1.25"),
            "constitutional": Decimal("1.5")
        }
        return multipliers.get(transparency_level, Decimal("1.0"))
    
    def calculate_tee_premium(self, tee_enabled: bool) -> Decimal:
        """Calculate TEE security premium"""
        return Decimal("1.3") if tee_enabled else Decimal("1.0")


@dataclass
class ConstitutionalTierMapping:
    """Mapping between user tiers and constitutional pricing"""
    
    tier: UserSizeTier
    h200_hour_base_rate: Decimal     # Base H200-hour rate for tier
    constitutional_discount: Decimal  # Discount for constitutional compliance
    audit_transparency_bonus: Decimal # Bonus for audit transparency
    max_h200_hours_per_month: Decimal # Monthly H200-hour limit
    
    def calculate_tier_price(self, h200_hours: Decimal, constitutional_level: str = "basic") -> Decimal:
        """Calculate final price for tier with constitutional adjustments"""
        base_cost = h200_hours * self.h200_hour_base_rate
        
        # Apply constitutional discount
        if constitutional_level != "basic":
            base_cost *= (Decimal("1.0") - self.constitutional_discount)
            
        # Apply transparency bonus
        if constitutional_level in ["full_audit", "constitutional"]:
            base_cost *= (Decimal("1.0") - self.audit_transparency_bonus)
            
        return base_cost


@dataclass
class MarketMetrics:
    """Comprehensive market metrics"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Overall market health
    total_supply: Decimal = Decimal("0")
    total_demand: Decimal = Decimal("0")
    overall_utilization: Decimal = Decimal("0")
    market_cap: Decimal = Decimal("0")  # Total value of all resources

    # Price metrics
    average_price: Decimal = Decimal("0")
    price_volatility: Decimal = Decimal("0")
    volume_weighted_average_price: Decimal = Decimal("0")

    # Lane-specific metrics
    lane_metrics: dict[ResourceLane, dict[str, Decimal]] = field(default_factory=dict)

    # Size tier metrics
    tier_metrics: dict[UserSizeTier, dict[str, Decimal]] = field(default_factory=dict)

    # Manipulation detection
    manipulation_score: Decimal = Decimal("0")  # 0-1 score
    suspicious_activity_count: int = 0

    # Market condition
    condition: MarketCondition = MarketCondition.NORMAL

    def calculate_market_health_score(self) -> Decimal:
        """Calculate overall market health score (0-1)"""

        # Factors contributing to market health
        liquidity_score = min(Decimal("1"), self.total_supply / max(Decimal("1"), self.total_demand))
        stability_score = max(Decimal("0"), Decimal("1") - self.price_volatility)
        manipulation_score = max(Decimal("0"), Decimal("1") - self.manipulation_score)

        # Weighted average
        health_score = (
            liquidity_score * Decimal("0.4") + stability_score * Decimal("0.4") + manipulation_score * Decimal("0.2")
        )

        return health_score


class DynamicPricingManager:
    """
    Advanced dynamic pricing manager for fog computing resources

    Features:
    - Lane-specific pricing for different resource types
    - Real-time supply/demand analysis
    - Anti-manipulation algorithms
    - Circuit breaker protection
    - Historical analytics and forecasting
    """

    def __init__(self, token_system=None, auction_engine=None, reputation_engine=None, 
                 constitutional_engine=None, audit_manager=None):
        self.token_system = token_system
        self.auction_engine = auction_engine
        
        # Reputation system integration
        self.reputation_engine = reputation_engine or BayesianReputationEngine()
        
        # Constitutional pricing integration
        self.constitutional_engine = constitutional_engine
        self.audit_manager = audit_manager or AuditTrailManager()
        
        # H200-hour reference specs for constitutional pricing
        self.h200_reference_specs = {
            "tops": Decimal("989"),       # H200 TOPS performance
            "memory_gb": Decimal("141"),  # H200 memory capacity 
            "power_watts": Decimal("700") # H200 power consumption
        }

        # Price bands for each resource lane
        self.price_bands: dict[ResourceLane, PriceBand] = {}
        self._initialize_price_bands()

        # Size-tier pricing configurations
        self.tier_pricing: dict[UserSizeTier, SizeTierPricing] = {}
        self._initialize_tier_pricing()
        
        # Constitutional tier mappings
        self.constitutional_tiers: dict[UserSizeTier, ConstitutionalTierMapping] = {}
        self._initialize_constitutional_tier_mappings()

        # Market state
        self.current_strategy = PricingStrategy.DYNAMIC
        self.market_metrics = MarketMetrics()
        self.anomaly_alerts: list[PricingAnomalyAlert] = []

        # Anti-manipulation
        self.manipulation_thresholds = {
            "volume_spike_threshold": Decimal("3.0"),  # 3x normal volume
            "price_spike_threshold": Decimal("0.25"),  # 25% rapid price change
            "coordination_window": timedelta(minutes=5),  # Time window for detecting coordination
            "min_trades_for_detection": 5,
        }

        # Configuration
        self.config = {
            "price_update_interval": 30,  # seconds
            "volatility_smoothing_factor": Decimal("0.8"),
            "demand_elasticity": Decimal("1.5"),
            "supply_elasticity": Decimal("1.2"),
            "emergency_circuit_breaker": True,
            "max_price_deviation": Decimal("0.5"),  # 50% max deviation from target
        }

        # Background tasks
        self._pricing_update_task: asyncio.Task | None = None
        self._anomaly_detection_task: asyncio.Task | None = None

        logger.info("Dynamic pricing manager initialized with constitutional features")

    async def start(self):
        """Start pricing manager background tasks"""

        self._pricing_update_task = asyncio.create_task(self._pricing_update_loop())
        self._anomaly_detection_task = asyncio.create_task(self._anomaly_detection_loop())

        logger.info("Pricing manager started")

    async def stop(self):
        """Stop pricing manager background tasks"""

        if self._pricing_update_task:
            self._pricing_update_task.cancel()
        if self._anomaly_detection_task:
            self._anomaly_detection_task.cancel()

        logger.info("Pricing manager stopped")

    def _initialize_price_bands(self):
        """Initialize price bands with default values"""

        default_bands = {
            ResourceLane.CPU: {
                "floor_price": Decimal("0.05"),
                "ceiling_price": Decimal("2.00"),
                "current_price": Decimal("0.15"),
                "target_price": Decimal("0.15"),
            },
            ResourceLane.MEMORY: {
                "floor_price": Decimal("0.01"),
                "ceiling_price": Decimal("0.50"),
                "current_price": Decimal("0.05"),
                "target_price": Decimal("0.05"),
            },
            ResourceLane.STORAGE: {
                "floor_price": Decimal("0.001"),
                "ceiling_price": Decimal("0.10"),
                "current_price": Decimal("0.01"),
                "target_price": Decimal("0.01"),
            },
            ResourceLane.BANDWIDTH: {
                "floor_price": Decimal("0.02"),
                "ceiling_price": Decimal("1.00"),
                "current_price": Decimal("0.10"),
                "target_price": Decimal("0.10"),
            },
            ResourceLane.GPU: {
                "floor_price": Decimal("0.50"),
                "ceiling_price": Decimal("10.00"),
                "current_price": Decimal("2.00"),
                "target_price": Decimal("2.00"),
            },
            ResourceLane.SPECIALIZED: {
                "floor_price": Decimal("0.20"),
                "ceiling_price": Decimal("5.00"),
                "current_price": Decimal("1.00"),
                "target_price": Decimal("1.00"),
            },
            ResourceLane.FEDERATED_INFERENCE: {
                "floor_price": Decimal("0.01"),
                "ceiling_price": Decimal("10.00"),
                "current_price": Decimal("0.50"),
                "target_price": Decimal("0.50"),
            },
            ResourceLane.FEDERATED_TRAINING: {
                "floor_price": Decimal("1.00"),
                "ceiling_price": Decimal("1000.00"),
                "current_price": Decimal("50.00"),
                "target_price": Decimal("50.00"),
            },
            ResourceLane.TEE_SECURE: {
                "floor_price": Decimal("0.75"),
                "ceiling_price": Decimal("15.00"),
                "current_price": Decimal("3.00"),
                "target_price": Decimal("3.00"),
            },
            ResourceLane.CONSTITUTIONAL: {
                "floor_price": Decimal("0.25"),
                "ceiling_price": Decimal("5.00"),
                "current_price": Decimal("1.50"),
                "target_price": Decimal("1.50"),
            },
        }

        for lane, params in default_bands.items():
            self.price_bands[lane] = PriceBand(lane=lane, **params)

    def _initialize_tier_pricing(self):
        """Initialize size-tier pricing configurations"""
        
        # Small tier: Mobile-first, budget-conscious
        self.tier_pricing[UserSizeTier.SMALL] = SizeTierPricing(
            tier=UserSizeTier.SMALL,
            inference_price_min=Decimal("0.01"),
            inference_price_max=Decimal("0.10"),
            inference_price_base=Decimal("0.05"),
            training_price_min=Decimal("1.00"),
            training_price_max=Decimal("10.00"),
            training_price_base=Decimal("5.00"),
            max_concurrent_jobs=5,
            guaranteed_uptime_percentage=Decimal("95"),
            max_latency_sla_ms=Decimal("1000"),
            volume_discount_threshold=50,
            volume_discount_percentage=Decimal("0.05"),
            priority_multiplier=Decimal("0.8"),
        )
        
        # Medium tier: Hybrid cloud-edge users
        self.tier_pricing[UserSizeTier.MEDIUM] = SizeTierPricing(
            tier=UserSizeTier.MEDIUM,
            inference_price_min=Decimal("0.10"),
            inference_price_max=Decimal("1.00"),
            inference_price_base=Decimal("0.50"),
            training_price_min=Decimal("10.00"),
            training_price_max=Decimal("100.00"),
            training_price_base=Decimal("50.00"),
            max_concurrent_jobs=20,
            guaranteed_uptime_percentage=Decimal("98"),
            max_latency_sla_ms=Decimal("500"),
            volume_discount_threshold=100,
            volume_discount_percentage=Decimal("0.10"),
            priority_multiplier=Decimal("1.0"),
        )
        
        # Large tier: Cloud-heavy users
        self.tier_pricing[UserSizeTier.LARGE] = SizeTierPricing(
            tier=UserSizeTier.LARGE,
            inference_price_min=Decimal("1.00"),
            inference_price_max=Decimal("10.00"),
            inference_price_base=Decimal("5.00"),
            training_price_min=Decimal("100.00"),
            training_price_max=Decimal("1000.00"),
            training_price_base=Decimal("500.00"),
            max_concurrent_jobs=50,
            guaranteed_uptime_percentage=Decimal("99"),
            max_latency_sla_ms=Decimal("200"),
            volume_discount_threshold=500,
            volume_discount_percentage=Decimal("0.15"),
            priority_multiplier=Decimal("1.2"),
        )
        
        # Enterprise tier: Dedicated enterprise with custom SLAs
        self.tier_pricing[UserSizeTier.ENTERPRISE] = SizeTierPricing(
            tier=UserSizeTier.ENTERPRISE,
            inference_price_min=Decimal("10.00"),
            inference_price_max=Decimal("100.00"),
            inference_price_base=Decimal("25.00"),
            training_price_min=Decimal("1000.00"),
            training_price_max=Decimal("10000.00"),
            training_price_base=Decimal("2500.00"),
            max_concurrent_jobs=200,
            guaranteed_uptime_percentage=Decimal("99.9"),
            max_latency_sla_ms=Decimal("50"),
            volume_discount_threshold=1000,
            volume_discount_percentage=Decimal("0.20"),
            priority_multiplier=Decimal("1.5"),
            dedicated_support=True,
            custom_pricing=True,
        )
        
        # Platinum tier: Constitutional enterprise tier
        self.tier_pricing[UserSizeTier.PLATINUM] = SizeTierPricing(
            tier=UserSizeTier.PLATINUM,
            inference_price_min=Decimal("15.00"),
            inference_price_max=Decimal("150.00"),
            inference_price_base=Decimal("35.00"),
            training_price_min=Decimal("1500.00"),
            training_price_max=Decimal("15000.00"),
            training_price_base=Decimal("3500.00"),
            max_concurrent_jobs=500,
            guaranteed_uptime_percentage=Decimal("99.95"),
            max_latency_sla_ms=Decimal("25"),
            volume_discount_threshold=2000,
            volume_discount_percentage=Decimal("0.25"),
            priority_multiplier=Decimal("2.0"),
            dedicated_support=True,
            custom_pricing=True,
        )
        
    def _initialize_constitutional_tier_mappings(self):
        """Initialize constitutional tier mappings with H200-hour rates"""
        
        # Bronze Tier - Constitutional democrats
        self.constitutional_tiers[UserSizeTier.BRONZE] = ConstitutionalTierMapping(
            tier=UserSizeTier.BRONZE,
            h200_hour_base_rate=Decimal("0.50"),      # $0.50 per H200-hour
            constitutional_discount=Decimal("0.05"),   # 5% constitutional compliance discount
            audit_transparency_bonus=Decimal("0.03"),  # 3% audit transparency bonus
            max_h200_hours_per_month=Decimal("100")    # 100 H200-hours monthly limit
        )
        
        # Silver Tier - Constitutional republicans  
        self.constitutional_tiers[UserSizeTier.SILVER] = ConstitutionalTierMapping(
            tier=UserSizeTier.SILVER,
            h200_hour_base_rate=Decimal("0.75"),      # $0.75 per H200-hour
            constitutional_discount=Decimal("0.08"),   # 8% constitutional compliance discount
            audit_transparency_bonus=Decimal("0.05"),  # 5% audit transparency bonus
            max_h200_hours_per_month=Decimal("500")    # 500 H200-hours monthly limit
        )
        
        # Gold Tier - Constitutional libertarians
        self.constitutional_tiers[UserSizeTier.GOLD] = ConstitutionalTierMapping(
            tier=UserSizeTier.GOLD,
            h200_hour_base_rate=Decimal("1.00"),      # $1.00 per H200-hour
            constitutional_discount=Decimal("0.10"),   # 10% constitutional compliance discount
            audit_transparency_bonus=Decimal("0.08"),  # 8% audit transparency bonus
            max_h200_hours_per_month=Decimal("2000")   # 2000 H200-hours monthly limit
        )
        
        # Platinum Tier - Constitutional enterprise
        self.constitutional_tiers[UserSizeTier.PLATINUM] = ConstitutionalTierMapping(
            tier=UserSizeTier.PLATINUM,
            h200_hour_base_rate=Decimal("1.50"),      # $1.50 per H200-hour
            constitutional_discount=Decimal("0.15"),   # 15% constitutional compliance discount
            audit_transparency_bonus=Decimal("0.12"),  # 12% audit transparency bonus
            max_h200_hours_per_month=Decimal("10000")  # 10000 H200-hours monthly limit
        )

    async def update_market_conditions(
        self,
        supply_data: dict[ResourceLane, Decimal],
        demand_data: dict[ResourceLane, Decimal],
        utilization_data: dict[ResourceLane, Decimal],
    ):
        """Update market conditions and trigger price updates"""

        total_supply = sum(supply_data.values())
        total_demand = sum(demand_data.values())

        # Update price bands with new market data
        for lane in ResourceLane:
            if lane in self.price_bands:
                band = self.price_bands[lane]
                band.supply_units = supply_data.get(lane, Decimal("0"))
                band.demand_units = demand_data.get(lane, Decimal("0"))
                band.utilization_rate = utilization_data.get(lane, Decimal("0"))

        # Update market metrics
        self.market_metrics.total_supply = total_supply
        self.market_metrics.total_demand = total_demand
        self.market_metrics.overall_utilization = (
            sum(utilization_data.values()) / len(utilization_data) if utilization_data else Decimal("0")
        )

        # Calculate new target prices
        await self._calculate_target_prices()

        # Update current prices based on strategy
        await self._update_prices_by_strategy()

        logger.info(
            f"Market conditions updated: "
            f"supply={float(total_supply):.2f}, "
            f"demand={float(total_demand):.2f}, "
            f"utilization={float(self.market_metrics.overall_utilization):.2f}"
        )

    async def get_resource_price(
        self, lane: ResourceLane, quantity: Decimal = Decimal("1"), duration_hours: Decimal = Decimal("1"),
        node_id: str = None
    ) -> dict[str, Any]:
        """Get current price for specific resource, optionally for a specific node"""

        if lane not in self.price_bands:
            raise ValueError(f"Unknown resource lane: {lane}")

        band = self.price_bands[lane]

        # Base price calculation
        base_cost = band.current_price * quantity * duration_hours

        # Volume discount/premium
        volume_multiplier = self._calculate_volume_multiplier(quantity)

        # Duration discount/premium
        duration_multiplier = self._calculate_duration_multiplier(duration_hours)

        # Supply/demand adjustment
        supply_demand_multiplier = self._calculate_supply_demand_multiplier(band)
        
        # Reputation-based pricing adjustment
        reputation_multiplier = Decimal("1.0")
        trust_score = 0.5  # Default neutral trust
        
        if node_id and self.reputation_engine:
            trust_score = self.reputation_engine.get_trust_score(node_id)
            # Higher trust nodes get price premium (up to 30% bonus)
            reputation_multiplier = Decimal("1.0") + (Decimal(str(trust_score)) - Decimal("0.5")) * Decimal("0.6")
            reputation_multiplier = max(Decimal("0.7"), min(Decimal("1.3"), reputation_multiplier))

        # Final price calculation
        final_price = base_cost * volume_multiplier * duration_multiplier * supply_demand_multiplier * reputation_multiplier

        return {
            "lane": lane.value,
            "base_price_per_unit_per_hour": float(band.current_price),
            "quantity": float(quantity),
            "duration_hours": float(duration_hours),
            "base_cost": float(base_cost),
            "volume_multiplier": float(volume_multiplier),
            "duration_multiplier": float(duration_multiplier),
            "supply_demand_multiplier": float(supply_demand_multiplier),
            "reputation_multiplier": float(reputation_multiplier),
            "trust_score": trust_score,
            "final_price": float(final_price),
            "currency": "USD",
            "price_band_info": {
                "floor_price": float(band.floor_price),
                "ceiling_price": float(band.ceiling_price),
                "volatility": float(band.calculate_volatility()),
                "utilization_rate": float(band.utilization_rate),
                "supply_demand_ratio": float(band.get_supply_demand_ratio()),
            },
        }

    async def get_federated_inference_price(
        self, 
        user_tier: UserSizeTier,
        model_size: str,
        requests_count: int = 1,
        participants_needed: int = 1,
        privacy_level: str = "medium",
        node_id: str = None
    ) -> dict[str, Any]:
        """Get price for federated inference based on user tier and requirements"""
        
        if user_tier not in self.tier_pricing:
            raise ValueError(f"Unknown user tier: {user_tier}")
        
        tier_config = self.tier_pricing[user_tier]
        
        # Get base price from tier configuration
        base_price_per_request = tier_config.inference_price_base
        
        # Model size multiplier
        model_multipliers = {
            "small": Decimal("0.5"),
            "medium": Decimal("1.0"),
            "large": Decimal("2.0"), 
            "xlarge": Decimal("4.0"),
        }
        model_multiplier = model_multipliers.get(model_size, Decimal("1.0"))
        
        # Privacy level multiplier
        privacy_multipliers = {
            "low": Decimal("1.0"),
            "medium": Decimal("1.2"),
            "high": Decimal("1.5"),
            "critical": Decimal("2.0"),
        }
        privacy_multiplier = privacy_multipliers.get(privacy_level, Decimal("1.2"))
        
        # Participants multiplier (coordination complexity)
        participants_multiplier = Decimal("1.0") + (Decimal(str(participants_needed - 1)) * Decimal("0.1"))
        
        # Volume discount
        volume_discount = Decimal("1.0")
        if requests_count >= tier_config.volume_discount_threshold:
            volume_discount = Decimal("1.0") - tier_config.volume_discount_percentage
            
        # Priority multiplier from tier
        priority_multiplier = tier_config.priority_multiplier
        
        # Market dynamics (use federated inference lane)
        if ResourceLane.FEDERATED_INFERENCE in self.price_bands:
            inference_band = self.price_bands[ResourceLane.FEDERATED_INFERENCE]
            market_multiplier = self._calculate_supply_demand_multiplier(inference_band)
        else:
            market_multiplier = Decimal("1.0")
            
        # Reputation adjustment
        reputation_multiplier = Decimal("1.0")
        trust_score = 0.5
        if node_id and self.reputation_engine:
            trust_score = self.reputation_engine.get_trust_score(node_id)
            reputation_multiplier = Decimal("1.0") + (Decimal(str(trust_score)) - Decimal("0.5")) * Decimal("0.3")
            
        # Calculate final price per request
        price_per_request = (
            base_price_per_request
            * model_multiplier
            * privacy_multiplier
            * participants_multiplier
            * volume_discount
            * priority_multiplier
            * market_multiplier
            * reputation_multiplier
        )
        
        # Ensure within tier bounds
        price_per_request = max(
            tier_config.inference_price_min,
            min(tier_config.inference_price_max, price_per_request)
        )
        
        total_cost = price_per_request * Decimal(str(requests_count))
        
        return {
            "workload_type": "federated_inference",
            "user_tier": user_tier.value,
            "model_size": model_size,
            "requests_count": requests_count,
            "participants_needed": participants_needed,
            "privacy_level": privacy_level,
            "price_per_request": float(price_per_request),
            "total_cost": float(total_cost),
            "currency": "USD",
            "pricing_breakdown": {
                "base_price": float(base_price_per_request),
                "model_multiplier": float(model_multiplier),
                "privacy_multiplier": float(privacy_multiplier),
                "participants_multiplier": float(participants_multiplier),
                "volume_discount": float(volume_discount),
                "priority_multiplier": float(priority_multiplier),
                "market_multiplier": float(market_multiplier),
                "reputation_multiplier": float(reputation_multiplier),
                "trust_score": trust_score,
            },
            "tier_info": {
                "max_concurrent_jobs": tier_config.max_concurrent_jobs,
                "guaranteed_uptime": float(tier_config.guaranteed_uptime_percentage),
                "max_latency_sla_ms": float(tier_config.max_latency_sla_ms),
                "dedicated_support": tier_config.dedicated_support,
            },
        }

    async def get_federated_training_price(
        self,
        user_tier: UserSizeTier,
        model_size: str,
        duration_hours: float,
        participants_needed: int,
        privacy_level: str = "high",
        reliability_requirement: str = "high",
        node_id: str = None
    ) -> dict[str, Any]:
        """Get price for federated training based on user tier and requirements"""
        
        if user_tier not in self.tier_pricing:
            raise ValueError(f"Unknown user tier: {user_tier}")
            
        tier_config = self.tier_pricing[user_tier]
        
        # Get base price from tier configuration
        base_price_per_hour = tier_config.training_price_base
        
        # Model size multiplier (training is more resource intensive)
        model_multipliers = {
            "small": Decimal("1.0"),
            "medium": Decimal("2.0"),
            "large": Decimal("4.0"),
            "xlarge": Decimal("8.0"),
        }
        model_multiplier = model_multipliers.get(model_size, Decimal("2.0"))
        
        # Privacy level multiplier (higher for training)
        privacy_multipliers = {
            "low": Decimal("1.0"),
            "medium": Decimal("1.3"),
            "high": Decimal("1.8"),
            "critical": Decimal("2.5"),
        }
        privacy_multiplier = privacy_multipliers.get(privacy_level, Decimal("1.8"))
        
        # Reliability requirement multiplier
        reliability_multipliers = {
            "best_effort": Decimal("0.8"),
            "standard": Decimal("1.0"),
            "high": Decimal("1.4"),
            "guaranteed": Decimal("2.0"),
        }
        reliability_multiplier = reliability_multipliers.get(reliability_requirement, Decimal("1.4"))
        
        # Participants multiplier (higher coordination costs)
        participants_multiplier = Decimal("1.0") + (Decimal(str(participants_needed - 1)) * Decimal("0.2"))
        
        # Duration discount (longer training gets discount)
        duration_multiplier = self._calculate_duration_multiplier(Decimal(str(duration_hours)))
        
        # Priority multiplier from tier
        priority_multiplier = tier_config.priority_multiplier
        
        # Market dynamics (use federated training lane)
        if ResourceLane.FEDERATED_TRAINING in self.price_bands:
            training_band = self.price_bands[ResourceLane.FEDERATED_TRAINING]
            market_multiplier = self._calculate_supply_demand_multiplier(training_band)
        else:
            market_multiplier = Decimal("1.0")
            
        # Reputation adjustment (more important for training)
        reputation_multiplier = Decimal("1.0")
        trust_score = 0.7  # Higher default for training
        if node_id and self.reputation_engine:
            trust_score = self.reputation_engine.get_trust_score(node_id)
            reputation_multiplier = Decimal("1.0") + (Decimal(str(trust_score)) - Decimal("0.5")) * Decimal("0.5")
            
        # Calculate final price per hour
        price_per_hour = (
            base_price_per_hour
            * model_multiplier
            * privacy_multiplier
            * reliability_multiplier
            * participants_multiplier
            * duration_multiplier
            * priority_multiplier
            * market_multiplier
            * reputation_multiplier
        )
        
        # Ensure within tier bounds
        price_per_hour = max(
            tier_config.training_price_min,
            min(tier_config.training_price_max, price_per_hour)
        )
        
        total_cost = price_per_hour * Decimal(str(duration_hours))
        
        return {
            "workload_type": "federated_training",
            "user_tier": user_tier.value,
            "model_size": model_size,
            "duration_hours": duration_hours,
            "participants_needed": participants_needed,
            "privacy_level": privacy_level,
            "reliability_requirement": reliability_requirement,
            "price_per_hour": float(price_per_hour),
            "total_cost": float(total_cost),
            "currency": "USD",
            "pricing_breakdown": {
                "base_price": float(base_price_per_hour),
                "model_multiplier": float(model_multiplier),
                "privacy_multiplier": float(privacy_multiplier),
                "reliability_multiplier": float(reliability_multiplier),
                "participants_multiplier": float(participants_multiplier),
                "duration_multiplier": float(duration_multiplier),
                "priority_multiplier": float(priority_multiplier),
                "market_multiplier": float(market_multiplier),
                "reputation_multiplier": float(reputation_multiplier),
                "trust_score": trust_score,
            },
            "tier_info": {
                "max_concurrent_jobs": tier_config.max_concurrent_jobs,
                "guaranteed_uptime": float(tier_config.guaranteed_uptime_percentage),
                "max_latency_sla_ms": float(tier_config.max_latency_sla_ms),
                "dedicated_support": tier_config.dedicated_support,
                "custom_pricing": tier_config.custom_pricing,
            },
        }

    async def get_bulk_pricing_quote(
        self, resource_requirements: dict[ResourceLane, tuple[Decimal, Decimal]]  # lane -> (quantity, duration)
    ) -> dict[str, Any]:
        """Get bulk pricing quote for multiple resources"""

        lane_quotes = {}
        total_cost = Decimal("0")

        for lane, (quantity, duration) in resource_requirements.items():
            quote = await self.get_resource_price(lane, quantity, duration)
            lane_quotes[lane.value] = quote
            total_cost += Decimal(str(quote["final_price"]))

        # Bulk discount for large orders
        bulk_multiplier = self._calculate_bulk_discount_multiplier(total_cost)
        final_total = total_cost * bulk_multiplier

        return {
            "lane_quotes": lane_quotes,
            "subtotal": float(total_cost),
            "bulk_discount_multiplier": float(bulk_multiplier),
            "total_cost": float(final_total),
            "currency": "USD",
            "quote_valid_until": (datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
            "market_condition": self.market_metrics.condition.value,
        }

    async def set_dynamic_reserve_price(
        self, lane: ResourceLane, target_utilization: Decimal = Decimal("0.8")
    ) -> Decimal:
        """Calculate and set dynamic reserve price for auction"""

        if lane not in self.price_bands:
            raise ValueError(f"Unknown resource lane: {lane}")

        band = self.price_bands[lane]

        # Base reserve price
        base_reserve = band.current_price * Decimal("0.9")  # 10% below market price

        # Utilization adjustment
        utilization_factor = band.utilization_rate / target_utilization
        utilization_adjustment = min(Decimal("2.0"), max(Decimal("0.5"), utilization_factor))

        # Supply scarcity adjustment
        supply_demand_ratio = band.get_supply_demand_ratio()
        scarcity_adjustment = Decimal("1") / max(Decimal("0.1"), min(Decimal("10"), supply_demand_ratio))

        # Volatility adjustment (higher volatility = higher reserve)
        volatility = band.calculate_volatility()
        volatility_adjustment = Decimal("1") + (volatility * Decimal("0.5"))

        # Calculate final reserve price
        reserve_price = base_reserve * utilization_adjustment * scarcity_adjustment * volatility_adjustment

        # Ensure reserve price is within band limits
        reserve_price = max(band.floor_price, min(band.ceiling_price * Decimal("0.8"), reserve_price))

        logger.info(
            f"Dynamic reserve price for {lane.value}: ${float(reserve_price):.6f} "
            f"(utilization={float(utilization_adjustment):.2f}, "
            f"scarcity={float(scarcity_adjustment):.2f}, "
            f"volatility={float(volatility_adjustment):.2f})"
        )

        return reserve_price

    async def detect_price_manipulation(self) -> list[PricingAnomalyAlert]:
        """Detect potential price manipulation"""

        alerts = []

        for lane, band in self.price_bands.items():
            # Check for unusual volatility
            volatility = band.calculate_volatility(hours=1)  # Last hour
            if volatility > self.manipulation_thresholds["price_spike_threshold"]:
                alert = PricingAnomalyAlert(
                    alert_id=f"volatility_{uuid.uuid4().hex[:8]}",
                    lane=lane,
                    anomaly_type="volatility",
                    description=f"Unusual volatility detected in {lane.value}: {float(volatility * 100):.2f}%",
                    current_price=band.current_price,
                    expected_price=band.target_price,
                    deviation_percentage=volatility,
                    suggested_action="Monitor closely, consider circuit breaker",
                )
                alert.calculate_severity()
                alerts.append(alert)

            # Check for price manipulation patterns
            if self._detect_coordinated_manipulation(band):
                alert = PricingAnomalyAlert(
                    alert_id=f"manipulation_{uuid.uuid4().hex[:8]}",
                    lane=lane,
                    anomaly_type="manipulation",
                    description=f"Potential coordinated manipulation in {lane.value}",
                    current_price=band.current_price,
                    expected_price=band.target_price,
                    deviation_percentage=abs((band.current_price - band.target_price) / band.target_price),
                    market_condition=MarketCondition.MANIPULATION,
                    suggested_action="Activate circuit breaker, investigate bidders",
                )
                alert.calculate_severity()
                alerts.append(alert)

        return alerts

    async def activate_circuit_breaker(self, lane: ResourceLane, reason: str):
        """Activate emergency circuit breaker for a resource lane"""

        if lane not in self.price_bands:
            return

        band = self.price_bands[lane]

        # Freeze price at current level
        self.current_strategy = PricingStrategy.CIRCUIT_BREAKER

        # Create emergency alert
        alert = PricingAnomalyAlert(
            alert_id=f"circuit_breaker_{uuid.uuid4().hex[:8]}",
            lane=lane,
            anomaly_type="circuit_breaker",
            severity="critical",
            description=f"Circuit breaker activated for {lane.value}: {reason}",
            current_price=band.current_price,
            expected_price=band.target_price,
            market_condition=MarketCondition.EMERGENCY,
            suggested_action="Manual intervention required",
        )

        self.anomaly_alerts.append(alert)

        logger.critical(
            f"CIRCUIT BREAKER ACTIVATED for {lane.value}: {reason}. "
            f"Price frozen at ${float(band.current_price):.6f}"
        )

        # Notify auction engine if available
        if self.auction_engine:
            # Would pause auctions in this lane
            pass

    async def get_market_analytics(self) -> dict[str, Any]:
        """Get comprehensive market analytics"""

        lane_analytics = {}

        for lane, band in self.price_bands.items():
            volatility = band.calculate_volatility()
            supply_demand_ratio = band.get_supply_demand_ratio()

            lane_analytics[lane.value] = {
                "current_price": float(band.current_price),
                "target_price": float(band.target_price),
                "floor_price": float(band.floor_price),
                "ceiling_price": float(band.ceiling_price),
                "supply_units": float(band.supply_units),
                "demand_units": float(band.demand_units),
                "utilization_rate": float(band.utilization_rate),
                "supply_demand_ratio": float(supply_demand_ratio),
                "volatility_24h": float(volatility),
                "is_volatile": band.is_volatile(),
                "price_trend": self._calculate_price_trend(band),
                "health_score": float(self._calculate_lane_health_score(band)),
            }

        market_health_score = self.market_metrics.calculate_market_health_score()

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "market_overview": {
                "total_supply": float(self.market_metrics.total_supply),
                "total_demand": float(self.market_metrics.total_demand),
                "overall_utilization": float(self.market_metrics.overall_utilization),
                "market_condition": self.market_metrics.condition.value,
                "pricing_strategy": self.current_strategy.value,
                "health_score": float(market_health_score),
            },
            "lane_analytics": lane_analytics,
            "anomaly_alerts": {
                "active_alerts": len([a for a in self.anomaly_alerts if not a.resolved_at]),
                "critical_alerts": len(
                    [a for a in self.anomaly_alerts if a.severity == "critical" and not a.resolved_at]
                ),
                "total_alerts_24h": len(
                    [a for a in self.anomaly_alerts if (datetime.now(UTC) - a.detected_at).days == 0]
                ),
            },
            "pricing_metrics": {
                "average_price_across_lanes": float(
                    sum(band.current_price for band in self.price_bands.values()) / len(self.price_bands)
                ),
                "most_volatile_lane": max(
                    self.price_bands.keys(), key=lambda lane: self.price_bands[lane].calculate_volatility()
                ).value,
                "highest_demand_lane": max(
                    self.price_bands.keys(), key=lambda lane: self.price_bands[lane].demand_units
                ).value,
                "manipulation_risk_score": float(self._calculate_manipulation_risk()),
            },
        }

    # Private methods

    async def _pricing_update_loop(self):
        """Background task for continuous price updates"""

        while True:
            try:
                await asyncio.sleep(self.config["price_update_interval"])

                # Update prices based on current strategy
                await self._update_prices_by_strategy()

                # Update market metrics
                await self._update_market_metrics()

            except Exception as e:
                logger.error(f"Error in pricing update loop: {e}")
                await asyncio.sleep(60)

    async def _anomaly_detection_loop(self):
        """Background task for anomaly detection"""

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Detect anomalies
                new_alerts = await self.detect_price_manipulation()
                self.anomaly_alerts.extend(new_alerts)

                # Clean up old resolved alerts
                cutoff = datetime.now(UTC) - timedelta(days=7)
                self.anomaly_alerts = [alert for alert in self.anomaly_alerts if alert.detected_at > cutoff]

                # Trigger circuit breakers if needed
                for alert in new_alerts:
                    if alert.severity == "critical":
                        await self.activate_circuit_breaker(alert.lane, alert.description)

            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(120)

    async def _calculate_target_prices(self):
        """Calculate target prices based on supply/demand"""

        for lane, band in self.price_bands.items():
            if band.supply_units == 0 and band.demand_units == 0:
                continue  # No market data

            # Supply/demand ratio
            ratio = band.get_supply_demand_ratio()

            # Base target price adjustment
            if ratio > Decimal("1.5"):  # High supply
                adjustment = Decimal("0.9")  # Lower prices
            elif ratio < Decimal("0.7"):  # Low supply
                adjustment = Decimal("1.1")  # Higher prices
            else:
                adjustment = Decimal("1.0")  # Stable prices

            # Utilization adjustment
            util_adjustment = Decimal("1") + (band.utilization_rate - Decimal("0.5")) * Decimal("0.2")

            # Calculate new target price
            new_target = band.current_price * adjustment * util_adjustment

            # Ensure target is within bounds
            new_target = max(band.floor_price, min(band.ceiling_price, new_target))

            band.target_price = new_target

    async def _update_prices_by_strategy(self):
        """Update prices based on current pricing strategy"""

        for lane, band in self.price_bands.items():
            if self.current_strategy == PricingStrategy.CIRCUIT_BREAKER:
                continue  # Don't update prices in emergency mode

            # Calculate price adjustment based on strategy
            if self.current_strategy == PricingStrategy.DYNAMIC:
                # Move towards target price with smoothing
                adjustment = (band.target_price - band.current_price) * Decimal("0.3")

            elif self.current_strategy == PricingStrategy.STABLE:
                # Slower movement towards target
                adjustment = (band.target_price - band.current_price) * Decimal("0.1")

            elif self.current_strategy == PricingStrategy.AGGRESSIVE:
                # Rapid price discovery
                adjustment = (band.target_price - band.current_price) * Decimal("0.5")

            else:  # CONSERVATIVE
                # Very slow changes
                adjustment = (band.target_price - band.current_price) * Decimal("0.05")

            # Apply adjustment
            new_price = band.current_price + adjustment
            band.update_price(new_price, enforce_limits=True)

    async def _update_market_metrics(self):
        """Update comprehensive market metrics"""

        # Calculate average price across all lanes
        total_price = sum(band.current_price for band in self.price_bands.values())
        self.market_metrics.average_price = total_price / len(self.price_bands)

        # Calculate overall volatility
        volatilities = [band.calculate_volatility() for band in self.price_bands.values()]
        self.market_metrics.price_volatility = sum(volatilities) / len(volatilities)

        # Determine market condition
        if self.market_metrics.price_volatility > Decimal("0.3"):
            self.market_metrics.condition = MarketCondition.VOLATILE
        elif self.market_metrics.total_demand > self.market_metrics.total_supply * Decimal("1.5"):
            self.market_metrics.condition = MarketCondition.HIGH_DEMAND
        elif self.market_metrics.total_supply > self.market_metrics.total_demand * Decimal("1.5"):
            self.market_metrics.condition = MarketCondition.LOW_DEMAND
        else:
            self.market_metrics.condition = MarketCondition.NORMAL

    def _calculate_volume_multiplier(self, quantity: Decimal) -> Decimal:
        """Calculate volume-based price multiplier"""

        if quantity < Decimal("10"):
            return Decimal("1.0")  # No discount
        elif quantity < Decimal("100"):
            return Decimal("0.95")  # 5% discount
        elif quantity < Decimal("1000"):
            return Decimal("0.90")  # 10% discount
        else:
            return Decimal("0.85")  # 15% discount

    def _calculate_duration_multiplier(self, duration_hours: Decimal) -> Decimal:
        """Calculate duration-based price multiplier"""

        if duration_hours < Decimal("1"):
            return Decimal("1.2")  # 20% premium for short tasks
        elif duration_hours < Decimal("24"):
            return Decimal("1.0")  # Standard rate
        elif duration_hours < Decimal("168"):  # 1 week
            return Decimal("0.95")  # 5% discount
        else:
            return Decimal("0.90")  # 10% discount for long tasks

    def _calculate_supply_demand_multiplier(self, band: PriceBand) -> Decimal:
        """Calculate supply/demand based multiplier"""

        ratio = band.get_supply_demand_ratio()

        if ratio > Decimal("2"):  # High supply
            return Decimal("0.9")  # 10% discount
        elif ratio < Decimal("0.5"):  # Low supply
            return Decimal("1.15")  # 15% premium
        else:
            return Decimal("1.0")  # Standard rate

    def _calculate_bulk_discount_multiplier(self, total_cost: Decimal) -> Decimal:
        """Calculate bulk discount for large orders"""

        if total_cost > Decimal("10000"):  # >$10k
            return Decimal("0.85")  # 15% bulk discount
        elif total_cost > Decimal("1000"):  # >$1k
            return Decimal("0.90")  # 10% bulk discount
        elif total_cost > Decimal("100"):  # >$100
            return Decimal("0.95")  # 5% bulk discount
        else:
            return Decimal("1.0")  # No bulk discount

    def _detect_coordinated_manipulation(self, band: PriceBand) -> bool:
        """Detect coordinated price manipulation patterns"""

        # Simplified detection - in production would be more sophisticated
        recent_volatility = band.calculate_volatility(hours=1)
        return recent_volatility > Decimal("0.4")  # >40% volatility in 1 hour

    def _calculate_price_trend(self, band: PriceBand) -> str:
        """Calculate price trend direction"""

        if len(band.price_history) < 2:
            return "stable"

        recent_prices = [price for _, price in band.price_history[-5:]]  # Last 5 data points

        if len(recent_prices) < 2:
            return "stable"

        if recent_prices[-1] > recent_prices[0] * Decimal("1.05"):
            return "rising"
        elif recent_prices[-1] < recent_prices[0] * Decimal("0.95"):
            return "falling"
        else:
            return "stable"

    def _calculate_lane_health_score(self, band: PriceBand) -> Decimal:
        """Calculate health score for a specific lane"""

        # Factors: price stability, supply/demand balance, utilization
        volatility_score = max(Decimal("0"), Decimal("1") - band.calculate_volatility())

        ratio = band.get_supply_demand_ratio()
        balance_score = Decimal("1") - abs(ratio - Decimal("1")) / max(ratio, Decimal("1"))
        balance_score = max(Decimal("0"), min(Decimal("1"), balance_score))

        utilization_score = min(Decimal("1"), band.utilization_rate / Decimal("0.8"))

        health_score = (volatility_score + balance_score + utilization_score) / Decimal("3")
        return health_score

    def _calculate_manipulation_risk(self) -> Decimal:
        """Calculate overall market manipulation risk score"""

        risk_factors = []

        for band in self.price_bands.values():
            volatility_risk = band.calculate_volatility()
            supply_demand_imbalance = abs(Decimal("1") - band.get_supply_demand_ratio())
            risk_factors.append(volatility_risk + supply_demand_imbalance * Decimal("0.5"))

        if not risk_factors:
            return Decimal("0")

        average_risk = sum(risk_factors) / len(risk_factors)
        return min(Decimal("1"), average_risk)
    
    async def calculate_h200_hour_equivalent(
        self,
        device_computing_power_tops: Decimal,
        utilization_rate: Decimal,
        time_hours: Decimal,
        device_id: str = None
    ) -> Dict[str, Any]:
        """Calculate H200-hour equivalent using formula: H200h(d) = (TOPS_d × u × t) / T_ref"""
        
        # Core H200-hour calculation
        h200_hours = (
            device_computing_power_tops * 
            utilization_rate * 
            time_hours
        ) / self.h200_reference_specs["tops"]
        
        # Calculate efficiency metrics
        power_efficiency_ratio = device_computing_power_tops / self.h200_reference_specs["power_watts"]
        
        result = {
            "device_id": device_id or f"device_{uuid.uuid4().hex[:8]}",
            "raw_computing_power_tops": float(device_computing_power_tops),
            "utilization_rate": float(utilization_rate),
            "time_hours": float(time_hours),
            "h200_hours_equivalent": float(h200_hours),
            "h200_reference_tops": float(self.h200_reference_specs["tops"]),
            "power_efficiency_ratio": float(power_efficiency_ratio),
            "calculation_formula": "H200h(d) = (TOPS_d × u × t) / T_ref",
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # Audit trail logging
        if self.audit_manager:
            self.audit_manager.log_calculation(
                "h200_hour_calculation",
                result,
                device_id,
                "h200_pricing_engine"
            )
        
        return result
    
    async def get_constitutional_h200_price(
        self,
        user_tier: UserSizeTier,
        device_computing_power_tops: Decimal,
        utilization_rate: Decimal,
        time_hours: Decimal,
        constitutional_level: str = "basic",
        tee_enabled: bool = False,
        device_id: str = None
    ) -> Dict[str, Any]:
        """Get constitutional H200-hour pricing with transparency and audit trail"""
        
        if user_tier not in self.constitutional_tiers:
            raise ValueError(f"Unknown constitutional tier: {user_tier}")
        
        # Calculate H200-hour equivalent
        h200_calculation = await self.calculate_h200_hour_equivalent(
            device_computing_power_tops, utilization_rate, time_hours, device_id
        )
        h200_hours = Decimal(str(h200_calculation["h200_hours_equivalent"]))
        
        # Get constitutional tier configuration
        tier_config = self.constitutional_tiers[user_tier]
        
        # Calculate base cost
        base_cost = tier_config.calculate_tier_price(h200_hours, constitutional_level)
        
        # Apply constitutional adjustments
        constitutional_multiplier = Decimal("1.0")
        adjustments = {}
        
        # Constitutional compliance discount
        if constitutional_level != "basic":
            const_discount = tier_config.constitutional_discount
            constitutional_multiplier -= const_discount
            adjustments["constitutional_discount"] = -float(const_discount)
        
        # Audit transparency bonus
        if constitutional_level in ["full_audit", "constitutional"]:
            audit_bonus = tier_config.audit_transparency_bonus
            constitutional_multiplier -= audit_bonus
            adjustments["audit_transparency_bonus"] = -float(audit_bonus)
        
        # TEE security premium
        if tee_enabled:
            tee_premium = Decimal("0.30")  # 30% premium for TEE
            constitutional_multiplier += tee_premium
            adjustments["tee_security_premium"] = float(tee_premium)
        
        final_cost = base_cost * constitutional_multiplier
        
        # Create comprehensive pricing result
        result = {
            "quote_id": str(uuid.uuid4()),
            "user_tier": user_tier.value,
            "constitutional_level": constitutional_level,
            "h200_calculation": h200_calculation,
            "pricing": {
                "h200_hours": float(h200_hours),
                "base_rate_per_h200_hour": float(tier_config.h200_hour_base_rate),
                "base_cost": float(base_cost),
                "constitutional_multiplier": float(constitutional_multiplier),
                "final_cost": float(final_cost),
                "adjustments": adjustments,
                "currency": "USD"
            },
            "tier_limits": {
                "max_h200_hours_monthly": float(tier_config.max_h200_hours_per_month),
                "constitutional_discount": float(tier_config.constitutional_discount),
                "audit_transparency_bonus": float(tier_config.audit_transparency_bonus)
            },
            "constitutional_features": {
                "transparency_enabled": constitutional_level != "basic",
                "audit_trail": constitutional_level in ["full_audit", "constitutional"],
                "tee_security": tee_enabled,
                "governance_participation": True
            },
            "quote_valid_until": (datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
            "generated_at": datetime.now(UTC).isoformat()
        }
        
        # Audit trail logging
        if self.audit_manager:
            self.audit_manager.log_pricing_quote(result, "constitutional_pricing_manager")
        
        return result
    
    async def get_tee_enhanced_pricing(
        self,
        lane: ResourceLane,
        quantity: Decimal = Decimal("1"),
        duration_hours: Decimal = Decimal("1"),
        tee_level: str = "basic",  # "basic", "enhanced", "confidential"
        node_id: str = None
    ) -> Dict[str, Any]:
        """Get TEE-enhanced workload pricing with security premiums"""
        
        # Get base resource pricing
        base_pricing = await self.get_resource_price(lane, quantity, duration_hours, node_id)
        base_cost = Decimal(str(base_pricing["final_price"]))
        
        # TEE security premiums by level
        tee_premiums = {
            "basic": Decimal("0.20"),      # 20% premium for basic TEE
            "enhanced": Decimal("0.35"),   # 35% premium for enhanced TEE
            "confidential": Decimal("0.50") # 50% premium for confidential computing
        }
        
        tee_premium = tee_premiums.get(tee_level, Decimal("0.20"))
        tee_enhanced_cost = base_cost * (Decimal("1.0") + tee_premium)
        
        # Additional constitutional compliance bonus
        constitutional_bonus = Decimal("0.05")  # 5% discount for constitutional compliance
        final_cost = tee_enhanced_cost * (Decimal("1.0") - constitutional_bonus)
        
        result = {
            "lane": lane.value,
            "tee_level": tee_level,
            "base_pricing": base_pricing,
            "tee_enhanced_pricing": {
                "base_cost": float(base_cost),
                "tee_premium_percentage": float(tee_premium * 100),
                "tee_enhanced_cost": float(tee_enhanced_cost),
                "constitutional_bonus": float(constitutional_bonus * 100),
                "final_cost": float(final_cost),
                "currency": "USD"
            },
            "tee_features": {
                "hardware_security": True,
                "encrypted_computation": True,
                "attestation_available": tee_level in ["enhanced", "confidential"],
                "confidential_computing": tee_level == "confidential",
                "constitutional_compliant": True
            },
            "generated_at": datetime.now(UTC).isoformat()
        }
        
        # Audit trail logging
        if self.audit_manager:
            self.audit_manager.log_calculation(
                "tee_enhanced_pricing",
                result,
                node_id,
                "tee_pricing_manager"
            )
        
        return result


# Global pricing manager instance
_pricing_manager: DynamicPricingManager | None = None


async def get_pricing_manager() -> DynamicPricingManager:
    """Get global pricing manager instance with constitutional features"""
    global _pricing_manager

    if _pricing_manager is None:
        # Initialize with constitutional and audit features
        audit_manager = AuditTrailManager()
        _pricing_manager = DynamicPricingManager(
            audit_manager=audit_manager
        )
        await _pricing_manager.start()

    return _pricing_manager


# Convenience functions for integration
async def get_current_resource_price(lane: str, quantity: float = 1.0, duration_hours: float = 1.0) -> dict[str, Any]:
    """Get current price for a specific resource"""

    resource_lane = ResourceLane(lane)
    manager = await get_pricing_manager()

    return await manager.get_resource_price(resource_lane, Decimal(str(quantity)), Decimal(str(duration_hours)))


async def update_resource_supply_demand(
    supply_data: dict[str, float], demand_data: dict[str, float], utilization_data: dict[str, float]
):
    """Update supply/demand data for pricing"""

    # Convert to appropriate format
    supply_decimal = {ResourceLane(k): Decimal(str(v)) for k, v in supply_data.items()}
    demand_decimal = {ResourceLane(k): Decimal(str(v)) for k, v in demand_data.items()}
    utilization_decimal = {ResourceLane(k): Decimal(str(v)) for k, v in utilization_data.items()}

    manager = await get_pricing_manager()
    await manager.update_market_conditions(supply_decimal, demand_decimal, utilization_decimal)


async def get_dynamic_reserve_price(lane: str, target_utilization: float = 0.8) -> float:
    """Get dynamic reserve price for auction"""

    resource_lane = ResourceLane(lane)
    manager = await get_pricing_manager()

    reserve_price = await manager.set_dynamic_reserve_price(resource_lane, Decimal(str(target_utilization)))

    return float(reserve_price)


# Constitutional H200-hour pricing convenience functions
async def get_h200_hour_quote(
    user_tier: str,
    device_tops: float,
    utilization_rate: float,
    time_hours: float,
    constitutional_level: str = "basic",
    tee_enabled: bool = False
) -> Dict[str, Any]:
    """Get H200-hour constitutional pricing quote"""
    
    manager = await get_pricing_manager()
    tier_enum = UserSizeTier(user_tier)
    
    return await manager.get_constitutional_h200_price(
        tier_enum,
        Decimal(str(device_tops)),
        Decimal(str(utilization_rate)),
        Decimal(str(time_hours)),
        constitutional_level,
        tee_enabled
    )


async def calculate_h200_equivalent(
    device_tops: float,
    utilization_rate: float,
    time_hours: float
) -> Dict[str, Any]:
    """Calculate H200-hour equivalent for device"""
    
    manager = await get_pricing_manager()
    
    return await manager.calculate_h200_hour_equivalent(
        Decimal(str(device_tops)),
        Decimal(str(utilization_rate)),
        Decimal(str(time_hours))
    )


async def get_tee_pricing_quote(
    lane: str,
    quantity: float = 1.0,
    duration_hours: float = 1.0,
    tee_level: str = "basic"
) -> Dict[str, Any]:
    """Get TEE-enhanced pricing quote"""
    
    manager = await get_pricing_manager()
    resource_lane = ResourceLane(lane)
    
    return await manager.get_tee_enhanced_pricing(
        resource_lane,
        Decimal(str(quantity)),
        Decimal(str(duration_hours)),
        tee_level
    )
