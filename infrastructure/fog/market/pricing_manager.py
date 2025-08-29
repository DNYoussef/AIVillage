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

    def __init__(self, token_system=None, auction_engine=None, reputation_engine=None):
        self.token_system = token_system
        self.auction_engine = auction_engine
        
        # Reputation system integration
        self.reputation_engine = reputation_engine or BayesianReputationEngine()

        # Price bands for each resource lane
        self.price_bands: dict[ResourceLane, PriceBand] = {}
        self._initialize_price_bands()

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

        logger.info("Dynamic pricing manager initialized")

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
        }

        for lane, params in default_bands.items():
            self.price_bands[lane] = PriceBand(lane=lane, **params)

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


# Global pricing manager instance
_pricing_manager: DynamicPricingManager | None = None


async def get_pricing_manager() -> DynamicPricingManager:
    """Get global pricing manager instance"""
    global _pricing_manager

    if _pricing_manager is None:
        _pricing_manager = DynamicPricingManager()
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
