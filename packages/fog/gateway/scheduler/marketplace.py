# Fog Computing Marketplace Engine
# Production-ready marketplace for fog computing resources and services

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from collections import defaultdict
import json
from decimal import Decimal
import uuid


logger = logging.getLogger(__name__)


class BidStatus(Enum):
    """Status of marketplace bids."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class BidType(Enum):
    """Types of marketplace bids."""
    COMPUTE = "compute"  # Computing resources
    STORAGE = "storage"  # Storage resources
    NETWORK = "network"  # Network bandwidth
    SERVICE = "service"  # Application services
    DATA = "data"        # Data processing
    ML_INFERENCE = "ml_inference"  # ML inference services
    HYBRID = "hybrid"    # Mixed resource types


class PricingTier(Enum):
    """Pricing tiers for marketplace services."""
    SPOT = "spot"        # Lowest price, interruptible
    STANDARD = "standard"  # Regular pricing
    PREMIUM = "premium"   # Higher price, guaranteed availability
    ENTERPRISE = "enterprise"  # Custom enterprise pricing


class ResourceQuality(Enum):
    """Quality levels for fog resources."""
    BASIC = "basic"      # Basic performance
    STANDARD = "standard"  # Standard performance
    HIGH = "high"        # High performance
    PREMIUM = "premium"   # Premium performance


@dataclass
class ResourceRequirements:
    """Requirements specification for fog resources."""
    
    cpu_cores: int = 1
    memory_gb: float = 1.0
    storage_gb: float = 10.0
    bandwidth_mbps: float = 10.0
    duration_minutes: int = 60
    quality: ResourceQuality = ResourceQuality.STANDARD
    location_preference: Optional[str] = None  # Geographic preference
    latency_requirement_ms: Optional[float] = None
    availability_requirement: float = 0.99  # 99% availability
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize requirements to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "storage_gb": self.storage_gb,
            "bandwidth_mbps": self.bandwidth_mbps,
            "duration_minutes": self.duration_minutes,
            "quality": self.quality.value,
            "location_preference": self.location_preference,
            "latency_requirement_ms": self.latency_requirement_ms,
            "availability_requirement": self.availability_requirement
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceRequirements':
        """Deserialize requirements from dictionary."""
        return cls(
            cpu_cores=data.get("cpu_cores", 1),
            memory_gb=data.get("memory_gb", 1.0),
            storage_gb=data.get("storage_gb", 10.0),
            bandwidth_mbps=data.get("bandwidth_mbps", 10.0),
            duration_minutes=data.get("duration_minutes", 60),
            quality=ResourceQuality(data.get("quality", ResourceQuality.STANDARD.value)),
            location_preference=data.get("location_preference"),
            latency_requirement_ms=data.get("latency_requirement_ms"),
            availability_requirement=data.get("availability_requirement", 0.99)
        )


@dataclass
class ResourceOffer:
    """Resource offer from fog providers."""
    
    provider_id: str
    bid_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bid_type: BidType = BidType.COMPUTE
    requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    price_per_hour: Decimal = field(default_factory=lambda: Decimal('0.10'))
    pricing_tier: PricingTier = PricingTier.STANDARD
    available_until: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour default
    location: Optional[str] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    reputation_score: float = 5.0  # 1-10 scale
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.quality_metrics:
            self.quality_metrics = {
                "reliability": 0.95,
                "performance": 0.90,
                "latency_ms": 50.0,
                "uptime": 0.99
            }
    
    def is_expired(self) -> bool:
        """Check if offer has expired."""
        return time.time() > self.available_until
    
    def calculate_total_cost(self) -> Decimal:
        """Calculate total cost for the requested duration."""
        duration_hours = Decimal(self.requirements.duration_minutes) / Decimal('60')
        return self.price_per_hour * duration_hours
    
    def matches_requirements(self, requirements: ResourceRequirements) -> bool:
        """Check if offer matches given requirements."""
        return (
            self.requirements.cpu_cores >= requirements.cpu_cores and
            self.requirements.memory_gb >= requirements.memory_gb and
            self.requirements.storage_gb >= requirements.storage_gb and
            self.requirements.bandwidth_mbps >= requirements.bandwidth_mbps and
            (not requirements.location_preference or 
             self.location == requirements.location_preference) and
            (not requirements.latency_requirement_ms or 
             self.quality_metrics.get("latency_ms", 100) <= requirements.latency_requirement_ms)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize offer to dictionary."""
        return {
            "provider_id": self.provider_id,
            "bid_id": self.bid_id,
            "bid_type": self.bid_type.value,
            "requirements": self.requirements.to_dict(),
            "price_per_hour": str(self.price_per_hour),
            "pricing_tier": self.pricing_tier.value,
            "available_until": self.available_until,
            "location": self.location,
            "quality_metrics": self.quality_metrics,
            "reputation_score": self.reputation_score,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceOffer':
        """Deserialize offer from dictionary."""
        return cls(
            provider_id=data["provider_id"],
            bid_id=data.get("bid_id", str(uuid.uuid4())),
            bid_type=BidType(data.get("bid_type", BidType.COMPUTE.value)),
            requirements=ResourceRequirements.from_dict(data.get("requirements", {})),
            price_per_hour=Decimal(data.get("price_per_hour", '0.10')),
            pricing_tier=PricingTier(data.get("pricing_tier", PricingTier.STANDARD.value)),
            available_until=data.get("available_until", time.time() + 3600),
            location=data.get("location"),
            quality_metrics=data.get("quality_metrics", {}),
            reputation_score=data.get("reputation_score", 5.0),
            created_at=data.get("created_at", time.time())
        )


@dataclass
class MarketplaceBid:
    """Bid in the marketplace."""
    
    bid_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_id: str = ""
    requirements: ResourceRequirements = field(default_factory=ResourceRequirements)
    max_price_per_hour: Decimal = field(default_factory=lambda: Decimal('1.00'))
    preferred_pricing_tier: PricingTier = PricingTier.STANDARD
    status: BidStatus = BidStatus.PENDING
    offers: List[ResourceOffer] = field(default_factory=list)
    selected_offer: Optional[ResourceOffer] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 1800)  # 30 minutes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_offer(self, offer: ResourceOffer) -> bool:
        """Add an offer to this bid."""
        if offer.matches_requirements(self.requirements) and \
           offer.price_per_hour <= self.max_price_per_hour:
            self.offers.append(offer)
            logger.debug(f"Added offer {offer.bid_id} to bid {self.bid_id}")
            return True
        return False
    
    def select_best_offer(self) -> Optional[ResourceOffer]:
        """Select the best offer based on price and quality."""
        if not self.offers:
            return None
        
        # Remove expired offers
        valid_offers = [offer for offer in self.offers if not offer.is_expired()]
        
        if not valid_offers:
            return None
        
        # Score offers based on price, reputation, and quality
        def score_offer(offer: ResourceOffer) -> float:
            # Normalize price (lower is better)
            price_score = 1.0 - min(float(offer.price_per_hour), 10.0) / 10.0
            
            # Reputation score (0-1 scale)
            reputation_score = min(offer.reputation_score, 10.0) / 10.0
            
            # Quality metrics score
            quality_score = (
                offer.quality_metrics.get("reliability", 0.9) * 0.3 +
                offer.quality_metrics.get("performance", 0.9) * 0.3 +
                offer.quality_metrics.get("uptime", 0.9) * 0.4
            )
            
            # Combined score (weighted)
            return price_score * 0.4 + reputation_score * 0.3 + quality_score * 0.3
        
        best_offer = max(valid_offers, key=score_offer)
        self.selected_offer = best_offer
        return best_offer
    
    def is_expired(self) -> bool:
        """Check if bid has expired."""
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize bid to dictionary."""
        return {
            "bid_id": self.bid_id,
            "requester_id": self.requester_id,
            "requirements": self.requirements.to_dict(),
            "max_price_per_hour": str(self.max_price_per_hour),
            "preferred_pricing_tier": self.preferred_pricing_tier.value,
            "status": self.status.value,
            "offers": [offer.to_dict() for offer in self.offers],
            "selected_offer": self.selected_offer.to_dict() if self.selected_offer else None,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata
        }


class MarketplaceEngine:
    """Fog computing marketplace engine."""
    
    def __init__(self, auction_interval: float = 30.0, max_active_bids: int = 1000):
        """
        Initialize marketplace engine.
        
        Args:
            auction_interval: Interval between auction rounds (seconds)
            max_active_bids: Maximum number of active bids
        """
        self.auction_interval = auction_interval
        self.max_active_bids = max_active_bids
        
        # Marketplace state
        self.active_bids: Dict[str, MarketplaceBid] = {}
        self.completed_bids: List[MarketplaceBid] = []
        self.registered_providers: Dict[str, Dict[str, Any]] = {}
        self.market_statistics: Dict[str, Any] = defaultdict(float)
        
        # Event handlers
        self.bid_handlers: List[Callable] = []
        self.match_handlers: List[Callable] = []
        
        # Control flags
        self.is_running = False
        
        logger.info(f"Initialized marketplace engine with {auction_interval}s auction interval")
    
    async def start(self) -> bool:
        """Start the marketplace engine."""
        try:
            self.is_running = True
            
            # Start auction loop
            asyncio.create_task(self._auction_loop())
            
            # Start cleanup loop
            asyncio.create_task(self._cleanup_loop())
            
            # Start statistics update loop
            asyncio.create_task(self._statistics_loop())
            
            logger.info("Started marketplace engine")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start marketplace engine: {e}")
            self.is_running = False
            return False
    
    async def stop(self):
        """Stop the marketplace engine."""
        self.is_running = False
        logger.info("Stopped marketplace engine")
    
    def register_provider(self, provider_id: str, provider_info: Dict[str, Any]) -> bool:
        """Register a fog computing provider."""
        try:
            self.registered_providers[provider_id] = {
                **provider_info,
                "registered_at": time.time(),
                "reputation_score": 5.0,  # Starting reputation
                "total_jobs": 0,
                "successful_jobs": 0
            }
            
            logger.info(f"Registered provider {provider_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register provider {provider_id}: {e}")
            return False
    
    def unregister_provider(self, provider_id: str) -> bool:
        """Unregister a provider."""
        if provider_id in self.registered_providers:
            del self.registered_providers[provider_id]
            logger.info(f"Unregistered provider {provider_id}")
            return True
        return False
    
    def submit_bid(self, requester_id: str, requirements: ResourceRequirements, max_price_per_hour: Decimal, 
                   preferred_tier: PricingTier = PricingTier.STANDARD, metadata: Dict[str, Any] = None) -> str:
        """
        Submit a bid for resources.
        
        Args:
            requester_id: ID of the requester
            requirements: Resource requirements
            max_price_per_hour: Maximum price willing to pay
            preferred_tier: Preferred pricing tier
            metadata: Additional metadata
        
        Returns:
            Bid ID
        """
        if len(self.active_bids) >= self.max_active_bids:
            raise ValueError("Maximum active bids reached")
        
        bid = MarketplaceBid(
            requester_id=requester_id,
            requirements=requirements,
            max_price_per_hour=max_price_per_hour,
            preferred_pricing_tier=preferred_tier,
            metadata=metadata or {}
        )
        
        self.active_bids[bid.bid_id] = bid
        
        # Notify handlers
        for handler in self.bid_handlers:
            try:
                asyncio.create_task(handler(bid))
            except Exception as e:
                logger.error(f"Error in bid handler: {e}")
        
        logger.info(f"Submitted bid {bid.bid_id} for requester {requester_id}")
        return bid.bid_id
    
    def submit_offer(self, provider_id: str, bid_id: str, offer: ResourceOffer) -> bool:
        """
        Submit an offer for a specific bid.
        
        Args:
            provider_id: ID of the provider
            bid_id: ID of the bid to offer for
            offer: Resource offer
        
        Returns:
            True if offer was accepted
        """
        if provider_id not in self.registered_providers:
            logger.warning(f"Provider {provider_id} not registered")
            return False
        
        if bid_id not in self.active_bids:
            logger.warning(f"Bid {bid_id} not found or expired")
            return False
        
        bid = self.active_bids[bid_id]
        if bid.is_expired() or bid.status != BidStatus.PENDING:
            logger.warning(f"Bid {bid_id} is expired or not pending")
            return False
        
        # Update offer with provider reputation
        provider_info = self.registered_providers[provider_id]
        offer.reputation_score = provider_info.get("reputation_score", 5.0)
        
        success = bid.add_offer(offer)
        if success:
            logger.info(f"Added offer from provider {provider_id} to bid {bid_id}")
        
        return success
    
    def get_bid_status(self, bid_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a bid."""
        bid = self.active_bids.get(bid_id)
        if not bid:
            # Check completed bids
            for completed_bid in self.completed_bids:
                if completed_bid.bid_id == bid_id:
                    bid = completed_bid
                    break
        
        if bid:
            return {
                "bid_id": bid.bid_id,
                "status": bid.status.value,
                "offers_count": len(bid.offers),
                "selected_offer": bid.selected_offer.to_dict() if bid.selected_offer else None,
                "created_at": bid.created_at,
                "expires_at": bid.expires_at
            }
        
        return None
    
    def cancel_bid(self, bid_id: str, requester_id: str) -> bool:
        """Cancel a bid."""
        bid = self.active_bids.get(bid_id)
        if not bid or bid.requester_id != requester_id:
            return False
        
        if bid.status == BidStatus.PENDING:
            bid.status = BidStatus.CANCELLED
            self._move_to_completed(bid_id)
            logger.info(f"Cancelled bid {bid_id}")
            return True
        
        return False
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        total_bids = len(self.active_bids) + len(self.completed_bids)
        completed_bids = len(self.completed_bids)
        successful_matches = len([b for b in self.completed_bids if b.status == BidStatus.COMPLETED])
        
        avg_price = Decimal('0')
        if successful_matches > 0:
            total_price = sum(b.selected_offer.price_per_hour for b in self.completed_bids 
                            if b.status == BidStatus.COMPLETED and b.selected_offer)
            avg_price = total_price / successful_matches
        
        return {
            "active_bids": len(self.active_bids),
            "completed_bids": completed_bids,
            "total_bids": total_bids,
            "successful_matches": successful_matches,
            "success_rate": successful_matches / completed_bids if completed_bids > 0 else 0,
            "registered_providers": len(self.registered_providers),
            "average_price_per_hour": str(avg_price),
            "market_statistics": dict(self.market_statistics)
        }
    
    def list_active_bids(self, bid_type: BidType = None) -> List[Dict[str, Any]]:
        """List active bids, optionally filtered by type."""
        bids = []
        for bid in self.active_bids.values():
            if bid_type is None or bid.requirements.quality.name.lower() == bid_type.value:
                bids.append({
                    "bid_id": bid.bid_id,
                    "requester_id": bid.requester_id,
                    "requirements": bid.requirements.to_dict(),
                    "max_price_per_hour": str(bid.max_price_per_hour),
                    "offers_count": len(bid.offers),
                    "expires_at": bid.expires_at
                })
        
        return sorted(bids, key=lambda x: x["expires_at"])
    
    def add_bid_handler(self, handler: Callable):
        """Add bid event handler."""
        self.bid_handlers.append(handler)
    
    def add_match_handler(self, handler: Callable):
        """Add match event handler."""
        self.match_handlers.append(handler)
    
    async def _auction_loop(self):
        """Main auction loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.auction_interval)
                await self._run_auction_round()
                
            except Exception as e:
                logger.error(f"Error in auction loop: {e}")
    
    async def _run_auction_round(self):
        """Run a single auction round."""
        matched_bids = []
        
        for bid_id, bid in list(self.active_bids.items()):
            if bid.is_expired():
                bid.status = BidStatus.EXPIRED
                self._move_to_completed(bid_id)
                continue
            
            if bid.status == BidStatus.PENDING and bid.offers:
                best_offer = bid.select_best_offer()
                if best_offer:
                    bid.status = BidStatus.ACCEPTED
                    matched_bids.append(bid)
                    logger.info(f"Matched bid {bid_id} with offer from provider {best_offer.provider_id}")
        
        # Notify match handlers
        for bid in matched_bids:
            for handler in self.match_handlers:
                try:
                    await handler(bid)
                except Exception as e:
                    logger.error(f"Error in match handler: {e}")
            
            self._move_to_completed(bid.bid_id)
    
    async def _cleanup_loop(self):
        """Cleanup expired bids and offers."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Remove expired bids
                expired_bid_ids = []
                for bid_id, bid in self.active_bids.items():
                    if bid.is_expired():
                        bid.status = BidStatus.EXPIRED
                        expired_bid_ids.append(bid_id)
                
                for bid_id in expired_bid_ids:
                    self._move_to_completed(bid_id)
                
                # Trim completed bids list
                if len(self.completed_bids) > 10000:
                    self.completed_bids = self.completed_bids[-5000:]
                
                logger.debug(f"Cleanup: removed {len(expired_bid_ids)} expired bids")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _statistics_loop(self):
        """Update market statistics."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update market statistics
                self.market_statistics["timestamp"] = time.time()
                self.market_statistics["active_bids"] = len(self.active_bids)
                self.market_statistics["registered_providers"] = len(self.registered_providers)
                
                # Calculate average response time
                recent_bids = [b for b in self.completed_bids if time.time() - b.created_at < 3600]  # Last hour
                if recent_bids:
                    avg_response_time = sum((b.expires_at - b.created_at) for b in recent_bids) / len(recent_bids)
                    self.market_statistics["avg_response_time_seconds"] = avg_response_time
                
            except Exception as e:
                logger.error(f"Error in statistics loop: {e}")
    
    def _move_to_completed(self, bid_id: str):
        """Move bid from active to completed."""
        if bid_id in self.active_bids:
            bid = self.active_bids.pop(bid_id)
            self.completed_bids.append(bid)
            
            # Update provider statistics if bid was successful
            if bid.status == BidStatus.ACCEPTED and bid.selected_offer:
                provider_id = bid.selected_offer.provider_id
                if provider_id in self.registered_providers:
                    provider = self.registered_providers[provider_id]
                    provider["total_jobs"] = provider.get("total_jobs", 0) + 1
                    provider["successful_jobs"] = provider.get("successful_jobs", 0) + 1
                    
                    # Update reputation score
                    success_rate = provider["successful_jobs"] / provider["total_jobs"]
                    provider["reputation_score"] = min(10.0, 5.0 + (success_rate - 0.5) * 10)


# Backward compatibility - try to import from actual infrastructure locations first
try:
    from infrastructure.fog.gateway.scheduler.marketplace import *
except ImportError:
    try:
        from infrastructure.fog.marketplace import *
    except ImportError:
        # Use the implementations defined above
        pass