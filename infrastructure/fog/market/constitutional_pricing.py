"""
Constitutional Pricing Engine for Fog Computing Market

Implements constitutional compliance pricing with transparency requirements:
- Transparent cost models with full audit trails
- Constitutional tier pricing (Bronze/Silver/Gold/Platinum)
- H200-hour equivalent pricing mathematics
- Democratic governance pricing adjustments
- Privacy-preserving cost calculation

Key Features:
- H200-hour pricing formula: H200h(d) = (TOPS_d × u × t) / T_ref
- Constitutional tier mappings with transparency bonuses
- Audit-compliant cost tracking
- Governance-driven pricing adjustments
- TEE-enhanced workload premium pricing
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import uuid

# Set high precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class ConstitutionalLevel(str, Enum):
    """Constitutional compliance levels"""
    
    BASIC = "basic"                   # Standard pricing
    ENHANCED = "enhanced"             # Enhanced transparency
    FULL_AUDIT = "full_audit"        # Full audit trail
    CONSTITUTIONAL = "constitutional" # Full constitutional compliance


class GovernanceVoteType(str, Enum):
    """Types of governance votes affecting pricing"""
    
    PRICING_ADJUSTMENT = "pricing_adjustment"
    TIER_RESTRUCTURE = "tier_restructure"
    CONSTITUTIONAL_UPGRADE = "constitutional_upgrade"
    TRANSPARENCY_REQUIREMENT = "transparency_requirement"


@dataclass
class H200EquivalentDevice:
    """Device specifications for H200-hour calculation"""
    
    device_id: str
    device_type: str                  # "gpu", "cpu", "specialized"
    computing_power_tops: Decimal     # TOPS_d - device computing power
    memory_gb: Decimal
    power_efficiency: Decimal         # TOPS/Watt
    
    # Performance characteristics
    fp16_performance: Decimal = Decimal("0")
    int8_performance: Decimal = Decimal("0")  
    mixed_precision_support: bool = True
    
    # Constitutional features
    privacy_hardware: bool = False    # TEE/confidential computing
    governance_participation: bool = False
    audit_logging_enabled: bool = True
    
    def __post_init__(self):
        """Ensure all values are Decimal"""
        for attr in ["computing_power_tops", "memory_gb", "power_efficiency"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))


@dataclass 
class ConstitutionalPricingTier:
    """Constitutional pricing tier with H200-hour rates"""
    
    tier_name: str
    tier_level: str                   # "bronze", "silver", "gold", "platinum"
    
    # H200-hour pricing
    h200_hour_base_rate: Decimal      # USD per H200-hour
    h200_hour_peak_rate: Decimal      # Peak demand rate
    h200_hour_off_peak_rate: Decimal  # Off-peak rate
    
    # Tier limits and guarantees (non-default fields must come before default fields)
    max_h200_hours_monthly: Decimal
    guaranteed_availability: Decimal  # Percentage
    max_queue_time_minutes: Decimal
    
    # Constitutional bonuses/discounts (fields with defaults)
    transparency_bonus: Decimal = Decimal("0.05")      # 5% discount for transparency
    governance_participation_bonus: Decimal = Decimal("0.03")  # 3% discount for governance
    privacy_premium: Decimal = Decimal("0.20")         # 20% premium for privacy features
    audit_trail_discount: Decimal = Decimal("0.10")    # 10% discount for audit compliance
    priority_multiplier: Decimal = Decimal("1.0")
    
    # Constitutional requirements
    requires_identity_verification: bool = False
    requires_governance_participation: bool = False
    enables_democratic_pricing: bool = True
    audit_level: ConstitutionalLevel = ConstitutionalLevel.BASIC
    
    def calculate_h200_price(
        self, 
        h200_hours: Decimal, 
        constitutional_features: Dict[str, bool] = None,
        time_of_day: str = "standard"
    ) -> Dict[str, Any]:
        """Calculate price with constitutional adjustments"""
        
        constitutional_features = constitutional_features or {}
        
        # Select base rate by time
        if time_of_day == "peak":
            base_rate = self.h200_hour_peak_rate
        elif time_of_day == "off_peak":
            base_rate = self.h200_hour_off_peak_rate
        else:
            base_rate = self.h200_hour_base_rate
            
        base_cost = h200_hours * base_rate
        
        # Apply constitutional adjustments
        adjustments = {}
        final_multiplier = Decimal("1.0")
        
        # Transparency bonus
        if constitutional_features.get("transparency_enabled", False):
            transparency_discount = self.transparency_bonus
            adjustments["transparency_bonus"] = -float(transparency_discount)
            final_multiplier -= transparency_discount
            
        # Governance participation bonus
        if constitutional_features.get("governance_participation", False):
            governance_discount = self.governance_participation_bonus
            adjustments["governance_bonus"] = -float(governance_discount)
            final_multiplier -= governance_discount
            
        # Privacy premium (for TEE/confidential computing)
        if constitutional_features.get("privacy_hardware", False):
            privacy_premium = self.privacy_premium
            adjustments["privacy_premium"] = float(privacy_premium)
            final_multiplier += privacy_premium
            
        # Audit trail discount
        if constitutional_features.get("audit_trail", False):
            audit_discount = self.audit_trail_discount
            adjustments["audit_trail_discount"] = -float(audit_discount)
            final_multiplier -= audit_discount
            
        # Priority tier multiplier
        final_multiplier *= self.priority_multiplier
        adjustments["priority_multiplier"] = float(self.priority_multiplier)
        
        final_cost = base_cost * final_multiplier
        
        return {
            "tier": self.tier_level,
            "h200_hours": float(h200_hours),
            "base_rate_per_h200_hour": float(base_rate),
            "base_cost": float(base_cost),
            "final_cost": float(final_cost),
            "total_savings": float(base_cost - final_cost) if final_cost < base_cost else 0.0,
            "total_premium": float(final_cost - base_cost) if final_cost > base_cost else 0.0,
            "final_multiplier": float(final_multiplier),
            "adjustments": adjustments,
            "currency": "USD",
            "constitutional_level": self.audit_level.value,
        }


@dataclass
class GovernancePricingVote:
    """Governance vote for pricing adjustments"""
    
    vote_id: str
    proposal_id: str
    vote_type: GovernanceVoteType
    
    # Voting details
    proposed_adjustment: Decimal      # Percentage adjustment
    target_tier: Optional[str] = None # Target tier for adjustment
    rationale: str = ""
    
    # Voting metadata
    proposer_id: str = ""
    voting_power: Decimal = Decimal("1.0")
    votes_for: Decimal = Decimal("0")
    votes_against: Decimal = Decimal("0")
    votes_abstain: Decimal = Decimal("0")
    
    # Timeline
    proposal_created: datetime = field(default_factory=lambda: datetime.now(UTC))
    voting_deadline: Optional[datetime] = None
    implemented_at: Optional[datetime] = None
    
    status: str = "pending"  # "pending", "passed", "failed", "implemented"
    
    def calculate_vote_outcome(self) -> str:
        """Calculate voting outcome"""
        total_votes = self.votes_for + self.votes_against + self.votes_abstain
        
        if total_votes == 0:
            return "pending"
            
        # Require 60% approval for pricing changes
        approval_rate = self.votes_for / total_votes
        quorum_threshold = Decimal("0.4")  # 40% participation required
        approval_threshold = Decimal("0.6")  # 60% approval required
        
        if total_votes >= quorum_threshold and approval_rate >= approval_threshold:
            return "passed"
        elif self.voting_deadline and datetime.now(UTC) > self.voting_deadline:
            return "failed"
        else:
            return "pending"


class ConstitutionalPricingEngine:
    """
    Constitutional pricing engine with H200-hour mathematics and governance
    
    Features:
    - H200-hour equivalent pricing: H200h(d) = (TOPS_d × u × t) / T_ref
    - Constitutional tier pricing with transparency bonuses
    - Governance-driven pricing adjustments
    - Audit trail compliance
    - TEE-enhanced workload premium pricing
    """
    
    def __init__(self, audit_manager=None):
        self.audit_manager = audit_manager
        
        # H200 reference specifications
        self.h200_reference = {
            "tops": Decimal("989"),       # H200 TOPS performance
            "memory_gb": Decimal("141"),  # H200 memory capacity
            "power_watts": Decimal("700") # H200 power consumption
        }
        
        # Initialize constitutional tiers
        self.constitutional_tiers: Dict[str, ConstitutionalPricingTier] = {}
        self._initialize_constitutional_tiers()
        
        # Governance system
        self.governance_votes: Dict[str, GovernancePricingVote] = {}
        self.implemented_adjustments: List[Dict[str, Any]] = []
        
        # Audit and transparency
        self.pricing_transparency_log: List[Dict[str, Any]] = []
        
        logger.info("Constitutional pricing engine initialized")
    
    def _initialize_constitutional_tiers(self):
        """Initialize constitutional pricing tiers"""
        
        # Bronze Tier - Constitutional democrats, mobile-first
        self.constitutional_tiers["bronze"] = ConstitutionalPricingTier(
            tier_name="Constitutional Bronze",
            tier_level="bronze",
            h200_hour_base_rate=Decimal("0.50"),      # $0.50 per H200-hour
            h200_hour_peak_rate=Decimal("0.65"),      # Peak rate
            h200_hour_off_peak_rate=Decimal("0.35"),  # Off-peak discount
            max_h200_hours_monthly=Decimal("100"),    # 100 H200-hours/month
            guaranteed_availability=Decimal("95.0"),   # 95% availability
            max_queue_time_minutes=Decimal("30"),      # 30 minute max queue
            transparency_bonus=Decimal("0.05"),        # 5% transparency bonus
            governance_participation_bonus=Decimal("0.02"), # 2% governance bonus
            requires_governance_participation=True,     # Constitutional requirement
            audit_level=ConstitutionalLevel.ENHANCED
        )
        
        # Silver Tier - Constitutional republicans, hybrid users
        self.constitutional_tiers["silver"] = ConstitutionalPricingTier(
            tier_name="Constitutional Silver", 
            tier_level="silver",
            h200_hour_base_rate=Decimal("0.75"),      # $0.75 per H200-hour
            h200_hour_peak_rate=Decimal("0.95"),
            h200_hour_off_peak_rate=Decimal("0.55"),
            max_h200_hours_monthly=Decimal("500"),    # 500 H200-hours/month
            guaranteed_availability=Decimal("98.0"),   # 98% availability
            max_queue_time_minutes=Decimal("15"),      # 15 minute max queue
            priority_multiplier=Decimal("1.1"),       # 10% priority boost
            transparency_bonus=Decimal("0.08"),        # 8% transparency bonus
            governance_participation_bonus=Decimal("0.05"), # 5% governance bonus
            requires_governance_participation=True,
            audit_level=ConstitutionalLevel.FULL_AUDIT
        )
        
        # Gold Tier - Constitutional libertarians, cloud-heavy
        self.constitutional_tiers["gold"] = ConstitutionalPricingTier(
            tier_name="Constitutional Gold",
            tier_level="gold", 
            h200_hour_base_rate=Decimal("1.00"),      # $1.00 per H200-hour
            h200_hour_peak_rate=Decimal("1.25"),
            h200_hour_off_peak_rate=Decimal("0.75"),
            max_h200_hours_monthly=Decimal("2000"),   # 2000 H200-hours/month
            guaranteed_availability=Decimal("99.0"),  # 99% availability
            max_queue_time_minutes=Decimal("5"),      # 5 minute max queue
            priority_multiplier=Decimal("1.25"),     # 25% priority boost
            transparency_bonus=Decimal("0.10"),       # 10% transparency bonus
            governance_participation_bonus=Decimal("0.08"), # 8% governance bonus
            privacy_premium=Decimal("0.15"),          # 15% privacy premium
            requires_governance_participation=True,
            audit_level=ConstitutionalLevel.CONSTITUTIONAL
        )
        
        # Platinum Tier - Constitutional enterprise, dedicated
        self.constitutional_tiers["platinum"] = ConstitutionalPricingTier(
            tier_name="Constitutional Platinum",
            tier_level="platinum",
            h200_hour_base_rate=Decimal("1.50"),      # $1.50 per H200-hour
            h200_hour_peak_rate=Decimal("1.75"),
            h200_hour_off_peak_rate=Decimal("1.25"),
            max_h200_hours_monthly=Decimal("10000"),  # 10000 H200-hours/month
            guaranteed_availability=Decimal("99.9"),  # 99.9% availability
            max_queue_time_minutes=Decimal("1"),      # 1 minute max queue
            priority_multiplier=Decimal("1.5"),      # 50% priority boost
            transparency_bonus=Decimal("0.15"),       # 15% transparency bonus
            governance_participation_bonus=Decimal("0.12"), # 12% governance bonus
            privacy_premium=Decimal("0.10"),          # 10% privacy premium (enterprise)
            audit_trail_discount=Decimal("0.05"),     # 5% audit compliance discount
            requires_identity_verification=True,
            requires_governance_participation=True,
            audit_level=ConstitutionalLevel.CONSTITUTIONAL
        )
    
    def calculate_h200_hours(
        self,
        device: H200EquivalentDevice,
        utilization_rate: Decimal,
        time_hours: Decimal
    ) -> Dict[str, Any]:
        """Calculate H200-hour equivalent using formula: H200h(d) = (TOPS_d × u × t) / T_ref"""
        
        # Core H200-hour calculation
        h200_hours = (
            device.computing_power_tops * 
            utilization_rate * 
            time_hours
        ) / self.h200_reference["tops"]
        
        # Calculate efficiency metrics
        power_efficiency_ratio = Decimal("0")
        if device.power_efficiency > 0:
            h200_power_efficiency = self.h200_reference["tops"] / self.h200_reference["power_watts"]
            power_efficiency_ratio = device.power_efficiency / h200_power_efficiency
        
        memory_ratio = device.memory_gb / self.h200_reference["memory_gb"]
        
        # Constitutional features assessment
        constitutional_score = Decimal("0")
        if device.privacy_hardware:
            constitutional_score += Decimal("0.3")
        if device.governance_participation:
            constitutional_score += Decimal("0.2") 
        if device.audit_logging_enabled:
            constitutional_score += Decimal("0.1")
        
        result = {
            "device_id": device.device_id,
            "device_type": device.device_type,
            "raw_computing_power_tops": float(device.computing_power_tops),
            "utilization_rate": float(utilization_rate),
            "time_hours": float(time_hours),
            "h200_hours_equivalent": float(h200_hours),
            "h200_reference_tops": float(self.h200_reference["tops"]),
            "efficiency_metrics": {
                "power_efficiency_ratio": float(power_efficiency_ratio),
                "memory_ratio": float(memory_ratio),
                "constitutional_score": float(constitutional_score)
            },
            "calculation_formula": "H200h(d) = (TOPS_d × u × t) / T_ref",
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # Log for audit trail
        if self.audit_manager:
            self.audit_manager.log_calculation(
                "h200_hour_calculation",
                result,
                device_id=device.device_id
            )
        
        return result
    
    async def get_constitutional_price_quote(
        self,
        tier: str,
        device: H200EquivalentDevice, 
        utilization_rate: Decimal,
        time_hours: Decimal,
        constitutional_features: Dict[str, bool] = None,
        time_of_day: str = "standard"
    ) -> Dict[str, Any]:
        """Get constitutional pricing quote with H200-hour calculation"""
        
        if tier not in self.constitutional_tiers:
            raise ValueError(f"Unknown constitutional tier: {tier}")
        
        # Calculate H200-hour equivalent
        h200_calculation = self.calculate_h200_hours(device, utilization_rate, time_hours)
        h200_hours = Decimal(str(h200_calculation["h200_hours_equivalent"]))
        
        # Get tier pricing
        tier_config = self.constitutional_tiers[tier]
        pricing_result = tier_config.calculate_h200_price(
            h200_hours, 
            constitutional_features,
            time_of_day
        )
        
        # Combine results
        quote = {
            "quote_id": str(uuid.uuid4()),
            "tier": tier,
            "tier_name": tier_config.tier_name,
            "constitutional_level": tier_config.audit_level.value,
            "h200_calculation": h200_calculation,
            "pricing": pricing_result,
            "tier_limits": {
                "max_h200_hours_monthly": float(tier_config.max_h200_hours_monthly),
                "guaranteed_availability": float(tier_config.guaranteed_availability),
                "max_queue_time_minutes": float(tier_config.max_queue_time_minutes),
                "requires_governance_participation": tier_config.requires_governance_participation
            },
            "constitutional_features": constitutional_features or {},
            "transparency": {
                "pricing_transparency": True,
                "audit_trail_available": True,
                "governance_participation_enabled": True,
                "democratic_pricing_adjustments": tier_config.enables_democratic_pricing
            },
            "quote_valid_until": (datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
            "generated_at": datetime.now(UTC).isoformat()
        }
        
        # Log transparency entry
        self.pricing_transparency_log.append({
            "quote_id": quote["quote_id"],
            "tier": tier,
            "h200_hours": float(h200_hours),
            "final_cost": pricing_result["final_cost"],
            "constitutional_features": constitutional_features,
            "timestamp": datetime.now(UTC).isoformat()
        })
        
        # Audit trail
        if self.audit_manager:
            self.audit_manager.log_pricing_quote(quote)
        
        return quote
    
    async def create_governance_vote(
        self,
        vote_type: GovernanceVoteType,
        proposed_adjustment: Decimal,
        rationale: str,
        proposer_id: str,
        target_tier: Optional[str] = None,
        voting_duration_hours: int = 168  # 1 week default
    ) -> str:
        """Create a governance vote for pricing adjustments"""
        
        vote_id = str(uuid.uuid4())
        voting_deadline = datetime.now(UTC) + timedelta(hours=voting_duration_hours)
        
        vote = GovernancePricingVote(
            vote_id=vote_id,
            proposal_id=f"proposal_{vote_id[:8]}",
            vote_type=vote_type,
            proposed_adjustment=proposed_adjustment,
            target_tier=target_tier,
            rationale=rationale,
            proposer_id=proposer_id,
            voting_deadline=voting_deadline
        )
        
        self.governance_votes[vote_id] = vote
        
        logger.info(
            f"Governance vote created: {vote_type.value} "
            f"for tier {target_tier or 'all'} "
            f"with adjustment {float(proposed_adjustment)}%"
        )
        
        return vote_id
    
    async def cast_governance_vote(
        self,
        vote_id: str,
        voter_id: str,
        vote_choice: str,  # "for", "against", "abstain"
        voting_power: Decimal = Decimal("1.0")
    ) -> bool:
        """Cast a vote in governance proposal"""
        
        if vote_id not in self.governance_votes:
            raise ValueError(f"Vote {vote_id} not found")
        
        vote = self.governance_votes[vote_id]
        
        # Check if voting is still open
        if vote.voting_deadline and datetime.now(UTC) > vote.voting_deadline:
            raise ValueError("Voting period has ended")
        
        # Record vote
        if vote_choice == "for":
            vote.votes_for += voting_power
        elif vote_choice == "against": 
            vote.votes_against += voting_power
        elif vote_choice == "abstain":
            vote.votes_abstain += voting_power
        else:
            raise ValueError("Invalid vote choice")
        
        # Update vote status
        vote.status = vote.calculate_vote_outcome()
        
        # Implement if passed
        if vote.status == "passed":
            await self._implement_governance_adjustment(vote)
        
        logger.info(f"Vote cast: {voter_id} voted {vote_choice} on {vote_id}")
        
        return True
    
    async def _implement_governance_adjustment(self, vote: GovernancePricingVote):
        """Implement approved governance pricing adjustment"""
        
        if vote.vote_type == GovernanceVoteType.PRICING_ADJUSTMENT:
            adjustment_factor = Decimal("1.0") + (vote.proposed_adjustment / Decimal("100"))
            
            if vote.target_tier and vote.target_tier in self.constitutional_tiers:
                # Adjust specific tier
                tier = self.constitutional_tiers[vote.target_tier]
                tier.h200_hour_base_rate *= adjustment_factor
                tier.h200_hour_peak_rate *= adjustment_factor
                tier.h200_hour_off_peak_rate *= adjustment_factor
                
                target = vote.target_tier
            else:
                # Adjust all tiers
                for tier in self.constitutional_tiers.values():
                    tier.h200_hour_base_rate *= adjustment_factor
                    tier.h200_hour_peak_rate *= adjustment_factor  
                    tier.h200_hour_off_peak_rate *= adjustment_factor
                
                target = "all_tiers"
            
            # Record implementation
            implementation = {
                "vote_id": vote.vote_id,
                "adjustment_type": vote.vote_type.value,
                "adjustment_percentage": float(vote.proposed_adjustment),
                "target": target,
                "implemented_at": datetime.now(UTC).isoformat(),
                "rationale": vote.rationale
            }
            
            self.implemented_adjustments.append(implementation)
            vote.implemented_at = datetime.now(UTC)
            vote.status = "implemented"
            
            logger.info(
                f"Governance adjustment implemented: "
                f"{float(vote.proposed_adjustment)}% for {target}"
            )
    
    async def get_pricing_transparency_report(self) -> Dict[str, Any]:
        """Generate pricing transparency report"""
        
        # Calculate tier statistics
        tier_stats = {}
        for tier_name, tier_config in self.constitutional_tiers.items():
            tier_stats[tier_name] = {
                "base_rate_per_h200_hour": float(tier_config.h200_hour_base_rate),
                "peak_rate_per_h200_hour": float(tier_config.h200_hour_peak_rate),
                "off_peak_rate_per_h200_hour": float(tier_config.h200_hour_off_peak_rate),
                "transparency_bonus": float(tier_config.transparency_bonus),
                "governance_bonus": float(tier_config.governance_participation_bonus),
                "privacy_premium": float(tier_config.privacy_premium),
                "constitutional_level": tier_config.audit_level.value,
                "governance_required": tier_config.requires_governance_participation
            }
        
        # Recent pricing activity
        recent_quotes = self.pricing_transparency_log[-100:]  # Last 100 quotes
        
        # Governance activity
        active_votes = [
            {
                "vote_id": vote.vote_id,
                "type": vote.vote_type.value,
                "proposed_adjustment": float(vote.proposed_adjustment),
                "target_tier": vote.target_tier,
                "status": vote.status,
                "votes_for": float(vote.votes_for),
                "votes_against": float(vote.votes_against),
                "voting_deadline": vote.voting_deadline.isoformat() if vote.voting_deadline else None
            }
            for vote in self.governance_votes.values()
            if vote.status in ["pending", "passed"]
        ]
        
        return {
            "report_generated_at": datetime.now(UTC).isoformat(),
            "h200_reference_specs": {
                "tops": float(self.h200_reference["tops"]),
                "memory_gb": float(self.h200_reference["memory_gb"]),
                "power_watts": float(self.h200_reference["power_watts"])
            },
            "pricing_formula": "H200h(d) = (TOPS_d × u × t) / T_ref",
            "constitutional_tiers": tier_stats,
            "recent_pricing_activity": {
                "total_quotes": len(self.pricing_transparency_log),
                "quotes_last_24h": len([
                    q for q in self.pricing_transparency_log 
                    if datetime.fromisoformat(q["timestamp"]) > datetime.now(UTC) - timedelta(days=1)
                ]),
                "recent_quotes": recent_quotes
            },
            "governance_activity": {
                "active_votes": active_votes,
                "implemented_adjustments": self.implemented_adjustments[-10:],  # Last 10
                "total_votes": len(self.governance_votes)
            },
            "transparency_features": {
                "public_pricing_formulas": True,
                "audit_trail_available": True,
                "governance_participation": True,
                "democratic_pricing_adjustments": True,
                "constitutional_compliance": True
            }
        }


# Global constitutional pricing engine instance
_constitutional_pricing_engine: ConstitutionalPricingEngine | None = None


async def get_constitutional_pricing_engine() -> ConstitutionalPricingEngine:
    """Get global constitutional pricing engine instance"""
    global _constitutional_pricing_engine
    
    if _constitutional_pricing_engine is None:
        _constitutional_pricing_engine = ConstitutionalPricingEngine()
    
    return _constitutional_pricing_engine


# Convenience functions for integration
async def get_h200_price_quote(
    tier: str,
    device_tops: float,
    utilization_rate: float,
    time_hours: float,
    constitutional_features: Dict[str, bool] = None
) -> Dict[str, Any]:
    """Get H200-hour price quote for device"""
    
    engine = await get_constitutional_pricing_engine()
    
    device = H200EquivalentDevice(
        device_id=f"device_{uuid.uuid4().hex[:8]}",
        device_type="gpu",
        computing_power_tops=Decimal(str(device_tops)),
        memory_gb=Decimal("32"),  # Default
        power_efficiency=Decimal("1.0")  # Default
    )
    
    return await engine.get_constitutional_price_quote(
        tier,
        device,
        Decimal(str(utilization_rate)),
        Decimal(str(time_hours)), 
        constitutional_features
    )