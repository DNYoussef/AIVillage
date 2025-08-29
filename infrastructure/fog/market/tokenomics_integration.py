"""
Tokenomics Integration for Market-Based Pricing

Integrates market-based pricing with existing fog tokenomics system:
- Token-based auction deposits and payments
- Market-driven token economics
- Dynamic token rewards based on market conditions
- Cross-system transaction coordination
- Token flow optimization

Key Features:
- Seamless integration with FogTokenSystem
- Market-based reward calculations
- Token escrow for auction deposits
- Payment processing for market transactions
- Token economics optimization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import logging
from typing import Any
import uuid

# Set precision for token calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


class TokenTransactionType(str, Enum):
    """Token transaction types for market operations"""

    AUCTION_DEPOSIT = "auction_deposit"
    AUCTION_REFUND = "auction_refund"
    MARKET_PAYMENT = "market_payment"
    MARKET_REWARD = "market_reward"
    PRICING_FEE = "pricing_fee"
    QUALITY_BONUS = "quality_bonus"
    MARKET_MAKER_REWARD = "market_maker_reward"


class EscrowStatus(str, Enum):
    """Escrow transaction status"""

    PENDING = "pending"
    HELD = "held"
    RELEASED = "released"
    REFUNDED = "refunded"
    EXPIRED = "expired"


@dataclass
class TokenEscrow:
    """Token escrow for auction deposits and payments"""

    escrow_id: str
    account_id: str
    amount: Decimal
    purpose: str  # "auction_deposit", "payment_guarantee", etc.

    # Escrow conditions
    release_conditions: dict[str, Any] = field(default_factory=dict)
    timeout_hours: int = 24

    # Status tracking
    status: EscrowStatus = EscrowStatus.PENDING
    held_at: datetime | None = None
    released_at: datetime | None = None

    # Associated transactions
    hold_tx_id: str | None = None
    release_tx_id: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.amount, Decimal):
            self.amount = Decimal(str(self.amount))

    def is_expired(self) -> bool:
        """Check if escrow has expired"""
        if not self.held_at:
            return False

        expiry = self.held_at + timedelta(hours=self.timeout_hours)
        return datetime.now(UTC) > expiry

    def can_release(self, conditions: dict[str, Any]) -> bool:
        """Check if release conditions are met"""
        if self.status != EscrowStatus.HELD:
            return False

        # Check each release condition
        for key, required_value in self.release_conditions.items():
            if key not in conditions or conditions[key] != required_value:
                return False

        return True


@dataclass
class MarketTokenMetrics:
    """Token flow metrics for market operations"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Volume metrics
    total_auction_volume: Decimal = Decimal("0")
    total_direct_payment_volume: Decimal = Decimal("0")
    total_deposits_held: Decimal = Decimal("0")
    total_rewards_distributed: Decimal = Decimal("0")

    # Transaction counts
    auction_transactions: int = 0
    market_payment_transactions: int = 0
    escrow_operations: int = 0

    # Market impact on tokenomics
    market_driven_inflation: Decimal = Decimal("0")  # Additional tokens minted due to market activity
    market_token_burn: Decimal = Decimal("0")  # Tokens burned from fees
    average_transaction_fee: Decimal = Decimal("0")

    # Quality metrics
    average_trust_score: Decimal = Decimal("0")
    market_participation_rate: Decimal = Decimal("0")  # % of token holders participating in markets


class TokenomicsIntegration:
    """
    Integration layer between market-based pricing and fog tokenomics

    Handles:
    - Token escrow for auction deposits
    - Market-based payment processing
    - Dynamic reward calculations
    - Token flow optimization
    """

    def __init__(self, token_system=None, auction_engine=None, pricing_manager=None):
        self.token_system = token_system
        self.auction_engine = auction_engine
        self.pricing_manager = pricing_manager

        # Escrow management
        self.active_escrows: dict[str, TokenEscrow] = {}
        self.escrow_history: list[TokenEscrow] = []

        # Market metrics
        self.market_metrics = MarketTokenMetrics()

        # Configuration
        self.config = {
            "auction_deposit_percentage": Decimal("0.1"),  # 10% of bid as deposit
            "market_maker_reward_rate": Decimal("0.005"),  # 0.5% of volume
            "quality_bonus_pool": Decimal("1000"),  # Daily quality bonus pool
            "transaction_fee_rate": Decimal("0.01"),  # 1% transaction fee
            "escrow_timeout_hours": 24,
            "min_escrow_amount": Decimal("1.0"),  # Minimum 1 FOG token
        }

        # Token flow optimization
        self.reward_multipliers = {
            "high_trust": Decimal("1.2"),
            "high_volume": Decimal("1.1"),
            "market_maker": Decimal("1.3"),
            "quality_provider": Decimal("1.15"),
        }

        # Background tasks
        self._escrow_monitor_task: asyncio.Task | None = None
        self._metrics_update_task: asyncio.Task | None = None

        logger.info("Tokenomics integration initialized")

    async def start(self):
        """Start tokenomics integration services"""

        self._escrow_monitor_task = asyncio.create_task(self._escrow_monitor_loop())
        self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())

        logger.info("Tokenomics integration started")

    async def stop(self):
        """Stop tokenomics integration services"""

        if self._escrow_monitor_task:
            self._escrow_monitor_task.cancel()
        if self._metrics_update_task:
            self._metrics_update_task.cancel()

        logger.info("Tokenomics integration stopped")

    async def create_auction_deposit(
        self, account_id: str, auction_id: str, bid_amount: Decimal, auction_duration_hours: int = 1
    ) -> str | None:
        """Create escrow for auction deposit"""

        if not self.token_system:
            logger.error("Token system not available for escrow")
            return None

        # Calculate deposit amount
        deposit_amount = bid_amount * self.config["auction_deposit_percentage"]
        deposit_amount = max(deposit_amount, self.config["min_escrow_amount"])

        # Check account balance
        account_info = self.token_system.get_account_balance(account_id)
        if account_info.get("error") or account_info.get("balance", 0) < float(deposit_amount):
            logger.error(f"Insufficient balance for deposit: {account_id}")
            return None

        # Create escrow
        escrow_id = f"escrow_{uuid.uuid4().hex[:8]}"

        escrow = TokenEscrow(
            escrow_id=escrow_id,
            account_id=account_id,
            amount=deposit_amount,
            purpose="auction_deposit",
            timeout_hours=auction_duration_hours + 2,  # Auction duration + settlement time
            release_conditions={"auction_id": auction_id, "auction_completed": True},
            metadata={
                "auction_id": auction_id,
                "bid_amount": float(bid_amount),
                "deposit_percentage": float(self.config["auction_deposit_percentage"]),
            },
        )

        # Hold tokens in escrow (simplified - would lock tokens in production)
        success = await self._hold_tokens_in_escrow(escrow)

        if success:
            self.active_escrows[escrow_id] = escrow

            logger.info(
                f"Created auction deposit escrow {escrow_id}: " f"{float(deposit_amount)} FOG for auction {auction_id}"
            )

            return escrow_id
        else:
            logger.error(f"Failed to hold tokens for escrow {escrow_id}")
            return None

    async def release_auction_deposit(
        self, escrow_id: str, auction_result: dict[str, Any], winner: bool = False
    ) -> bool:
        """Release auction deposit based on auction outcome"""

        if escrow_id not in self.active_escrows:
            logger.error(f"Escrow not found: {escrow_id}")
            return False

        escrow = self.active_escrows[escrow_id]

        if winner:
            # Winner keeps deposit and it goes toward payment
            success = await self._convert_deposit_to_payment(escrow, auction_result)
        else:
            # Non-winner gets deposit refunded
            success = await self._refund_deposit(escrow)

        if success:
            # Move to history
            self.escrow_history.append(escrow)
            del self.active_escrows[escrow_id]

            logger.info(f"Released auction deposit {escrow_id}, winner={winner}")

        return success

    async def process_market_payment(
        self,
        payer_id: str,
        payee_id: str,
        amount: Decimal,
        payment_type: str = "market_transaction",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Process market-based payment between accounts"""

        if not self.token_system:
            logger.error("Token system not available for payments")
            return None

        # Calculate transaction fee
        transaction_fee = amount * self.config["transaction_fee_rate"]
        amount + transaction_fee

        # Process payment through token system
        success = await self.token_system.transfer(
            from_account=payer_id,
            to_account=payee_id,
            amount=float(amount),
            description=f"Market payment: {payment_type}",
        )

        if success:
            # Record transaction in metrics
            self.market_metrics.total_direct_payment_volume += amount
            self.market_metrics.market_payment_transactions += 1
            self.market_metrics.market_token_burn += transaction_fee

            # Update average transaction fee
            total_fees = (
                self.market_metrics.average_transaction_fee * (self.market_metrics.market_payment_transactions - 1)
                + transaction_fee
            )
            self.market_metrics.average_transaction_fee = total_fees / self.market_metrics.market_payment_transactions

            tx_id = f"market_tx_{uuid.uuid4().hex[:8]}"

            logger.info(f"Processed market payment {tx_id}: " f"{float(amount)} FOG from {payer_id} to {payee_id}")

            return tx_id
        else:
            logger.error(f"Market payment failed: {payer_id} -> {payee_id}")
            return None

    async def calculate_market_reward(
        self, provider_id: str, market_activity: dict[str, Any], base_contribution_reward: Decimal
    ) -> Decimal:
        """Calculate enhanced reward based on market participation"""

        # Base reward from existing contribution system
        total_reward = base_contribution_reward

        # Market participation bonuses
        trust_score = Decimal(str(market_activity.get("trust_score", 0.5)))
        volume_provided = Decimal(str(market_activity.get("volume_provided", 0)))
        quality_score = Decimal(str(market_activity.get("quality_score", 0.5)))

        # High trust bonus
        if trust_score > Decimal("0.8"):
            trust_bonus = base_contribution_reward * (self.reward_multipliers["high_trust"] - Decimal("1"))
            total_reward += trust_bonus

        # High volume bonus
        if volume_provided > Decimal("100"):  # $100+ in market activity
            volume_bonus = base_contribution_reward * (self.reward_multipliers["high_volume"] - Decimal("1"))
            total_reward += volume_bonus

        # Quality provider bonus
        if quality_score > Decimal("0.9"):
            quality_bonus = base_contribution_reward * (self.reward_multipliers["quality_provider"] - Decimal("1"))
            total_reward += quality_bonus

        # Market maker bonus (for providing liquidity)
        if market_activity.get("market_maker", False):
            mm_bonus = base_contribution_reward * (self.reward_multipliers["market_maker"] - Decimal("1"))
            total_reward += mm_bonus

        logger.debug(
            f"Calculated market reward for {provider_id}: "
            f"base={float(base_contribution_reward)}, "
            f"total={float(total_reward)} "
            f"(multiplier={float(total_reward / base_contribution_reward):.2f}x)"
        )

        return total_reward

    async def distribute_quality_bonuses(self) -> dict[str, Decimal]:
        """Distribute daily quality bonus pool based on market performance"""

        if not self.auction_engine:
            return {}

        # Get recent auction results for quality scoring
        await self.auction_engine.get_market_statistics()

        # Simplified quality bonus distribution
        # In production, would analyze detailed performance metrics

        quality_pool = self.config["quality_bonus_pool"]
        bonuses = {}

        # Distribute based on trust scores and market participation
        # This is simplified - production would have sophisticated scoring

        total_participants = 10  # Simplified
        bonus_per_participant = quality_pool / Decimal(str(total_participants))

        for i in range(total_participants):
            participant_id = f"provider_{i}"
            bonuses[participant_id] = bonus_per_participant

        # Process bonus payments
        for participant_id, bonus_amount in bonuses.items():
            if self.token_system:
                success = await self.token_system._mint_reward_tokens(
                    participant_id, int(bonus_amount * Decimal(10**18))
                )

                if success:
                    self.market_metrics.total_rewards_distributed += bonus_amount

                    logger.info(f"Distributed quality bonus: {float(bonus_amount)} FOG to {participant_id}")

        return bonuses

    async def get_token_flow_analytics(self) -> dict[str, Any]:
        """Get comprehensive token flow analytics for market operations"""

        # Calculate token velocity in market operations
        total_market_volume = self.market_metrics.total_auction_volume + self.market_metrics.total_direct_payment_volume

        # Token supply impact
        net_token_change = self.market_metrics.total_rewards_distributed - self.market_metrics.market_token_burn

        # Market health indicators
        escrow_utilization = len(self.active_escrows) / max(1, len(self.active_escrows) + len(self.escrow_history))

        return {
            "volume_metrics": {
                "total_market_volume": float(total_market_volume),
                "auction_volume": float(self.market_metrics.total_auction_volume),
                "direct_payment_volume": float(self.market_metrics.total_direct_payment_volume),
                "deposits_held": float(self.market_metrics.total_deposits_held),
                "rewards_distributed": float(self.market_metrics.total_rewards_distributed),
            },
            "token_supply_impact": {
                "net_token_change": float(net_token_change),
                "market_driven_inflation": float(self.market_metrics.market_driven_inflation),
                "tokens_burned_from_fees": float(self.market_metrics.market_token_burn),
                "inflation_rate_from_markets": float(net_token_change / max(Decimal("1000000"), total_market_volume)),
            },
            "transaction_metrics": {
                "auction_transactions": self.market_metrics.auction_transactions,
                "market_payment_transactions": self.market_metrics.market_payment_transactions,
                "escrow_operations": self.market_metrics.escrow_operations,
                "average_transaction_fee": float(self.market_metrics.average_transaction_fee),
            },
            "market_health": {
                "escrow_utilization": escrow_utilization,
                "average_trust_score": float(self.market_metrics.average_trust_score),
                "market_participation_rate": float(self.market_metrics.market_participation_rate),
            },
            "escrow_status": {
                "active_escrows": len(self.active_escrows),
                "total_escrowed_amount": float(sum(e.amount for e in self.active_escrows.values())),
                "expired_escrows": len([e for e in self.active_escrows.values() if e.is_expired()]),
            },
        }

    async def optimize_token_rewards(
        self, market_conditions: dict[str, Any], supply_demand_data: dict[str, Any]
    ) -> dict[str, Decimal]:
        """Optimize token reward rates based on market conditions"""

        optimization_factors = {}

        # Base optimization based on market conditions
        market_condition = market_conditions.get("condition", "normal")

        if market_condition == "high_demand":
            # Increase rewards to incentivize more supply
            optimization_factors["supply_incentive"] = Decimal("1.3")
        elif market_condition == "low_demand":
            # Reduce rewards to prevent oversupply
            optimization_factors["supply_incentive"] = Decimal("0.8")
        else:
            optimization_factors["supply_incentive"] = Decimal("1.0")

        # Utilization-based optimization
        utilization_rate = Decimal(str(supply_demand_data.get("utilization_rate", 0.5)))

        if utilization_rate < Decimal("0.3"):  # Low utilization
            optimization_factors["utilization_adjustment"] = Decimal("0.9")
        elif utilization_rate > Decimal("0.8"):  # High utilization
            optimization_factors["utilization_adjustment"] = Decimal("1.2")
        else:
            optimization_factors["utilization_adjustment"] = Decimal("1.0")

        # Quality-based optimization
        avg_quality = Decimal(str(market_conditions.get("average_quality_score", 0.5)))

        if avg_quality > Decimal("0.8"):
            optimization_factors["quality_premium"] = Decimal("1.1")
        else:
            optimization_factors["quality_premium"] = Decimal("1.0")

        logger.info(
            f"Token reward optimization: "
            f"supply_incentive={float(optimization_factors['supply_incentive']):.2f}, "
            f"utilization_adjustment={float(optimization_factors['utilization_adjustment']):.2f}, "
            f"quality_premium={float(optimization_factors['quality_premium']):.2f}"
        )

        return optimization_factors

    # Private methods

    async def _hold_tokens_in_escrow(self, escrow: TokenEscrow) -> bool:
        """Hold tokens in escrow (simplified implementation)"""

        if not self.token_system:
            return True  # Skip if no token system

        # In production, this would lock tokens in escrow contract
        # For now, just verify balance and mark as held

        account_info = self.token_system.get_account_balance(escrow.account_id)
        if account_info.get("error") or account_info.get("balance", 0) < float(escrow.amount):
            return False

        escrow.status = EscrowStatus.HELD
        escrow.held_at = datetime.now(UTC)
        escrow.hold_tx_id = f"hold_{uuid.uuid4().hex[:8]}"

        # Update metrics
        self.market_metrics.total_deposits_held += escrow.amount
        self.market_metrics.escrow_operations += 1

        return True

    async def _convert_deposit_to_payment(self, escrow: TokenEscrow, auction_result: dict[str, Any]) -> bool:
        """Convert winning bid deposit to payment"""

        if not self.token_system:
            return True

        # Release escrow and apply to payment
        escrow.status = EscrowStatus.RELEASED
        escrow.released_at = datetime.now(UTC)
        escrow.release_tx_id = f"convert_{uuid.uuid4().hex[:8]}"

        # Update metrics
        self.market_metrics.total_deposits_held -= escrow.amount
        self.market_metrics.total_auction_volume += escrow.amount
        self.market_metrics.auction_transactions += 1

        logger.info(f"Converted deposit {escrow.escrow_id} to payment: {float(escrow.amount)} FOG")
        return True

    async def _refund_deposit(self, escrow: TokenEscrow) -> bool:
        """Refund deposit to account"""

        if not self.token_system:
            return True

        # Release tokens back to account
        escrow.status = EscrowStatus.REFUNDED
        escrow.released_at = datetime.now(UTC)
        escrow.release_tx_id = f"refund_{uuid.uuid4().hex[:8]}"

        # Update metrics
        self.market_metrics.total_deposits_held -= escrow.amount

        logger.info(f"Refunded deposit {escrow.escrow_id}: {float(escrow.amount)} FOG to {escrow.account_id}")
        return True

    async def _escrow_monitor_loop(self):
        """Monitor escrows for expiration and automatic release"""

        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                expired_escrows = []

                for escrow_id, escrow in self.active_escrows.items():
                    if escrow.is_expired():
                        expired_escrows.append(escrow_id)

                # Handle expired escrows
                for escrow_id in expired_escrows:
                    escrow = self.active_escrows[escrow_id]

                    # Auto-refund expired escrows
                    await self._refund_deposit(escrow)

                    # Move to history
                    escrow.status = EscrowStatus.EXPIRED
                    self.escrow_history.append(escrow)
                    del self.active_escrows[escrow_id]

                    logger.warning(f"Auto-refunded expired escrow {escrow_id}")

                if expired_escrows:
                    logger.info(f"Processed {len(expired_escrows)} expired escrows")

            except Exception as e:
                logger.error(f"Error in escrow monitor loop: {e}")
                await asyncio.sleep(600)

    async def _metrics_update_loop(self):
        """Update token flow metrics"""

        while True:
            try:
                await asyncio.sleep(3600)  # Update hourly

                # Distribute daily quality bonuses (simplified timing)
                await self.distribute_quality_bonuses()

                # Update market participation metrics
                if self.token_system:
                    network_stats = self.token_system.get_network_stats()
                    total_accounts = network_stats.get("total_accounts", 1)

                    # Estimate market participation (simplified)
                    market_participants = len(set([e.account_id for e in self.active_escrows.values()]))
                    self.market_metrics.market_participation_rate = Decimal(
                        str(market_participants / max(1, total_accounts))
                    )

            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(1800)


# Global integration instance
_tokenomics_integration: TokenomicsIntegration | None = None


async def get_tokenomics_integration() -> TokenomicsIntegration:
    """Get global tokenomics integration instance"""
    global _tokenomics_integration

    if _tokenomics_integration is None:
        _tokenomics_integration = TokenomicsIntegration()
        await _tokenomics_integration.start()

    return _tokenomics_integration


# Convenience functions
async def create_market_deposit(account_id: str, auction_id: str, bid_amount: float) -> str | None:
    """Create market deposit for auction participation"""

    integration = await get_tokenomics_integration()
    return await integration.create_auction_deposit(account_id, auction_id, Decimal(str(bid_amount)))


async def process_market_transaction(
    payer_id: str, payee_id: str, amount: float, tx_type: str = "market"
) -> str | None:
    """Process market-based token transaction"""

    integration = await get_tokenomics_integration()
    return await integration.process_market_payment(payer_id, payee_id, Decimal(str(amount)), tx_type)


async def get_market_token_analytics() -> dict[str, Any]:
    """Get market-based token analytics"""

    integration = await get_tokenomics_integration()
    return await integration.get_token_flow_analytics()
