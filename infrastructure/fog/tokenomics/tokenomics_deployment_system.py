"""
Tokenomics Deployment and Economic Systems

Complete deployment system for FOG token economics including:
- Automated token distribution and rewards
- Economic incentive mechanisms
- Credit systems for resource usage
- Staking and governance integration
- Cross-chain compatibility and DEX integration
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from ..governance.dao_operational_system import DAOOperationalSystem
from ..monitoring.slo_monitor import SLOMonitor
from .fog_token_system import FogTokenSystem

# Set decimal precision for financial calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


class EconomicEventType(Enum):
    """Types of economic events in the system."""

    COMPUTE_CONTRIBUTION = "compute_contribution"
    RESOURCE_CONSUMPTION = "resource_consumption"
    STAKING_REWARD = "staking_reward"
    GOVERNANCE_PARTICIPATION = "governance_participation"
    MARKETPLACE_TRANSACTION = "marketplace_transaction"
    PENALTY_APPLIED = "penalty_applied"
    BONUS_AWARDED = "bonus_awarded"
    LIQUIDITY_PROVISION = "liquidity_provision"
    CROSS_CHAIN_TRANSFER = "cross_chain_transfer"


class IncentiveTier(Enum):
    """Incentive tiers for different participant levels."""

    BRONZE = "bronze"  # 0-1K FOG
    SILVER = "silver"  # 1K-10K FOG
    GOLD = "gold"  # 10K-100K FOG
    PLATINUM = "platinum"  # 100K-1M FOG
    DIAMOND = "diamond"  # 1M+ FOG


@dataclass
class EconomicMetrics:
    """Economic performance metrics."""

    timestamp: datetime

    # Token metrics
    total_supply: int
    circulating_supply: int
    staked_supply: int
    burnt_tokens: int

    # Market metrics
    market_cap_usd: float = 0.0
    token_price_usd: float = 0.0
    trading_volume_24h: float = 0.0

    # Network metrics
    total_participants: int = 0
    active_nodes: int = 0
    compute_hours_contributed: float = 0.0
    revenue_generated: float = 0.0

    # Staking metrics
    staking_ratio: float = 0.0
    avg_staking_duration_days: float = 0.0
    staking_rewards_distributed: int = 0

    # Governance metrics
    governance_participation_rate: float = 0.0
    proposals_submitted: int = 0
    voting_power_distribution_gini: float = 0.0


@dataclass
class IncentiveRule:
    """Economic incentive rule definition."""

    rule_id: str
    name: str
    description: str
    event_type: EconomicEventType

    # Reward calculation
    base_reward_amount: int  # in wei
    multiplier_rules: dict[str, float] = field(default_factory=dict)
    tier_multipliers: dict[IncentiveTier, float] = field(default_factory=dict)

    # Conditions
    min_threshold: float = 0.0
    max_reward_per_day: int = 0  # 0 = unlimited
    cooldown_hours: int = 0

    # Validity
    enabled: bool = True
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass
class EconomicParticipant:
    """Economic system participant profile."""

    participant_id: str
    wallet_address: str

    # Economic stats
    total_earned: int = 0
    total_spent: int = 0
    total_staked: int = 0
    current_tier: IncentiveTier = IncentiveTier.BRONZE

    # Contribution metrics
    compute_hours_contributed: float = 0.0
    storage_gb_contributed: float = 0.0
    bandwidth_gb_contributed: float = 0.0
    uptime_percentage: float = 100.0
    quality_score: float = 100.0

    # Participation history
    first_contribution: datetime | None = None
    last_activity: datetime | None = None
    consecutive_days_active: int = 0

    # Staking info
    staking_start_date: datetime | None = None
    staking_duration_days: int = 0
    compound_staking_enabled: bool = False

    # Reputation and bonuses
    reputation_score: int = 100
    bonus_multiplier: float = 1.0
    penalty_multiplier: float = 1.0


@dataclass
class RewardDistribution:
    """Record of reward distribution."""

    distribution_id: str
    participant_id: str
    event_type: EconomicEventType

    # Reward details
    base_amount: int
    bonus_amount: int
    penalty_amount: int
    final_amount: int

    # Context
    rule_id: str
    tier: IncentiveTier
    multipliers_applied: dict[str, float]
    timestamp: datetime

    # Transaction info
    transaction_id: str | None = None
    confirmed: bool = False


class TokenomicsDeploymentSystem:
    """
    Complete tokenomics deployment and economic management system.

    Features:
    - Automated reward distribution
    - Dynamic economic incentives
    - Multi-tier participant system
    - Staking and governance integration
    - Market-responsive token mechanics
    - Cross-chain compatibility
    """

    def __init__(
        self,
        token_system: FogTokenSystem,
        dao_system: DAOOperationalSystem | None = None,
        slo_monitor: SLOMonitor | None = None,
        data_dir: str = "tokenomics_data",
    ):
        self.token_system = token_system
        self.dao_system = dao_system
        self.slo_monitor = slo_monitor

        # Data storage
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "tokenomics.db"

        # Economic state
        self.participants: dict[str, EconomicParticipant] = {}
        self.incentive_rules: dict[str, IncentiveRule] = {}
        self.reward_history: list[RewardDistribution] = []
        self.economic_metrics_history: list[EconomicMetrics] = []

        # Configuration
        self.config = self._load_config()

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

        # Initialize system
        self._init_database()
        self._load_existing_data()
        self._initialize_incentive_rules()

        logger.info("Tokenomics Deployment System initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load tokenomics configuration."""
        return {
            "rewards": {
                "base_compute_reward_per_hour": 10,  # FOG tokens
                "quality_bonus_max": 50,  # percentage
                "consistency_bonus": 20,  # percentage for consistent contributors
                "early_adopter_bonus": 100,  # percentage for first 1000 participants
                "referral_bonus": 5,  # percentage of referee's rewards
                "governance_participation_reward": 100,  # FOG tokens per vote
            },
            "staking": {
                "base_apy": 0.05,  # 5% annual
                "tier_apy_bonuses": {
                    "silver": 0.01,  # +1%
                    "gold": 0.02,  # +2%
                    "platinum": 0.03,  # +3%
                    "diamond": 0.05,  # +5%
                },
                "min_staking_period_days": 7,
                "compound_frequency_days": 1,
                "early_unstaking_penalty": 0.1,  # 10%
            },
            "tiers": {
                "bronze_min": 0,
                "silver_min": 1000,
                "gold_min": 10000,
                "platinum_min": 100000,
                "diamond_min": 1000000,
            },
            "economics": {
                "inflation_rate_annual": 0.02,  # 2%
                "burn_rate_transaction": 0.001,  # 0.1% of transaction
                "treasury_allocation": 0.1,  # 10% of rewards to treasury
                "dev_fund_allocation": 0.05,  # 5% to development fund
                "marketing_allocation": 0.02,  # 2% to marketing
            },
            "market": {
                "price_oracle_update_minutes": 15,
                "volatility_circuit_breaker": 0.2,  # 20% price change
                "liquidity_incentive_pool_percentage": 0.05,  # 5%
            },
        }

    def _init_database(self):
        """Initialize tokenomics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Participants table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                wallet_address TEXT UNIQUE NOT NULL,
                total_earned INTEGER DEFAULT 0,
                total_spent INTEGER DEFAULT 0,
                total_staked INTEGER DEFAULT 0,
                current_tier TEXT DEFAULT 'bronze',
                compute_hours_contributed REAL DEFAULT 0.0,
                storage_gb_contributed REAL DEFAULT 0.0,
                bandwidth_gb_contributed REAL DEFAULT 0.0,
                uptime_percentage REAL DEFAULT 100.0,
                quality_score REAL DEFAULT 100.0,
                first_contribution TIMESTAMP,
                last_activity TIMESTAMP,
                consecutive_days_active INTEGER DEFAULT 0,
                staking_start_date TIMESTAMP,
                staking_duration_days INTEGER DEFAULT 0,
                compound_staking_enabled BOOLEAN DEFAULT FALSE,
                reputation_score INTEGER DEFAULT 100,
                bonus_multiplier REAL DEFAULT 1.0,
                penalty_multiplier REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Incentive rules table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS incentive_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                event_type TEXT NOT NULL,
                base_reward_amount INTEGER NOT NULL,
                multiplier_rules TEXT DEFAULT '{}',  -- JSON
                tier_multipliers TEXT DEFAULT '{}',  -- JSON
                min_threshold REAL DEFAULT 0.0,
                max_reward_per_day INTEGER DEFAULT 0,
                cooldown_hours INTEGER DEFAULT 0,
                enabled BOOLEAN DEFAULT TRUE,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Reward distributions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reward_distributions (
                distribution_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                base_amount INTEGER NOT NULL,
                bonus_amount INTEGER DEFAULT 0,
                penalty_amount INTEGER DEFAULT 0,
                final_amount INTEGER NOT NULL,
                rule_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                multipliers_applied TEXT DEFAULT '{}',  -- JSON
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                transaction_id TEXT,
                confirmed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (participant_id) REFERENCES participants(participant_id),
                FOREIGN KEY (rule_id) REFERENCES incentive_rules(rule_id)
            )
        """
        )

        # Economic metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS economic_metrics (
                metric_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_supply INTEGER NOT NULL,
                circulating_supply INTEGER NOT NULL,
                staked_supply INTEGER NOT NULL,
                burnt_tokens INTEGER DEFAULT 0,
                market_cap_usd REAL DEFAULT 0.0,
                token_price_usd REAL DEFAULT 0.0,
                trading_volume_24h REAL DEFAULT 0.0,
                total_participants INTEGER DEFAULT 0,
                active_nodes INTEGER DEFAULT 0,
                compute_hours_contributed REAL DEFAULT 0.0,
                revenue_generated REAL DEFAULT 0.0,
                staking_ratio REAL DEFAULT 0.0,
                avg_staking_duration_days REAL DEFAULT 0.0,
                staking_rewards_distributed INTEGER DEFAULT 0,
                governance_participation_rate REAL DEFAULT 0.0,
                proposals_submitted INTEGER DEFAULT 0,
                voting_power_distribution_gini REAL DEFAULT 0.0
            )
        """
        )

        # Economic events log
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS economic_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                participant_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount INTEGER DEFAULT 0,
                details TEXT,  -- JSON
                processed BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_participants_tier ON participants(current_tier)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_participants_wallet ON participants(wallet_address)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rewards_participant ON reward_distributions(participant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rewards_timestamp ON reward_distributions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON economic_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON economic_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON economic_metrics(timestamp)")

        conn.commit()
        conn.close()

        logger.info("Tokenomics database initialized")

    def _load_existing_data(self):
        """Load existing tokenomics data."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Load participants
        cursor.execute("SELECT * FROM participants")
        for row in cursor.fetchall():
            participant = EconomicParticipant(
                participant_id=row["participant_id"],
                wallet_address=row["wallet_address"],
                total_earned=row["total_earned"],
                total_spent=row["total_spent"],
                total_staked=row["total_staked"],
                current_tier=IncentiveTier(row["current_tier"]),
                compute_hours_contributed=row["compute_hours_contributed"],
                storage_gb_contributed=row["storage_gb_contributed"],
                bandwidth_gb_contributed=row["bandwidth_gb_contributed"],
                uptime_percentage=row["uptime_percentage"],
                quality_score=row["quality_score"],
                first_contribution=datetime.fromisoformat(row["first_contribution"])
                if row["first_contribution"]
                else None,
                last_activity=datetime.fromisoformat(row["last_activity"]) if row["last_activity"] else None,
                consecutive_days_active=row["consecutive_days_active"],
                staking_start_date=datetime.fromisoformat(row["staking_start_date"])
                if row["staking_start_date"]
                else None,
                staking_duration_days=row["staking_duration_days"],
                compound_staking_enabled=bool(row["compound_staking_enabled"]),
                reputation_score=row["reputation_score"],
                bonus_multiplier=row["bonus_multiplier"],
                penalty_multiplier=row["penalty_multiplier"],
            )
            self.participants[participant.participant_id] = participant

        # Load incentive rules
        cursor.execute("SELECT * FROM incentive_rules WHERE enabled = TRUE")
        for row in cursor.fetchall():
            rule = IncentiveRule(
                rule_id=row["rule_id"],
                name=row["name"],
                description=row["description"],
                event_type=EconomicEventType(row["event_type"]),
                base_reward_amount=row["base_reward_amount"],
                multiplier_rules=json.loads(row["multiplier_rules"]),
                tier_multipliers={IncentiveTier(k): v for k, v in json.loads(row["tier_multipliers"]).items()}
                if row["tier_multipliers"]
                else {},
                min_threshold=row["min_threshold"],
                max_reward_per_day=row["max_reward_per_day"],
                cooldown_hours=row["cooldown_hours"],
                enabled=bool(row["enabled"]),
                start_date=datetime.fromisoformat(row["start_date"]) if row["start_date"] else None,
                end_date=datetime.fromisoformat(row["end_date"]) if row["end_date"] else None,
            )
            self.incentive_rules[rule.rule_id] = rule

        conn.close()

        logger.info(f"Loaded {len(self.participants)} participants and {len(self.incentive_rules)} incentive rules")

    def _initialize_incentive_rules(self):
        """Initialize default incentive rules."""

        # Compute contribution rewards
        if "compute_contribution" not in self.incentive_rules:
            rule = IncentiveRule(
                rule_id="compute_contribution",
                name="Compute Contribution Rewards",
                description="Rewards for contributing computing resources",
                event_type=EconomicEventType.COMPUTE_CONTRIBUTION,
                base_reward_amount=self._to_wei(self.config["rewards"]["base_compute_reward_per_hour"]),
                multiplier_rules={
                    "quality_bonus": self.config["rewards"]["quality_bonus_max"] / 100,
                    "consistency_bonus": self.config["rewards"]["consistency_bonus"] / 100,
                    "early_adopter_bonus": self.config["rewards"]["early_adopter_bonus"] / 100,
                },
                tier_multipliers={
                    IncentiveTier.BRONZE: 1.0,
                    IncentiveTier.SILVER: 1.1,
                    IncentiveTier.GOLD: 1.25,
                    IncentiveTier.PLATINUM: 1.5,
                    IncentiveTier.DIAMOND: 2.0,
                },
                min_threshold=0.1,  # Minimum 0.1 compute hours
                cooldown_hours=1,
            )
            self.incentive_rules[rule.rule_id] = rule

        # Governance participation rewards
        if "governance_participation" not in self.incentive_rules:
            rule = IncentiveRule(
                rule_id="governance_participation",
                name="Governance Participation Rewards",
                description="Rewards for participating in DAO governance",
                event_type=EconomicEventType.GOVERNANCE_PARTICIPATION,
                base_reward_amount=self._to_wei(self.config["rewards"]["governance_participation_reward"]),
                tier_multipliers={
                    IncentiveTier.BRONZE: 1.0,
                    IncentiveTier.SILVER: 1.2,
                    IncentiveTier.GOLD: 1.5,
                    IncentiveTier.PLATINUM: 2.0,
                    IncentiveTier.DIAMOND: 3.0,
                },
                cooldown_hours=24,  # Once per day per governance action
            )
            self.incentive_rules[rule.rule_id] = rule

        # Staking rewards
        if "staking_rewards" not in self.incentive_rules:
            rule = IncentiveRule(
                rule_id="staking_rewards",
                name="Staking Rewards",
                description="Daily rewards for staking FOG tokens",
                event_type=EconomicEventType.STAKING_REWARD,
                base_reward_amount=0,  # Calculated dynamically based on APY
                tier_multipliers={
                    IncentiveTier.BRONZE: 1.0,
                    IncentiveTier.SILVER: 1.2,  # +20%
                    IncentiveTier.GOLD: 1.4,  # +40%
                    IncentiveTier.PLATINUM: 1.7,  # +70%
                    IncentiveTier.DIAMOND: 2.0,  # +100%
                },
                cooldown_hours=24,  # Daily rewards
            )
            self.incentive_rules[rule.rule_id] = rule

    def _to_wei(self, amount: float) -> int:
        """Convert decimal amount to wei (18 decimals)."""
        return int(Decimal(str(amount)) * Decimal(10**18))

    def _from_wei(self, amount: int) -> Decimal:
        """Convert wei amount to decimal."""
        return Decimal(amount) / Decimal(10**18)

    async def start(self):
        """Start the tokenomics system."""
        if self._running:
            return

        logger.info("Starting Tokenomics Deployment System")
        self._running = True

        # Start background tasks
        tasks = [
            self._reward_processor(),
            self._staking_rewards_distributor(),
            self._tier_updater(),
            self._economic_metrics_collector(),
            self._market_monitor(),
            self._compliance_checker(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("Tokenomics system started")

    async def stop(self):
        """Stop the tokenomics system."""
        if not self._running:
            return

        logger.info("Stopping Tokenomics Deployment System")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Tokenomics system stopped")

    # Participant Management

    async def register_participant(self, wallet_address: str) -> str:
        """Register a new economic participant."""

        # Check if already registered
        for participant in self.participants.values():
            if participant.wallet_address == wallet_address:
                return participant.participant_id

        # Verify wallet exists in token system
        account_info = self.token_system.get_account_balance(wallet_address)
        if "error" in account_info:
            raise ValueError(f"Wallet {wallet_address} not found in token system")

        participant_id = str(uuid.uuid4())
        participant = EconomicParticipant(
            participant_id=participant_id,
            wallet_address=wallet_address,
            current_tier=self._calculate_tier(account_info["balance"]),
        )

        self.participants[participant_id] = participant
        await self._save_participant(participant)

        # Log economic event
        await self._log_economic_event(
            EconomicEventType.MARKETPLACE_TRANSACTION,
            participant_id,
            details={"action": "participant_registered", "wallet": wallet_address},
        )

        logger.info(f"Registered economic participant: {participant_id} ({wallet_address})")
        return participant_id

    def _calculate_tier(self, balance: float) -> IncentiveTier:
        """Calculate participant tier based on token balance."""
        tiers = self.config["tiers"]

        if balance >= tiers["diamond_min"]:
            return IncentiveTier.DIAMOND
        elif balance >= tiers["platinum_min"]:
            return IncentiveTier.PLATINUM
        elif balance >= tiers["gold_min"]:
            return IncentiveTier.GOLD
        elif balance >= tiers["silver_min"]:
            return IncentiveTier.SILVER
        else:
            return IncentiveTier.BRONZE

    # Reward Distribution

    async def distribute_compute_reward(
        self, participant_id: str, compute_hours: float, quality_metrics: dict[str, float]
    ) -> str:
        """Distribute rewards for compute contribution."""

        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self.participants[participant_id]

        # Get base reward rule
        rule = self.incentive_rules.get("compute_contribution")
        if not rule or not rule.enabled:
            raise ValueError("Compute contribution rewards not enabled")

        # Calculate base reward
        base_amount = int(rule.base_reward_amount * compute_hours)

        # Calculate bonuses
        bonus_amount = 0
        multipliers_applied = {}

        # Quality bonus
        quality_score = quality_metrics.get("quality_score", 100.0)
        if quality_score > 90:
            quality_bonus = (quality_score - 90) / 10 * rule.multiplier_rules.get("quality_bonus", 0)
            bonus_amount += int(base_amount * quality_bonus)
            multipliers_applied["quality_bonus"] = quality_bonus

        # Consistency bonus
        if participant.consecutive_days_active >= 7:
            consistency_bonus = rule.multiplier_rules.get("consistency_bonus", 0)
            bonus_amount += int(base_amount * consistency_bonus)
            multipliers_applied["consistency_bonus"] = consistency_bonus

        # Early adopter bonus
        total_participants = len(self.participants)
        if total_participants <= 1000:
            early_bonus = rule.multiplier_rules.get("early_adopter_bonus", 0)
            bonus_amount += int(base_amount * early_bonus)
            multipliers_applied["early_adopter_bonus"] = early_bonus

        # Tier multiplier
        tier_multiplier = rule.tier_multipliers.get(participant.current_tier, 1.0)
        tier_bonus = int((base_amount + bonus_amount) * (tier_multiplier - 1.0))
        multipliers_applied["tier_multiplier"] = tier_multiplier

        # Apply penalties
        penalty_amount = int((base_amount + bonus_amount + tier_bonus) * (1.0 - participant.penalty_multiplier))

        # Calculate final amount
        final_amount = base_amount + bonus_amount + tier_bonus - penalty_amount

        # Create reward distribution record
        distribution = RewardDistribution(
            distribution_id=str(uuid.uuid4()),
            participant_id=participant_id,
            event_type=EconomicEventType.COMPUTE_CONTRIBUTION,
            base_amount=base_amount,
            bonus_amount=bonus_amount + tier_bonus,
            penalty_amount=penalty_amount,
            final_amount=final_amount,
            rule_id=rule.rule_id,
            tier=participant.current_tier,
            multipliers_applied=multipliers_applied,
            timestamp=datetime.utcnow(),
        )

        # Distribute tokens
        success = await self.token_system.transfer(
            from_account="treasury",  # Treasury pays rewards
            to_account=participant.wallet_address,
            amount=float(self._from_wei(final_amount)),
            description=f"Compute contribution reward: {compute_hours}h",
        )

        if success:
            distribution.confirmed = True

            # Update participant stats
            participant.total_earned += final_amount
            participant.compute_hours_contributed += compute_hours
            participant.last_activity = datetime.utcnow()
            participant.quality_score = quality_score

            if participant.first_contribution is None:
                participant.first_contribution = datetime.utcnow()

            # Update tier
            account_info = self.token_system.get_account_balance(participant.wallet_address)
            if "balance" in account_info:
                new_tier = self._calculate_tier(account_info["balance"])
                if new_tier != participant.current_tier:
                    participant.current_tier = new_tier

            await self._save_participant(participant)

            logger.info(
                f"Distributed {self._from_wei(final_amount)} FOG to {participant_id} "
                f"for {compute_hours}h compute contribution"
            )

        # Save reward record
        self.reward_history.append(distribution)
        await self._save_reward_distribution(distribution)

        return distribution.distribution_id

    async def distribute_governance_reward(
        self, participant_id: str, governance_action: str, vote_power: int = 0
    ) -> str:
        """Distribute rewards for governance participation."""

        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self.participants[participant_id]
        rule = self.incentive_rules.get("governance_participation")

        if not rule or not rule.enabled:
            raise ValueError("Governance participation rewards not enabled")

        # Calculate reward amount
        base_amount = rule.base_reward_amount

        # Tier multiplier
        tier_multiplier = rule.tier_multipliers.get(participant.current_tier, 1.0)
        final_amount = int(base_amount * tier_multiplier)

        # Create reward distribution
        distribution = RewardDistribution(
            distribution_id=str(uuid.uuid4()),
            participant_id=participant_id,
            event_type=EconomicEventType.GOVERNANCE_PARTICIPATION,
            base_amount=base_amount,
            bonus_amount=final_amount - base_amount,
            penalty_amount=0,
            final_amount=final_amount,
            rule_id=rule.rule_id,
            tier=participant.current_tier,
            multipliers_applied={"tier_multiplier": tier_multiplier},
            timestamp=datetime.utcnow(),
        )

        # Distribute tokens
        success = await self.token_system.transfer(
            from_account="treasury",
            to_account=participant.wallet_address,
            amount=float(self._from_wei(final_amount)),
            description=f"Governance participation reward: {governance_action}",
        )

        if success:
            distribution.confirmed = True
            participant.total_earned += final_amount
            participant.last_activity = datetime.utcnow()
            await self._save_participant(participant)

        self.reward_history.append(distribution)
        await self._save_reward_distribution(distribution)

        logger.info(f"Distributed governance reward: {self._from_wei(final_amount)} FOG to {participant_id}")
        return distribution.distribution_id

    # Staking System

    async def stake_tokens(self, participant_id: str, amount: float, compound_enabled: bool = False) -> bool:
        """Stake tokens for a participant."""

        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")

        participant = self.participants[participant_id]

        # Stake tokens in token system
        success = await self.token_system.stake_tokens(participant.wallet_address, amount)

        if success:
            participant.total_staked += self._to_wei(amount)
            participant.staking_start_date = datetime.utcnow()
            participant.compound_staking_enabled = compound_enabled

            await self._save_participant(participant)

            await self._log_economic_event(
                EconomicEventType.STAKING_REWARD,
                participant_id,
                amount=self._to_wei(amount),
                details={"action": "stake", "compound": compound_enabled},
            )

            logger.info(f"Staked {amount} FOG for participant {participant_id}")

        return success

    async def calculate_staking_rewards(self, participant_id: str) -> int:
        """Calculate pending staking rewards for a participant."""

        if participant_id not in self.participants:
            return 0

        participant = self.participants[participant_id]

        if not participant.staking_start_date or participant.total_staked == 0:
            return 0

        # Get staked balance from token system
        account_info = self.token_system.get_account_balance(participant.wallet_address)
        if "error" in account_info:
            return 0

        staked_balance = self._to_wei(account_info["staked_balance"])

        # Calculate time staked
        days_staked = (datetime.utcnow() - participant.staking_start_date).days
        if days_staked < 1:
            return 0

        # Base APY
        base_apy = self.config["staking"]["base_apy"]

        # Tier bonus
        tier_bonus = self.config["staking"]["tier_apy_bonuses"].get(participant.current_tier.value, 0)

        # Total APY
        total_apy = base_apy + tier_bonus

        # Calculate daily rewards
        daily_rate = total_apy / 365
        daily_reward = int(staked_balance * daily_rate)

        # Total rewards for days staked
        total_rewards = daily_reward * days_staked

        return total_rewards

    # Background Tasks

    async def _reward_processor(self):
        """Background task to process pending reward distributions."""
        while self._running:
            try:
                # Process any pending economic events
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM economic_events WHERE processed = FALSE LIMIT 10")
                events = cursor.fetchall()

                for event in events:
                    event_type = EconomicEventType(event[1])
                    participant_id = event[2]
                    event[4]
                    details = json.loads(event[5]) if event[5] else {}

                    try:
                        if event_type == EconomicEventType.COMPUTE_CONTRIBUTION:
                            await self.distribute_compute_reward(
                                participant_id, details.get("compute_hours", 1.0), details.get("quality_metrics", {})
                            )
                        elif event_type == EconomicEventType.GOVERNANCE_PARTICIPATION:
                            await self.distribute_governance_reward(participant_id, details.get("action", "vote"))

                        # Mark as processed
                        cursor.execute("UPDATE economic_events SET processed = TRUE WHERE event_id = ?", (event[0],))
                        conn.commit()

                    except Exception as e:
                        logger.error(f"Error processing economic event {event[0]}: {e}")

                conn.close()
                await asyncio.sleep(60)  # Process every minute

            except Exception as e:
                logger.error(f"Error in reward processor: {e}")
                await asyncio.sleep(300)

    async def _staking_rewards_distributor(self):
        """Background task to distribute daily staking rewards."""
        while self._running:
            try:
                current_time = datetime.utcnow()

                for participant_id, participant in self.participants.items():
                    if participant.total_staked > 0 and participant.staking_start_date:
                        # Check if 24 hours passed since last reward
                        time_since_staking = current_time - participant.staking_start_date

                        if time_since_staking.total_seconds() >= 86400:  # 24 hours
                            rewards = await self.calculate_staking_rewards(participant_id)

                            if rewards > 0:
                                # Distribute staking rewards
                                success = await self.token_system.transfer(
                                    from_account="treasury",
                                    to_account=participant.wallet_address,
                                    amount=float(self._from_wei(rewards)),
                                    description="Daily staking rewards",
                                )

                                if success:
                                    participant.total_earned += rewards
                                    participant.staking_start_date = current_time  # Reset for next reward cycle
                                    await self._save_participant(participant)

                                    logger.info(
                                        f"Distributed staking rewards: {self._from_wei(rewards)} FOG to {participant_id}"
                                    )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in staking rewards distributor: {e}")
                await asyncio.sleep(1800)

    async def _tier_updater(self):
        """Background task to update participant tiers."""
        while self._running:
            try:
                for participant_id, participant in self.participants.items():
                    # Get current balance
                    account_info = self.token_system.get_account_balance(participant.wallet_address)
                    if "balance" in account_info:
                        new_tier = self._calculate_tier(account_info["balance"])

                        if new_tier != participant.current_tier:
                            old_tier = participant.current_tier
                            participant.current_tier = new_tier
                            await self._save_participant(participant)

                            logger.info(
                                f"Updated participant {participant_id} tier: {old_tier.value} -> {new_tier.value}"
                            )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in tier updater: {e}")
                await asyncio.sleep(1800)

    async def _economic_metrics_collector(self):
        """Background task to collect economic metrics."""
        while self._running:
            try:
                network_stats = self.token_system.get_network_stats()

                metrics = EconomicMetrics(
                    timestamp=datetime.utcnow(),
                    total_supply=int(network_stats["current_supply"]),
                    circulating_supply=int(network_stats["current_supply"] - network_stats["total_staked"]),
                    staked_supply=int(network_stats["total_staked"]),
                    burnt_tokens=0,  # Calculate from transaction fees
                    total_participants=len(self.participants),
                    active_nodes=network_stats["total_validators"],
                    compute_hours_contributed=sum(p.compute_hours_contributed for p in self.participants.values()),
                    staking_ratio=network_stats["total_staked"] / network_stats["current_supply"]
                    if network_stats["current_supply"] > 0
                    else 0,
                    staking_rewards_distributed=int(network_stats["total_rewards_distributed"]),
                )

                self.economic_metrics_history.append(metrics)
                await self._save_economic_metrics(metrics)

                # Keep only last 1000 entries
                if len(self.economic_metrics_history) > 1000:
                    self.economic_metrics_history = self.economic_metrics_history[-1000:]

                await asyncio.sleep(900)  # Collect every 15 minutes

            except Exception as e:
                logger.error(f"Error in economic metrics collector: {e}")
                await asyncio.sleep(300)

    async def _market_monitor(self):
        """Background task to monitor market conditions."""
        while self._running:
            try:
                # Mock market monitoring - in production would use real price feeds
                # Monitor for circuit breaker conditions, liquidity issues, etc.

                await asyncio.sleep(900)  # Check every 15 minutes

            except Exception as e:
                logger.error(f"Error in market monitor: {e}")
                await asyncio.sleep(300)

    async def _compliance_checker(self):
        """Background task to check economic compliance."""
        while self._running:
            try:
                # Check for unusual reward patterns
                recent_rewards = [r for r in self.reward_history if (datetime.utcnow() - r.timestamp).hours < 24]

                if recent_rewards:
                    total_distributed_24h = sum(r.final_amount for r in recent_rewards)

                    # Alert if daily distribution exceeds threshold
                    if total_distributed_24h > self._to_wei(100000):  # 100K FOG per day
                        logger.warning(
                            f"High reward distribution detected: {self._from_wei(total_distributed_24h)} FOG in 24h"
                        )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in compliance checker: {e}")
                await asyncio.sleep(300)

    # Database Operations

    async def _save_participant(self, participant: EconomicParticipant):
        """Save participant to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO participants
            (participant_id, wallet_address, total_earned, total_spent, total_staked, current_tier,
             compute_hours_contributed, storage_gb_contributed, bandwidth_gb_contributed,
             uptime_percentage, quality_score, first_contribution, last_activity, consecutive_days_active,
             staking_start_date, staking_duration_days, compound_staking_enabled,
             reputation_score, bonus_multiplier, penalty_multiplier, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                participant.participant_id,
                participant.wallet_address,
                participant.total_earned,
                participant.total_spent,
                participant.total_staked,
                participant.current_tier.value,
                participant.compute_hours_contributed,
                participant.storage_gb_contributed,
                participant.bandwidth_gb_contributed,
                participant.uptime_percentage,
                participant.quality_score,
                participant.first_contribution.isoformat() if participant.first_contribution else None,
                participant.last_activity.isoformat() if participant.last_activity else None,
                participant.consecutive_days_active,
                participant.staking_start_date.isoformat() if participant.staking_start_date else None,
                participant.staking_duration_days,
                participant.compound_staking_enabled,
                participant.reputation_score,
                participant.bonus_multiplier,
                participant.penalty_multiplier,
            ),
        )

        conn.commit()
        conn.close()

    async def _save_reward_distribution(self, distribution: RewardDistribution):
        """Save reward distribution to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO reward_distributions
            (distribution_id, participant_id, event_type, base_amount, bonus_amount, penalty_amount,
             final_amount, rule_id, tier, multipliers_applied, timestamp, transaction_id, confirmed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                distribution.distribution_id,
                distribution.participant_id,
                distribution.event_type.value,
                distribution.base_amount,
                distribution.bonus_amount,
                distribution.penalty_amount,
                distribution.final_amount,
                distribution.rule_id,
                distribution.tier.value,
                json.dumps(distribution.multipliers_applied),
                distribution.timestamp.isoformat(),
                distribution.transaction_id,
                distribution.confirmed,
            ),
        )

        conn.commit()
        conn.close()

    async def _save_economic_metrics(self, metrics: EconomicMetrics):
        """Save economic metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO economic_metrics
            (metric_id, timestamp, total_supply, circulating_supply, staked_supply, burnt_tokens,
             market_cap_usd, token_price_usd, trading_volume_24h, total_participants, active_nodes,
             compute_hours_contributed, revenue_generated, staking_ratio, avg_staking_duration_days,
             staking_rewards_distributed, governance_participation_rate, proposals_submitted,
             voting_power_distribution_gini)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(uuid.uuid4()),
                metrics.timestamp.isoformat(),
                metrics.total_supply,
                metrics.circulating_supply,
                metrics.staked_supply,
                metrics.burnt_tokens,
                metrics.market_cap_usd,
                metrics.token_price_usd,
                metrics.trading_volume_24h,
                metrics.total_participants,
                metrics.active_nodes,
                metrics.compute_hours_contributed,
                metrics.revenue_generated,
                metrics.staking_ratio,
                metrics.avg_staking_duration_days,
                metrics.staking_rewards_distributed,
                metrics.governance_participation_rate,
                metrics.proposals_submitted,
                metrics.voting_power_distribution_gini,
            ),
        )

        conn.commit()
        conn.close()

    async def _log_economic_event(
        self, event_type: EconomicEventType, participant_id: str, amount: int = 0, details: dict[str, Any] = None
    ):
        """Log economic event."""
        event_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO economic_events
            (event_id, event_type, participant_id, timestamp, amount, details, processed)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, FALSE)
        """,
            (event_id, event_type.value, participant_id, amount, json.dumps(details) if details else None),
        )

        conn.commit()
        conn.close()

    # Public Query Methods

    async def get_economic_summary(self) -> dict[str, Any]:
        """Get comprehensive economic system summary."""
        if not self.economic_metrics_history:
            return {}

        latest_metrics = self.economic_metrics_history[-1]
        self.token_system.get_network_stats()

        # Participant tier distribution
        tier_distribution = {}
        for tier in IncentiveTier:
            tier_distribution[tier.value] = len([p for p in self.participants.values() if p.current_tier == tier])

        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "token_metrics": {
                "total_supply": float(self._from_wei(latest_metrics.total_supply)),
                "circulating_supply": float(self._from_wei(latest_metrics.circulating_supply)),
                "staked_supply": float(self._from_wei(latest_metrics.staked_supply)),
                "staking_ratio": latest_metrics.staking_ratio,
            },
            "network_metrics": {
                "total_participants": latest_metrics.total_participants,
                "active_nodes": latest_metrics.active_nodes,
                "compute_hours_contributed": latest_metrics.compute_hours_contributed,
                "staking_rewards_distributed": float(self._from_wei(latest_metrics.staking_rewards_distributed)),
            },
            "participant_distribution": tier_distribution,
            "reward_statistics": {
                "total_rewards_24h": sum(
                    r.final_amount
                    for r in self.reward_history
                    if (datetime.utcnow() - r.timestamp).total_seconds() < 86400
                ),
                "avg_reward_per_participant": sum(p.total_earned for p in self.participants.values())
                / len(self.participants)
                if self.participants
                else 0,
            },
        }

    async def get_participant_profile(self, participant_id: str) -> dict[str, Any] | None:
        """Get detailed participant profile."""
        if participant_id not in self.participants:
            return None

        participant = self.participants[participant_id]
        account_info = self.token_system.get_account_balance(participant.wallet_address)

        return {
            "participant_id": participant.participant_id,
            "wallet_address": participant.wallet_address,
            "current_tier": participant.current_tier.value,
            "token_balance": account_info.get("balance", 0),
            "staked_balance": account_info.get("staked_balance", 0),
            "total_earned": float(self._from_wei(participant.total_earned)),
            "total_spent": float(self._from_wei(participant.total_spent)),
            "contributions": {
                "compute_hours": participant.compute_hours_contributed,
                "storage_gb": participant.storage_gb_contributed,
                "bandwidth_gb": participant.bandwidth_gb_contributed,
            },
            "performance": {
                "uptime_percentage": participant.uptime_percentage,
                "quality_score": participant.quality_score,
                "reputation_score": participant.reputation_score,
                "consecutive_days_active": participant.consecutive_days_active,
            },
            "staking": {
                "is_staking": participant.staking_start_date is not None,
                "staking_start_date": participant.staking_start_date.isoformat()
                if participant.staking_start_date
                else None,
                "compound_enabled": participant.compound_staking_enabled,
                "pending_rewards": float(self._from_wei(await self.calculate_staking_rewards(participant_id))),
            },
        }
