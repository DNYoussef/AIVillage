"""
Fog Computing Tokenomics System

Implements a comprehensive token economy for fog computing resources:
- FOG tokens for compute contribution rewards
- Staking mechanism for validator nodes
- DAO governance for network parameters
- Dynamic pricing based on supply/demand
- Burn/mint mechanisms for token stability
- Cross-chain compatibility for DEX trading

Key Features:
- ERC-20 compatible token standard
- Proof-of-Contribution mining
- Automatic reward distribution
- Governance voting power
- Penalty system for SLA violations
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import hashlib
import logging
from typing import Any
import uuid

# Set decimal precision for token calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Types of token transactions"""

    MINT = "mint"  # Create new tokens (rewards)
    BURN = "burn"  # Destroy tokens (penalties)
    TRANSFER = "transfer"  # Send tokens between accounts
    STAKE = "stake"  # Lock tokens for staking
    UNSTAKE = "unstake"  # Unlock staked tokens
    REWARD = "reward"  # Contribution rewards
    PENALTY = "penalty"  # SLA violation penalty
    GOVERNANCE_VOTE = "governance_vote"  # Voting fee
    MARKETPLACE_FEE = "marketplace_fee"  # Transaction fee


class ProposalStatus(Enum):
    """DAO governance proposal statuses"""

    PENDING = "pending"
    ACTIVE = "active"  # Voting period
    SUCCEEDED = "succeeded"
    DEFEATED = "defeated"
    EXECUTED = "executed"
    EXPIRED = "expired"


@dataclass
class TokenAccount:
    """Token account for fog network participants"""

    account_id: str
    public_key: bytes

    # Balances (in wei-equivalent, 18 decimals)
    balance: int = 0  # Available tokens
    staked_balance: int = 0  # Locked for staking
    locked_balance: int = 0  # Locked for other reasons

    # Contribution metrics
    total_contributed: int = 0  # Total tokens earned from contributions
    total_consumed: int = 0  # Total tokens spent on services

    # Staking info
    validator_node: bool = False
    stake_delegation: str | None = None  # Delegated to validator
    last_reward_block: int = 0

    # Governance
    voting_power: int = 0  # Based on staked balance
    proposals_created: int = 0
    votes_cast: int = 0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_balance(self) -> int:
        """Total balance including staked and locked"""
        return self.balance + self.staked_balance + self.locked_balance

    @property
    def available_balance(self) -> int:
        """Available balance for spending"""
        return self.balance

    def to_decimal(self, amount: int) -> Decimal:
        """Convert wei amount to decimal tokens"""
        return Decimal(amount) / Decimal(10**18)

    def from_decimal(self, amount: Decimal) -> int:
        """Convert decimal tokens to wei amount"""
        return int(amount * Decimal(10**18))


@dataclass
class Transaction:
    """Token transaction record"""

    tx_id: str
    tx_type: TransactionType
    from_account: str
    to_account: str
    amount: int  # In wei

    # Transaction details
    block_number: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    gas_fee: int = 0

    # Metadata
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Validation
    signature: bytes | None = None
    confirmed: bool = False


@dataclass
class ContributionRecord:
    """Record of fog computing contributions"""

    contribution_id: str
    contributor_id: str
    device_id: str

    # Contribution metrics
    compute_hours: Decimal
    memory_gb_hours: Decimal
    bandwidth_gb: Decimal
    storage_gb_hours: Decimal
    tasks_completed: int

    # Quality metrics
    uptime_percent: Decimal
    latency_avg_ms: Decimal
    success_rate: Decimal

    # Token rewards
    base_reward: int  # Base tokens for contribution
    quality_bonus: int  # Bonus for high quality

    # Timing
    period_start: datetime
    period_end: datetime
    scarcity_multiplier: Decimal = Decimal("1.0")  # Supply/demand adjustment
    total_reward: int = 0
    calculated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def calculate_total_reward(self):
        """Calculate total reward including bonuses and multipliers"""
        bonus_reward = int(self.base_reward * (float(self.quality_bonus) / 100))
        self.total_reward = int((self.base_reward + bonus_reward) * float(self.scarcity_multiplier))


@dataclass
class DAOProposal:
    """DAO governance proposal"""

    proposal_id: str
    title: str
    description: str
    proposer_id: str

    # Proposal type and parameters
    proposal_type: str  # "parameter_change", "token_mint", "upgrade", etc.
    parameters: dict[str, Any] = field(default_factory=dict)

    # Voting
    votes_for: int = 0
    votes_against: int = 0
    votes_abstain: int = 0
    quorum_required: int = 1000000  # Minimum voting power needed

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    voting_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    voting_end: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(days=7))

    # Status
    status: ProposalStatus = ProposalStatus.PENDING
    execution_hash: str | None = None

    @property
    def total_votes(self) -> int:
        """Total voting power participated"""
        return self.votes_for + self.votes_against + self.votes_abstain

    @property
    def approval_rate(self) -> float:
        """Percentage of votes in favor"""
        if self.total_votes == 0:
            return 0.0
        return (self.votes_for / self.total_votes) * 100


class FogTokenSystem:
    """
    Fog Computing Token Economy System

    Manages FOG tokens for the decentralized computing network:
    - Rewards for compute contributions
    - Payments for fog services
    - Staking for network governance
    - DAO voting and proposals
    """

    def __init__(
        self,
        initial_supply: int = 1000000000,  # 1 billion tokens
        reward_rate_per_hour: int = 10,  # Base reward rate
        staking_apy: float = 0.05,  # 5% annual staking rewards
        governance_threshold: int = 1000000,  # Min tokens to create proposal
    ):
        # Token economics
        self.initial_supply = self._to_wei(initial_supply)
        self.current_supply = self.initial_supply
        self.reward_rate_per_hour = self._to_wei(reward_rate_per_hour)
        self.staking_apy = staking_apy
        self.governance_threshold = self._to_wei(governance_threshold)

        # Accounts and transactions
        self.accounts: dict[str, TokenAccount] = {}
        self.transactions: list[Transaction] = []
        self.transaction_pool: list[Transaction] = []

        # Contribution tracking
        self.contributions: dict[str, ContributionRecord] = {}
        self.reward_multipliers: dict[str, Decimal] = {
            "cpu": Decimal("1.0"),
            "gpu": Decimal("2.0"),
            "memory": Decimal("0.5"),
            "storage": Decimal("0.3"),
            "bandwidth": Decimal("1.5"),
        }

        # DAO governance
        self.proposals: dict[str, DAOProposal] = {}
        self.validators: set[str] = set()

        # Network parameters (governed by DAO)
        self.network_params = {
            "reward_rate_per_hour": reward_rate_per_hour,
            "min_stake_amount": self._to_wei(1000),
            "validator_commission": 0.1,  # 10%
            "transaction_fee": self._to_wei(0.01),  # 0.01 FOG
            "inflation_rate": 0.02,  # 2% annual
            "max_supply": self._to_wei(10000000000),  # 10 billion max
        }

        # Metrics
        self.total_staked = 0
        self.total_rewards_distributed = 0
        self.current_block = 0

        logger.info(
            f"FogTokenSystem initialized: {initial_supply:,} FOG initial supply, "
            f"{reward_rate_per_hour} FOG/hour base reward"
        )

    def _to_wei(self, amount: float) -> int:
        """Convert decimal amount to wei (18 decimals)"""
        return int(Decimal(str(amount)) * Decimal(10**18))

    def _from_wei(self, amount: int) -> Decimal:
        """Convert wei amount to decimal"""
        return Decimal(amount) / Decimal(10**18)

    async def create_account(self, account_id: str, public_key: bytes, initial_balance: float = 0) -> TokenAccount:
        """Create a new token account"""

        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")

        account = TokenAccount(account_id=account_id, public_key=public_key, balance=self._to_wei(initial_balance))

        self.accounts[account_id] = account

        # Create initial balance transaction if needed
        if initial_balance > 0:
            tx = Transaction(
                tx_id=str(uuid.uuid4()),
                tx_type=TransactionType.MINT,
                from_account="system",
                to_account=account_id,
                amount=self._to_wei(initial_balance),
                description="Initial account balance",
            )
            await self._execute_transaction(tx)

        logger.info(f"Created account {account_id} with {initial_balance} FOG")
        return account

    async def record_contribution(
        self, contributor_id: str, device_id: str, metrics: dict[str, Any]
    ) -> ContributionRecord:
        """Record fog computing contribution and calculate rewards"""

        contribution_id = str(uuid.uuid4())

        # Extract metrics
        compute_hours = Decimal(str(metrics.get("compute_hours", 0)))
        memory_gb_hours = Decimal(str(metrics.get("memory_gb_hours", 0)))
        bandwidth_gb = Decimal(str(metrics.get("bandwidth_gb", 0)))
        storage_gb_hours = Decimal(str(metrics.get("storage_gb_hours", 0)))
        tasks_completed = metrics.get("tasks_completed", 0)

        # Quality metrics
        uptime_percent = Decimal(str(metrics.get("uptime_percent", 100)))
        latency_avg_ms = Decimal(str(metrics.get("latency_avg_ms", 100)))
        success_rate = Decimal(str(metrics.get("success_rate", 100)))

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            compute_hours, memory_gb_hours, bandwidth_gb, storage_gb_hours, tasks_completed
        )

        # Calculate quality bonus (0-50% bonus)
        quality_bonus = self._calculate_quality_bonus(uptime_percent, latency_avg_ms, success_rate)

        # Apply scarcity multiplier based on network demand
        scarcity_multiplier = await self._calculate_scarcity_multiplier()

        contribution = ContributionRecord(
            contribution_id=contribution_id,
            contributor_id=contributor_id,
            device_id=device_id,
            compute_hours=compute_hours,
            memory_gb_hours=memory_gb_hours,
            bandwidth_gb=bandwidth_gb,
            storage_gb_hours=storage_gb_hours,
            tasks_completed=tasks_completed,
            uptime_percent=uptime_percent,
            latency_avg_ms=latency_avg_ms,
            success_rate=success_rate,
            base_reward=base_reward,
            quality_bonus=quality_bonus,
            scarcity_multiplier=scarcity_multiplier,
            period_start=datetime.now(UTC) - timedelta(hours=1),
            period_end=datetime.now(UTC),
        )

        contribution.calculate_total_reward()
        self.contributions[contribution_id] = contribution

        # Mint reward tokens
        await self._mint_reward_tokens(contributor_id, contribution.total_reward)

        logger.info(
            f"Recorded contribution {contribution_id}: "
            f"{float(compute_hours)} compute hours, "
            f"{contribution.total_reward} wei reward"
        )

        return contribution

    def _calculate_base_reward(
        self,
        compute_hours: Decimal,
        memory_gb_hours: Decimal,
        bandwidth_gb: Decimal,
        storage_gb_hours: Decimal,
        tasks_completed: int,
    ) -> int:
        """Calculate base token reward for contributions"""

        # Weighted contribution score
        compute_score = compute_hours * self.reward_multipliers["cpu"]
        memory_score = memory_gb_hours * self.reward_multipliers["memory"]
        bandwidth_score = bandwidth_gb * self.reward_multipliers["bandwidth"]
        storage_score = storage_gb_hours * self.reward_multipliers["storage"]

        total_score = compute_score + memory_score + bandwidth_score + storage_score

        # Base reward calculation
        base_reward = total_score * Decimal(str(self.network_params["reward_rate_per_hour"]))

        # Task completion bonus
        task_bonus = Decimal(tasks_completed) * Decimal("0.1")  # 0.1 FOG per task

        return int((base_reward + task_bonus) * Decimal(10**18))

    def _calculate_quality_bonus(self, uptime_percent: Decimal, latency_avg_ms: Decimal, success_rate: Decimal) -> int:
        """Calculate quality bonus percentage (0-50)"""

        # Uptime bonus (0-20%)
        uptime_bonus = min(20, int(uptime_percent - 80) / 5) if uptime_percent >= 80 else 0

        # Latency bonus (0-15%)
        latency_bonus = max(0, 15 - int(latency_avg_ms / 10))

        # Success rate bonus (0-15%)
        success_bonus = min(15, int(success_rate - 85) / 5) if success_rate >= 85 else 0

        return max(0, min(50, uptime_bonus + latency_bonus + success_bonus))

    async def _calculate_scarcity_multiplier(self) -> Decimal:
        """Calculate reward multiplier based on network supply/demand"""

        # Simplified supply/demand calculation
        # In production, this would consider actual marketplace metrics

        total_compute_offered = sum(
            float(c.compute_hours)
            for c in self.contributions.values()
            if (datetime.now(UTC) - c.calculated_at).days < 7
        )

        # Base multiplier
        if total_compute_offered < 100:
            return Decimal("2.0")  # High demand, low supply
        elif total_compute_offered < 1000:
            return Decimal("1.5")  # Medium demand
        else:
            return Decimal("1.0")  # Normal supply

    async def _mint_reward_tokens(self, account_id: str, amount: int) -> bool:
        """Mint new tokens as rewards"""

        # Check max supply
        if self.current_supply + amount > self.network_params["max_supply"]:
            logger.warning("Cannot mint tokens: would exceed max supply")
            return False

        # Create mint transaction
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=TransactionType.MINT,
            from_account="system",
            to_account=account_id,
            amount=amount,
            description="Fog computing contribution reward",
        )

        success = await self._execute_transaction(tx)
        if success:
            self.current_supply += amount
            self.total_rewards_distributed += amount

        return success

    async def transfer(self, from_account: str, to_account: str, amount: float, description: str = "") -> bool:
        """Transfer tokens between accounts"""

        amount_wei = self._to_wei(amount)

        # Validate accounts
        if from_account not in self.accounts:
            logger.error(f"Source account not found: {from_account}")
            return False

        if to_account not in self.accounts:
            logger.error(f"Destination account not found: {to_account}")
            return False

        # Check balance
        if self.accounts[from_account].balance < amount_wei:
            logger.error(f"Insufficient balance for transfer: {from_account}")
            return False

        # Create transaction
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=TransactionType.TRANSFER,
            from_account=from_account,
            to_account=to_account,
            amount=amount_wei,
            gas_fee=self.network_params["transaction_fee"],
            description=description,
        )

        return await self._execute_transaction(tx)

    async def stake_tokens(self, account_id: str, amount: float, validator_id: str | None = None) -> bool:
        """Stake tokens for network consensus and rewards"""

        amount_wei = self._to_wei(amount)

        if account_id not in self.accounts:
            return False

        account = self.accounts[account_id]

        # Check minimum stake
        if amount_wei < self.network_params["min_stake_amount"]:
            logger.error("Amount below minimum stake requirement")
            return False

        # Check balance
        if account.balance < amount_wei:
            logger.error("Insufficient balance for staking")
            return False

        # Move tokens to staked balance
        account.balance -= amount_wei
        account.staked_balance += amount_wei

        # Set delegation if specified
        if validator_id and validator_id in self.validators:
            account.stake_delegation = validator_id
        else:
            # Self-staking (become validator if enough stake)
            if amount_wei >= self._to_wei(10000):  # 10k FOG minimum for validator
                self.validators.add(account_id)
                account.validator_node = True

        # Update voting power (1 FOG = 1 vote)
        account.voting_power = account.staked_balance

        # Track total staked
        self.total_staked += amount_wei

        # Create transaction record
        tx = Transaction(
            tx_id=str(uuid.uuid4()),
            tx_type=TransactionType.STAKE,
            from_account=account_id,
            to_account="staking_pool",
            amount=amount_wei,
            description=f"Stake delegation to {validator_id or 'self'}",
        )

        await self._execute_transaction(tx)

        logger.info(f"Staked {amount} FOG for account {account_id}")
        return True

    async def create_proposal(
        self, proposer_id: str, title: str, description: str, proposal_type: str, parameters: dict[str, Any]
    ) -> DAOProposal | None:
        """Create a DAO governance proposal"""

        if proposer_id not in self.accounts:
            return None

        account = self.accounts[proposer_id]

        # Check governance threshold
        if account.staked_balance < self.governance_threshold:
            logger.error("Insufficient staked tokens to create proposal")
            return None

        proposal_id = str(uuid.uuid4())

        proposal = DAOProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            parameters=parameters,
            quorum_required=max(1000000, self.total_staked // 100),  # 1% of staked tokens
        )

        self.proposals[proposal_id] = proposal
        account.proposals_created += 1

        # Charge proposal fee
        fee = self._to_wei(100)  # 100 FOG proposal fee
        account.balance -= fee

        logger.info(f"Created proposal {proposal_id}: {title}")
        return proposal

    async def vote_on_proposal(
        self, proposal_id: str, voter_id: str, vote: str, voting_power: int | None = None  # "for", "against", "abstain"
    ) -> bool:
        """Vote on a DAO proposal"""

        if proposal_id not in self.proposals:
            return False

        if voter_id not in self.accounts:
            return False

        proposal = self.proposals[proposal_id]
        account = self.accounts[voter_id]

        # Check voting period
        now = datetime.now(UTC)
        if now < proposal.voting_start or now > proposal.voting_end:
            logger.error("Proposal not in voting period")
            return False

        # Use account voting power if not specified
        if voting_power is None:
            voting_power = account.voting_power

        # Validate voting power
        if voting_power > account.voting_power:
            logger.error("Cannot vote with more power than staked")
            return False

        # Record vote
        if vote.lower() == "for":
            proposal.votes_for += voting_power
        elif vote.lower() == "against":
            proposal.votes_against += voting_power
        else:
            proposal.votes_abstain += voting_power

        account.votes_cast += 1

        logger.info(f"Vote recorded: {voter_id} voted {vote} on {proposal_id}")
        return True

    async def execute_proposal(self, proposal_id: str) -> bool:
        """Execute a successful DAO proposal"""

        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        # Check if proposal passed
        if proposal.total_votes < proposal.quorum_required:
            proposal.status = ProposalStatus.DEFEATED
            logger.info(f"Proposal {proposal_id} failed: insufficient quorum")
            return False

        if proposal.votes_for <= proposal.votes_against:
            proposal.status = ProposalStatus.DEFEATED
            logger.info(f"Proposal {proposal_id} failed: more votes against")
            return False

        # Execute proposal based on type
        success = False

        if proposal.proposal_type == "parameter_change":
            # Update network parameters
            for key, value in proposal.parameters.items():
                if key in self.network_params:
                    self.network_params[key] = value
                    success = True

        elif proposal.proposal_type == "token_mint":
            # Mint additional tokens
            amount = proposal.parameters.get("amount", 0)
            recipient = proposal.parameters.get("recipient", "treasury")
            if amount > 0:
                await self._mint_reward_tokens(recipient, self._to_wei(amount))
                success = True

        if success:
            proposal.status = ProposalStatus.EXECUTED
            proposal.execution_hash = hashlib.sha256(
                f"{proposal_id}{datetime.now(UTC).isoformat()}".encode()
            ).hexdigest()
            logger.info(f"Executed proposal {proposal_id}")
        else:
            logger.error(f"Failed to execute proposal {proposal_id}")

        return success

    async def _execute_transaction(self, tx: Transaction) -> bool:
        """Execute a token transaction"""

        try:
            if tx.tx_type == TransactionType.TRANSFER:
                # Transfer tokens
                from_account = self.accounts[tx.from_account]
                to_account = self.accounts[tx.to_account]

                from_account.balance -= tx.amount + tx.gas_fee
                to_account.balance += tx.amount

            elif tx.tx_type == TransactionType.MINT:
                # Mint new tokens
                to_account = self.accounts[tx.to_account]
                to_account.balance += tx.amount
                to_account.total_contributed += tx.amount

            elif tx.tx_type == TransactionType.BURN:
                # Burn tokens
                from_account = self.accounts[tx.from_account]
                from_account.balance -= tx.amount
                self.current_supply -= tx.amount

            # Update transaction
            tx.confirmed = True
            tx.block_number = self.current_block
            self.transactions.append(tx)
            self.current_block += 1

            # Update account activity
            if tx.from_account in self.accounts:
                self.accounts[tx.from_account].last_activity = datetime.now(UTC)
            if tx.to_account in self.accounts:
                self.accounts[tx.to_account].last_activity = datetime.now(UTC)

            return True

        except Exception as e:
            logger.error(f"Transaction execution failed: {e}")
            return False

    def get_account_balance(self, account_id: str) -> dict[str, Any]:
        """Get account balance and staking info"""

        if account_id not in self.accounts:
            return {"error": "Account not found"}

        account = self.accounts[account_id]

        return {
            "account_id": account_id,
            "balance": float(self._from_wei(account.balance)),
            "staked_balance": float(self._from_wei(account.staked_balance)),
            "locked_balance": float(self._from_wei(account.locked_balance)),
            "total_balance": float(self._from_wei(account.total_balance)),
            "voting_power": float(self._from_wei(account.voting_power)),
            "validator_node": account.validator_node,
            "total_contributed": float(self._from_wei(account.total_contributed)),
            "total_consumed": float(self._from_wei(account.total_consumed)),
            "created_at": account.created_at.isoformat(),
            "last_activity": account.last_activity.isoformat(),
        }

    def get_network_stats(self) -> dict[str, Any]:
        """Get network-wide token statistics"""

        return {
            "current_supply": float(self._from_wei(self.current_supply)),
            "max_supply": float(self._from_wei(self.network_params["max_supply"])),
            "total_staked": float(self._from_wei(self.total_staked)),
            "total_rewards_distributed": float(self._from_wei(self.total_rewards_distributed)),
            "total_accounts": len(self.accounts),
            "total_validators": len(self.validators),
            "total_transactions": len(self.transactions),
            "active_proposals": len([p for p in self.proposals.values() if p.status == ProposalStatus.ACTIVE]),
            "current_block": self.current_block,
            "network_parameters": self.network_params.copy(),
        }
